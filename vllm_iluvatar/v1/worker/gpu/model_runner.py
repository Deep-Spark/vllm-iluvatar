# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/model_runner.py

Iluvatar GPU model runner customizations:

* ``IluvatarGPUModelRunner(GPUModelRunner)`` -- overrides
  ``initialize_kv_cache`` / ``update_requests`` / ``shutdown`` to allocate,
  bind, copy-on-write, and clear the ILU int8 KV-cache *scale* buffers that
  live alongside the community KV caches (flash KINT8/KVINT8 separate per-layer
  V + K-scale, and MLA-INT8 tail-carved scales). Active when
  ``VLLM_ILU_FLASH_ATTN_QUANT_TYPE`` / ``VLLM_ILU_MLA_ATTN_QUANT_TYPE`` is set.

The V2 PP + MTP draft-token relay that previously lived here (runner init /
``update_pp_decode_requests`` / ``sample_tokens`` wrappers wired by
``patch_v2_pp_mtp.py``) has been retired: upstream vLLM now carries equivalent
V2 PP + MTP support (``PPHandler.receive``/``broadcast``/``broadcast_drafts``
and ``GPUModelRunner.update_pp_decode_requests``/``sample_tokens``), so the
Iluvatar plugin no longer overrides those paths.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.worker import get_aux_output_connector
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hisparse.binding import init_hisparse_kv_cache
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.attn_utils import (
    FastPrefillHelper,
    get_shared_kv_cache_layers,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    ModelCudaGraphManager,
    has_compiled_submodule,
)
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR, get_kv_connector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import get_max_chunk_logits
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.ubatch_utils import maybe_build_ubatch_runner
from vllm.v1.worker.utils import (
    allocate_kv_cache,
    bind_kv_cache_to_layers,
    copy_kv_cache_blocks_inplace,
)

logger = init_logger(__name__)

# Iluvatar GPU model runner customizations.
#
# ``IluvatarGPUModelRunner(GPUModelRunner)`` -- subclass overriding KV init
# (optional int8 scale caches). Swapped in by
# ``IluWorker._try_replace_model_runner_class`` for quant KV runs only.
# See vllm_iluvatar/v1/worker/gpu_worker.py.


def _recover_int8_buf(kv_caches: dict[str, torch.Tensor], device: torch.device):
    """Recover the shared int8 backing buffer from any KV view's storage."""
    first_view = next(iter(kv_caches.values()))
    storage = first_view.untyped_storage()
    buf_int8 = torch.empty(0, dtype=torch.int8, device=device)
    buf_int8.set_(storage, 0, (storage.nbytes(),), (1,))
    return buf_int8


def allocate_kv_cache_scale(
    kv_caches: dict[str, torch.Tensor],
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int] | None = None,
    vllm_config: VllmConfig | None = None,
) -> dict[str, list[torch.Tensor | None]]:
    """Allocate per-layer K-scale (and V for KINT8) buffers.

    Two modes, selected per packed ``KVCacheTensor``:

    * KINT8 (flash): the V2 backing buffer holds K ONLY (dense
      ``[B, H, N, head_size]`` int8, ``state_content_size_bytes = head_size``),
      so K is already dense/contiguous and matches the ILU KINT8 kernels'
      fixed-dense stride assumption. V is NOT interleaved into C (that
      interleaving made the K view non-contiguous with token stride =
      head_size + 2*head_size_v, mismatching the kernels and producing
      garbage decode). Instead allocate a FRESH dense bf16 V buffer
      ``[num_blocks, H, block_size, head_size_v]`` and a FRESH dense float32
      K-scale buffer ``[num_blocks, H, block_size]`` PER LAYER. Returns
      ``{layer_name -> [k_scale_buffer, v_buffer]}`` (slot 0 = K-scale,
      slot 1 = V); ``_forward_k_int8`` reads them as ``key_scale_cache`` /
      ``value_cache``. Nothing is carved from the shared backing buffer.

    * KVINT8 / MLA-INT8: carve per-layer K-scale views out of the SHARED KV
      backing buffer (B2), two placement modes matching
      ``get_kv_cache_config_from_groups``:
        - Block-outermost (co-located): every block carries the spec-region's
          scale pages right after its KV pages (``block_stride`` includes
          scale); layer ``l``'s scale in block ``b`` sits at
          ``scale_offset + b * block_stride + l * scale_page`` (strided
          float32 ``[num_blocks, H, block_size]``).
        - Layer-outermost (tail): scale pages appended as a contiguous tail;
          each layer's scale is a contiguous float32
          ``[num_blocks, H, block_size]`` slice of the tail.
      The fixed-size region (KVINT8 static V scale) is appended once after
      the per-block / KV region and carved PER-LAYER as a contiguous int8
      slice. Returns ``{layer_name -> [scale_view, fixed_view]}`` (slot 0 =
      K-scale, slot 1 = fixed V-scale).
    """
    kv_scale_caches: dict[str, list[torch.Tensor | None]] = {}

    tensors = kv_cache_config.kv_cache_tensors
    if not tensors or not kv_caches:
        return kv_scale_caches

    num_blocks = kv_cache_config.num_blocks

    # Flash int8 (KINT8 / KVINT8): the V2 backing buffer holds K ONLY (dense
    # int8); V, K-scale (per-token), and -- for KVINT8 -- V-scale (static) are
    # allocated as SEPARATE dense buffers per layer here. Do NOT touch the
    # shared backing buffer. Non-flash-int8 tensors fall through to the
    # shared-carve logic below.
    flash_int8_tensors = [t for t in tensors if getattr(t, "is_flash_int8", False)]
    if flash_int8_tensors:
        import vllm.envs as _envs

        quant = _envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE
        # KINT8: V keeps the model dtype (bf16/fp16); no V-scale.
        # KVINT8: V is int8; plus a static per-head V-scale (float32).
        if quant == "KINT8":
            v_dtype = (
                vllm_config.model_config.dtype
                if vllm_config is not None
                else torch.bfloat16
            )
        else:  # KVINT8
            v_dtype = torch.int8
        for tensor in flash_int8_tensors:
            v_head_size = getattr(tensor, "v_head_size", 0) or 0
            # KVINT8: load the calibrated per-head V-scale from the file once per
            # tensor group. The file keys are "{layer_idx}" (or "mtp.{layer_idx}");
            # load_static_v_scales already slices per TP rank. Without this the
            # V-scale buffer stays all-zeros and the decode kernel dequantizes V
            # to 0, producing garbage output (layout LBHNC is already correct).
            v_scales_map: dict[str, torch.Tensor] | None = None
            if quant == "KVINT8":
                from vllm_iluvatar.v1.attention.ilu_static_quant import (
                    load_static_v_scales,
                )

                _first_view = kv_caches[next(iter(tensor.layers))]
                v_scales_map = load_static_v_scales(
                    num_kv_heads=_first_view.shape[1],
                    head_size=v_head_size,
                    device=device,
                )
            for layer_name in tensor.layers:
                kv_view = kv_caches[layer_name]
                _nb, _h, _bs = kv_view.shape[0], kv_view.shape[1], kv_view.shape[2]
                # K-scale: [num_blocks, H, block_size] float32 (one scale per
                # (block, head, token)); V: [num_blocks, H, block_size, v_head].
                # Both dense/contiguous -- the ILU int8 kernels require it.
                k_scale = torch.zeros(
                    (_nb, _h, _bs), dtype=torch.float32, device=device
                )
                v_buf = torch.zeros(
                    (_nb, _h, _bs, v_head_size), dtype=v_dtype, device=device
                )
                # KVINT8 static V-scale: [num_kv_heads, v_head_size] float32
                # (one per head, not per token), loaded from the calibration
                # file. None for KINT8.
                v_scale = None
                if quant == "KVINT8":
                    layer_idx = extract_layer_index(layer_name)
                    key = f"mtp.{layer_idx}" if "mtp" in layer_name else f"{layer_idx}"
                    # MTP draft layers are not in the calibrated scale file
                    # (keys are the base layers "0".."N"). Reuse the last base
                    # layer, matching the 0.27 KVINT8 fallback.
                    if key not in v_scales_map:
                        if "mtp" not in layer_name:
                            raise KeyError(
                                f"KVINT8 v_scale file has no entry for layer {key}"
                            )
                        base_layer_keys = [
                            scale_key
                            for scale_key in v_scales_map
                            if str(scale_key).isdecimal()
                        ]
                        if not base_layer_keys:
                            raise RuntimeError(
                                "No numeric base-layer key found in the KVINT8 V scale file"
                            )
                        fallback_idx = max(base_layer_keys, key=int)
                        logger.warning(
                            "[KVINT8] layer_idx=%s not in v_scale file, "
                            "fallback to layer_idx:%s",
                            key,
                            fallback_idx,
                        )
                        key = fallback_idx
                    v_scale = v_scales_map[key]
                    assert v_scale.shape == (_h, v_head_size), (
                        f"KVINT8 v_scale for {key}: shape {tuple(v_scale.shape)} "
                        f"!= ({_h}, {v_head_size})"
                    )
                # slot 0 = K-scale, slot 1 = V, slot 2 = V-scale (KVINT8 only).
                kv_scale_caches[layer_name] = [k_scale, v_buf, v_scale]

    non_flash_int8_tensors = [
        t for t in tensors if not getattr(t, "is_flash_int8", False)
    ]
    if not non_flash_int8_tensors:
        return kv_scale_caches

    tensors = non_flash_int8_tensors
    scale_sizes = [getattr(t, "scale_size", 0) or 0 for t in tensors]
    if sum(scale_sizes) == 0:
        return kv_scale_caches

    buf_int8 = _recover_int8_buf(kv_caches, device)

    if sum(scale_sizes) > 0:
        assert buf_int8.numel() % 4 == 0, (
            "ILU int8 scale: backing buffer size must be 4-byte aligned for "
            f"float32 scale views, got {buf_int8.numel()} bytes."
        )
    buf_f32 = buf_int8.view(torch.float32)

    # Scale lives in a contiguous tail (not co-located inside blocks); carve a
    # per-layer contiguous float32 [num_blocks, H, block_size] slice out of it.
    sizes_set = {t.size for t in tensors}
    assert len(sizes_set) == 1
    total_size = sizes_set.pop()
    scale_total = sum(scale_sizes)
    scale_off = total_size - scale_total
    for tensor in tensors:
        s = getattr(tensor, "scale_size", 0) or 0
        num_layers = len(tensor.layers)
        scale_page = s // (num_blocks * num_layers) if s else 0
        for li, layer_name in enumerate(tensor.layers):
            sv: torch.Tensor | None = None
            if s:
                layer_off = scale_off + li * scale_page * num_blocks
                assert layer_off % 4 == 0, (
                    "ILU int8 scale: tail scale offset must be 4-byte "
                    f"aligned, got {layer_off}."
                )
                sv = torch.as_strided(
                    buf_f32,
                    (num_blocks, scale_page // 4),
                    (scale_page // 4, 1),
                    storage_offset=layer_off // 4,
                )
                # Reshape the contiguous 2D [num_blocks, per_block_floats] tail
                # slice to the 3D layout the consuming kernel expects.
                # Flash int8 (KINT8/KVINT8) uses a SEPARATE buffer (above), so
                # this tail carve is MLA-INT8 only in practice. MLA stores TWO
                # adjacent float32 scales per token (kc_scale + kpe_scale); the
                # write kernel (vllm_concat_and_cache_mla_int8) indexes them as
                # block_offset * scale_dim_stride + {0,1}, so scale_dim_stride
                # (= stride(1)) must be 2 -> shape [num_blocks, block_size, 2].
                # Generalize via scales_per_token = (scale_page//4) / block_size
                # so any per-token scale count is handled.
                kv_view = kv_caches[layer_name]
                block_size = kv_view.shape[2]
                scales_per_token = (scale_page // 4) // block_size
                sv = sv.view(kv_view.shape[0], block_size, scales_per_token)
            kv_scale_caches[layer_name] = [sv]
        scale_off += s

    return kv_scale_caches


def bind_kv_cache_scale(
    kv_scale_caches: dict[str, list[torch.Tensor | None]],
    forward_context: dict[str, Any],
    runner_kv_scale_caches: list[torch.Tensor | list[torch.Tensor]],
    num_attn_module: int,
) -> None:
    """Bind ILU KV-cache scale tensors to the runner list and attention layers.

    Mirrors the community ``bind_kv_cache`` ordering (layer-index order, with
    ``num_attn_module`` for dual-attention models) so the runner's flat scale
    list stays aligned with ``runner_kv_caches``. Each attention layer receives
    its ``[k_scale, v_scale]`` pair via the ``kv_cache_scale`` attribute that
    the ILU attention backends (ilu_flash_attn / mla_attention) read at forward
    time.
    """
    assert len(runner_kv_scale_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_scale_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # Same multi-layer-in-one-block case as community bind_kv_cache
            # (e.g. encoder-decoder cross/self attention sharing an index).
            current_platform.check_runner_kv_caches_multi_layer()
        for layer_name in layer_names:
            runner_kv_scale_caches.append(kv_scale_caches[layer_name])

    # Bind kv_scale_caches to forward context. Each layer's bind_kv_cache_scale
    # stores its [k_scale, v_scale] views the way bind_kv_cache stores the KV
    # view; both Attention (patch_attention) and MambaBase (patch_mamba_abstract)
    # expose the hook, so the call is uniform across hybrid forward_context.
    for layer_name, kv_scale_cache in kv_scale_caches.items():
        forward_context[layer_name].bind_kv_cache_scale(kv_scale_cache)


def init_kv_cache(
    runner_kv_scale_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    kv_cache_allocation_context: AbstractContextManager | None = None,
    *,
    block_tables: BlockTables | None = None,
) -> dict[str, Any]:
    allocation_context = kv_cache_allocation_context or nullcontext()
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    with allocation_context:
        if vllm_config.attention_config.hisparse_config is not None:
            assert block_tables is not None
            kv_caches = init_hisparse_kv_cache(
                kv_cache_config,
                device,
                kernel_block_sizes,
                vllm_config,
                forward_context,
                block_tables,
            )
        else:
            kv_caches = allocate_kv_cache(
                kv_cache_config,
                device,
                layout,
                kernel_block_sizes,
            )
        kv_scale_caches = allocate_kv_cache_scale(
            kv_caches,
            kv_cache_config,
            device,
            layout,
            kernel_block_sizes,
            vllm_config=vllm_config,
        )

    for layer_name, target in get_shared_kv_cache_layers(vllm_config).items():
        kv_caches[layer_name] = kv_caches[target]
        if target in kv_scale_caches:
            kv_scale_caches[layer_name] = kv_scale_caches[target]

    num_attn_module = (
        2
        if vllm_config.model_config.hf_config.model_type
        in ("longcat_flash", "longcat_flash_ngram")
        else 1
    )
    bindable_caches = {
        name: cache for name, cache in kv_caches.items() if name in forward_context
    }
    bind_kv_cache_to_layers(
        bindable_caches,
        forward_context,
        num_attn_module,
        kv_cache_groups=kv_cache_config.kv_cache_groups,
    )
    bind_kv_cache_scale(
        kv_scale_caches, forward_context, runner_kv_scale_caches, num_attn_module
    )
    return kv_caches


class IluvatarGPUModelRunner(GPUModelRunner):
    # [ILU override] initialize_kv_cache
    # Delta: correct the upstream two-plane stride for ILU's three-plane KINT8
    # cache after delegating the common hybrid-layout handling to vLLM.
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # GPUWorker finalizes the PD interleave before KV cache initialization.
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        dcp_sharded = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
            dcp_sharded.append(spec.dcp_sharded)
            # Let each cache type account for CP. Attention KV is DCP-sharded,
            # while Mamba/GDN recurrent state is replicated across DCP ranks.
            max_num_blocks = spec.max_num_blocks_per_req(
                self.vllm_config, block_table_max_model_len
            )
            # Preserve each cache type's alignment requirements after applying
            # its topology-aware block-table width.
            max_num_blocks = get_block_table_width(
                max_num_blocks,
                spec.block_size,
                token_alignment=spec.block_table_token_alignment,
            )
            max_num_blocks_per_group.append(max_num_blocks)

        target_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            # Adaptive verification validates target attention separately.
            target_attn_layer_names = {
                layer_name
                for group in self.kv_cache_config.kv_cache_groups
                for layer_name in group.layer_names
            } - self.speculator.draft_attn_layer_names
        draft_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            draft_attn_layer_names = self.speculator.draft_attn_layer_names
        # Metadata builders select attention kernels that need JIT warmup.
        with self.jit_warmup_registry.activate():
            (
                self.attn_groups,
                attn_cg_support,
                self.kernel_block_sizes,
            ) = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
                draft_layer_names=draft_attn_layer_names,
            )
        additional_attn_cg_support = self.model_state.get_additional_cg_support()
        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
        # The speculator clears the flag at load time when the checkpoint has
        # no confidence head, so it holds the effective value.
        self.adaptive_verification = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=getattr(
                self.speculator, "enable_adaptive_verification", False
            ),
            attn_groups=self.attn_groups,
            req_states=self.req_states,
            query_start_loc=self.input_buffers.query_start_loc,
            num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            max_total_logits=get_max_chunk_logits(self.vocab_size),
            vllm_config=self.vllm_config,
            target_layer_names=target_attn_layer_names,
            additional_attn_cg_support=additional_attn_cg_support,
        )

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            slot_mapping_enabled=slot_mapping_enabled,
            dcp_sharded=dcp_sharded,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        self.ubatch_runner = maybe_build_ubatch_runner(
            self.vllm_config,
            self.device,
            self.model_state,
            self.attn_groups,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        if self.speculator is not None:
            self.speculator.pcp_manager = self.pcp_manager
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config,
            self.kv_cache_config,
            use_replayssm=self.cache_config.use_replayssm,
        )
        piecewise_capture_available = bool(
            envs.VLLM_USE_BREAKABLE_CUDAGRAPH or has_compiled_submodule(self.model)
        )
        if self.adaptive_verification is not None:
            self.compilation_config.cudagraph_mode = resolve_adaptive_cudagraph_mode(
                self.compilation_config.cudagraph_mode,
                piecewise_capture_available=piecewise_capture_available,
            )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
            piecewise_capture_available=piecewise_capture_available,
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
            varlen_decode=self.adaptive_verification is not None,
            ubatch_runner=self.ubatch_runner,
        )
        if self.cache_config.kv_sharing_fast_prefill and self.pcp_manager is None:
            self.fast_prefill = FastPrefillHelper(
                self.cudagraph_manager, self.max_num_tokens
            )
        check_attention_cp_compatibility(self.vllm_config, target_attn_layer_names)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        # Capture warmup providers that depend on allocated KV-cache strides.
        self.kv_scale_caches: list[torch.Tensor | list[torch.Tensor]] = []
        with self.jit_warmup_registry.activate():
            kv_caches_dict = init_kv_cache(
                self.kv_scale_caches,
                self.compilation_config.static_forward_context,
                self.kv_cache_config,
                self.device,
                self.kernel_block_sizes,
                self.vllm_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
                block_tables=self.block_tables,
            )
        self.kv_caches = [
            cache for cache in kv_caches_dict.values() if cache.device == self.device
        ]
        if is_profiling:
            self.kv_connector = NO_OP_KV_CONNECTOR
        else:
            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

            # AuxOutput connector requires resolved kv_cache_config.
            if self.vllm_config.aux_output_config.enabled:
                self.aux_output_connector = get_aux_output_connector(
                    self.model, self.vllm_config, kv_cache_config
                )

        ilu_int8_quant = (
            envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8")
            or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8"
        )
        if ilu_int8_quant and self.kv_connector is not NO_OP_KV_CONNECTOR:
            raise NotImplementedError(
                "ILU int8 KV-cache quantization is not supported with a "
                "kv_connector because scale buffers are not transferred"
            )

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # [ILU override] delegate to community update_requests (which does the
        # V2 kv_caches copy-on-write via copy_kv_cache_blocks_inplace), then
        # additionally copy-on-write the ILU KINT8 SEPARATE buffers (V + K-scale,
        # allocated per layer by allocate_kv_cache_scale and stored in
        # self.kv_scale_caches). The V2 copy only moves K (KINT8 stores K alone
        # in the V2 buffer); V and K-scale live outside the V2 buffer and would
        # otherwise keep stale data on a prefix-cache block copy, corrupting
        # dequant/attention for the copied prefix tokens.
        super().update_requests(scheduler_output)
        if scheduler_output.kv_cache_block_copies:
            self._ilu_copy_kv_cache_scale_blocks(scheduler_output.kv_cache_block_copies)

    def _ilu_copy_kv_cache_scale_blocks(self, block_copies) -> None:
        # Copy manager blocks for every flash-int8 (KINT8/KVINT8) SEPARATE
        # per-block buffer in self.kv_scale_caches. Each entry is
        # [k_scale, v_buf, v_scale]: k_scale is [num_kernel_blocks, H, bs]
        # float32 (per-token), v_buf is [num_kernel_blocks, H, bs, v_head]
        # (bf16 for KINT8 / int8 for KVINT8), both per-block and must be
        # copy-on-written with the K (V2) blocks. v_scale (KVINT8 only) is a
        # STATIC per-head [H, v_head] float32 -- NOT per-block, must NOT be
        # copied (it is identical across blocks). Non-flash-int8 entries
        # ([scale_view, fixed_view]) are carved out of the V2 backing buffer
        # and are already moved by the community
        # copy_kv_cache_blocks_inplace(self.kv_caches, ...) call, so skip them.
        num_kernel_blocks = self.kv_caches[0].shape[0] if self.kv_caches else 0
        scale_bufs: list[torch.Tensor] = []
        for entry in self.kv_scale_caches:
            if not isinstance(entry, (list, tuple)):
                continue
            for t in entry:
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                # Skip V2-backed views (already copied via self.kv_caches).
                if self._ilu_shares_v2_storage(t):
                    continue
                # Skip static per-head V-scale (2D, not per-block) -- it is
                # identical across blocks and must not be block-copied.
                if t.ndim < 3 or t.shape[0] != num_kernel_blocks:
                    continue
                scale_bufs.append(t)
        if not scale_bufs:
            return
        copy_kv_cache_blocks_inplace(
            scale_bufs, self.kv_cache_config.num_blocks, block_copies
        )

    def _ilu_shares_v2_storage(self, t: torch.Tensor) -> bool:
        if not self.kv_caches:
            return False
        v2_ptr = self.kv_caches[0].untyped_storage().data_ptr()
        try:
            return t.untyped_storage().data_ptr() == v2_ptr
        except Exception:
            return False

    def shutdown(self) -> None:
        # super().shutdown() is the community V2 shutdown: it clears
        # self.kv_caches and calls clear_layer_kv_caches(...) (which the ILU
        # patch extends to also detach layer.kv_cache_scale). This override
        # only adds the runner-level scale list clear (parallel to
        # self.kv_caches), created by the ILU scale wiring.
        super().shutdown()
        if hasattr(self, "kv_scale_caches"):
            self.kv_scale_caches.clear()


def make_pcp_pp_sample_tokens(original_sample_tokens):
    """Use request rows, rather than PCP segment rows, for PP state handoff."""

    @wraps(original_sample_tokens)
    def sample_tokens(self, grammar_output):
        state = self.execute_model_state
        manager = self.pcp_manager
        if state is not None and manager is not None and not self.is_last_pp_rank:
            # Forward and PP p2p have already consumed the local, padded token
            # slice. The non-last PP branch only receives sampled tokens and
            # advances request state; both must use the same global batch as
            # the last PP stage's sampler. Local DualChunkSwap rows can repeat
            # a request index or omit a request entirely on an empty rank.
            # No hidden-state gather is needed on this stage.
            global_batch = manager._global_batch
            assert global_batch is not None
            self.execute_model_state = state._replace(input_batch=global_batch)
        return original_sample_tokens(self, grammar_output)

    return sample_tokens
