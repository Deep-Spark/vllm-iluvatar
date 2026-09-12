# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-Runner-V2 ``ModelState`` for Qwen3.8-Flash-Next (``qwen4_exp``).

Qwen4Exp carries the PLE n-gram token history and the partial sum of the QSA
compression group as per-request state. The PLE dilated-conv window is instead
stored in the padding of a GDN mamba page, so APC can restore it together with
the recurrent GDN state without adding a KV-cache group. This keeps the cache
at three spec buckets (QSA main KV, QSA compressed K, GDN) and therefore five
KV-cache groups, identical whether or not PLE is enabled. See
``LongcatNgramModelState`` in ``vllm/model_executor/models/longcat_flash_ngram.py``
for the upstream n-gram-state precedent.

``MambaHybridModelState`` is the required base (not ``DefaultModelState``): the
36 GDN layers need the ``is_prefilling`` / ``num_accepted_tokens`` plumbing it
provides.
"""

import math
from typing import Any

import torch
import torch.nn as nn
from ixformer.inference.functions.qwen38 import (
    build_mamba_align_layout,
    qwen_mamba_align_advance,
    qwen_mamba_align_postcopy,
    qwen_mamba_align_precopy,
)

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .indexer import QSAPoolCarry

logger = init_logger(__name__)


class Qwen4ExpModelState(MambaHybridModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        self._logged_kv_cache_layout = False
        self._align_layout = None
        self._align_block_size = 0
        self._apc = bool(vllm_config.cache_config.enable_prefix_caching)
        self._compress_ratio = int(
            vllm_config.model_config.hf_text_config.indexer_compress_ratio
        )
        # _QSAStateCache checks this at model construction, where block_size can
        # still be the default; Platform.update_block_size_for_backend only
        # settles it afterwards, so re-check the final value. A resume is only
        # group-aligned if the scheduler block holds whole compression groups.
        assert self.cache_config.block_size % self._compress_ratio == 0, (
            f"Qwen4Exp needs block_size divisible by the QSA compression ratio, "
            f"got block_size={self.cache_config.block_size} "
            f"compress_ratio={self._compress_ratio}"
        )
        self.qsa_carry = self._init_qsa_carry(model)
        # FULL graph replay does not call model(); capture-time tensor addresses
        # are replayed directly. Stage every per-step model input in storage
        # owned for the lifetime of this ModelState.
        self._state_indices_buf = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.ple_layers = _find_ple_layers(model)
        self.ple = self.ple_layers[0] if self.ple_layers else None
        self._ple_state_indices_ready = False
        if self.ple is None:
            logger.info_once(
                "Qwen4ExpModelState active (PLE disabled: ple_layer_ids is empty)."
            )
            return

        text_config = vllm_config.model_config.hf_text_config
        self.ngram_ctx_len = max(int(text_config.ngram_size) - 1, 0)
        self.ngram_eos_token_id = int(self.ple.ple_embedding.eos_token_id)
        self._query_start_loc_buf = torch.zeros(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self._ngram_ctx_buf = torch.full(
            (self.max_num_reqs, self.ngram_ctx_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=device,
        )
        # Rows are dense PLE-host rows, not all Mamba groups. This makes the
        # buffer size knowable before KV groups are initialized for capture.
        self._ple_idx_buf = torch.full(
            (len(self.ple_layers), self.max_num_reqs),
            NULL_BLOCK_ID,
            dtype=torch.int32,
            device=device,
        )
        self._ple_group_ids: tuple[int, ...] | None = None
        # Scratch for the per-step n-gram left-context gather (see
        # _build_ngram_context); sized once so no per-step allocation is needed.
        self._ngram_col_offsets = torch.arange(
            self.ngram_ctx_len, dtype=torch.int32, device=device
        )
        logger.info_once(
            "Qwen4ExpModelState active (PLE enabled: %d GDN-page window(s), "
            "ngram_ctx_len=%d).",
            len(self.ple_layers),
            self.ngram_ctx_len,
        )

    def _init_qsa_carry(self, model: nn.Module) -> "QSAPoolCarry | None":
        """Give every QSA indexer its row block of the shared group accumulator."""
        indexers = [
            module
            for module in model.modules()
            if type(module).__name__ == "QSAIndexer"
        ]
        if not indexers:
            return None
        head_dims = {int(indexer.index_head_dim) for indexer in indexers}
        assert (
            len(head_dims) == 1
        ), f"Qwen4Exp QSA layers must share one indexer head dim, got {head_dims}"
        carry = QSAPoolCarry(
            acc=torch.zeros(
                (len(indexers), self.max_num_reqs, head_dims.pop()),
                dtype=torch.float32,
                device=self.device,
            )
        )
        for row, indexer in enumerate(indexers):
            indexer.pool_carry = carry
            indexer.pool_row = row
        for module in model.modules():
            if type(module).__name__ == "Qwen4ExpModel":
                module.qsa_pool_carry = carry
        logger.info_once(
            "Qwen4ExpModelState QSA group accumulator: %s float32.",
            tuple(carry.acc.shape),
        )
        return carry

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        num_computed = new_req_data.num_computed_tokens
        if num_computed:
            # A prefix hit or preemption resume lands on a scheduler-block
            # boundary, and block_size % compress_ratio == 0, so the request
            # resumes with no partially accumulated compression group: zeroing
            # the QSA carry below is the restore. The PLE window resumes from
            # its GDN-page snapshot and needs no per-request reset here.
            assert self._apc, "nonzero resume requires prefix caching"
            assert num_computed % self._compress_ratio == 0, (
                f"Qwen4Exp QSA resume needs a compression-group boundary, got "
                f"num_computed_tokens={num_computed} with "
                f"compress_ratio={self._compress_ratio}"
            )
        if self.qsa_carry is not None:
            self.qsa_carry.acc[:, req_index].zero_()

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Advance the align state column and migrate state across blocks.

        Same contract as the base implementation, implemented exclusively by
        ixformer CUDA ops on CoreX30.
        """
        self._log_kv_cache_layout_once(kv_cache_config)
        num_reqs = input_batch.num_reqs
        if self.ple is not None:
            self._gather_ple_state_indices(input_batch, block_tables, kv_cache_config)
            self._ple_state_indices_ready = True
        if not self._align_mode or num_reqs == 0:
            return
        _, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        layout = self._ensure_align_layout(kv_cache_config, block_tables)

        # Both ops run every step: they fast-exit per request when the row did
        # not cross a block boundary, which is cheaper than the async-D2H of the
        # real num_computed that skipping the launch would need.
        qwen_mamba_align_advance(
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            mamba_spec.block_size,
            backend="ixformer_native",
        )
        qwen_mamba_align_precopy(
            layout,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
            num_reqs,
            backend="ixformer_native",
        )

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        """Save the running state to the block-aligned position after sampling.

        Passing ``num_computed_tokens=None`` down suppresses the base class's
        Triton align post-copy (its only use of that argument) while keeping
        its accepted-token bookkeeping; the copy then runs through ixformer.
        """
        layout = self._align_layout
        if layout is None or num_computed_tokens is None:
            super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
            return
        super().postprocess_state(idx_mapping, num_sampled, None)

        num_reqs = idx_mapping.shape[0]
        if num_reqs == 0:
            return
        qwen_mamba_align_postcopy(
            layout,
            self.num_accepted_tokens_gpu,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            idx_mapping,
            num_reqs,
            self._align_block_size,
            backend="ixformer_native",
        )

    def _ensure_align_layout(
        self,
        kv_cache_config: KVCacheConfig,
        block_tables: tuple[torch.Tensor, ...],
    ):
        """Build the ixformer address tables for the mamba state cache once.

        Mirrors ``MambaSpecDecodeGPUContext.initialize_from_forward_context``:
        the state tensors and the persistent block tables have stable
        ``data_ptr()`` for the engine's lifetime, so the tables are captured on
        the first real batch and reused every step.
        """
        if self._align_layout is not None:
            return self._align_layout
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        states, groups, is_conv = collect_mamba_states(
            kv_cache_config,
            mamba_group_ids,
            self.model.get_mamba_state_copy_func(),
            self.vllm_config.compilation_config.static_forward_context,
        )
        self._align_layout = build_mamba_align_layout(
            states,
            groups,
            is_conv,
            [block_tables[group_id] for group_id in mamba_group_ids],
            is_conv_state_dim_first(),
        )
        self._align_block_size = mamba_spec.block_size
        logger.info_once(
            "Qwen4Exp mamba align via ixformer: %d states over %d groups, "
            "conv_state_dim_first=%s.",
            len(states),
            len(mamba_group_ids),
            is_conv_state_dim_first(),
        )
        return self._align_layout

    def _ensure_ple_group_ids(self, kv_cache_config: KVCacheConfig) -> tuple[int, ...]:
        """Bind each PLE host to a dense row in the persistent index buffer."""

        if self._ple_group_ids is not None:
            return self._ple_group_ids
        layer_to_group_id = {
            layer_name: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        group_ids = []
        for row, ple in enumerate(self.ple_layers):
            host_prefix = ple.state_host.prefix
            if host_prefix not in layer_to_group_id:
                raise RuntimeError(
                    f"PLE state host {host_prefix!r} is absent from KV cache groups"
                )
            ple.state_group_row = row
            group_ids.append(layer_to_group_id[host_prefix])
        self._ple_group_ids = tuple(group_ids)
        return self._ple_group_ids

    def _gather_ple_state_indices(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
    ) -> torch.Tensor:
        """Gather the same current block id used by each GDN mamba group."""

        group_ids = self._ensure_ple_group_ids(kv_cache_config)
        num_reqs = input_batch.num_reqs
        padded_num_reqs = input_batch.num_reqs_after_padding
        out = self._ple_idx_buf[:, :padded_num_reqs]
        out.fill_(NULL_BLOCK_ID)
        for row, group_id in enumerate(group_ids):
            spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            table = mamba_get_block_table_tensor(
                block_tables[group_id][:num_reqs],
                input_batch.seq_lens[:num_reqs],
                spec,
                self.cache_config.mamba_cache_mode,
            )
            out[row, :num_reqs].copy_(table[:, 0], non_blocking=True)
        return out

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        # Capture dummy inputs are built before attention metadata. Resolve PLE
        # host rows here, still before the captured model forward executes.
        if self.ple is not None:
            self._ensure_ple_group_ids(kv_cache_config)
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )

    def _log_kv_cache_layout_once(self, kv_cache_config: KVCacheConfig) -> None:
        """Dump the per-group block/page layout that decides APC hit length.

        The compressed-K group's page is a fraction of the main KV page, so
        ``unify_kv_cache_spec_page_size`` may scale its ``block_size`` up
        instead of padding its page. Cache hits are rounded down to the LCM of
        the group block sizes, so an LCM above ``max_model_len`` silently pins
        the hit rate at zero -- worth one line in the log rather than a bisect.
        """
        if self._logged_kv_cache_layout:
            return
        self._logged_kv_cache_layout = True
        block_sizes = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_sizes.append(spec.block_size)
            logger.warning(
                "Qwen4Exp KV group %d: %s layers=%d block_size=%d page_bytes=%d",
                group_id,
                type(spec).__name__,
                len(group.layer_names),
                spec.block_size,
                spec.page_size_bytes,
            )
        logger.warning(
            "Qwen4Exp KV layout: num_blocks=%d cache_block_size=%d "
            "hit_alignment(LCM)=%d hash_block_size(GCD)=%d max_model_len=%d "
            "prefix_caching=%s mamba_cache_mode=%s zeroing=%s",
            kv_cache_config.num_blocks,
            self.cache_config.block_size,
            math.lcm(*block_sizes),
            math.gcd(*block_sizes),
            self.max_model_len,
            self.cache_config.enable_prefix_caching,
            self.cache_config.mamba_cache_mode,
            kv_cache_config.needs_kv_cache_zeroing,
        )
        compilation_config = self.vllm_config.compilation_config
        max_capture = compilation_config.max_cudagraph_capture_size
        decode_max_bs = self.max_num_reqs * (
            int(self.vllm_config.num_speculative_tokens) + 1
        )
        if max_capture is not None:
            decode_max_bs = min(decode_max_bs, max_capture)
        logger.warning(
            "Qwen4Exp CUDA graph: mode=%s capture_sizes=%s "
            "decode_cudagraph_max_bs=%d max_capture_size=%s",
            compilation_config.cudagraph_mode,
            compilation_config.cudagraph_capture_sizes,
            decode_max_bs,
            max_capture,
        )

    def _stage_state_indices(self, input_batch: InputBatch) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        padded_num_reqs = input_batch.num_reqs_after_padding
        out = self._state_indices_buf[:padded_num_reqs]
        out[:num_reqs].copy_(input_batch.idx_mapping[:num_reqs], non_blocking=True)
        out[num_reqs:].fill_(-1)
        return out

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs = input_batch.num_reqs
        padded_num_reqs = input_batch.num_reqs_after_padding
        state_indices = self._stage_state_indices(input_batch)
        if self.ple is None:
            model_inputs["state_indices"] = state_indices
            return model_inputs

        query_start_loc = self._query_start_loc_buf[: padded_num_reqs + 1]
        query_start_loc.copy_(
            input_batch.query_start_loc[: padded_num_reqs + 1], non_blocking=True
        )
        model_inputs["query_start_loc"] = query_start_loc
        if not self._ple_state_indices_ready:
            # Initial memory profiling skips attention preparation, so
            # preprocess_state() has no block tables from which to gather the
            # PLE page ids. Treat that batch as stateless just like cudagraph
            # capture: run the math with EOS left context and skip every
            # PLE/QSA state write.
            model_inputs["state_indices"] = None
            ngram_context = self._ngram_ctx_buf[:padded_num_reqs]
            ngram_context.fill_(self.ngram_eos_token_id)
            model_inputs["ngram_context"] = ngram_context
            model_inputs["ple_state_indices"] = None
            model_inputs["ple_conv_state"] = None
            return model_inputs
        self._ple_state_indices_ready = False

        # Consumed by the PLE conv directly and forwarded to the QSA carry by
        # Qwen4ExpModel.forward, so both read the same step's slot map.
        model_inputs["state_indices"] = state_indices
        ngram_context = self._ngram_ctx_buf[:padded_num_reqs]
        self._build_ngram_context(
            state_indices[:num_reqs], req_states, ngram_context[:num_reqs]
        )
        ngram_context[num_reqs:].fill_(self.ngram_eos_token_id)
        model_inputs["ngram_context"] = ngram_context
        model_inputs["ple_state_indices"] = self._ple_idx_buf[:, :padded_num_reqs]
        model_inputs["ple_conv_state"] = None
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        # -1 keeps QSA carry reads/writes inert during capture while preserving
        # the non-None branch that replay needs.
        model_inputs["state_indices"] = self._state_indices_buf[:num_reqs].fill_(-1)
        if self.ple is None:
            return model_inputs
        base = num_tokens // num_reqs
        extra = num_tokens % num_reqs
        lens = [base + 1 if i >= num_reqs - extra else base for i in range(num_reqs)]
        starts = [0]
        for length in lens:
            starts.append(starts[-1] + length)
        query_start_loc = self._query_start_loc_buf[: num_reqs + 1]
        query_start_loc.copy_(
            torch.tensor(starts, dtype=torch.int32, device=self.device)
        )
        model_inputs["query_start_loc"] = query_start_loc
        model_inputs["ngram_context"] = self._ngram_ctx_buf[:num_reqs].fill_(
            self.ngram_eos_token_id
        )
        model_inputs["ple_state_indices"] = self._ple_idx_buf[:, :num_reqs].fill_(
            NULL_BLOCK_ID
        )
        model_inputs["ple_conv_state"] = None
        return model_inputs

    def _build_ngram_context(
        self,
        state_indices: torch.Tensor,
        req_states: RequestState,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Left context ``[num_reqs, ngram_size - 1]`` for the n-gram hash.

        The context is the tail of each request's already-computed tokens, so it
        is *derived* rather than carried: V2 keeps ``all_token_ids`` and
        ``num_computed_tokens`` current on GPU (sampled tokens are appended by
        ``post_update``), which means there is no rolling buffer to get out of
        sync across chunked-prefill steps or to restore on resume. Positions
        before the start of the sequence read as EOS, matching the HF reference's
        context-boundary handling.
        """
        if self.ngram_ctx_len == 0:
            return out
        slots = state_indices.to(torch.long)
        num_computed = req_states.num_computed_tokens.gpu[slots]
        # Columns [ncomp - ctx_len, ncomp) of each request's token history.
        cols = num_computed.unsqueeze(1).to(torch.int64) + (
            self._ngram_col_offsets.to(torch.int64) - self.ngram_ctx_len
        )
        context = req_states.all_token_ids.gpu[slots.unsqueeze(1), cols.clamp_min(0)]
        out.copy_(
            torch.where(cols >= 0, context, self.ngram_eos_token_id).to(torch.int32)
        )
        return out


def collect_mamba_states(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: list[int],
    copy_funcs: tuple[Any, ...],
    forward_context: dict[str, Any],
) -> tuple[list[torch.Tensor], list[int], list[bool]]:
    """Flatten the mamba state tensors into (layer, state-type) order.

    The order must match ``MambaSpecDecodeGPUContext``'s metadata arrays --
    group-major, then layer, then state type -- because the block tables are
    handed over per group and indexed by the group slot recorded here. The
    conv/temporal split comes from the model's copy functions, as upstream.
    """
    is_conv_by_type = [func is get_conv_copy_spec for func in copy_funcs]
    states: list[torch.Tensor] = []
    groups: list[int] = []
    is_conv: list[bool] = []
    for group_slot, group_id in enumerate(mamba_group_ids):
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
            layer = forward_context[layer_name]
            kv_caches = layer.kv_cache
            assert len(kv_caches) == len(copy_funcs), (
                f"{layer_name}: {len(kv_caches)} state tensors for "
                f"{len(copy_funcs)} state types"
            )
            for state_type_idx, state in enumerate(kv_caches):
                is_ple_window = (
                    state_type_idx == len(kv_caches) - 1
                    and getattr(layer, "ple_window_shape", None) is not None
                )
                if is_ple_window and not getattr(layer, "hosts_ple_window", False):
                    continue
                states.append(state)
                groups.append(group_slot)
                is_conv.append(is_conv_by_type[state_type_idx])
    return states, groups, is_conv


def _find_ple_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    """Locate all PLE layers in deterministic decoder-layer order."""

    return tuple(
        sorted(
            (
                module
                for module in model.modules()
                if type(module).__name__ == "Qwen4ExpPLELayer"
            ),
            key=lambda module: int(module.layer_idx),
        )
    )
