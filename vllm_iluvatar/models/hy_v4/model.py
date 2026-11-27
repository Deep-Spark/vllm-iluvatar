# SPDX-License-Identifier: Apache-2.0
"""Inference-only HY V4 model for Iluvatar.

Corresponds to: vllm/models/hy_v4/nvidia/model.py (upstream #54160)

Target   : ``HYV4ForCausalLM``, the arch named by the Hy4-preview checkpoint.
Why      : the Iluvatar vLLM baseline has no ``hy_v4`` model at all, and
           community vLLM must stay byte-identical, so the graph is registered
           from the plugin instead of landing #54160 in vLLM.
How      : ``ModelRegistry.register_model`` under the checkpoint's own arch name
           (see ``register_model`` below). The graph reuses the community
           infrastructure that already exists in the baseline -- ``MLAAttention``
           (which ``IluPlatform`` routes to ``IluFlashMLASparseBackend``),
           ``IluSparseAttnIndexer``, ``FusedMoEFactory``, ``GateLinear`` -- so the
           Iluvatar deltas are confined to three places: the sink-capable sparse
           backend (`.flashmla_sparse`), ixformer's fused iHC ops (`.hc`) and
           the MXFP8 indexer-``wk`` dequant (`.weight_utils`).
Scope    : Hy4-preview, no MTP (the MTP weights are dropped in ``load_weights``).
Remove   : when the Iluvatar vLLM baseline carries hy_v4 natively.
"""

from __future__ import annotations

import os
import time
import typing
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from itertools import islice

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS

from .attention import (
    _HYV4_DSA_FP8_CACHE_DTYPES,
    HYV4MLAAttention,
    Indexer,
    build_hy4_ropes,
    compute_skip_topk_layers,
    hyv4_layer_skip_topk,
    is_skip_topk_indexer_weight,
)
from .fp8_indexer import hy4_fp8_indexer_forward_offload_sender
from .hc import HYV4HCHeadLayer, HYV4HCLayer
from .moe import HYV4FeedForward, HYV4MoEFused
from .topk_index_cache import HYV4TopkIndexCache
from .weight_utils import try_load_quantized_indexer_wk

logger = init_logger(__name__)

# IntermediateTensors key carrying the DSA top-k buffer across a pipeline
# boundary whose first layer is shared. See ``HYV4Model``.
_TOPK_INDICES_KEY = "topk_indices"
_OFFLOAD_FIRST_TOPK_KEY = "offload_first_topk"
_PP_ACT_SCALE_KEY = "hidden_states_scale"


def _pp_act_fp8_fused() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_PP_ACT_FP8"]()


def _ihc_fp8_res() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_IHC_FP8_RES"]()


def _ihc_fuse_norm() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_FUSE_IHC_NORM"]()


def _pp_offload_first_indexer() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_PP_OFFLOAD_FIRST_INDEXER"]()


def _pp_pack_tensors() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_PP_PACK_TENSORS"]()


def _pp_stage_timer_enabled() -> bool:
    return _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_STAGE_TIMER"]()


def compute_pp_packed_row_bytes(hidden_cols: int, topk_cols: int) -> int:
    """Per-token byte width of a plane-major packed PP boundary tensor."""
    return hidden_cols + topk_cols * 4 + 4


def pack_hy4_pp_boundary(
    pack_buf_2d: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    topk: torch.Tensor | None = None,
    *,
    topk_cols: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack ``(q, topk, scale)`` into a single contiguous ``[N, W]`` uint8 buffer.

    Layout in the flat ``N * W`` byte view is plane-major so every unpacked view
    stays contiguous and 4-byte (in fact 4096-byte for HY4) aligned while
    preserving ``dim 0 == num_tokens`` for ``sync_and_gather_intermediate_tensors``:
      1. ``q``     : ``[0, N * hidden_cols)`` uint8 -> ``(N, hidden_cols)``
      2. ``topk``  : ``[N * hidden_cols, + N * topk_cols * 4)`` int32 -> ``(N, topk_cols)``
      3. ``scale`` : ``[..., N * W)`` float32 -> ``(N,)``

    If ``topk is None`` and ``topk_cols > 0``, returns a writable ``(N, topk_cols)``
    int32 view into plane 2 so callers (such as ``forward_offload``) can write
    directly into the wire buffer with zero extra copy.
    """
    num_tokens, hidden_cols = q.shape
    row_bytes = compute_pp_packed_row_bytes(hidden_cols, topk_cols)
    out_2d = pack_buf_2d[:num_tokens]
    assert out_2d.shape == (num_tokens, row_bytes) and out_2d.is_contiguous()
    flat = out_2d.view(-1)
    o_q = num_tokens * hidden_cols
    o_tk = o_q + num_tokens * topk_cols * 4
    o_end = o_tk + num_tokens * 4

    flat[:o_q].view(num_tokens, hidden_cols).copy_(q)
    flat[o_tk:o_end].view(torch.float32).copy_(scale.reshape(num_tokens))

    topk_view: torch.Tensor | None = None
    if topk_cols > 0:
        topk_view = flat[o_q:o_tk].view(torch.int32).view(num_tokens, topk_cols)
        if topk is not None:
            topk_view.copy_(topk[:num_tokens])
    return out_2d, topk_view


def unpack_hy4_pp_boundary(
    packed_2d: torch.Tensor,
    hidden_cols: int,
    topk_cols: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Zero-copy unpack of a ``[N, W]`` plane-major PP boundary tensor."""
    if not packed_2d.is_contiguous():
        packed_2d = packed_2d.contiguous()
    num_tokens = packed_2d.shape[0]
    row_bytes = compute_pp_packed_row_bytes(hidden_cols, topk_cols)
    assert packed_2d.shape == (num_tokens, row_bytes)
    flat = packed_2d.view(-1)
    o_q = num_tokens * hidden_cols
    o_tk = o_q + num_tokens * topk_cols * 4
    o_end = o_tk + num_tokens * 4

    q = flat[:o_q].view(num_tokens, hidden_cols)
    scale = flat[o_tk:o_end].view(torch.float32)
    topk = (
        flat[o_q:o_tk].view(torch.int32).view(num_tokens, topk_cols)
        if topk_cols > 0
        else None
    )
    return q, scale, topk


_pp_act_fp8_fused_logged = False
_ihc_fp8_res_logged = False
_ihc_fuse_norm_logged = False


def _quant_pp_act(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import ixformer.inference.functions as ixfops

    flat = hidden if hidden.dim() == 2 else hidden.flatten(1)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    q, scale = ixfops.fp8_per_token_dynamic_scaled(flat)
    return q.view(torch.uint8), scale.reshape(-1)


def _dequant_pp_act(q_bytes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    import ixformer.inference.functions as ixfops

    return ixfops.hy4_pp_fp8_dequant(q_bytes, scale)


def _ihc_pre_fp8_boundary(
    hc_pre: nn.Module,
    q_bytes: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import ixformer.inference.functions as ixfops

    cached = getattr(hc_pre, "_ihc_w_fp8_cache", None)
    if cached is None:
        w_bf16 = hc_pre.hc_fn.weight.to(torch.bfloat16).contiguous()
        w_fp8, w_scale = ixfops.fp8_per_token_dynamic_scaled(w_bf16)
        cached = (w_fp8.contiguous(), w_scale.reshape(-1).contiguous())
        hc_pre._ihc_w_fp8_cache = cached
    w_fp8, w_scale = cached
    tokens = q_bytes.shape[0]
    n = 2 * hc_pre.hc_mult
    if scale.dim() != 1:
        scale = scale.reshape(tokens)
    mixes = torch.empty((tokens, n), dtype=torch.bfloat16, device=q_bytes.device)
    torch.ops._ixformer_C.fp8_per_channel_mm(
        q_bytes.view(torch.float8_e4m3fn),
        w_fp8,
        scale,
        w_scale,
        mixes,
        None,
        None,
        1.0,
        0.0,
        False,
    )
    return ixfops.ihc_pre_epilogue_fp8(
        q_bytes,
        scale,
        mixes,
        hc_pre.hc_scale,
        hc_pre.hc_base,
        norm_eps=float(hc_pre.layernorm_epsilon),
        hc_eps=float(hc_pre.hc_eps),
        magnitude=float(hc_pre.magnitude),
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        q_out=q_out,
        scale_out=scale_out,
    )


def _ihc_post_fp8_boundary(
    x: torch.Tensor,
    q_bytes: torch.Tensor,
    scale: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    import ixformer.inference.functions as ixfops

    return ixfops.fuse_ihc_post_fp8(x, q_bytes, scale, post)


def _ihc_post_quant_fp8_boundary(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import ixformer.inference.functions as ixfops

    return ixfops.fuse_ihc_post_quant_fp8(x, residual, post)


def _ihc_post_fp8io(
    x: torch.Tensor,
    q_bytes: torch.Tensor,
    scale: torch.Tensor,
    post: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import ixformer.inference.functions as ixfops

    return ixfops.fuse_ihc_post_fp8io(x, q_bytes, scale, post)


def _normalize_hyv4_config(config: PretrainedConfig) -> PretrainedConfig:
    """Populate the aliases consumed by the shared MoE implementation."""
    config.router_scaling_factor = config.routed_scaling_factor
    config.num_experts = config.n_routed_experts
    config.expert_hidden_dim = config.moe_intermediate_size
    config.num_shared_experts = config.n_shared_experts
    config.route_norm = config.norm_topk_prob
    return config


class HYV4DecoderLayer(nn.Module):
    """One HY V4 decoder layer: MLA attention plus a dense or MoE MLP.

    When ``config.enable_ihc`` is set the layer runs on ``hc_mult`` residual
    channels and each sub-block is wrapped by an `HYV4HCLayer` boundary;
    otherwise it uses the standard single-stream residual.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        topk_index_cache: HYV4TopkIndexCache | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.enable_ihc = getattr(config, "enable_ihc", False)
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = HYV4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
            topk_index_cache=topk_index_cache,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_idx] == "dense":
            self.mlp = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV4MoEFused(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
            self.block_type = "moe"
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.enable_ihc:
            return self._forward_ihc(positions, hidden_states)
        return self._forward_normal(positions, hidden_states, residual)

    def _forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single-residual-stream forward (iHC disabled)."""
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def _run_ihc_pre_and_norm(
        self,
        hc_layer: HYV4HCLayer,
        layernorm: RMSNorm,
        hidden_states: torch.Tensor,
        *,
        fuse_fp8: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        return _run_ihc_pre_and_norm_impl(
            hc_layer, layernorm, hidden_states, fuse_fp8=fuse_fp8
        )

    def _run_ihc_pre_fp8_boundary_and_norm(
        self,
        hc_pre: nn.Module,
        layernorm: RMSNorm,
        q_bytes: torch.Tensor,
        scale: torch.Tensor,
        *,
        fuse_fp8: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_ihc_pre_fp8_boundary_and_norm_impl(
            hc_pre, layernorm, q_bytes, scale, fuse_fp8=fuse_fp8
        )

    def _forward_ihc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None]:
        """iHC forward: each sub-block reduces and re-scatters the channels."""
        from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
            clear_activation_prequant,
        )

        global _ihc_fp8_res_logged
        tokens = (
            hidden_states[0].shape[0]
            if isinstance(hidden_states, tuple)
            else hidden_states.shape[0]
        )
        keep_fp8_res = _ihc_fp8_res() and tokens >= 128
        if keep_fp8_res and not _ihc_fp8_res_logged:
            _ihc_fp8_res_logged = True
            logger.info("iHC residual FP8: tokens>=%d", tokens)

        if isinstance(hidden_states, tuple):
            q_bytes, scale = hidden_states
            if keep_fp8_res or (_pp_act_fp8_fused() and tokens >= 128):
                try:
                    hidden_states, post_gates = self._run_ihc_pre_fp8_boundary_and_norm(
                        self.hc_attn_layer.hc_pre,
                        self.input_layernorm,
                        q_bytes,
                        scale,
                    )
                    hidden_states = self.self_attn(
                        positions=positions,
                        hidden_states=hidden_states,
                    )
                finally:
                    clear_activation_prequant()
                if keep_fp8_res:
                    hidden_states = _ihc_post_fp8io(
                        hidden_states, q_bytes, scale, post_gates
                    )
                else:
                    hidden_states = _ihc_post_fp8_boundary(
                        hidden_states, q_bytes, scale, post_gates
                    )
            else:
                hidden_states = _dequant_pp_act(q_bytes, scale)
                try:
                    hidden_states, post_gates, residual = self._run_ihc_pre_and_norm(
                        self.hc_attn_layer, self.input_layernorm, hidden_states
                    )
                    hidden_states = self.self_attn(
                        positions=positions,
                        hidden_states=hidden_states,
                    )
                finally:
                    clear_activation_prequant()
                hidden_states = self.hc_attn_layer.post(
                    hidden_states, residual, post_gates
                )
        else:
            try:
                hidden_states, post_gates, residual = self._run_ihc_pre_and_norm(
                    self.hc_attn_layer, self.input_layernorm, hidden_states
                )
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )
            finally:
                clear_activation_prequant()
            if keep_fp8_res:
                hidden_states = _ihc_post_quant_fp8_boundary(
                    hidden_states, residual, post_gates
                )
            else:
                hidden_states = self.hc_attn_layer.post(
                    hidden_states, residual, post_gates
                )

        if isinstance(hidden_states, tuple):
            q_bytes, scale = hidden_states
            try:
                hidden_states, post_gates = self._run_ihc_pre_fp8_boundary_and_norm(
                    self.hc_mlp_layer.hc_pre,
                    self.post_attention_layernorm,
                    q_bytes,
                    scale,
                )
                hidden_states = self.mlp(hidden_states)
            finally:
                clear_activation_prequant()
            hidden_states = _ihc_post_fp8io(hidden_states, q_bytes, scale, post_gates)
        else:
            try:
                hidden_states, post_gates, residual = self._run_ihc_pre_and_norm(
                    self.hc_mlp_layer, self.post_attention_layernorm, hidden_states
                )
                hidden_states = self.mlp(hidden_states)
            finally:
                clear_activation_prequant()
            if getattr(self, "_emit_pp_fp8", False) or keep_fp8_res:
                assert post_gates is not None
                hidden_states = _ihc_post_quant_fp8_boundary(
                    hidden_states, residual, post_gates
                )
            else:
                hidden_states = self.hc_mlp_layer.post(
                    hidden_states, residual, post_gates
                )

        # Under iHC the residual is carried inside hidden_states.
        return hidden_states, None


def _run_ihc_pre_and_norm_impl(
    hc_layer: HYV4HCLayer,
    layernorm: RMSNorm,
    hidden_states: torch.Tensor,
    *,
    fuse_fp8: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    global _ihc_fuse_norm_logged
    hidden_states = hc_layer.prepare_input(hidden_states)
    if _ihc_fuse_norm() and hc_layer.can_fuse_norm(hidden_states):
        tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[2]
        q_out = None
        scale_out = None
        if fuse_fp8 and tokens > 4:
            q_out = torch.empty(
                (tokens, hidden_dim),
                dtype=torch.float8_e4m3fn,
                device=hidden_states.device,
            )
            scale_out = torch.empty(
                (tokens,), dtype=torch.float32, device=hidden_states.device
            )
        normed, post_gates, residual = hc_layer.pre(
            hidden_states,
            rms_weight=layernorm.weight.data,
            rms_eps=float(layernorm.variance_epsilon),
            q_out=q_out,
            scale_out=scale_out,
        )
        if q_out is not None and scale_out is not None:
            from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
                register_activation_prequant,
            )

            register_activation_prequant(normed, q_out, scale_out)
        if not _ihc_fuse_norm_logged:
            _ihc_fuse_norm_logged = True
            logger.info(
                "HYV4 iHC pre + RMSNorm (+ FP8 quant) fusion armed "
                "(VLLM_ILU_HYV4_FUSE_IHC_NORM=1)"
            )
        return normed, post_gates, residual

    reduced, post_gates, residual = hc_layer.pre(hidden_states)
    return layernorm(reduced), post_gates, residual


def _run_ihc_pre_fp8_boundary_and_norm_impl(
    hc_pre: nn.Module,
    layernorm: RMSNorm,
    q_bytes: torch.Tensor,
    scale: torch.Tensor,
    *,
    fuse_fp8: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _ihc_fuse_norm():
        tokens = q_bytes.shape[0]
        hidden_dim = layernorm.weight.shape[0]
        q_out = None
        scale_out = None
        if fuse_fp8 and tokens > 4:
            q_out = torch.empty(
                (tokens, hidden_dim),
                dtype=torch.float8_e4m3fn,
                device=q_bytes.device,
            )
            scale_out = torch.empty(
                (tokens,), dtype=torch.float32, device=q_bytes.device
            )
        normed, post_gates = _ihc_pre_fp8_boundary(
            hc_pre,
            q_bytes,
            scale,
            rms_weight=layernorm.weight.data,
            rms_eps=float(layernorm.variance_epsilon),
            q_out=q_out,
            scale_out=scale_out,
        )
        if q_out is not None and scale_out is not None:
            from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
                register_activation_prequant,
            )

            register_activation_prequant(normed, q_out, scale_out)
        return normed, post_gates

    reduced, post_gates = _ihc_pre_fp8_boundary(hc_pre, q_bytes, scale)
    return layernorm(reduced), post_gates


def _compute_pp_first_indexer_offload_hops(
    config: PretrainedConfig,
    cache_config: CacheConfig | None,
    pp_world_size: int,
    *,
    pcp_size: int = 1,
) -> set[int]:
    """Return sender PP ranks ``r`` that co-compute stage ``r+1``'s first full indexer.

    Requires TP=1 and PCP=1: the helper's K-insert / MQA path is PCP=1-only, and
    ``compute_offload_split_t`` also refuses PCP>1.
    """
    if (
        not _pp_offload_first_indexer()
        or pp_world_size <= 1
        or pcp_size > 1
        or not hasattr(config, "index_topk")
        or config.index_topk % 2 != 0
        or getattr(config, "q_lora_rank", None) is None
        or cache_config is None
        or (cache_config.cache_dtype or "auto") not in _HYV4_DSA_FP8_CACHE_DTYPES
        or get_tensor_model_parallel_world_size() != 1
    ):
        return set()

    cuts = [
        get_pp_indices(config.num_hidden_layers, rank, pp_world_size)
        for rank in range(pp_world_size)
    ]
    full_counts = [
        sum(
            1
            for layer_idx in range(s, e)
            if not hyv4_layer_skip_topk(config, layer_idx)
        )
        for s, e in cuts
    ]
    active_senders: set[int] = set()
    for r in range(pp_world_size - 1):
        if (r - 1) in active_senders:
            continue
        s_curr, e_curr = cuts[r]
        s_next, e_next = cuts[r + 1]
        if e_curr <= s_curr or e_next <= s_next:
            continue
        if hyv4_layer_skip_topk(config, s_next):
            continue
        if full_counts[r] < full_counts[r + 1]:
            active_senders.add(r)
    return active_senders


class HYV4OffloadFirstIndexerAttn(nn.Module):
    """Mirrored self-attn subset (``fused_qkv_a_proj`` + ``q_a_layernorm`` + ``indexer``).

    Lives at ``layers[end_layer].self_attn`` on predecessor helper ranks while
    marking all unused main-MLA projections as ``PPMissingLayer`` so checkpoint
    loading reuses the exact canonical prefixes and FP8 online quant rules.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.indexer_rope_emb, self.indexer_mrope_emb = build_hy4_ropes(
            self.qk_rope_head_dim,
            max_position_embeddings,
            config.rope_parameters,
        )
        dummy_topk = torch.empty(
            (1, config.index_topk),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.indexer = Indexer(
            vllm_config,
            config,
            self.hidden_size,
            self.q_lora_rank,
            quant_config,
            cache_config,
            dummy_topk,
            f"{prefix}.indexer",
        )
        self.indexer.mrope_emb = self.indexer_mrope_emb

        self.q_b_proj = PPMissingLayer()
        self.q_proj = PPMissingLayer()
        self.kv_a_proj_with_mqa = PPMissingLayer()
        self.kv_a_layernorm = PPMissingLayer()
        self.kv_b_proj = PPMissingLayer()
        self.o_proj = PPMissingLayer()
        self.linear_gate = PPMissingLayer()


class HYV4OffloadFirstIndexerLayer(nn.Module):
    """Mirrored next-stage entry layer computing K-cache sync + tail top-k offload."""

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = getattr(config, "enable_ihc", False)
        self.half_topk = config.index_topk // 2
        self._wire_buf = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens, self.half_topk),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = HYV4OffloadFirstIndexerAttn(
            vllm_config=vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if self.enable_ihc:
            self.hc_attn_layer = HYV4HCLayer(
                config, layer_idx, prefix=f"{prefix}.hc_attn_layer"
            )
        else:
            self.hc_attn_layer = PPMissingLayer()
        self.mlp = PPMissingLayer()
        self.post_attention_layernorm = PPMissingLayer()
        self.hc_mlp_layer = PPMissingLayer()

    def _get_wire_buf(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if self._wire_buf.shape[0] < num_tokens or self._wire_buf.device != device:
            self._wire_buf = torch.empty(
                (num_tokens, self.half_topk),
                dtype=torch.int32,
                device=device,
            )
        return self._wire_buf[:num_tokens]

    def forward_offload(
        self,
        positions: torch.Tensor,
        pp_hidden: torch.Tensor,
        pp_scale: torch.Tensor | None,
        residual: torch.Tensor | None,
        wire_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
            clear_activation_prequant,
        )

        num_tokens = pp_hidden.shape[0]
        if wire_out is None:
            wire_out = self._get_wire_buf(num_tokens, pp_hidden.device)
        attn = self.self_attn
        supports_wq_b_prequant = HYV4MLAAttention._supports_activation_prequant(
            attn.indexer.wq_b
        )
        try:
            if self.enable_ihc:
                assert isinstance(self.hc_attn_layer, HYV4HCLayer)
                if pp_scale is not None:
                    if num_tokens >= 128:
                        normed, _ = _run_ihc_pre_fp8_boundary_and_norm_impl(
                            self.hc_attn_layer.hc_pre,
                            self.input_layernorm,
                            pp_hidden,
                            pp_scale,
                        )
                    else:
                        deq = _dequant_pp_act(pp_hidden, pp_scale)
                        normed, _, _ = _run_ihc_pre_and_norm_impl(
                            self.hc_attn_layer, self.input_layernorm, deq
                        )
                else:
                    normed, _, _ = _run_ihc_pre_and_norm_impl(
                        self.hc_attn_layer, self.input_layernorm, pp_hidden
                    )
            else:
                h = (
                    _dequant_pp_act(pp_hidden, pp_scale)
                    if pp_scale is not None
                    else pp_hidden
                )
                if residual is not None:
                    h = h + residual
                normed = self.input_layernorm(h)

            return hy4_fp8_indexer_forward_offload_sender(
                indexer=attn.indexer,
                normed_hidden_states=normed,
                fused_qkv_a_proj=attn.fused_qkv_a_proj,
                q_a_layernorm=attn.q_a_layernorm,
                q_lora_rank=attn.q_lora_rank,
                kv_lora_and_rope_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
                supports_wq_b_prequant=supports_wq_b_prequant,
                positions=positions,
                rotary_emb=attn.indexer_rope_emb,
                wire_out=wire_out,
            )
        finally:
            clear_activation_prequant()
            attn.indexer._prefetched_mrope_table = None


class HYV4Model(nn.Module):
    """HY V4 backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = _normalize_hyv4_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_sparse = hasattr(config, "index_topk")
        if self.is_sparse:
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            # One cache per model, shared by every layer's sparse MLA impl, so
            # the 57 "shared" indexer layers reuse the global-slot conversion
            # and topk_length the preceding full layer already produced.
            self.topk_index_cache = HYV4TopkIndexCache()
        else:
            self.topk_indices_buffer = None
            self.topk_index_cache = None
        self.config = config
        self.quant_config = quant_config
        self.enable_ihc = getattr(config, "enable_ihc", False)

        if get_pp_group().is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                topk_index_cache=self.topk_index_cache,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        if self.enable_ihc:
            # iHC head layer: merge the residual channels into one stream.
            if get_pp_group().is_last_rank:
                self.hc_head = HYV4HCHeadLayer(
                    config,
                    hidden_size=config.hidden_size,
                    hc_mult=config.hc_mult,
                    hc_eps=config.hc_eps,
                    prefix=f"{prefix}.hc_head",
                )
            else:
                self.hc_head = PPMissingLayer()
            self._pp_intermediate_keys = ("hidden_states",)
            self._pp_intermediate_hidden = config.hc_mult * config.hidden_size
        else:
            self._pp_intermediate_keys = ("hidden_states", "residual")
            self._pp_intermediate_hidden = config.hidden_size

        # ``topk_indices_buffer`` lives on this rank, while the full/shared
        # split is global. A stage whose first layer is shared would otherwise
        # read whatever the previous forward left in its own buffer. Carry the
        # buffer only across those boundaries. A stage that already starts on a
        # full layer sends nothing, and pipeline-parallel size 1 sends nothing.
        # ``end_layer`` is the next stage's ``start_layer``, so the send and
        # recv key sets match.
        self.recv_topk = (
            self.is_sparse
            and not get_pp_group().is_first_rank
            and hyv4_layer_skip_topk(config, self.start_layer)
        )
        self.send_topk = (
            self.is_sparse
            and not get_pp_group().is_last_rank
            and hyv4_layer_skip_topk(config, self.end_layer)
        )
        pp_group = get_pp_group()
        offload_senders = _compute_pp_first_indexer_offload_hops(
            config,
            cache_config,
            pp_group.world_size,
            pcp_size=vllm_config.parallel_config.prefill_context_parallel_size,
        )
        self.send_offload_first_topk = pp_group.rank_in_group in offload_senders
        self.recv_offload_first_topk = (
            pp_group.rank_in_group > 0
            and (pp_group.rank_in_group - 1) in offload_senders
        )
        self.offload_first_indexer_layer: HYV4OffloadFirstIndexerLayer | None = None
        if self.send_offload_first_topk:
            target_layer = self.end_layer
            assert cache_config is not None
            helper = HYV4OffloadFirstIndexerLayer(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_idx=target_layer,
                prefix=f"{prefix}.layers.{target_layer}",
            )
            self.layers[target_layer] = helper
            self.offload_first_indexer_layer = helper
            logger.info(
                "HYV4 PP first-indexer offload sender armed on PP rank %d -> %d "
                "(target layer %d)",
                pp_group.rank_in_group,
                pp_group.rank_in_group + 1,
                target_layer,
            )
        elif self.recv_offload_first_topk:
            logger.info(
                "HYV4 PP first-indexer offload receiver armed on PP rank %d <- %d "
                "(entry layer %d)",
                pp_group.rank_in_group,
                pp_group.rank_in_group - 1,
                self.start_layer,
            )

        if (
            _pp_act_fp8_fused()
            and self.enable_ihc
            and not get_pp_group().is_last_rank
            and self.end_layer > self.start_layer
        ):
            self.layers[self.end_layer - 1]._emit_pp_fp8 = True

        self._use_pp_packed_wire = (
            pp_group.world_size > 1
            and self.enable_ihc
            and _pp_act_fp8_fused()
            and _pp_pack_tensors()
            and get_tensor_model_parallel_world_size() == 1
        )
        index_topk = int(getattr(config, "index_topk", 0))
        self._pp_recv_topk_cols = (
            index_topk
            if self.recv_topk
            else (index_topk // 2 if self.recv_offload_first_topk else 0)
        )
        self._pp_send_topk_cols = (
            index_topk
            if self.send_topk
            else (index_topk // 2 if self.send_offload_first_topk else 0)
        )
        self._pp_recv_packed_row_bytes = compute_pp_packed_row_bytes(
            self._pp_intermediate_hidden, self._pp_recv_topk_cols
        )
        self._pp_send_packed_row_bytes = compute_pp_packed_row_bytes(
            self._pp_intermediate_hidden, self._pp_send_topk_cols
        )
        self._pp_max_batched_tokens = int(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._pp_pack_bufs: list[torch.Tensor] = []
        self._pp_pack_slot = 0
        self._stage_timer_enabled = _pp_stage_timer_enabled()
        self._stage_timer_steps = 0
        self._stage_timer_max_steps = int(
            os.environ.get("VLLM_ILU_HYV4_STAGE_TIMER_STEPS", "8")
        )
        if self._use_pp_packed_wire:
            logger.info(
                "HYV4 PP single-tensor boundary packing armed on PP rank %d "
                "(recv_row_bytes=%d, send_row_bytes=%d)",
                pp_group.rank_in_group,
                self._pp_recv_packed_row_bytes if not pp_group.is_first_rank else 0,
                self._pp_send_packed_row_bytes if not pp_group.is_last_rank else 0,
            )

        # MoE hyperparameters (consumed by EPLB).
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        example_layer: HYV4MoEFused | None = None
        for layer in self.layers:
            if not isinstance(layer, HYV4DecoderLayer):
                continue

            if layer.block_type == "moe":
                assert isinstance(layer.mlp, HYV4MoEFused)
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.layers:
            if not isinstance(layer, HYV4DecoderLayer):
                continue
            if isinstance(layer.mlp, HYV4MoEFused):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id) for weights, fp8
        # weight scales and fp8 activation scales.
        if not hasattr(self, "_cached_expert_params_mapping"):
            self._cached_expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
        return self._cached_expert_params_mapping

    def _maybe_log_stage_timer(
        self,
        layer_ms: list[tuple[int, float]],
        offload_ms: float,
        *,
        active: bool,
    ) -> None:
        if not active or not layer_ms:
            return
        total = sum(ms for _, ms in layer_ms) + offload_ms
        parts = [f"L{idx}:{ms:.2f}" for idx, ms in layer_ms]
        if offload_ms > 0:
            parts.append(f"offload:{offload_ms:.2f}")
        logger.info(
            "HYV4_STAGE_TIMER rank=%d steps=%d total_ms=%.2f %s",
            get_pp_group().rank_in_group,
            self._stage_timer_steps + 1,
            total,
            " ".join(parts),
        )
        self._stage_timer_steps += 1

    def _next_pp_pack_send_buf(
        self, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        cap = max(num_tokens, self._pp_max_batched_tokens, 1)
        if (
            not self._pp_pack_bufs
            or self._pp_pack_bufs[0].shape[0] < num_tokens
            or self._pp_pack_bufs[0].device != device
        ):
            self._pp_pack_bufs = [
                torch.empty(
                    (cap, self._pp_send_packed_row_bytes),
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(2)
            ]
        buf = self._pp_pack_bufs[self._pp_pack_slot]
        self._pp_pack_slot ^= 1
        return buf

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        if self._use_pp_packed_wire:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self._pp_recv_packed_row_bytes),
                        dtype=torch.uint8,
                        device=device,
                    )
                }
            )
        tensors = {}
        for key in self._pp_intermediate_keys:
            buf_dtype = (
                torch.uint8 if _pp_act_fp8_fused() and key == "hidden_states" else dtype
            )
            tensors[key] = torch.zeros(
                (batch_size, self._pp_intermediate_hidden),
                dtype=buf_dtype,
                device=device,
            )
        if _pp_act_fp8_fused():
            tensors[_PP_ACT_SCALE_KEY] = torch.zeros(
                (batch_size,), dtype=torch.float32, device=device
            )
        if self.recv_topk:
            tensors[_TOPK_INDICES_KEY] = torch.zeros(
                (batch_size, self.config.index_topk),
                dtype=torch.int32,
                device=device,
            )
        if self.recv_offload_first_topk:
            tensors[_OFFLOAD_FIRST_TOPK_KEY] = torch.zeros(
                (batch_size, self.config.index_topk // 2),
                dtype=torch.int32,
                device=device,
            )
        return IntermediateTensors(tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.topk_index_cache is not None:
            self.topk_index_cache.clear()
        offload_recv_wire: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        elif self._use_pp_packed_wire:
            assert intermediate_tensors is not None
            q, scale, unpacked_topk = unpack_hy4_pp_boundary(
                intermediate_tensors["hidden_states"],
                self._pp_intermediate_hidden,
                self._pp_recv_topk_cols,
            )
            if q.shape[0] >= 128:
                hidden_states = (q, scale)
            else:
                hidden_states = _dequant_pp_act(q, scale)
            residual = None
            if self.recv_topk:
                assert (
                    unpacked_topk is not None and self.topk_indices_buffer is not None
                )
                self.topk_indices_buffer[: unpacked_topk.shape[0]].copy_(unpacked_topk)
                if self.topk_index_cache is not None:
                    self.topk_index_cache.clear()
            elif self.recv_offload_first_topk:
                offload_recv_wire = unpacked_topk
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if _pp_act_fp8_fused():
                scale = intermediate_tensors[_PP_ACT_SCALE_KEY]
                if self.enable_ihc and hidden_states.shape[0] >= 128:
                    if scale.dim() != 1:
                        scale = scale.reshape(hidden_states.shape[0])
                    hidden_states = (hidden_states, scale)
                else:
                    hidden_states = _dequant_pp_act(hidden_states, scale)
            # In iHC mode the flattened [num_tokens, hc*h] tensor from the
            # previous PP stage is reshaped back to 3D by the first layer's
            # prepare_input, and residual is unused.
            residual = None if self.enable_ihc else intermediate_tensors["residual"]
            if self.recv_topk:
                topk_indices = intermediate_tensors[_TOPK_INDICES_KEY]
                assert self.topk_indices_buffer is not None
                self.topk_indices_buffer[: topk_indices.shape[0]].copy_(topk_indices)
                if self.topk_index_cache is not None:
                    self.topk_index_cache.clear()
            if self.recv_offload_first_topk:
                offload_recv_wire = intermediate_tensors[_OFFLOAD_FIRST_TOPK_KEY]

        first_entry_indexer = None
        if self.recv_offload_first_topk and self.start_layer < self.end_layer:
            assert offload_recv_wire is not None
            entry_layer = self.layers[self.start_layer]
            assert isinstance(entry_layer, HYV4DecoderLayer)
            first_entry_indexer = entry_layer.self_attn.indexer
            assert first_entry_indexer is not None
            first_entry_indexer._offload_recv_wire = offload_recv_wire

        stage_timer_active = (
            self._stage_timer_enabled
            and self._stage_timer_steps < self._stage_timer_max_steps
        )
        layer_ms: list[tuple[int, float]] = []
        offload_ms = 0.0
        try:
            if stage_timer_active:
                torch.cuda.synchronize()
            for layer_idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
            ):
                if stage_timer_active:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                hidden_states, residual = layer(positions, hidden_states, residual)
                if stage_timer_active:
                    torch.cuda.synchronize()
                    layer_ms.append((layer_idx, (time.perf_counter() - t0) * 1e3))
        finally:
            if first_entry_indexer is not None:
                first_entry_indexer._offload_recv_wire = None

        if not get_pp_group().is_last_rank:
            global _pp_act_fp8_fused_logged
            if self._use_pp_packed_wire:
                if isinstance(hidden_states, tuple):
                    q, scale = hidden_states
                else:
                    q, scale = _quant_pp_act(hidden_states.flatten(1))
                if not _pp_act_fp8_fused_logged:
                    _pp_act_fp8_fused_logged = True
                    logger.info(
                        "PP activation FP8 fused + single-tensor packed: "
                        "uint8 %s + scale %s (row_bytes=%d)",
                        tuple(q.shape),
                        tuple(scale.shape),
                        self._pp_send_packed_row_bytes,
                    )
                num_tokens = q.shape[0]
                send_buf_2d = self._next_pp_pack_send_buf(num_tokens, q.device)
                packed_2d, wire_view = pack_hy4_pp_boundary(
                    send_buf_2d,
                    q,
                    scale,
                    topk=(
                        self.topk_indices_buffer[:num_tokens]
                        if self.send_topk and self.topk_indices_buffer is not None
                        else None
                    ),
                    topk_cols=self._pp_send_topk_cols,
                )
                if self.offload_first_indexer_layer is not None:
                    assert wire_view is not None
                    if stage_timer_active:
                        torch.cuda.synchronize()
                        t_off = time.perf_counter()
                    self.offload_first_indexer_layer.forward_offload(
                        positions=positions,
                        pp_hidden=q,
                        pp_scale=scale,
                        residual=None,
                        wire_out=wire_view,
                    )
                    if stage_timer_active:
                        torch.cuda.synchronize()
                        offload_ms = (time.perf_counter() - t_off) * 1e3
                self._maybe_log_stage_timer(
                    layer_ms, offload_ms, active=stage_timer_active
                )
                return IntermediateTensors({"hidden_states": packed_2d})

            pp_hidden: torch.Tensor
            pp_scale: torch.Tensor | None = None
            if isinstance(hidden_states, tuple):
                q, scale = hidden_states
                num_tokens = q.shape[0]
                pp_hidden = q
                pp_scale = scale
                tensors = {"hidden_states": q, _PP_ACT_SCALE_KEY: scale}
                if not _pp_act_fp8_fused_logged:
                    _pp_act_fp8_fused_logged = True
                    logger.info(
                        "PP activation FP8 fused iHC: uint8 %s + scale %s",
                        tuple(q.shape),
                        tuple(scale.shape),
                    )
            else:
                num_tokens = hidden_states.shape[0]
                flat = hidden_states.flatten(1) if self.enable_ihc else hidden_states
                if _pp_act_fp8_fused():
                    q, scale = _quant_pp_act(flat)
                    pp_hidden = q
                    pp_scale = scale
                    tensors = {"hidden_states": q, _PP_ACT_SCALE_KEY: scale}
                    if not self.enable_ihc:
                        tensors["residual"] = residual
                    if not _pp_act_fp8_fused_logged:
                        _pp_act_fp8_fused_logged = True
                        logger.info(
                            "PP activation FP8 fused: bf16 %s -> uint8 %s + scale %s",
                            tuple(flat.shape),
                            tuple(q.shape),
                            tuple(scale.shape),
                        )
                elif self.enable_ihc:
                    pp_hidden = flat
                    tensors = {"hidden_states": flat}
                else:
                    pp_hidden = hidden_states
                    tensors = {"hidden_states": hidden_states, "residual": residual}
            if self.send_topk:
                assert self.topk_indices_buffer is not None
                tensors[_TOPK_INDICES_KEY] = self.topk_indices_buffer[:num_tokens]
            if self.offload_first_indexer_layer is not None:
                if stage_timer_active:
                    torch.cuda.synchronize()
                    t_off = time.perf_counter()
                tensors[_OFFLOAD_FIRST_TOPK_KEY] = (
                    self.offload_first_indexer_layer.forward_offload(
                        positions=positions,
                        pp_hidden=pp_hidden,
                        pp_scale=pp_scale,
                        residual=None if self.enable_ihc else residual,
                    )
                )
                if stage_timer_active:
                    torch.cuda.synchronize()
                    offload_ms = (time.perf_counter() - t_off) * 1e3
            self._maybe_log_stage_timer(layer_ms, offload_ms, active=stage_timer_active)
            return IntermediateTensors(tensors)
        self._maybe_log_stage_timer(layer_ms, offload_ms, active=stage_timer_active)
        if self.enable_ihc:
            if isinstance(hidden_states, tuple):
                q, scale = hidden_states
                flat = _dequant_pp_act(q, scale)
                hidden_states = flat.view(
                    flat.shape[0], self.config.hc_mult, self.config.hidden_size
                )
            if _ihc_fuse_norm() and self.hc_head.can_fuse_norm(hidden_states):
                return self.hc_head(
                    hidden_states,
                    rms_weight=self.norm.weight.data,
                    rms_eps=float(self.norm.variance_epsilon),
                )
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual

        return self.norm(hidden_states)

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Quantized indexer wk dequant buffer (weight and scale arrive
        # separately). Uses the plugin helper, not the community one: the
        # baseline derives a single square block shape and would read Hy4's
        # uint8 E8M0 MXFP8 scales as float multipliers.
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        skip_topk_layers = compute_skip_topk_layers(self.config)

        # Must not be cached on `self`: `process_weights_after_loading` swaps in
        # kernel-specific expert weights, and a cache outliving this call would
        # pin the pre-shuffle storage (OOM on large MoE) and make a later reload
        # target orphaned tensors.
        params_dict = dict(self.named_parameters())
        # Split per-expert mapping (V3 style): experts.0.gate_proj.weight
        split_expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        # Sink weights are sharded like the q/k/v linears.
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // sink_tp_size
        head_rank_start = n_local_head * sink_tp_rank
        head_rank_end = n_local_head * (sink_tp_rank + 1)
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        # Fused expert mapping: experts.gate_up_proj (all experts in one tensor).
        # The packed weights are owned by the RoutedExperts submodule, so the
        # targets are experts.routed_experts.[base_layer.]w{13,2}_weight.
        fused_expert_prefix = f".experts.routed_experts.{base_layer}"
        fused_expert_params_mapping = [
            (f"{fused_expert_prefix}w13_weight", ".experts.gate_up_proj", 0, "w1"),
            (f"{fused_expert_prefix}w2_weight", ".experts.down_proj", 0, "w2"),
        ]
        num_experts = getattr(self.config, "num_experts", 0)

        def _should_skip_missing_param(param_name: str) -> bool:
            # Sparse checkpoints may contain indexer weights for layers that
            # fall back to dense MLA on unsupported GPUs.
            return ".indexer." in param_name

        def _is_split_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in split (per-expert) format."""
            # Split format: mlp.experts.<id>.gate_proj.weight
            return bool(re.search(r"\.experts\.\d+\.", weight_name))

        def _is_fused_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in fused (all-experts-packed) format."""
            return ".experts.gate_up_proj" in weight_name or (
                ".experts.down_proj" in weight_name
                and not _is_split_expert_weight(weight_name)
            )

        for name, loaded_weight in weights:
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            if try_load_quantized_indexer_wk(
                name,
                loaded_weight,
                pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Names and stacked shard ids are normalized upstream by
            # AutoWeightsLoader, through quant_config.get_cache_scale_mapper()
            # and hf_to_vllm_mapper.
            stacked_shard_id = getattr(loaded_weight, "shard_id", None)
            if ".indexer.wk." in name:
                name = name.replace(".indexer.wk.", ".indexer.wk_weights_proj.")
                stacked_shard_id = 0
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            # Determine per-weight whether this is fused or split format.
            is_fused_expert = _is_fused_expert_weight(name)
            expert_params_mapping = (
                fused_expert_params_mapping
                if is_fused_expert
                else split_expert_params_mapping
            )

            is_expert_weight = False
            loaded_expert_param_names: set[str] = set()
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                # This is an expert weight and must not be attempted as any
                # other kind of weight later on.
                is_expert_weight = True

                # Do not modify `name`: the loop may continue past this point.
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if is_fused_expert:
                    if "experts.gate_up_proj" in name:
                        chunks = loaded_weight.chunk(2, dim=-2)
                        success_w1 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[0], "w1", num_experts
                        )
                        success_w3 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[1], "w3", num_experts
                        )
                        success = success_w1 and success_w3
                    else:
                        success = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight,
                            shard_id,
                            num_experts,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    # Split per-expert format (V3 style).
                    if name_mapped not in params_dict:
                        if _should_skip_missing_param(name_mapped):
                            continue
                        logger.warning_once(
                            "Skipping unknown checkpoint weight: %s",
                            name_mapped,
                        )
                        continue
                    param = params_dict[name_mapped]
                    # Ask the weight loader whether it succeeded, otherwise we
                    # may skip experts that have other available replicas.
                    weight_loader: Callable[..., typing.Any] = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                if success:
                    if not is_fused_expert:
                        loaded_expert_param_names.add(name_mapped)
                        continue
                    name = name_mapped
                    break
            else:
                if loaded_expert_param_names:
                    loaded_params.update(loaded_expert_param_names)
                    continue
                if "learnable_sink_param" in name:
                    if is_pp_missing_parameter(name, self) or name not in params_dict:
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    param = params_dict[name]
                    with torch.no_grad():
                        param[:n].copy_(narrow_weight.to(param.dtype))
                else:
                    if is_expert_weight:
                        # An expert weight that is not mapped to this rank.
                        continue
                    if "gate.e_score_correction_bias" in name:
                        name = name.replace(
                            "gate.e_score_correction_bias", "expert_bias"
                        )
                    if "router.gate." in name:
                        name = name.replace("router.", "")
                    if "hc_fn" in name:
                        name = name.replace("hc_fn", "hc_fn.weight")
                    if "hc_head_fn" in name:
                        name = name.replace("hc_head_fn", "hc_head_fn.weight")

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        if _should_skip_missing_param(name):
                            continue
                        logger.warning_once(
                            "Skipping unknown checkpoint weight: %s",
                            name,
                        )
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if stacked_shard_id is None:
                        weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(name)

        return loaded_params


class HYV4ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
            ".indexer.weights_proj.": (".indexer.wk_weights_proj.", 1),
        }
    )
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        # MLA runs both latent down-projections as one GEMM.
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        # The indexer fuses wk and weights_proj into one GEMM.
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = HYV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            # The head stays in the model dtype; ``enable_lm_head_fp32`` is
            # surfaced as ``head_dtype`` on the config (see HYV4Config), which
            # makes LogitsProcessor accumulate the projection straight into
            # fp32 instead of materializing an fp32 copy of the weight.
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # With tie_word_embeddings, embed_tokens is kept on the last rank
            # (see HYV4Model.__init__) so the weight can be shared here.
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # MTP layers are loaded by the MTP head, not by the target model. They
        # appear as model.mtp_layers.<i>.* and as the layer ids past the backbone.
        mtp_start = self.config.num_hidden_layers
        drop_prefixes: dict[str, str | None] = {"model.mtp_layers.": None}
        for i in range(getattr(self.config, "num_nextn_predict_layers", 0)):
            drop_prefixes[f"model.layers.{mtp_start + i}."] = None
        if self.config.tie_word_embeddings:
            drop_prefixes["lm_head."] = None
        drop_mapper = WeightsMapper(orig_to_new_prefix=drop_prefixes)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper | drop_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def register_model() -> None:
    """Register the HY V4 arch and its HF config.

    Invoked by ``vllm_iluvatar.ilu_register.apply_oot_registrations`` during
    register_general. The config registration has to happen here too (rather
    than as an import side effect of ``config.py``): ``ModelConfig`` is built
    long after plugin init but ``get_config`` runs before any model module is
    imported, so ``_CONFIG_REGISTRY`` must already hold ``hy_v4`` by then.
    """
    from vllm import ModelRegistry

    from .config import register_configs

    register_configs()

    # Import for its ``@ReasoningParserManager.register_module("hy_v4")`` side
    # effect so ``--reasoning-parser hy_v4`` resolves. Kept out of module import
    # time (vllm.reasoning pulls in the serving stack) and off the hot path.
    from . import reasoning as _hy_v4_reasoning  # noqa: F401

    # Import for its ``@MULTIMODAL_REGISTRY.register_processor`` side effect so the
    # VL model's image/video processor is registered before the model is built.
    from . import vl_model as _hy_v4_vl_model  # noqa: F401

    # Registered under the checkpoint's own arch name: ``config.json`` says
    # ``HYV4ForCausalLM`` and community vLLM has no entry for it, so this is a
    # new registration rather than an override of a community model.
    ModelRegistry.register_model(
        "HYV4ForCausalLM",
        "vllm_iluvatar.models.hy_v4.model:HYV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "HYV4MTPModel",
        "vllm_iluvatar.models.hy_v4.mtp:HYV4MTP",
    )
    # VL composition for the hy_v4_vl checkpoint: vision tower + HYV4ForCausalLM,
    # registered under the checkpoint's own arch name.
    ModelRegistry.register_model(
        "HYV4VLForConditionalGeneration",
        "vllm_iluvatar.models.hy_v4.vl_model:HYV4VLForConditionalGeneration",
    )
