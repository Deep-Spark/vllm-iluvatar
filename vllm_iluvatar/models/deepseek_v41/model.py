"""DeepSeek-V4.1-Flash on Iluvatar, raw FP8 checkpoint.

The text backbone (``IluDeepseekV41ForCausalLM``) is wrapped by
``vl_model.IluDeepseekV41VLForCausalLM`` for vision checkpoints. Text-only
checkpoints reach this class through that wrapper with the tower stubbed out.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from contextlib import nullcontext
from copy import copy
from itertools import islice

import torch
from torch import nn
from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
)
from vllm.model_executor.offloader import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
    producer_kv_only_layer,
    producer_last_layer,
)
from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS
from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    clear_activation_prequant,
    register_activation_prequant,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)
from vllm_iluvatar.models.deepseek_v4.fp8_model import (
    IluDeepseekV4Fp8ForCausalLM,
    IluDeepseekV4Fp8MLP,
    IluDeepseekV4Fp8Model,
    _make_fp8_weights_mapper,
)
from vllm_iluvatar.models.deepseek_v41 import mhc
from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.engram import (
    EngramLayout,
    IluEngram,
    NgramHashState,
    _window_metadata,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan, local_kv_source

logger = init_logger(f"vllm.{__name__}")

def _ll_shared_dispatch_overlap_enabled() -> bool:
    """Decode NVSM/LL: overlap shared MLP on the auxiliary stream. Default off."""
    return bool(
        _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP"]()
    )


def _allreduce_mhc_post_supported(config, tp_size: int) -> bool:
    return (
        envs.VLLM_ILU_V41_ALLREDUCE_MHC_POST
        and tp_size in (2, 4, 8)
        and getattr(config, "hidden_size", None) in (4096, 5120, 7168)
        and getattr(config, "hc_mult", None) == 4
    )


# Engram tables are loaded from mmap, not the vLLM weight iterator.
_ENGRAM_SKIP = re.compile(r"(^|\.)engram\.")

# ``kDeadId`` in the hash kernel (``engram_hash_out`` pad argument),
# ``DEAD_ID`` in community ``deepseek_v41/common/engram.py``: the token-cache
# value that stops an n-gram. Only reachable through ``token_map``.
_ENGRAM_DEAD_ID = -1


def _image_dead_token_id(hf_config) -> int | None:
    """The token id whose positions are dead for n-gram hashing, if any.

    Vision checkpoints fill every image-span position with ``image_token_id``
    (community ``deepseek_v41/common/mm_preprocess.py``), and those positions
    must not enter an n-gram. Text-only checkpoints have no such id.
    """
    if not getattr(hf_config, "vision_n_layers", 0):
        return None
    image_token_id = getattr(hf_config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError(
            "config declares a vision tower but no image_token_id; engram "
            "hashing needs it to recognize image-span positions"
        )
    return int(image_token_id)


def default_aux_hidden_state_layers(hf_config) -> tuple[int, ...]:
    """Aux capture layers for a V4.1 DSpark draft."""
    return tuple(getattr(hf_config, "dspark_target_layer_ids", None) or ())


def _make_weights_mapper() -> WeightsMapper:
    return _make_fp8_weights_mapper()


def _is_pure_prefill(attn_metadata: object, num_tokens: int) -> bool:
    """Return whether metadata describes a prefill-only batch of this size."""
    if not isinstance(attn_metadata, dict):
        return False
    metadata = next(
        (
            value
            for value in attn_metadata.values()
            if hasattr(value, "num_prefill_tokens")
            and hasattr(value, "num_decode_tokens")
        ),
        None,
    )
    if metadata is None:
        return False
    return (
        int(metadata.num_decode_tokens) == 0
        and int(metadata.num_prefill_tokens) == num_tokens
    )


class IluDeepseekV41MoE(nn.Module, IluFusedMoEEP):
    """V4.1 routed FFN.

    V4.1 has no hash-routed layers, so this owns its routing construction
    instead of relying on the V4 class and its ``num_hash_layers`` contract.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        tp_size = get_tensor_model_parallel_world_size()

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.tid2eid = None
        self.gate.e_score_correction_bias = None
        self.gate.bias_vl = None
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

        self.image_sentinel_lo = 0
        if getattr(config, "vision_n_layers", 0) > 0:
            # Vision checkpoints route image sentinel tokens with bias_vl
            # instead of e_score_correction_bias. Created on every MoE layer.
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.image_sentinel_lo = IMAGE_SENTINEL_BASE_ID

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = IluDeepseekV4Fp8MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                swiglu_limit=config.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        eplb_config = parallel_config.eplb_config
        n_physical_experts = config.n_routed_experts + eplb_config.num_redundant_experts
        if n_physical_experts % tp_size != 0:
            raise ValueError(
                f"n_physical_experts={n_physical_experts} must be divisible "
                f"by tp_size={tp_size}"
            )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=None,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
            swiglu_limit=config.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
        )
        # DP ranks jointly form the expert-parallel group.  The generic
        # FusedMoE runner alone does not dispatch/combine activations across
        # that group on Iluvatar; the platform mixin owns that collective.
        self.init_ep()
        # Decode NVSM/LL: run shared MLP on the aux stream so it overlaps
        # IBGDA dispatch. GLM already wires this; V4.1 previously left the
        # mixin default (False) and serialized shared compute behind dispatch.
        self._ll_shared_dispatch_overlap_enabled = _ll_shared_dispatch_overlap_enabled()

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.bias_vl is not None and input_ids is None:
            raise ValueError("DeepSeek V4.1 vision MoE routing requires input_ids.")
        shape = hidden_states.shape
        if self._should_use_forward_dp_ep():
            return self.forward_dp_ep(hidden_states, input_ids).view(shape)
        return self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        ).view(shape)


class IluDeepseekV41DecoderLayer(nn.Module):
    """Independent V4.1 decoder layer with optional Engram write."""

    _fuse_attn_allreduce_post = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        *,
        is_kv_forwarder: bool = False,
        is_kv_sender: bool = False,
        kv_source_override: int | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.layer_id = extract_layer_index(prefix)
        self.producer_kv_only = producer_kv_only_layer(vllm_config) == self.layer_id
        tp_size = get_tensor_model_parallel_world_size()
        self._fuse_attn_allreduce_post = _allreduce_mhc_post_supported(config, tp_size)
        self.attn = IluDeepseekV41Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            is_kv_forwarder=is_kv_forwarder,
            is_kv_sender=is_kv_sender,
            kv_source_override=kv_source_override,
            defer_o_proj_reduce=self._fuse_attn_allreduce_post,
        )
        self.ffn = (
            PPMissingLayer()
            if self.producer_kv_only
            else IluDeepseekV41MoE(vllm_config, prefix=f"{prefix}.ffn")
        )
        if self.producer_kv_only:
            logger.info(
                "DeepSeek V4.1 PD terminal layer %s runs cache writes only; "
                "index selection, attention output and FFN are skipped; "
                "FFN weights are not allocated",
                self.layer_id,
            )
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        # Set by IluDeepseekV41Model for the engram layers only.
        self.engram: IluEngram | None = None
        self._hc_fn_io_cache: dict[str, torch.Tensor] = {}

    def _hc_pre_mixes(
        self,
        x: torch.Tensor,
        name: str,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix_in: torch.Tensor,
        sqrsum: torch.Tensor | None = None,
        norm: torch.nn.Module | None = None,
        want_fp8: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        cached = self._hc_fn_io_cache.get(name)
        if cached is None or cached.dtype != x.dtype:
            cached = hc_fn.to(x.dtype).contiguous()
            self._hc_fn_io_cache[name] = cached
        can_fold = norm is not None and sqrsum is not None
        post, comb, h, pre, h_fp8, h_scale = mhc.hc_pre_mixes(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix_in,
            norm_eps=self.rms_norm_eps,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            post_mult=self.hc_post_alpha,
            hc_fn_io=cached,
            sqrsum=sqrsum,
            layer_norm_weight=norm.weight if can_fold else None,
            layer_norm_eps=self.rms_norm_eps,
            want_fp8=want_fp8 and can_fold,
        )
        if norm is not None and not can_fold:
            h = norm(h)
        return post, comb, h, pre, h_fp8, h_scale

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        *,
        ffn_rows: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(x, residual, post_mix, res_mix, ffn_pre)``; FFN post is delayed."""
        # Each post hands the pre that follows it the sum of squares of the
        # residual it just wrote, so the pre can stream instead of reducing it
        # again. sqrsum stays None wherever that chain is broken: the first
        # layer has no post, and engram rewrites the residual after one.
        sqrsum = None
        if residual is None:
            residual = x
        else:
            residual, sqrsum = mhc.hc_post_sqrsum(x, residual, post_mix, res_mix)

        try:
            valid_prefix = get_forward_context().additional_kwargs.get(
                "prefix_decode_tokens"
            )
        except AssertionError:
            valid_prefix = None
        if self.engram is not None and self.engram._rows is not None:
            if valid_prefix is None:
                residual, sqrsum = self.engram(
                    residual, self.engram._hash_ids, self.engram._token_mask
                )
            elif valid_prefix:
                mask = self.engram._token_mask
                prefix_residual, _ = self.engram(
                    residual[:valid_prefix],
                    self.engram._hash_ids[:valid_prefix],
                    mask[:valid_prefix] if mask is not None else None,
                )
                residual[:valid_prefix].copy_(prefix_residual)
                sqrsum = None
            else:
                sqrsum = None

        attn_post, attn_comb, h, attn_pre, h_fp8, h_scale = self._hc_pre_mixes(
            residual,
            "attn",
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
            sqrsum,
            self.attn_norm,
            want_fp8=True,
        )
        if getattr(self, "producer_kv_only", False):
            # Only a discarded producer token consumes this placeholder. Keep
            # the delayed-post contract so PP/sampling/connector completion is
            # unchanged, without executing the terminal layer's FFN. Padded
            # ranks still skip cache writes for rows that have no local request.
            if valid_prefix is None:
                if h_fp8 is not None:
                    register_activation_prequant(h, h_fp8, h_scale)
                try:
                    h = self.attn(positions, h, None, cache_only=True)
                finally:
                    if h_fp8 is not None:
                        clear_activation_prequant()
            elif valid_prefix:
                h[:valid_prefix].copy_(
                    self.attn(
                        positions[:valid_prefix],
                        h[:valid_prefix],
                        None,
                        cache_only=True,
                    )
                )
            return h, residual, attn_post, attn_comb, attn_pre
        # The fused_wqa_wkv fp8 GEMM reads this exact normed ``h``; hand it the
        # per-token quant the pre-norm kernel already produced so it skips its own.
        if valid_prefix is None:
            if h_fp8 is not None:
                register_activation_prequant(h, h_fp8, h_scale)
            try:
                h = self.attn(positions, h, None)
            finally:
                if h_fp8 is not None:
                    clear_activation_prequant()
        else:
            # Padded ranks still participate in MoE collectives, but never issue
            # attention/cache writes for rows that have no local request.
            if valid_prefix:
                h[:valid_prefix].copy_(
                    self.attn(positions[:valid_prefix], h[:valid_prefix], None)
                )
            h[valid_prefix:].zero_()
        if self._fuse_attn_allreduce_post:
            residual, attn_sqrsum = mhc.allreduce_hc_post(
                h,
                residual,
                attn_post,
                attn_comb,
                group=get_tp_group().device_group,
            )
        else:
            residual, attn_sqrsum = mhc.hc_post_sqrsum(
                h, residual, attn_post, attn_comb
            )

        ffn_post, ffn_comb, h, ffn_pre, _, _ = self._hc_pre_mixes(
            residual,
            "ffn",
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
            attn_sqrsum,
            self.ffn_norm,
        )
        if ffn_rows is None:
            h = self.ffn(h, input_ids)
        else:
            h = self._forward_tail_ffn(h, input_ids, ffn_rows)
        return h, residual, ffn_post, ffn_comb, ffn_pre

    def _forward_tail_ffn(self, h, input_ids, rows):
        if rows is None:
            return self.ffn(h, input_ids)
        if not rows:
            return torch.zeros_like(h)
        # Fresh pinned storage is stream-owned during the asynchronous copy;
        # a mutable persistent host index buffer could race the next PP step.
        indices = torch.tensor(
            rows, dtype=torch.int64, device="cpu", pin_memory=h.is_cuda
        ).to(device=h.device, non_blocking=h.is_cuda)
        selected = h.index_select(0, indices)
        selected_ids = None if input_ids is None else input_ids.index_select(0, indices)
        context = get_forward_context()
        mask = getattr(context, "is_padding", None)
        scoped = copy(context) if mask is not None else context
        if mask is not None:
            scoped.is_padding = mask.index_select(0, indices)
        try:
            with override_forward_context(scoped):
                output = self.ffn(selected, selected_ids)
        finally:
            if scoped is not context:
                context.moe_layer_index = scoped.moe_layer_index
        result = torch.zeros_like(h)
        result.index_copy_(0, indices, output)
        return result


class IluDeepseekV41Model(IluDeepseekV4Fp8Model, EagleModelMixin):
    """V4.1 backbone: V4.1 attention, the engram layers, and the pipeline cut.

    ``supports_aux_hidden_states_over_pp`` is the contract the V2 runner checks
    before a DSpark draft. Capture stays ``idx + 1`` (the id V4.1 stores), which
    is what ``EagleModelMixin._cache_aux_pp_layout`` counts with ``bisect_right``.
    Aux captured on an earlier PP rank is packed into the intermediate tensors
    and concatenated on the last rank, where the drafter runs.

    The constructor is V4.1's own rather than the V4 parent's, because the
    layer list is where the pipeline plan has to be applied: which layer
    replicates a stranded KV group, which one mirrors its records for the next
    rank, and which cache every consumer should read. Weight loading is still
    the parent's -- the FP8 checkpoint layout is identical and there is nothing
    to gain from a second copy of it.
    """

    decoder_layer_cls = IluDeepseekV41DecoderLayer
    supports_aux_hidden_states_over_pp = True
    main_kv_scale_width = 0
    main_kv_scale_rows = None

    # Shared-expert gate_up, plus attention's wq_a/wkv (both read the same normed
    # hidden state, so they are one GEMM), plus the compressor's wkv/wgate, which
    # both read the layer's hidden state.
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
        ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self._producer_ffn_layer = (
            producer_last_layer(vllm_config)
            if envs.VLLM_ILU_DS41_PRODUCER_TAIL_FFN
            and self.parallel_config.tensor_parallel_size == 1
            and self.parallel_config.data_parallel_size == 1
            else None
        )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        # V4.1 has no global MHC head and no MTP draft, so neither the
        # ``hc_head_*`` parameters nor the residual buffer the V4 parent
        # allocates for one exist here.
        self._mtp_hidden_buffer = None
        # Filled by the inherited load_weights, read by the load verifier.
        self.consumed_ckpt_names: set[str] = set()
        # Set by IluDeepseekV41ForCausalLM.set_aux_hidden_state_layers when a
        # DSpark draft is active.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.topk_indices_buffer = torch.full(
            (max_tokens, config.index_topk), -1, dtype=torch.int32, device=self.device
        )
        self._initial_pre_mix = torch.zeros(
            max_tokens,
            self.hc_mult,
            dtype=torch.float32,
            device=self.device,
        )
        self._initial_pre_mix[:, 0] = 1.0

        pp = get_pp_group()
        if pp.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
            self.embed_tokens = get_offloader().wrap_modules(
                iter([self.embed_tokens]), prefix=f"{prefix}.embed_tokens"
            )[0]
        else:
            self.embed_tokens = PPMissingLayer()

        self._build_layers(vllm_config, prefix, pp)
        self._build_transfer_buffers(max_tokens)

        if pp.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self._bind_index_buffers(max_tokens)

        disable_engram = os.getenv("VLLM_ILU_V41_DISABLE_ENGRAM", "0") != "0"
        self.engram_layer_ids = (
            []
            if disable_engram
            else list(getattr(config, "engram_layer_ids", ()) or ())
        )
        if disable_engram:
            logger.info(
                "DeepSeek V4.1 Engram disabled for migration step one "
                "(VLLM_ILU_V41_DISABLE_ENGRAM=1)"
            )
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: NgramHashState | None = None
        self.image_dead_token_id = _image_dead_token_id(config)
        if self.engram_layer_ids:
            model_config = vllm_config.model_config
            self.model_path = model_config.model
            self._build_engram(config, model_config)
        # Flipped at the end of load_weights. The listener is up before that,
        # and it refuses a snapshot until this is set.
        self._weights_ready = False

    # ---- construction -----------------------------------------------------

    def _build_layers(self, vllm_config: VllmConfig, prefix: str, pp) -> None:
        """The local layers, each told what the pipeline cut made of it."""
        config = self.config
        kv_sources = tuple(getattr(config, "kv_source_layer_ids", None) or ())
        index_sources = tuple(getattr(config, "index_source_layer_ids", None) or ())
        compress_ratios = tuple(
            int(r) for r in (getattr(config, "compress_ratios", None) or ())
        )
        plan = build_rank_plan(
            config.num_hidden_layers,
            kv_sources,
            index_sources,
            compress_ratios,
            pp.rank_in_group,
            pp.world_size,
            # -1 unless a shortlist is actually allocated, so the plan and
            # ``_bind_index_buffers`` cannot disagree about whether there is
            # anything to carry.
            candidate_source=(
                int(getattr(config, "candidate_source_layer_id", -1))
                if int(getattr(config, "candidate_topk_blocks", 0)) > 0
                else -1
            ),
            last_layer=producer_last_layer(vllm_config),
        )
        self.pp_plan = plan
        self.start_layer, self.end_layer = plan.start_layer, plan.end_layer

        # Only the last KV source on this rank has to mirror its records, and
        # only when the next rank is actually stranded: an earlier source's
        # mirror would be overwritten before the rank finishes anyway, and a
        # mirror nobody receives is pure cost.
        local_sources = [
            s for s in kv_sources if plan.start_layer <= s < plan.end_layer
        ]
        self.kv_sender_layer = (
            max(local_sources) if local_sources and plan.sends_kv else None
        )
        # With no local source to refill it, the outgoing buffer is whatever
        # arrived, so a group spanning three or more ranks needs this rank to
        # pass it along.
        self.relays_kv = plan.sends_kv and self.kv_sender_layer is None

        def build(layer_id: int) -> nn.Module:
            real_source = None
            if (
                layer_id < len(compress_ratios)
                and compress_ratios[layer_id] > 0
                and kv_sources
            ):
                eligible = [s for s in kv_sources if s <= layer_id]
                real_source = max(eligible) if eligible else None
            return self.decoder_layer_cls(
                vllm_config,
                prefix=f"{prefix}.layers.{layer_id}",
                topk_indices_buffer=self.topk_indices_buffer,
                is_kv_forwarder=layer_id == plan.forwarder,
                is_kv_sender=layer_id == self.kv_sender_layer,
                kv_source_override=local_kv_source(layer_id, real_source, plan),
            )

        self.layers = nn.ModuleList(
            [PPMissingLayer() for _ in range(plan.start_layer)]
            + get_offloader().wrap_modules(
                build(layer_id) for layer_id in range(plan.start_layer, plan.end_layer)
            )
            + [
                PPMissingLayer()
                for _ in range(plan.end_layer, config.num_hidden_layers)
            ]
        )

        last_layer = producer_last_layer(vllm_config)
        if last_layer is not None:
            logger.info(
                "DeepSeek V4.1 PD prefiller stops after layer %s (last kv "
                "source); rank %s/%s runs [%s, %s)",
                last_layer,
                pp.rank_in_group,
                pp.world_size,
                plan.start_layer,
                plan.end_layer,
            )

        if plan.recvs_kv or plan.sends_kv or plan.recvs_candidates:
            logger.info(
                "DeepSeek V4.1 pipeline rank %s/%s owns layers [%s, %s); "
                "kv forwarder=%s (replicating source %s), kv sender=%s, "
                "relays=%s; receives kv=%s topk=%s candidates=%s; "
                "sends kv=%s topk=%s candidates=%s",
                pp.rank_in_group,
                pp.world_size,
                plan.start_layer,
                plan.end_layer,
                plan.forwarder,
                plan.forwarded_kv_source,
                self.kv_sender_layer,
                self.relays_kv,
                plan.recvs_kv,
                plan.recvs_topk,
                plan.recvs_candidates,
                plan.sends_kv,
                plan.sends_topk,
                plan.sends_candidates,
            )

    def _build_transfer_buffers(self, max_tokens: int) -> None:
        """Dense per-token record buffers for the hop out of this rank.

        Only a rank that sends records allocates them: the local kv source
        mirrors into them, or -- for a group spanning three or more ranks --
        the forwarder relays into them what it received. The receiving side
        needs no buffer of its own, because the receive allocates from the
        sender's metadata; it does need the record geometry, which is why that
        is read off a layer whether or not anything is allocated here.
        """
        self.main_kv_rows: torch.Tensor | None = None
        self.main_kv_scale_rows: torch.Tensor | None = None
        self.indexer_kv_rows: torch.Tensor | None = None
        self.main_kv_record: tuple[int, torch.dtype] | None = None
        self.indexer_kv_record: tuple[int, torch.dtype] | None = None
        sample = self.layers[self.start_layer].attn
        self.main_kv_scale_width = sample.main_scale_transfer_width
        self.main_kv_record = (sample.main_transfer_width, sample.main_transfer_dtype)
        self.indexer_kv_record = (
            sample.indexer_transfer_width,
            sample.indexer_transfer_dtype,
        )
        if not self.pp_plan.sends_kv:
            return
        assert self.kv_sender_layer is not None or self.pp_plan.recvs_kv, (
            f"rank bounds [{self.start_layer}, {self.end_layer}) have to send "
            "kv records but own no source and receive none to relay"
        )

        main_width, main_dtype = self.main_kv_record
        indexer_width, indexer_dtype = self.indexer_kv_record
        self.main_kv_rows = torch.empty(
            max_tokens, main_width, dtype=main_dtype, device=self.device
        )
        if self.main_kv_scale_width:
            self.main_kv_scale_rows = torch.empty(
                max_tokens,
                self.main_kv_scale_width,
                dtype=torch.float16,
                device=self.device,
            )
        self.indexer_kv_rows = torch.empty(
            max_tokens, indexer_width, dtype=indexer_dtype, device=self.device
        )
        if self.kv_sender_layer is not None:
            sender = self.layers[self.kv_sender_layer].attn
            sender.main_kv_rows = self.main_kv_rows
            sender.main_kv_scale_rows = self.main_kv_scale_rows
            assert sender.indexer is not None, (
                f"layer {self.kv_sender_layer} is a kv source and must own an "
                "indexer to mirror its index keys"
            )
            sender.indexer.k_row_buffer = self.indexer_kv_rows

    def _bind_index_buffers(self, max_tokens: int) -> None:
        """Preallocate the selection buffers the indexers share across layers.

        ``topk_lens`` because V4.1's selection is ragged and our kernels take
        the count explicitly, and the candidate list because the indexer at
        ``candidate_source_layer_id`` publishes a block mask that the ratio-1
        indexers above it score against.

        Allocating once is not only about the per-step cost: an indexer writing
        into a buffer of a fixed address is what lets the selection cross layers
        without any layer holding a reference to another layer's tensor.
        """
        config = self.config
        device = self.topk_indices_buffer.device
        self.topk_lens_buffer = torch.zeros(
            max_tokens, dtype=torch.int32, device=device
        )
        blocks = getattr(config, "candidate_topk_blocks", 0)
        if getattr(config, "candidate_source_layer_id", -1) >= 0 and blocks > 0:
            self.candidate_idxs_buffer = torch.full(
                (max_tokens, blocks), -1, dtype=torch.int32, device=device
            )
            self.candidate_lens_buffer = torch.zeros(
                max_tokens, dtype=torch.int32, device=device
            )
        else:
            self.candidate_idxs_buffer = None
            self.candidate_lens_buffer = None

        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                attn.bind_index_buffers(
                    self.topk_lens_buffer,
                    self.candidate_idxs_buffer,
                    self.candidate_lens_buffer,
                )

    def _build_engram(self, config, model_config) -> None:
        """Create the shared token history and one IluEngram per engram layer.

        The compressed token map is derived from the tokenizer (about 5 s for
        129280 ids) and asserted against ``engram_compressed_vocab_size``: every
        hash multiplier is derived from that number, so a mismatch would silently
        rehash all 189 GiB of tables.
        """
        local_ids = [
            lid
            for lid in self.engram_layer_ids
            if not isinstance(self.layers[lid], PPMissingLayer)
        ]
        if not local_ids:
            return

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        token_map, compressed_vocab = build_compressed_token_map(tokenizer)
        expected = config.engram_compressed_vocab_size
        if compressed_vocab != expected:
            raise ValueError(
                f"engram compressed vocab is {compressed_vocab} but the config "
                f"declares {expected}; every hash multiplier derives from this "
                "number, so the tables would be rehashed incorrectly"
            )
        if self.image_dead_token_id is not None:
            token_map[self.image_dead_token_id] = _ENGRAM_DEAD_ID

        layout = EngramLayout.from_config(config)
        assert layout is not None
        self.engram_layout = layout
        multipliers = compute_hash_multipliers(
            self.engram_layer_ids, config.engram_max_ngram_size, compressed_vocab
        )
        swa = self.layers[local_ids[0]].attn.swa_cache_layer
        self.engram_hash = NgramHashState(
            layout,
            swa,
            token_map,
            multipliers,
            token_map[config.engram_pad_token_id],
            self.device,
        )
        object.__setattr__(swa, "_engram_hash_state", self.engram_hash)
        self._engram_dummy = torch.full(
            (self.topk_indices_buffer.shape[0], layout.n_hash_cols),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        prefetch_stream = torch.cuda.Stream(device=self.device)
        for layer_id in self.engram_layer_ids:
            if layer_id not in local_ids:
                continue
            self.layers[layer_id].engram = IluEngram(
                config,
                layer_id,
                layout.primes[layout.layer_ids.index(layer_id)],
                multipliers[layout.layer_ids.index(layer_id)],
                token_map[config.engram_pad_token_id],
                self.device,
                prefetch_stream=prefetch_stream,
            )

    # ---- pipeline ---------------------------------------------------------

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """The posted multi-stream hidden states, the leftover ``pre_mix``, and
        whatever the hop into this rank carries.

        This is exactly ``recvs_*``, no wider: the runner copies the keys it
        received into these buffers and then hands the model *these* keys, so a
        key declared here that rank-1 never sends would reach the layers
        holding the previous step's rows rather than raising.
        """
        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
        }
        try:
            is_first_rank = get_pp_group().is_first_rank
        except AssertionError:
            is_first_rank = True
        if not is_first_rank:
            tensors["input_ids"] = torch.zeros(
                (batch_size,), dtype=torch.int32, device=device
            )
        if self.pp_plan.recvs_kv:
            assert self.main_kv_record is not None
            assert self.indexer_kv_record is not None
            main_width, main_dtype = self.main_kv_record
            indexer_width, indexer_dtype = self.indexer_kv_record
            tensors["main_kv"] = torch.zeros(
                (batch_size, main_width), dtype=main_dtype, device=device
            )
            if self.main_kv_scale_width:
                tensors["main_kv_scales"] = torch.zeros(
                    (batch_size, self.main_kv_scale_width),
                    dtype=torch.float16,
                    device=device,
                )
            tensors["indexer_kv"] = torch.zeros(
                (batch_size, indexer_width), dtype=indexer_dtype, device=device
            )
        if self.pp_plan.recvs_topk:
            tensors["topk"] = torch.zeros(
                (batch_size, self.config.index_topk),
                dtype=torch.int32,
                device=device,
            )
        if self.pp_plan.recvs_candidates:
            assert self.candidate_idxs_buffer is not None
            tensors["candidates"] = torch.zeros(
                (batch_size, self.candidate_idxs_buffer.shape[1]),
                dtype=torch.int32,
                device=device,
            )
        return IntermediateTensors(tensors)

    def _accept_remote_kv(
        self,
        intermediate_tensors: IntermediateTensors,
        positions: torch.Tensor,
    ) -> None:
        """Hand a stranded group's records to this rank's forwarder."""
        if not self.pp_plan.recvs_kv:
            return
        forwarder = self.pp_plan.forwarder
        assert forwarder is not None
        main_rows = intermediate_tensors["main_kv"]
        indexer_rows = intermediate_tensors["indexer_kv"]
        topk_rows = intermediate_tensors["topk"] if self.pp_plan.recvs_topk else None
        self.layers[forwarder].attn.accept_remote_kv(
            main_rows,
            indexer_rows,
            topk_rows,
            positions,
            main_scale_rows=(
                intermediate_tensors["main_kv_scales"]
                if self.main_kv_scale_width
                else None
            ),
        )
        if self.relays_kv:
            # No local source will refill the outgoing buffer, so the group
            # continues past this rank only if we pass on what we received.
            assert self.main_kv_rows is not None
            assert self.indexer_kv_rows is not None
            num_tokens = main_rows.shape[0]
            self.main_kv_rows[:num_tokens].copy_(main_rows)
            if self.main_kv_scale_width:
                assert self.main_kv_scale_rows is not None
                self.main_kv_scale_rows[:num_tokens].copy_(
                    intermediate_tensors["main_kv_scales"]
                )
            self.indexer_kv_rows[:num_tokens].copy_(indexer_rows)

    def _accept_remote_candidates(
        self, intermediate_tensors: IntermediateTensors
    ) -> None:
        """Take over the shortlist published on an earlier rank.

        Done by every rank the shortlist reaches, forwarder or not and reader
        or not: a rank with no index source of its own still has to hand it to
        the next one, and copying into the local buffer is what makes the send
        at the end of :meth:`forward` pick it up.
        """
        if not self.pp_plan.recvs_candidates:
            return
        assert self.candidate_idxs_buffer is not None
        rows = intermediate_tensors["candidates"]
        num_tokens = rows.shape[0]
        self.candidate_idxs_buffer[:num_tokens].copy_(rows)

    def _engram_layers(self):
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                yield engram

    def _prepare_engram(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Hash once, then prefetch local heads or synchronously gather across DP.

        Across-DP lookups are collectives. A replica that skips the real hash still
        submits dead ids so the other replicas' rows, which live on this rank's
        shard, are still read.
        """
        self._mark_batch_image_tokens(input_ids)
        ready = (
            input_ids is not None
            and self.engram_hash is not None
            and self.engram_hash.ensure_cache()
        )
        md = _window_metadata() if ready else None
        if ready and md is not None and md.slot_mapping is not None:
            n = min(input_ids.size(0), positions.size(0))
            hashes = self.engram_hash(
                input_ids[:n], positions[:n], md, lookback_token_ids
            )
            keep = md.is_valid_token[:n]
            if self.image_dead_token_id is not None:
                keep = keep & (input_ids[:n] != self.image_dead_token_id)
            for engram in self._engram_layers():
                engram.prepare_embeddings(hashes[:, engram.layer_hash_index])
                engram._token_mask = keep
            return
        if self.engram_layout is None or not any(
            engram.embed._dp_group is not None for engram in self._engram_layers()
        ):
            return
        t = 0 if input_ids is None else input_ids.size(0)
        dummy = self._engram_dummy[:t]
        dummy.fill_(-1)
        for engram in self._engram_layers():
            engram.prepare_embeddings(dummy)
            engram._rows = None

    def _mark_batch_image_tokens(self, input_ids: torch.Tensor | None) -> None:
        # Leave the predicate on the device. ``.any().item()`` would sync the
        # compute stream here, while the previous step's CUDA graph may still
        # be running. The first MoE op reads it after this forward has queued
        # more launches.
        pred = None
        if (
            self.image_dead_token_id is not None
            and input_ids is not None
            and input_ids.numel() > 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            pred = (input_ids == self.image_dead_token_id).any()
        try:
            ctx = get_forward_context()
            # Reset the resolved flag together with the predicate so a reader
            # that runs before the first MoE layer cannot see the previous
            # step's answer. False is the conservative default.
            ctx.ilu_v41_has_image_tokens = False
            ctx.ilu_v41_image_token_pred = pred
        except AssertionError:
            # ``get_forward_context`` asserts when no forward context is set
            # (tests, capture setup). Engram image masking does not use this
            # flag: ``_prepare_engram`` still ANDs ``input_ids != dead_id``.
            logger.debug("V4.1 image-token flag skipped: forward context is not set")

    producer_prefix_state = None
    consumer_prefix_state = None

    def _compressed_kv_workspace(self):
        if self.start_layer >= self.end_layer:
            return None
        attn = getattr(self.layers[self.start_layer], "attn", None)
        return getattr(attn, "compressed_kv_workspace", None)

    def compressed_kv_forward_scope(self):
        """Scope the complete local forward, including producer-prefix replay."""
        workspace = self._compressed_kv_workspace()
        return nullcontext() if workspace is None else workspace.forward_scope()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        with self.compressed_kv_forward_scope():
            context = self.consumer_prefix_state
            if context is not None:
                return context.forward(
                    self._forward,
                    input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                    lookback_token_ids,
                )
            return self._forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
            )

    def _forward_full(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Embed -> local layers -> leftover FFN post -> collapse or PP send.

        Returns ``(hidden_states, aux)`` instead of a bare tensor when a DSpark
        draft asked for aux layers. Aux captured on an earlier PP rank is packed
        into the intermediate tensors and concatenated on the last rank, which
        is where the drafter runs. PP=1 leaves the packed keys empty and returns
        only this rank's captures.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
            pre_mix = self._initial_pre_mix[: hidden_states.shape[0]]
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            pre_mix = intermediate_tensors["pre_mix"]
            input_ids = intermediate_tensors["input_ids"]
            # Before any layer runs: the stranded group's caches have to hold
            # this step's rows by the time the first consumer reads them.
            self._accept_remote_kv(intermediate_tensors, positions)
            self._accept_remote_candidates(intermediate_tensors)

        self._prepare_engram(input_ids, positions, lookback_token_ids)
        residual = post_mix = res_mix = None
        aux_hidden_states: list[torch.Tensor] = []
        ffn_rows = None
        source_layer = getattr(self, "_producer_ffn_layer", None)
        if (
            source_layer is not None
            and self.end_layer == source_layer + 1
            and source_layer + 1 not in self.aux_hidden_state_layers
            and self.producer_prefix_state is not None
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
        ):
            ffn_rows = self.producer_prefix_state.tail_ffn_rows(positions.shape[0])
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer), self.start_layer
        ):
            if idx == source_layer and ffn_rows is not None:
                hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    ffn_rows=ffn_rows,
                )
            else:
                hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                )
            if (idx + 1) in self.aux_hidden_state_layers:
                # ``hidden_states`` is this layer's delayed FFN output, not the
                # hc stream, so folding it is what the next layer would do first.
                # Mean-pool the hc_mult copies to the [T, H] the DSpark combiner
                # wants -- a plain mean, not the learned ``pre_mix`` collapse.
                aux_hidden_states.append(
                    mhc.hc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1)
                )

        hidden_states = mhc.hc_post(hidden_states, residual, post_mix, res_mix)
        if self.producer_prefix_state is not None:
            self.producer_prefix_state.capture(self, hidden_states, pre_mix, positions)
        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states, "pre_mix": pre_mix}
            num_tokens = positions.shape[0]
            attn_metadata = get_forward_context().attn_metadata
            pp_quant_senders = envs.VLLM_ILU_V41_PREFILL_PP_QUANT_SENDERS
            sender_ranks = {
                rank.strip() for rank in pp_quant_senders.split(",") if rank.strip()
            }
            if (
                pp_quant_senders
                and num_tokens > 128
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                and get_tensor_model_parallel_world_size() == 1
                and _is_pure_prefill(attn_metadata, num_tokens)
                and (
                    pp_quant_senders == "all"
                    or str(get_pp_group().rank_in_group) in sender_ranks
                )
            ):
                from ._prefill_pp_codec import encode

                tensors["hidden_states"], tensors["prefill_pp_scale"] = encode(
                    hidden_states
                )
            if self.pp_plan.sends_kv:
                assert self.main_kv_rows is not None
                assert self.indexer_kv_rows is not None
                tensors["main_kv"] = self.main_kv_rows[:num_tokens]
                if self.main_kv_scale_width:
                    assert self.main_kv_scale_rows is not None
                    tensors["main_kv_scales"] = self.main_kv_scale_rows[:num_tokens]
                tensors["indexer_kv"] = self.indexer_kv_rows[:num_tokens]
            if self.pp_plan.sends_topk:
                # No designated sender: every index source rewrites this
                # buffer, so at the end of the rank it already holds the
                # latest selection -- this rank's, or the one it took over.
                tensors["topk"] = self.topk_indices_buffer[:num_tokens]
            if self.pp_plan.sends_candidates:
                # Same story as the top-k: whichever rank last wrote the
                # shortlist left it in this buffer, so the send needs no
                # designated sender.
                assert self.candidate_idxs_buffer is not None
                tensors["candidates"] = self.candidate_idxs_buffer[:num_tokens]
            if input_ids is None:
                raise ValueError(
                    "DeepSeek V4.1 pipeline stage has no input_ids to forward."
                )
            if input_ids.dtype != torch.int32:
                input_ids = input_ids.to(torch.int32)
            tensors["input_ids"] = input_ids
            tensors.update(self.pack_local_aux_hidden_states(aux_hidden_states))
            return IntermediateTensors(tensors)
        hidden_states = mhc.hc_pre(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        aux_hidden_states = (
            self.collect_remote_aux_hidden_states(intermediate_tensors)
            + aux_hidden_states
        )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    _forward = _forward_full

    def load_engram_weights(self) -> None:
        """Load the engram tensors that bypass the vLLM weight path."""
        for layer_id in self.engram_layer_ids:
            layer = self.layers[layer_id]
            if (
                isinstance(layer, PPMissingLayer)
                or getattr(layer, "engram", None) is None
            ):
                continue
            layer.engram.load_from_checkpoint(self.model_path)


class IluDeepseekV41ForCausalLM(IluDeepseekV4Fp8ForCausalLM, SupportsEagle3):
    """DeepSeek-V4.1-Flash on Iluvatar, raw FP8 checkpoint."""

    is_attention_free = False

    model_cls = IluDeepseekV41Model
    hf_to_vllm_mapper = _make_weights_mapper()

    _LOAD_SKIP_SUBSTRS = ("mtp.",)

    # The runner hands ``input_ids`` to the model even when it also builds
    # ``inputs_embeds``: both the MoE image routing and the engram hashing read
    # raw ids. ``IluDeepseekV41ModelState`` forwards them past the first PP rank.
    requires_raw_input_tokens = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        raw_input_ids: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids if input_ids is not None else raw_input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Keep only the text target weights for migration step one."""

        def text_weights():
            for name, tensor in weights:
                if _ENGRAM_SKIP.search(name):
                    continue
                if any(part in name for part in self._LOAD_SKIP_SUBSTRS):
                    continue
                yield name, tensor

        loaded = super().load_weights(text_weights())
        if getattr(self.model, "engram_layer_ids", None):
            self.model.load_engram_weights()
        self.model._weights_ready = True
        return loaded

    @property
    def get_mtp_target_hidden_states(self):
        """Absent on purpose: V4.1 has no MTP draft.

        The parent hands the drafter its pre-hc_head residual buffer, which V4.1
        never fills (``_mtp_hidden_buffer`` is None). The runner probes for this
        hook with ``hasattr`` and then indexes whatever it gets back, so
        inheriting it means a ``NoneType`` subscript; raising ``AttributeError``
        is what makes ``hasattr`` answer False for an inherited name. DSpark does
        not want the residual anyway -- it builds its own input from the aux
        hidden states, and ignores ``last_hidden_states`` whenever aux is present.
        """
        raise AttributeError(
            "IluDeepseekV41ForCausalLM has no MTP draft; DSpark reads the aux "
            "hidden states instead of the pre-hc_head residual"
        )

    @staticmethod
    def get_model_state_cls():
        """Iluvatar V4.1 state: SWA bounded replay, PP raw token ids, prefix reuse."""
        from vllm_iluvatar.models.deepseek_v41.model_state import (
            IluDeepseekV41ModelState,
        )

        return IluDeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Gather chunk predecessors even when SWA pages were not cached."""
        state = self.model.engram_hash
        return state.lookback_depth if state is not None else 0

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # Sorts the ids and fills the PP slot cache the runner's
        # ``reserve_aux_intermediate_tensor_slots`` reads.
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return default_aux_hidden_state_layers(self.config)
