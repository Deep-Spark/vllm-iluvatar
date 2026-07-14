# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4 W4A8 model.

Adapted from :mod:`vllm.model_executor.models.deepseek_v4`. The ilu fork
diverges in three places:

1. **Quantization scheme** - this model expects checkpoints converted by
   ``my_tests/deepseek/deepseek-v4/convert_w4a8/convert_v4_to_w4a8.py``,
   which write a fresh ``quantization_config`` block whose
   ``quant_method == "ilu_deepseek_v4_w4a8"`` (handled by
   :class:`IluDeepseekV4W4A8Config`). Only ``IluDeepseekV4ForCausalLM``
   should be paired with that config; the upstream ``DeepseekV4ForCausalLM``
   continues to handle the original FP4/FP8 layouts.

2. **MegaMoE removed** - the deep-gemm MegaMoE path, its triton input
   staging kernel, the bespoke expert params mapping and the
   ``deepseek_v4_mega_moe_experts`` custom op are all gone. Only the
   :class:`FusedMoE` route remains.

3. **Attention path** - imports :class:`IluDeepseekV4Indexer` /
   :class:`IluDeepseekV4MLAModules` /
   :class:`IluDeepseekV4MultiHeadLatentAttentionWrapper` from
   :mod:`vllm_iluvatar.model_executor.models.deepseek_v4.attention`. KV cache is
   bf16 throughout (no ``fp8_ds_mla``).

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
It registers both the causal-LM arch and the MTP draft arch.

The weights mapper accounts for three concrete differences vs upstream:

- The offline converter renames ``.scale`` to ``.weight_scale`` already, so
  the regex ``\\.scale$ -> .weight_scale_inv`` is dropped (vLLM's internal
  parameter names for both W4A8 MoE and W8A8 Linear are ``weight_scale``).

- ``"head.weight"``, ``"embed.weight"``, ``".ffn.gate.bias"``,
  ``".attn.compressor."`` and ``".shared_experts.w2"`` are renamed as in
  upstream's mapper (vLLM groups gate/up into a single
  ``gate_up_proj`` and exposes the bias as ``e_score_correction_bias``).

- The ``stacked_params_mapping`` keeps the ``("gate_up_proj", "w1"/"w3", ...)``
  shards (covers shared_experts), the
  ``("attn.fused_wqa_wkv", "attn.wq_a/wkv", ...)`` shards and the
  ``("compressor.fused_wkv_wgate", "compressor.wkv/wgate", ...)`` shards;
  there is nothing MegaMoE-specific to remove.
"""

import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import ixformer.inference.functions as ixfops  # type: ignore[import]
import regex as re
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.multi_stream_utils import AuxStreamType

from vllm_iluvatar.model_executor.models.deepseek_v4.attention import (
    IluDeepseekV4Indexer,
    IluDeepseekV4MLAModules,
    IluDeepseekV4MultiHeadLatentAttentionWrapper,
)

logger = init_logger(__name__)

# A/B/C switch for the mhc_pre operator via IXFORMER_DS4_MHC_PRE ∈ {v1,v2,v3}:
#   v1 (default) : original fused ds4_mhc_pre
#   v2           : bf16 GEMM + single warp-synchronous epilogue (ds4_mhc_pre_v2)
#   v3           : bf16 GEMM + split epilogue rms/Sinkhorn-coef/apply
#                  (ds4_mhc_pre_v3, occupancy-isolation experiment)
# Legacy IXFORMER_DS4_MHC_PRE_V2=1 still selects v2 when the new var is unset.
_DS4_MHC_PRE_VERSION = os.environ.get("IXFORMER_DS4_MHC_PRE", "").strip().lower()
if not _DS4_MHC_PRE_VERSION:
    _DS4_MHC_PRE_VERSION = (
        "v2" if os.environ.get("IXFORMER_DS4_MHC_PRE_V2", "1") == "1" else "v1"
    )
if _DS4_MHC_PRE_VERSION not in ("v1", "v2", "v3"):
    logger.warning(
        "IXFORMER_DS4_MHC_PRE=%s is invalid; falling back to v1",
        _DS4_MHC_PRE_VERSION,
    )
    _DS4_MHC_PRE_VERSION = "v1"
if _DS4_MHC_PRE_VERSION == "v2":
    logger.info("DeepSeek-V4 mhc_pre: using ds4_mhc_pre_v2 (bf16 GEMM + epilogue)")
elif _DS4_MHC_PRE_VERSION == "v3":
    logger.info(
        "DeepSeek-V4 mhc_pre: using ds4_mhc_pre_v3 (bf16 GEMM + split epilogue)"
    )


# ---------------------------------------------------------------------------
# Inlined helpers (originally in vllm.model_executor.models.deepseek_v4).
# ---------------------------------------------------------------------------
# We deliberately do NOT `from vllm.model_executor.models.deepseek_v4 import ...`,
# because that module pulls in `vllm.model_executor.layers.deepseek_v4_attention`
# -> `vllm.model_executor.layers.sparse_attn_indexer`, which calls
# `direct_register_custom_op("vllm::sparse_attn_indexer", ...)`. The ilu fork
# also registers the same op name via
# `vllm.model_executor.layers.ilu_sparse_attn_indexer`
# (pulled in by `deepseek_v2.py`), so importing both leads to a duplicate
# torch.library schema registration at worker startup.
#
# `IluDeepseekV4MLP` is byte-identical to upstream. The model-output multi-head
# fusion (originally `hc_head` eager PyTorch) is replaced by the fused
# ixformer kernel `ixfops.ds4_hc_head` at the call site.


class IluDeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()
        
         # add fused check for act + quant
        if isinstance(self.act_fn, SiluAndMulWithClamp) and \
                hasattr(self.down_proj, 'scheme'):
            from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
                IluCompressedTensorsW8A8Int8,
            )
            if isinstance(self.down_proj.scheme, IluCompressedTensorsW8A8Int8):
                self.act_fn.enable_fuse_quant()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class IluDeepseekV4MoE(nn.Module):
    """Routed + shared expert MoE for ilu fork (FusedMoE only)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        # FusedMoE always wants int32 indices on the routing path.
        self.hash_indices_dtype = torch.int32

        if is_hash_moe:
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = IluDeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self._init_fused_moe_experts(config, quant_config, prefix)

    def _init_fused_moe_experts(
        self,
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0
        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
            input_ids = input_ids.to(dtype=self.hash_indices_dtype)

        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        return final_hidden_states.view(org_shape)


class IluDeepseekV4Attention(nn.Module):
    """Top-level attention for an ilu DeepSeek-V4 layer.

    Wires the per-layer modules required by ``load_weights`` and delegates
    to :class:`IluDeepseekV4MultiHeadLatentAttentionWrapper`. ``forward``
    (via the wrapper) raises ``NotImplementedError`` at stage 1.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size

        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window

        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        # Per-head sparse-MLA attn sink (ixformer kernels require
        # `attn_sink.shape == [h_q]`; h_q == n_local_heads, no head padding).
        # Init to -inf so a missing/partial load leaves the sink in the
        # "no effect" state (an attended row with -inf sink behaves as if
        # the sink weren't merged into the online softmax).
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5
        # The new ilu_deepseek_v4_w4a8 quant_config doesn't carry scale_fmt;
        # default to ue8m0 for forward-time helpers (stage 1 doesn't use it).
        self.scale_fmt = (
            (config.quantization_config or {}).get("scale_fmt", "ue8m0")
            if isinstance(config.quantization_config, dict)
            else "ue8m0"
        )

        rope_parameters = config.rope_parameters
        rope_parameters["rope_theta"] = (
            config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        rope_parameters["mscale"] = 0
        rope_parameters["mscale_all_dim"] = 0
        rope_parameters["is_deepseek_v4"] = True
        rope_parameters["rope_dim"] = self.rope_head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
            dtype=config.torch_dtype,
        )

        self.indexer = None
        if self.compress_ratio == 4:
            self.indexer = IluDeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=vllm_config.cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
            )

        mla_modules = IluDeepseekV4MLAModules(
            vllm_config=vllm_config,
            fused_wqa_wkv=self.fused_wqa_wkv,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            rotary_emb=self.rotary_emb,
            indexer=self.indexer,
            indexer_rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=aux_stream,
        )
        self.mla_attn = IluDeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.softmax_scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            v_head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            mla_modules=mla_modules,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ):
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class IluDeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = IluDeepseekV4Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=(
                aux_stream_dict.get(AuxStreamType.Attention)
                if aux_stream_dict is not None
                else None
            ),
        )
        self.ffn = IluDeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        # All hc_* tensors are bf16/fp32 plain weights — load_weights walks
        # them via default_weight_loader.
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
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

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if _DS4_MHC_PRE_VERSION == "v3":
            mhc_pre = ixfops.ds4_mhc_pre_v3
        elif _DS4_MHC_PRE_VERSION == "v2":
            mhc_pre = ixfops.ds4_mhc_pre_v2
        else:
            mhc_pre = ixfops.ds4_mhc_pre
        post_mix, comb_mix, layer_input = mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, comb_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return ixfops.ds4_mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # v2 feeds hc_*_fn straight into the same-dtype cuinfer GEMM
        # (ixformer_linear), so cast the fp32 weights to x's dtype once on the
        # first forward and cache in-place; later forwards hit the dtype guard
        # and no-op. v1 / v3 keep fp32 fn.
        if _DS4_MHC_PRE_VERSION == "v2":
            if self.hc_attn_fn.dtype != x.dtype:
                self.hc_attn_fn.data = self.hc_attn_fn.data.to(x.dtype)
            if self.hc_ffn_fn.dtype != x.dtype:
                self.hc_ffn_fn.data = self.hc_ffn_fn.data.to(x.dtype)

        residual = x
        x, post, comb = self.hc_pre(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class IluDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(1)]
        self.aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] = {
            AuxStreamType.Attention: aux_stream_list[0],
        }

        self.device = current_platform.device_type
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: IluDeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_dict=self.aux_stream_dict,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        # Pre-hc_head residual stream buffer for the MTP draft. Only allocated
        # on the last PP rank -- that's where MTP target hidden states are
        # produced.
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states of
        # shape (num_tokens, hc_mult, hidden_size) -- V4 expands the token
        # embedding to hc_mult streams before the first decoder layer and
        # keeps that shape until ds4_hc_head() collapses it on the last rank.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions, input_ids)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (only on last rank).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = ixfops.ds4_hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            rms_norm_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Stacked params shared by attention / shared_experts / compressor.
        # Routed experts are handled below via expert_mapping.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip params that don't live on this PP rank.
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # Note: the offline converter writes routed-expert
                    # weight_scale as fp32 (channel) or [n_groups, ...]
                    # group-int4 scales -- never as float8_e8m0fnu -- so we
                    # do *not* need the upstream uint8-bitcast workaround.
                    if is_pp_missing_parameter(name, self):
                        continue
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


def _make_ilu_deepseek_v4_weights_mapper() -> WeightsMapper:
    """Translate ckpt names produced by ``convert_v4_to_w4a8.py`` to vLLM's
    internal parameter names.

    The converter already renamed ``.scale`` to ``.weight_scale`` (matching
    both W4A8 MoE method's ``w13_weight_scale`` / ``w2_weight_scale`` and
    W8A8 Linear scheme's ``weight_scale``), so no scale-suffix regex is
    required here. Other rules mirror upstream (prefix lifts, gate.bias to
    e_score_correction_bias, ``shared_experts.w2`` -> ``.down_proj``, and
    moving compressor under ``mla_attn``).
    """
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        # Reserved for future-compat; no-op today.
        orig_to_new_regex={
            re.compile(r"(?!)"): r"",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class IluDeepseekV4ForCausalLM(nn.Module, SupportsPP):
    """Top-level causal-LM for ilu fork's W4A8 DeepSeek-V4."""

    model_cls = IluDeepseekV4Model
    hf_to_vllm_mapper = _make_ilu_deepseek_v4_weights_mapper()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # No expert_dtype dispatch -- this class is bound to W4A8 only.
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids.shape[0] in [1, 1024]:
            torch.cuda.profiler.start()
            result = self.model(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
            torch.cuda.profiler.stop()
            return result
        else:
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return getattr(self.model, "_mtp_hidden_buffer", None)

    # NOTE: applied uniformly to skip_substrs in AutoWeightsLoader and to
    # the post-load missing-param check below.
    _LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("mtp.",)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=list(self._LOAD_SKIP_SUBSTRS))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Strict reverse check: every nn.Parameter must be filled from the
        # checkpoint. vLLM's DefaultModelLoader only runs this check when
        # ``model_config.quantization is None`` (see
        # ``vllm/model_executor/model_loader/default_loader.py``), so for
        # ``--quantization ilu_deepseek_v4_w4a8`` we have to do it ourselves.
        # Without this guard a missing scale / weight name mapping silently
        # falls back to the parameter's init value and produces wrong results.
        expected: set[str] = set()
        for name, _ in self.named_parameters():
            if any(s in name for s in self._LOAD_SKIP_SUBSTRS):
                continue
            if is_pp_missing_parameter(name, self):
                continue
            expected.add(name)

        missing = expected - loaded
        if missing:
            preview = sorted(missing)[:20]
            raise ValueError(
                f"[IluDeepseekV4ForCausalLM.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                f"checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )

        # Optional sanity stats (helps first-run dispatch verification: routed
        # vs. shared expert vs. bf16 base counts should be plausible).
        n_routed = sum(1 for n in loaded if ".experts." in n and "shared" not in n)
        n_shared = sum(1 for n in loaded if ".shared_experts." in n)
        n_other = len(loaded) - n_routed - n_shared
        logger.info(
            "[IluDeepseekV4ForCausalLM.load_weights] loaded total=%d "
            "(routed_experts=%d, shared_experts=%d, other=%d)",
            len(loaded),
            n_routed,
            n_shared,
            n_other,
        )
        return loaded

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def register_model() -> None:
    """Register the Iluvatar DeepSeek-V4 W4A8 causal-LM and MTP draft archs.

    Invoked by ``vllm_iluvatar.ilu_register.apply_oot_registrations`` during
    register_general. The class paths are resolved lazily by ModelRegistry, so
    importing mtp.py / attention.py is deferred until the model is loaded (the
    ``vllm::ilu_deepseek_v4_attention`` custom op registers on that import).
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "IluDeepseekV4ForCausalLM",
        "vllm_iluvatar.model_executor.models.deepseek_v4.model:"
        "IluDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "IluDeepSeekV4MTPModel",
        "vllm_iluvatar.model_executor.models.deepseek_v4.mtp:IluDeepSeekV4MTP",
    )
