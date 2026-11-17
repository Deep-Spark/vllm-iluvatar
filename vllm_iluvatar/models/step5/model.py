"""Step5 text model for Iluvatar.

Full layers use the native sparse chain. The explicit S1 dense switch remains
available for the staged eager baseline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from ixformer.inference.functions.models.step3_5.float.mul_sigmoid import (
    step3_5_mul_sigmoid,
)
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, MoERunner
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.step3p5 import Step3p5Model as _Step3p5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_iluvatar import envs
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
    set_max_num_seqs,
)
from vllm_iluvatar.model_executor.layers.quantization.common.fp8_input import (
    IluFP8Input,
)
from vllm_iluvatar.model_executor.models.step3p5 import (
    IluStep3p5Attention,
    IluStep3p5MLP,
)

from .config import Step5Config
from .logits import Step5LogitsProcessor
from .native_layers import Step5GateLinear, Step5VocabParallelEmbedding
from .runtime_contract import validate_step5_runtime


def _checkpoint_parameter_audit(model: nn.Module) -> tuple[set[str], list[str]]:
    from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
    from vllm.utils.torch_utils import is_quantized_kv_cache

    expected, exemptions = (set(), [])
    for name, param in model.named_parameters():
        if getattr(param, "ilu_derived_at_load", False):
            exemptions.append(name)
            continue
        if isinstance(param, KVCacheScaleParameter):
            parent_name, leaf = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            if leaf in ("q_scale", "k_scale", "v_scale", "prob_scale") and (
                not is_quantized_kv_cache(parent.kv_cache_dtype)
                or getattr(parent, "_ilu_step5_external_fp8_scales", False)
            ):
                exemptions.append(name)
                continue
        expected.add(name)
    return (expected, exemptions)


class Step5RMSNorm(nn.Module):
    """Step5 zero-centered RMSNorm with FP32 checkpoint parameters."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, output_dtype=None, round_input_to_bf16=False
    ) -> torch.Tensor:
        from ixformer.inference.functions.models.step5.norm.float.rmsnorm import rmsnorm

        output, _ = rmsnorm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            output_dtype=output_dtype,
            round_input_to_bf16=round_input_to_bf16,
            backend="ixformer_native" if hidden_states.is_cuda else "torch",
        )
        return output

    def forward_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ):
        from ixformer.inference.functions.models.step5.norm.float.rmsnorm import rmsnorm

        residual_output = torch.empty(
            hidden_states.shape, device=hidden_states.device, dtype=torch.float32
        )
        return rmsnorm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            residual=residual,
            output_dtype=torch.bfloat16,
            residual_output=residual_output,
            backend="ixformer_native" if hidden_states.is_cuda else "torch",
        )


class Step5QKRMSNorm(Step5RMSNorm):
    """Step5 attention-only fixed arithmetic; checkpoint names remain identical."""

    def forward(
        self, hidden_states: torch.Tensor, output_dtype=None, round_input_to_bf16=False
    ):
        from ixformer.inference.functions.models.step5.norm.float.qk_rmsnorm import (
            qk_rmsnorm,
        )

        if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
            return super().forward(hidden_states, output_dtype, round_input_to_bf16)
        return qk_rmsnorm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            output_dtype=output_dtype,
            backend="ixformer_native",
        )


def _layer_value(values: list[Any] | None, layer_idx: int, default: Any) -> Any:
    if not values:
        return default
    if layer_idx >= len(values):
        raise ValueError(
            f"Step5 per-layer setting has {len(values)} entries, but layer {layer_idx} was requested"
        )
    return values[layer_idx]


class Step5Attention(IluStep3p5Attention):
    """Native sparse full attention and dense sliding-window attention."""

    def __init__(
        self,
        *,
        config: Step5Config,
        cache_config: CacheConfig | None,
        quant_config: Any,
        prefix: str,
    ) -> None:
        layer_idx = extract_layer_index(prefix)
        layer_type = _layer_value(config.layer_types, layer_idx, "full_attention")
        from .fp8_attention import sparse_fp8_enabled

        # Full layers: FP8 paged KV + sparse_gqa_fp8. SWA layers: same FP8 KV
        # dtype so IluFlashAttention takes _forward_fp8 / reshape_and_cache_fp8
        # (backend already supports sliding_window + FP8).
        fp8_kv = sparse_fp8_enabled()
        fp8_sparse = (
            fp8_kv
            and layer_type == "full_attention"
            and bool(
                (getattr(config, "sparse_config", None) or {}).get("enabled", False)
            )
            and (envs.VLLM_ILU_STEP5_SPARSE_ATTENTION)
        )
        fp8_swa = fp8_kv and layer_type != "full_attention"
        if fp8_sparse or fp8_swa:
            from .fp8_attention import sparse_fp8_cache_config

            cache_config = sparse_fp8_cache_config(cache_config)
        partial_rotary_factor = float(
            _layer_value(config.partial_rotary_factors, layer_idx, 1.0)
        )
        attention_setting = getattr(config, "attention_other_setting", None)
        if attention_setting and layer_type == attention_setting.get("attention_type"):
            num_heads = int(attention_setting["num_attention_heads"])
            num_kv_heads = int(attention_setting["num_attention_groups"])
            head_dim = int(attention_setting["head_dim"])
        else:
            num_heads = int(config.num_attention_heads)
            num_kv_heads = int(config.num_attention_groups)
            head_dim = int(config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=config.max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=config.rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=prefix,
            sliding_window=getattr(config, "sliding_window", None),
            use_head_wise_attn_gate=config.use_head_wise_attn_gate,
            layer_types=config.layer_types,
            use_rope_layers=getattr(config, "use_rope_layers", None),
            yarn_only_types=getattr(config, "yarn_only_types", None),
            swa_num_attention_heads=getattr(config, "swa_num_attention_heads", None),
            partial_rotary_factor=1.0,
        )
        qk_norm_cls = Step5QKRMSNorm if self.head_dim == 192 else Step5RMSNorm
        self.q_norm = qk_norm_cls(self.head_dim, config.rms_norm_eps)
        self.k_norm = qk_norm_cls(self.head_dim, config.rms_norm_eps)
        self.attn._ilu_step5_native_decode = True
        # SWA FP8 KV has no checkpoint q/k/v/prob_scale; use unit scales via
        # IluFlashAttention._forward_fp8 (same contract as sparse full-attn).
        if fp8_swa:
            self.attn._ilu_step5_external_fp8_scales = True
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(
                f"invalid Step5 rotary dimension {self.rotary_dim} for head_dim={self.head_dim}"
            )
        rope_scaling = getattr(config, "rope_scaling", None)
        if (
            getattr(config, "yarn_only_types", None)
            and layer_type not in config.yarn_only_types
        ):
            rope_scaling = None
        rope_parameters = dict(rope_scaling or {})
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dtype=self.qkv_proj.params_dtype,
        )
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            sparse = getattr(config, "sparse_config", {}) or {}
            proxy_dim = int(sparse.get("proxy_dim", 256))
            index_heads = int(sparse.get("sparse_indexer_num_heads", 16))
            index_k_heads = int(sparse.get("sparse_indexer_num_k_heads", 1))
            from .sparse_attention import indexer_parallelism

            provider_tp = indexer_parallelism(
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
                int(sparse.get("num_provider_groups", 4)),
            )
            self.sparse_indexer_q = ColumnParallelLinear(
                config.hidden_size,
                index_heads * proxy_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.sparse_indexer_q",
                **provider_tp,
            )
            self.sparse_indexer_k = ColumnParallelLinear(
                config.hidden_size,
                index_k_heads * proxy_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.sparse_indexer_k",
                disable_tp=True,
            )
            self.sparse_indexer_z = ColumnParallelLinear(
                config.hidden_size,
                index_k_heads * proxy_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.sparse_indexer_z",
                disable_tp=True,
            )
            self.sparse_indexer_w = ColumnParallelLinear(
                config.hidden_size,
                index_heads,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.sparse_indexer_w",
                **provider_tp,
            )
            self.sparse_indexer_q_norm = Step5RMSNorm(proxy_dim, config.rms_norm_eps)
            self.sparse_indexer_k_norm = nn.LayerNorm(
                proxy_dim, eps=config.rms_norm_eps
            )
            self.ssmax_s = nn.Parameter(
                torch.zeros(num_heads, dtype=torch.float32), requires_grad=False
            )
        self.sparse_enabled = (
            self.is_full_attention
            and bool(
                (getattr(config, "sparse_config", None) or {}).get("enabled", False)
            )
            and (envs.VLLM_ILU_STEP5_SPARSE_ATTENTION)
        )
        if self.sparse_enabled:
            if self.head_dim != 192 or num_kv_heads != 4:
                raise ValueError(
                    "Step5 sparse attention requires D192 and four KV providers"
                )
            from vllm.config import get_current_vllm_config

            from .sparse_attention import Step5SparseAttention, validate_sparse_geometry

            validate_sparse_geometry(self.num_heads, self.num_kv_heads, self.head_dim)
            self.sparse_path = Step5SparseAttention(
                get_current_vllm_config(), config, prefix, rope_parameters
            )
            if fp8_sparse:
                from .fp8_attention import configure_sparse_fp8_attention

                configure_sparse_fp8_attention(
                    self.attn, self.sparse_path.csa.ring.requested_capacity
                )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, prequantized=None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(
            prequantized if prequantized is not None else hidden_states
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape = q.shape
        k_shape = k.shape
        from .decode_fusions import (
            can_fuse_gate,
            can_fuse_qk_norm_rope,
            fusion_actual_tokens,
        )

        fusion_actual = fusion_actual_tokens(self.attn, qkv.shape[0], device=qkv.device)
        if can_fuse_qk_norm_rope(
            qkv,
            self.rotary_emb.cos_sin_cache,
            fusion_actual,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim if self.use_rope else 0,
        ):
            from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
                qk_norm_rope,
            )

            q, k = qk_norm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                fusion_actual,
                self.num_heads,
                self.num_kv_heads,
                self.q_norm.variance_epsilon,
                self.rotary_dim if self.use_rope else 0,
                self.rotary_emb.is_neox_style,
            )
        else:
            q = self.q_norm(
                q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
            ).view(q_shape)
            k = self.k_norm(
                k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
            ).view(k_shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)
        if self.sparse_enabled:
            index_q, _ = self.sparse_indexer_q(hidden_states)
            index_k, _ = self.sparse_indexer_k(hidden_states)
            index_z, _ = self.sparse_indexer_z(hidden_states)
            index_w, _ = self.sparse_indexer_w(hidden_states)
            attn_output = self.sparse_path(
                q.view(-1, self.num_heads, self.head_dim),
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                index_q,
                index_k,
                index_z,
                index_w,
                positions,
                self.sparse_indexer_q_norm.weight,
                self.sparse_indexer_k_norm.weight,
                self.sparse_indexer_k_norm.bias,
            ).view(q_shape)
        else:
            attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            gate, _ = self.g_proj(
                prequantized if prequantized is not None else hidden_states
            )
            if can_fuse_gate(self.o_proj, attn_output, fusion_actual):
                from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
                    gate_fp8,
                )

                q_gate, gate_scale = gate_fp8(attn_output, gate, fusion_actual)
                # RowParallelLinear still owns bias and TP reduction. Its FP8
                # method consumes these already quantized activations directly.
                output, _ = self.o_proj(IluFP8Input(q_gate, gate_scale))
                return output
            gated_output = step3_5_mul_sigmoid(
                attn_output.view(
                    *attn_output.shape[:-1], self.num_heads, self.head_dim
                ),
                gate.unsqueeze(-1),
            ).view(*attn_output.shape)
            attn_output = gated_output
        output, _ = self.o_proj(attn_output)
        return output


class Step5MLP(IluStep3p5MLP):
    def __init__(
        self,
        config,
        hidden_size,
        intermediate_size,
        hidden_act,
        quant_config=None,
        reduce_results=True,
        prefix="",
    ):
        super().__init__(
            config,
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config,
            reduce_results,
            prefix,
        )
        # Dense rows are independent; unlike routed MoE there is no gather map.
        # All rows, including allocated padding, keep the original elementwise
        # semantics. vLLM consumes only valid rows; no cross-row reduction occurs.
        # Reserve the prefill and decode/MTP budgets before CUDA graph capture.
        # The buffer address must remain stable for every later graph replay.
        from .decode_fusions import dense_row_mask_capacity

        self.register_buffer(
            "_decode_dense_rows",
            torch.zeros(dense_row_mask_capacity(), dtype=torch.int32),
            persistent=False,
        )

    def forward(self, hidden_states):
        from .decode_fusions import can_fuse_swiglu_fp8, dense_valid_row_mask

        gate_up, _ = self.gate_up_proj(hidden_states)
        rows = gate_up.shape[0]
        # Dense rows have no EP gather map; sign(row_mask)>=0 marks every row valid.
        self._decode_dense_rows, row_mask = dense_valid_row_mask(
            self._decode_dense_rows, rows, gate_up.device
        )
        # Prefill and decode: fuse SwiGLU-step + per-token FP8 into down_proj.
        if can_fuse_swiglu_fp8(self.down_proj, gate_up, row_mask):
            from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
                swiglu_fp8,
            )

            q, scales = swiglu_fp8(
                gate_up,
                row_mask,
                float(self.limit or 0.0),
            )
            output, _ = self.down_proj(IluFP8Input(q, scales.unsqueeze(-1)))
            return output
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step5MoE(nn.Module, IluFusedMoEEP):
    """Step5 router/shared/routed experts with an FP32 router."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.layer_idx = extract_layer_index(prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_ep_group().device_group.size()
        self.n_routed_experts = int(config.moe_num_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.fuse_all_reduce = self.tp_size > 1 and get_dp_group().world_size == 1
        self.gate = Step5GateLinear(
            config.hidden_size,
            config.moe_num_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.share_expert = Step5MLP(
            config=config,
            hidden_size=config.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=not self.fuse_all_reduce,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert",
        )
        swiglu_limit = float(_layer_value(config.swiglu_limits, self.layer_idx, 7.0))
        self.experts = FusedMoEFactory(
            shared_experts=None,
            gate=self.gate,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=not self.fuse_all_reduce,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation="swiglustep",
            swiglu_limit=swiglu_limit,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            apply_routed_scale_to_output=False,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )
        self.init_ep()
        self.experts.routed_experts._ilu_step5_native_activation = True
        moe_parallel = self.experts.moe_config.moe_parallel_config
        fp32_ep = (
            envs.VLLM_ILU_STEP5_FP32_EP
            and get_dp_group().world_size > 1
            and moe_parallel.use_ep
            and (
                moe_parallel.all2all_backend
                in ("naive", "allgather_reducescatter", "ag-rs", "ag_rs")
            )
            and self._should_use_forward_dp_ep()
        )
        if fp32_ep:
            raise ValueError(
                "FP32 EP communication is unsupported; disable VLLM_ILU_STEP5_FP32_EP (legacy VLLM_STEP5_FP32_EP)"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, shape[-1])
        shared_output = self.share_expert(hidden_states)
        if self._should_use_forward_dp_ep():
            routed_output = self.forward_dp_ep(hidden_states)
        else:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        from ixformer.inference.functions.models.step5.norm.float.residual_add import (
            residual_add,
        )

        output = residual_add(shared_output, routed_output, backend="ixformer_native")
        if self.fuse_all_reduce:
            output = get_tp_group().all_reduce(output)
        return output.view(shape)


class Step5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Step5Config):
            raise TypeError(f"expected Step5Config, got {type(config).__name__}")
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        cache_config = vllm_config.cache_config
        if cache_config is not None:
            import copy

            cache_config = copy.copy(cache_config)
            cache_config.sliding_window = None
        self.self_attn = Step5Attention(
            config=config,
            cache_config=cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        moe_layers = getattr(config, "moe_layer_list", None)
        if moe_layers is None:
            enum = getattr(config, "moe_layers_enum", None)
            moe_layers = (
                [int(value) for value in enum.split(",")]
                if enum
                else list(range(1, config.num_hidden_layers))
            )
        self.use_moe = layer_idx in set(moe_layers)
        if self.use_moe:
            self.moe = Step5MoE(vllm_config, prefix=f"{prefix}.moe")
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Step5RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def _for_params(self, value: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            return value.to(torch.bfloat16)
        return value

    def _for_residual(self, value: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            return value.to(torch.float32)
        return value

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.fp32_residual_connection:
            from ixformer.inference.functions.models.step5.norm.float.residual_add import (
                residual_add,
            )

            hidden_states, residual = self.forward_with_residual(
                positions, hidden_states
            )
            return residual_add(hidden_states, residual, backend="ixformer_native")
        residual = hidden_states
        from .decode_fusions import can_share_norm_quant, fusion_actual_tokens

        actual = fusion_actual_tokens(
            self.self_attn.attn, hidden_states.shape[0], device=hidden_states.device
        )
        prepared = None
        if can_share_norm_quant(self.self_attn, hidden_states, actual):
            from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
                norm_fp8,
            )

            # residual=None: fuse RMSNorm + dyn FP8 for qkv/g; keep BF16 residual path.
            hidden_states, _, prepared = norm_fp8(
                hidden_states,
                None,
                self.input_layernorm.weight,
                actual,
                self.input_layernorm.variance_epsilon,
            )
            prepared = IluFP8Input(*prepared)
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self._for_params(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, prequantized=prepared)
        hidden_states = self._for_residual(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._for_params(hidden_states)
        hidden_states = (
            self.moe(hidden_states) if self.use_moe else self.mlp(hidden_states)
        )
        return self._for_residual(hidden_states) + residual

    def forward_with_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry FFN and FP32 residual until the next fused input norm.

        The public forward still returns the materialized sum. This internal
        path removes one standalone addition per layer inside each PP stage.
        """
        if not self.fp32_residual_connection:
            raise RuntimeError("residual-stream path requires FP32 residual config")
        from .decode_fusions import can_share_norm_quant, fusion_actual_tokens

        actual = fusion_actual_tokens(
            self.self_attn.attn, hidden_states.shape[0], device=hidden_states.device
        )
        prepared = None
        if can_share_norm_quant(self.self_attn, hidden_states, actual):
            from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
                norm_fp8,
            )

            hidden_states, residual, prepared = norm_fp8(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                actual,
                self.input_layernorm.variance_epsilon,
            )
        else:
            hidden_states, residual = self.input_layernorm.forward_residual(
                hidden_states, residual
            )
        if prepared is not None:
            prepared = IluFP8Input(*prepared)
        hidden_states = self.self_attn(positions, hidden_states, prequantized=prepared)
        hidden_states, residual = self.post_attention_layernorm.forward_residual(
            hidden_states, residual
        )
        hidden_states = (
            self.moe(hidden_states) if self.use_moe else self.mlp(hidden_states)
        )
        return (hidden_states, residual)


class Step5Model(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Step5Config):
            raise TypeError(f"expected Step5Config, got {type(config).__name__}")
        self.config = config
        self.vocab_size = config.vocab_size
        self.moe_num_experts = config.moe_num_experts
        self.fp32_residual_connection = config.fp32_residual_connection
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = Step5VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step5DecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        if self.fp32_residual_connection:
            dtype = torch.float32
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                )
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        carry_residual = self.fp32_residual_connection
        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            if carry_residual:
                hidden_states, residual = self.layers[layer_idx].forward_with_residual(
                    positions, hidden_states, residual
                )
            else:
                hidden_states = self.layers[layer_idx](positions, hidden_states)
        if carry_residual and residual is not None:
            from ixformer.inference.functions.models.step5.norm.float.residual_add import (
                residual_add,
            )

            hidden_states = residual_add(
                hidden_states, residual, backend="ixformer_native"
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.moe.fp8_to_fp4 import (
            IluFp8ToFp4OnlineMoEMethod,
        )

        from .online_moe_loader import Step5OnlineExpertLoader

        online = {}
        for index in range(self.start_layer, self.end_layer):
            layer = self.layers[index]
            if layer.use_moe:
                experts = layer.moe.experts.routed_experts
                if isinstance(experts.quant_method, IluFp8ToFp4OnlineMoEMethod):
                    online[index] = experts
        if not online:
            return _Step3p5Model.load_weights(self, weights)
        paired = getattr(self, "_online_expert_loader", None)
        if paired is None:
            paired = Step5OnlineExpertLoader(
                self.vllm_config.model_config.model, online, self.moe_num_experts
            )
            self._online_expert_loader = paired
        loaded = _Step3p5Model.load_weights(
            self, paired.remaining_weights(weights, finalize=False)
        )
        return loaded | paired.loaded_names


class Step5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    _enable_weights_track_by_default = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".share_expert.": ".moe.share_expert."},
        orig_to_new_prefix={"vision_model.": None},
        orig_to_new_suffix={"vit_large_projector.weight": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        validate_step5_runtime(vllm_config)
        set_max_num_seqs(vllm_config.scheduler_config.max_num_seqs)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = Step5Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = Step5LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.moe_layers: list[MoERunner] = []
        example: Step5MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer, Step5DecoderLayer) and layer.use_moe:
                example = layer.moe
                self.moe_layers.append(layer.moe.experts)
        if example is None:
            raise ValueError("Step5 requires at least one MoE layer")
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example.n_logical_experts
        self.num_physical_experts = example.n_physical_experts
        self.num_local_physical_experts = example.n_local_physical_experts
        self.num_routed_experts = example.n_routed_experts
        self.num_redundant_experts = example.n_redundant_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(
            hidden_states,
            output_dtype=torch.bfloat16,
            round_input_to_bf16=self.fp32_residual_connection,
        )
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def cpu_sources():
            iterator = iter(weights)
            while True:
                with torch.device("cpu"):
                    item = next(iterator, None)
                if item is None:
                    break
                yield item

        loaded = AutoWeightsLoader(self).load_weights(
            cpu_sources(), mapper=self.hf_to_vllm_mapper
        )
        paired = getattr(self.model, "_online_expert_loader", None)
        if paired is not None:
            paired.finalize()
        expected, _ = _checkpoint_parameter_audit(self)
        missing = expected - loaded
        if missing:
            raise ValueError(
                f"Step5 checkpoint parameters not loaded: {sorted(missing)}"
            )
        for layer in self.model.layers[self.model.start_layer : self.model.end_layer]:
            if layer.use_moe:
                layer.moe.gate.prepare_native_weights()
        cpu_parameters = [
            name for name, param in self.named_parameters() if not param.is_cuda
        ]
        if cpu_parameters:
            raise RuntimeError(
                f"Step5 requires GPU-resident parameters: {cpu_parameters}"
            )
        return loaded


def register_model() -> None:
    from vllm import ModelRegistry

    target = "vllm_iluvatar.models.step5.model:Step5ForCausalLM"
    ModelRegistry.register_model("Step5ForCausalLM", target)
    ModelRegistry.register_model("Step5ForConditionalGeneration", target)


__all__ = ["Step5ForCausalLM", "Step5Model"]
