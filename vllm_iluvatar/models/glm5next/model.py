# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (GLM5Next) on Iluvatar."""

from __future__ import annotations

import os

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import hc_contract, hc_expand
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import Glm4vDummyInputsBuilder
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.glm5next.common.model import (
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextMLP,
    Glm5NextModel,
    Glm5NextMoE,
)
from vllm.models.glm5next.common.multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import IluGlm5NextMLAAttention, glm5next_topk_buffer_width
from .kda import IluGlm5NextLinearAttention

_ILU_GLM5_PROFILE = os.environ.get("ILU_GLM5_PROFILE")


def _ilu_mhc_pre_v2(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = fn.to(dtype=residual.dtype)
    rms_weight = (
        norm_weight.to(dtype=residual.dtype) if norm_weight is not None else None
    )
    fuse_norm = rms_weight is not None and ixfops.enable_hc_pre_norm_fusion(
        *residual.shape, dtype=residual.dtype
    )
    post, comb, layer_input = ixfops.ds4_mhc_pre_v2(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_mult,
        sinkhorn_repeat=sinkhorn_repeat,
        need_rms=fuse_norm,
        rms_weight=rms_weight if fuse_norm else None,
        layer_rms_eps=norm_eps,
    )
    if rms_weight is not None and not fuse_norm:
        layer_input = ixfops.rms_norm(layer_input, rms_weight, norm_eps)
    return post, comb, layer_input


def _ilu_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return ixfops.mhc_post(x, residual, post, comb)


class IluGlm5NextMLP(Glm5NextMLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results,
            is_sequence_parallel=is_sequence_parallel,
            prefix=prefix,
            swiglu_limit=swiglu_limit,
        )

        if isinstance(self.act_fn, SiluAndMulWithClamp) and hasattr(
            self.down_proj, "scheme"
        ):
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


class IluGateLinear(GateLinear):
    def forward(self, x):
        output = ixfops.linear(x, self.weight)
        return self._return(output)


class IluGlm5NextMoE(Glm5NextMoE):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super(Glm5NextMoE, self).__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = IluGateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.bfloat16)
            )
        else:
            self.gate.e_score_correction_bias = None

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = IluGlm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class IluGlm5NextDecoderLayer(Glm5NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super(Glm5NextDecoderLayer, self).__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if is_kda_layer:
            self.self_attn = IluGlm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = IluGlm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = IluGlm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = IluGlm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        return _ilu_mhc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.mhc_post_mult_value,
            self.mhc_sinkhorn_iterations,
            norm_weight,
            norm_eps,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return _ilu_mhc_post(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        residual = _ilu_mhc_post(x, residual, post, comb)
        new_post, new_comb, layer_input = self.hc_pre(
            residual, hc_fn, hc_scale, hc_base, norm_weight, norm_eps
        )
        return residual, new_post, new_comb, layer_input


class IluGlm5NextModel(Glm5NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextModel, self).__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = glm5next_topk_buffer_width(vllm_config, topk_tokens, kpool)
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return IluGlm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        n = self.config.mhc_num_residual_streams
        d = self.config.hidden_size
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, d), dtype=dtype, device=device
                ),
                "residual": torch.zeros((batch_size, n, d), dtype=dtype, device=device),
                "post": torch.zeros(
                    (batch_size, n, 1), dtype=torch.float32, device=device
                ),
                "comb": torch.zeros(
                    (batch_size, n, n), dtype=torch.float32, device=device
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if _ILU_GLM5_PROFILE is None:
            return self._forward_impl(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        nt = positions.numel()
        arm = (_ILU_GLM5_PROFILE == "decode" and nt <= 6) or (
            _ILU_GLM5_PROFILE == "prefill" and nt > 64
        )
        if arm:
            torch.cuda.profiler.start()
        try:
            return self._forward_impl(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        finally:
            if arm:
                torch.cuda.profiler.stop()

    def _forward_impl(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            post = intermediate_tensors["post"]
            comb = intermediate_tensors["comb"]

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "post": post,
                    "comb": comb,
                }
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights):
        return super().load_weights(
            w for w in weights if not is_pp_missing_parameter(w[0], self)
        )


class IluGlm5NextForCausalLM(Glm5NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextForCausalLM, self).__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = IluGlm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class IluGlm5NextForConditionalGeneration(Glm5NextForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )


def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm5NextForCausalLM",
        "vllm_iluvatar.models.glm5next.model:IluGlm5NextForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "vllm_iluvatar.models.glm5next.model:IluGlm5NextForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Glm5NextMTPModel",
        "vllm_iluvatar.models.glm5next.mtp:IluGlm5NextMTP",
    )
