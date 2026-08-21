# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM-5.2 W4A8 model (self-contained attention).

Implements GLM-5.2 (``GlmMoeDsa`` arch = DeepSeek-V3.2: MLA + DSA lightning
indexer + fine-grained MoE) bound to the W4A8 mixed-quant scheme
(``compressed-tensors`` / ``IluQuantConfig``). The attention *algorithm* chain is fully
self-contained in :mod:`.attention` (no community
``MultiHeadLatentAttentionWrapper`` / ``MLAAttention`` / ``SparseMLAAttentionImpl``
in the runtime path), aligned with ``deepseek_v4``.

Target   : arch "IluGlmMoeDsaForCausalLM" (additive; does NOT override the
           community deepseek_v2.py).
Baseline : community deepseek_v2.py V3.2 path (structure only; kernels are ilu).
How      : ``register_model()`` points the arch at ``IluGlmMoeDsaForCausalLM``.
Scope    : GLM-5.2 W4A8 on the Iluvatar platform (MTP layer 78 skipped).

GLM-5.2 architecture (see ``config.json``):
- 78 backbone layers, ``first_k_dense_replace=3`` (layers 0-2 dense MLP,
  3-77 fine-grained MoE: 256 routed / 8 activated + 1 shared).
- MLA: ``q_lora_rank=2048``, ``kv_lora_rank=512``, ``qk_nope=192`` /
  ``qk_rope=64`` / ``v_head_dim=256``, 64 heads.
- DSA lightning indexer on ``full`` layers (``index_topk=2048``,
  ``index_topk_freq=4``, ``index_skip_topk_offset=3``).
- MoE: sigmoid scoring, ``noaux_tc`` (``e_score_correction_bias``),
  ``routed_scaling_factor=2.5``, ``norm_topk_prob=True``.
- MTP (``num_nextn_predict_layers=1``, layer 78) is skipped in this build.

W4A8 mixed quant (produced by
``my_tests/glm5.2/weight_convert/convert_glm52_to_w4a8.py``):
- routed experts -> INT4 W4A8 (``IluCompressedTensorsW4A8Int8MoEMethod``),
- shared experts / dense MLP / MLA fused_qkv_a/q_b/o_proj / indexer.wq_b ->
  INT8 W8A8 (``IluCompressedTensorsW8A8Int8``),
- kv_b_proj -> INT8 W8A16 grouped marlin (``IluGlmKVBProjW8A16Method ``: split
  into W_UK/W_UV, consumed via ``ixf_ops.marlin_w8a16``),
- indexer wk/weights_proj + norms/embed/head/gate -> BF16.

The MLA ``q_a_proj`` + ``kv_a_proj_with_mqa`` fusion is done at *load time* via
``fused_qkv_a_proj`` (``MergedColumnParallelLinear``) using the
``stacked_params_mapping`` below -- the checkpoint keeps the two source tensors
separate, so no re-conversion is needed and there is no runtime merge.
"""

from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, cast

import torch
import torch.nn as nn
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.models.glm_moe_dsa.attention import (
    IluGlmMoeDsaMLAAttention,
)

logger = init_logger(__name__)


@IluvatarQuantRegistry.register_method
class IluGlmKVBProjW8A16Method(LinearMethodBase):
    """W8A16 grouped-BMM method for GLM-5.2 ``kv_b_proj``."""

    def __init__(self, parent_config, ilu_params: dict[str, Any]) -> None:
        self.parent_config = parent_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.parameter import (
            ChannelQuantScaleParameter,
            ModelWeightParameter,
        )

        out_total = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(out_total, input_size_per_partition, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((out_total, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Split ``kv_b_proj`` into W_UK/W_UV and Marlin-repack each."""
        import ixformer.inference.functions as ixf_ops

        geom = getattr(layer, "bmm_geom", None)
        assert geom is not None, (
            "IluGlmKVBProjW8A16Method requires layer.bmm_geom to be set by "
            "the attention wrapper before process_weights_after_loading."
        )
        num_heads: int = geom.num_heads
        nope: int = geom.qk_nope_head_dim
        v: int = geom.v_head_dim
        kv_lora: int = geom.kv_lora_rank

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        assert weight.shape == (num_heads * (nope + v), kv_lora), (
            f"kv_b_proj weight shape {tuple(weight.shape)} does not match "
            f"({num_heads} * ({nope} + {v}), {kv_lora}) implied by "
            f"bmm_geom={geom!r}"
        )
        assert weight_scale.shape == (num_heads * (nope + v), 1), (
            f"kv_b_proj weight_scale shape {tuple(weight_scale.shape)} != "
            f"({num_heads * (nope + v)}, 1)"
        )

        w = weight.view(num_heads, nope + v, kv_lora)
        s = weight_scale.view(num_heads, nope + v, 1)
        w_uk, w_uv = w.split([nope, v], dim=1)
        w_uk_s, w_uv_s = s.split([nope, v], dim=1)

        layer.W_UK = ixf_ops.marlin_w8_weight_repack(
            weights=w_uk.contiguous(),
            weight_format="int8",
            reformat="k16n16_grouped_n",
        )
        layer.W_UK_S = w_uk_s.contiguous()
        layer.W_UV = ixf_ops.marlin_w8_weight_repack(
            weights=w_uv.permute(0, 2, 1).contiguous(),
            weight_format="int8",
            reformat="k16n16",
        )
        layer.W_UV_S = w_uv_s.contiguous()

        delattr(layer, "weight")
        delattr(layer, "weight_scale")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "IluGlmKVBProjW8A16Method.apply must not be called; kv_b_proj is "
            "consumed by IluGlmMoeDsaSparseMLA via "
            "marlin_w8a16(W_UK/W_UV, W_UK_S/W_UV_S)."
        )


def _get_moe_router_dtype(config) -> torch.dtype | None:
    """GLM-5.2 routes in fp32 but older configs do not expose
    ``moe_router_dtype``; mirror the community special-case."""
    router_dtype = getattr(config, "moe_router_dtype", None)
    if getattr(config, "model_type", None) == "glm_moe_dsa":
        return None
    if router_dtype == "float32":
        return torch.float32
    return None


class IluGlmMoeDsaMLP(nn.Module):
    """Dense MLP (first_k_dense_replace layers) + shared-expert MLP.

    ``gate_proj`` + ``up_proj`` are fused into ``gate_up_proj``
    (MergedColumnParallelLinear); at runtime the W4A8 config dispatches
    ``mlp.gate_up_proj`` / ``mlp.down_proj`` to the W8A8 INT8 linear method.
    GLM uses plain SwiGLU (silu) with no ``swiglu_limit``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class IluGlmMoeDsaMoE(nn.Module):
    """Fine-grained MoE (256 routed / 8 activated + 1 shared) for GLM-5.2.

    sigmoid scoring + ``noaux_tc`` correction bias + grouped top-k (n_group=1).
    ``routed_scaling_factor=2.5`` flows to ``FusedMoE`` and is applied on the
    output side (``_ilu_output_scale`` via ``moe_output_reduce_sum`` for the
    W4A8 kernel; ``apply_routed_scale_to_output=True`` so the router does not
    also scale). GLM has no ``swiglu_limit`` so it is deliberately not set.
    """

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu supported."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts)
            )
        else:
            self.gate.e_score_correction_bias = None

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = IluGlmMoeDsaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "sigmoid"),
            routed_scaling_factor=self.routed_scaling_factor,
            # aiter applies routed_scaling_factor internally; on Iluvatar aiter
            # is off, so we apply it on the output side.
            apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled(),
            e_score_correction_bias=self.gate.e_score_correction_bias,
            router_logits_dtype=self.gate.out_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        return final_hidden_states.view(num_tokens, hidden_dim)


class IluGlmMoeDsaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = IluGlmMoeDsaMLAAttention(
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
            topk_indices_buffer=topk_indices_buffer,
        )

        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        )
        if is_moe_layer:
            self.mlp = IluGlmMoeDsaMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = IluGlmMoeDsaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class IluGlmMoeDsaModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # GLM-5.2 is always sparse (V3.2 DSA indexer): the shared top-k buffer
        # is written by ``full`` layers and reused by ``skip_topk`` layers.
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: IluGlmMoeDsaDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Fused MLA q_a + kv_a: merged at load (replaces _inject_merge_qkv_a).
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj).
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Only ``full`` layers build an indexer; the checkpoint ships indexer
        # weights only for those (plus the skipped MTP layer), but guard
        # defensively by dropping indexer weights whose layer has none built.
        indexer_present_prefixes = {
            n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
        }

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if ".indexer." in name and (
                name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
            ):
                # This layer has no indexer built; drop its checkpoint weights.
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip experts (handled below via expert_params_mapping) so we
                # don't mangle ``mlp.experts.N.gate_proj`` into gate_up_proj.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class IluGlmMoeDsaForCausalLM(nn.Module, SupportsPP):
    """Top-level causal-LM for ilu GLM-5.2 W4A8 (GlmMoeDsa arch)."""

    model_cls = IluGlmMoeDsaModel

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    # MTP (layer 78) is skipped: applied to AutoWeightsLoader skip_substrs and
    # the strict missing-param check below.
    _LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("model.layers.78.",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model_dtype = vllm_config.model_config.dtype

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=list(self._LOAD_SKIP_SUBSTRS))
        loaded = loader.load_weights(weights)

        # Strict reverse check: every nn.Parameter (except skipped MTP / PP
        # missing) must be filled from the checkpoint. vLLM's DefaultModelLoader
        # only runs this when ``model_config.quantization is None``; we must do it ourselves so a
        # missing scale / name-mapping does not silently fall back to init.
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
                f"[IluGlmMoeDsaForCausalLM.load_weights] "
                f"{len(missing)} parameter(s) were not initialized from the "
                f"checkpoint. Showing up to 20:\n  " + "\n  ".join(preview)
            )

        # NOTE: the absorbed MLA up-proj (kv_b_proj -> W_UK/W_UV marlin) is now
        # built deterministically by ``IluGlmKVBProjW8A16Method .process_weights_after_loading``,
        # invoked by the loader's ``utils.process_weights_after_loading`` pass
        # (kv_b_proj is a Linear with a quant_method). No explicit early call
        # here -- and no BF16 absorb fallback.

        n_routed = sum(1 for n in loaded if ".experts." in n and "shared" not in n)
        n_shared = sum(1 for n in loaded if ".shared_experts." in n)
        n_other = len(loaded) - n_routed - n_shared
        logger.info(
            "[IluGlmMoeDsaForCausalLM.load_weights] loaded total=%d "
            "(routed_experts=%d, shared_experts=%d, other=%d)",
            len(loaded),
            n_routed,
            n_shared,
            n_other,
        )
        return loaded


def register_model() -> None:
    """Register the Iluvatar GLM-5.2 W4A8 causal-LM and MTP draft archs.

    Invoked by ``vllm_iluvatar.ilu_register.apply_oot_registrations`` during
    register_general. The class paths are resolved lazily by ModelRegistry, so
    importing this module's heavy attention deps (the plugin-local
    ``vllm::ilu_glm_sparse_attn_indexer`` op) -- and ``mtp.py`` -- is deferred
    until the model is loaded.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "IluGlmMoeDsaForCausalLM",
        "vllm_iluvatar.models.glm_moe_dsa.model:IluGlmMoeDsaForCausalLM",
    )
    ModelRegistry.register_model(
        "IluGlmMoeDsaMTPModel",
        "vllm_iluvatar.models.glm_moe_dsa.mtp:IluGlmMoeDsaMTP",
    )
