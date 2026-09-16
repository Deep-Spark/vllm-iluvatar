# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 compressed-tensors (W4A8) implementation.

Registers ``IluMiniMaxM3SparseFor*``. MiniMax ``mxfp8`` lives in ``mxfp8/``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.minimax_m3.compressed_tensors.attention import (
    IluMiniMaxM3Attention,
    IluMiniMaxM3SparseAttention,
)
from vllm_iluvatar.models.minimax_m3.compressed_tensors.sparse_attention import (
    resolve_sparse_attention_cfg,
)

# Community VL helpers. The early compatibility patch makes importing the
# upstream common package safe without loading NVIDIA-only model entry points.
from vllm.models.minimax_m3.common.mm_preprocess import (
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
)
from vllm.models.minimax_m3.common.vision_tower import (
    MiniMaxVLVisionModel,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    IluQuantConfig,
)

logger = init_logger(__name__)


def _fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TP all-reduce (if needed) then Gemma RMSNorm + residual.

    Mirrors upstream MiniMax-M3 defer-AR into Gemma Norm; Iluvatar has no
    FlashInfer fused AR+Norm kernel, so this is all_reduce then ``norm``.
    """
    if get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return norm(hidden_states, residual)

# Applied to AutoWeightsLoader skip list and quantized missing-param checks.
_LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("mtp.",)
# Runtime attention quantization state is initialized by vLLM rather than loaded
# from the checkpoint.  Only dense layers own the nested ``.attn`` module.
_LOAD_CHECK_SKIP_SUFFIXES: tuple[str, ...] = tuple(
    f".attn.{name}"
    for name in (
        "q_scale",
        "k_scale",
        "v_scale",
        "q_zero_point",
        "k_zero_point",
        "v_zero_point",
    )
)


def _assert_m3_parameters_loaded(
    model: nn.Module,
    loaded: set[str],
    *,
    skip_substrs: tuple[str, ...] = _LOAD_SKIP_SUBSTRS,
    kv_cache_dtype: str = "auto",
) -> None:
    """Strict reverse check for compressed-tensors (IluQuantConfig) loads."""
    skip_runtime_attn_quant = kv_cache_dtype in ("auto", "", None)
    expected: set[str] = set()
    for name, _ in model.named_parameters():
        if any(s in name for s in skip_substrs):
            continue
        if skip_runtime_attn_quant and name.endswith(_LOAD_CHECK_SKIP_SUFFIXES):
            continue
        if is_pp_missing_parameter(name, model):
            continue
        expected.add(name)

    missing = expected - loaded
    if missing:
        preview = sorted(missing)[:20]
        raise ValueError(
            f"[IluMiniMaxM3.load_weights] {len(missing)} parameter(s) were not "
            f"initialized from the checkpoint. Showing up to 20:\n  "
            + "\n  ".join(preview)
        )


def _log_m3_quant_load_stats(loaded: set[str]) -> None:
    n_routed = sum(1 for n in loaded if ".experts." in n and "shared" not in n)
    n_shared = sum(1 for n in loaded if ".shared_experts." in n)
    n_other = len(loaded) - n_routed - n_shared
    logger.info(
        "[IluMiniMaxM3.load_weights] compressed-tensors loaded total=%d "
        "(routed_experts=%d, shared_experts=%d, other=%d)",
        len(loaded),
        n_routed,
        n_shared,
        n_other,
    )

def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    """Layer ids whose attention runs the extra sparse "index" branch."""
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether this layer's MLP is a sparse MoE block (vs a dense MLP)."""
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _is_ilu_quant_config(quant_config: QuantizationConfig | None) -> bool:
    return isinstance(quant_config, IluQuantConfig)


class IluMiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the leading dense layers)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        # gate * sigmoid(alpha * gate) * (up + beta), with both halves clamped.
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class IluMiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE block with a routing-bias correction and a shared
    expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        # Sigmoid routing uses a per-expert score-correction bias for selection.
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                IluMiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        # Router weights are stored in fp32; GateLinear upcasts the bf16
        # activations and computes the gate in fp32 (fp32 router logits).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: IluMiniMaxM3MLP | None = None
        if self.n_shared_experts:
            self.shared_experts = IluMiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 (gate_up_proj) is loaded packed via MergedColumnParallelLinear
            # ([all gates; all ups]), so use the uninterleaved SwiGLU-OAI variant
            # rather than the interleaved gpt-oss layout.
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=config.swiglu_beta,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); GateLinear casts to fp32.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class IluMiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

        # Complete the preceding dense MLP's deferred all-reduce
        # (reduce_results=False), fused into this layer's input_layernorm.
        # Disable this fusion when PP is set. MoE still reduces in-layer
        # (no fused AR+Norm kernel on Iluvatar → defer-AR had ~0 e2e gain).
        self.fuse_input_allreduce = (
            layer_id > 0
            and not _is_moe_layer(config, layer_id - 1)
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        )

        is_sparse_attention_layer = layer_id in _sparse_attention_layer_ids(config)

        if is_sparse_attention_layer:
            self.self_attn = IluMiniMaxM3SparseAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            self.self_attn = IluMiniMaxM3Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        # Dense layers store the FFN under `mlp`; MoE layers under
        # `block_sparse_moe` -- matching the checkpoint's naming.
        self.is_moe_layer = _is_moe_layer(config, layer_id)
        if self.is_moe_layer:
            self.block_sparse_moe = IluMiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = IluMiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=vllm_config.parallel_config.pipeline_parallel_size > 1,
            )

        # config.use_gemma_norm is True for M3 -> Gemma-style RMSNorm
        # (OOT: IluGemmaRMSNorm).
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Align with Ilu DeepSeek-V4: keep the whole layer inside FULL
        # cudagraph capture (no eager_break). MSA/MoE must be CG-safe.
        if self.fuse_input_allreduce and residual is not None:
            hidden_states, residual = _fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = _fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual


class IluMiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        sparse_cfg = getattr(config, "sparse_attention_config", None)
        topk_indices_buffer = None
        if sparse_cfg is not None:
            resolved = resolve_sparse_attention_cfg(config)
            tp_size = get_tensor_model_parallel_world_size()
            # Same shard as IluMiniMaxM3SparseAttention.num_idx_heads
            # (index heads == KV heads, including TP replication).
            num_index_heads = max(1, resolved["num_index_heads"] // tp_size)
            max_num_batched_tokens = int(
                vllm_config.scheduler_config.max_num_batched_tokens
            )
            capture_sizes = (
                vllm_config.compilation_config.cudagraph_capture_sizes or []
            )
            max_capture = max((int(s) for s in capture_sizes), default=0)
            # Token-major topk has no 4-align requirement.
            num_tokens = max(max_num_batched_tokens, max_capture)
            topk_blocks = resolved["topk_blocks"]
            device = vllm_config.device_config.device
            # One token-major [T, H, K] workspace for every sparse layer.
            # ``buf[:n]`` is a contiguous dense kernel input, so decode CUDA
            # graph and eager prefill reuse the same storage.
            self.register_buffer(
                "topk_indices_buffer",
                torch.empty(
                    num_tokens,
                    num_index_heads,
                    topk_blocks,
                    dtype=torch.int32,
                    device=device,
                ),
                persistent=False,
            )
            topk_indices_buffer = self.topk_indices_buffer

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: IluMiniMaxM3DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Dense MLP uses reduce_results=False when PP==1 and relies on the next
        # layer's input norm to all-reduce. If the last configured text layer is
        # dense, complete that deferred AR here before the final Gemma RMSNorm.
        self._complete_deferred_dense_mlp_ar = (
            get_pp_group().is_last_rank
            and get_tensor_model_parallel_world_size() > 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and self.end_layer > self.start_layer
            and not _is_moe_layer(config, self.end_layer - 1)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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

        # EAGLE3 is not yet compatible with pipeline parallel
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self._complete_deferred_dense_mlp_ar:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Checkpoint experts use w1=gate, w2=down, w3=up.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # q/k/v_proj -> fused qkv_proj; gate_proj/up_proj -> fused gate_up_proj
        # (dense MLP and shared expert). On sparse layers the indexer
        # index_q/index_k_proj fold into the same fused qkv_proj
        # (MinimaxM3QKVParallelLinearWithIndexer); these entries simply never match on
        # dense layers, whose checkpoints have no index_*_proj weights. Leading
        # dots keep `q_proj`/`k_proj` from matching `index_q_proj`/`index_k_proj`
        # (preceded by `_`, not `.`).
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # The MTP module is not modeled yet.
            if "mtp." in name:
                continue

            # compressed-tensors W4/W8 uses ``weight_scale``. ModelOpt FP8 /
            # MXFP8 checkpoints store the same tensor as ``weight_scale_inv``.
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; don't let the
                # stacked mapping rewrite them.
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Unmapped checkpoint tensors are skipped (e.g. MTP).
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IluMiniMaxM3SparseForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    """Iluvatar MiniMax-M3 causal LM. Eagle3/MTP draft is not implemented."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"language_model.": ""},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.model = IluMiniMaxM3Model(
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
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
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
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self, skip_substrs=list(_LOAD_SKIP_SUBSTRS)
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class IluMiniMaxM3SparseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    """VL wrapper; text backbone is ``IluMiniMaxM3SparseForCausalLM``."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        assert self.multimodal_config is not None
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_config.hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            vision_config = config.vision_config
            self.vision_tower = MiniMaxVLVisionModel(
                config=PretrainedConfig.from_dict(vision_config),
                text_hidden_size=text_hidden_size,
                projector_hidden_size=projector_hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["IluMiniMaxM3SparseForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = image_input["pixel_values"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        image_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = video_input["pixel_values_videos"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        video_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object):
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                multimodal_embeddings.extend(
                    self._process_image_input(multimodal_input)
                )
            if modality == "video":
                multimodal_embeddings.extend(
                    self._process_video_input(multimodal_input)
                )

        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self, skip_substrs=list(_LOAD_SKIP_SUBSTRS)
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if _is_ilu_quant_config(self.quant_config):
            kv_dtype = getattr(
                self.language_model, "kv_cache_dtype", "auto"
            )
            _assert_m3_parameters_loaded(
                self, loaded, kv_cache_dtype=kv_dtype
            )
            _log_m3_quant_load_stats(loaded)
        return loaded
