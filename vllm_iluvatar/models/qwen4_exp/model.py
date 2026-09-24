# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .attention import Qwen4ExpQSAAttention
from .config import Qwen4ExpConfig, Qwen4ExpTextConfig
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .indexer import QSAPoolCarry
from .ple import Qwen4ExpPLELayer
from .ple_utils import ple_conv_window_state_shape


class _Qwen4ExpGDNStateMixin:
    """Add the PLE convolution window as a third state in each GDN page."""

    ple_window_shape: tuple[int, int] | None = None
    ple_window_dtype: torch.dtype | None = None
    hosts_ple_window = False
    _gdn_internal_state_view = False

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        shapes = tuple(super().get_state_shape())
        if self.ple_window_shape is None or self._gdn_internal_state_view:
            return shapes
        return (*shapes, self.ple_window_shape)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = tuple(super().get_state_dtype())
        if self.ple_window_shape is None or self._gdn_internal_state_view:
            return dtypes
        assert self.ple_window_dtype is not None
        return (*dtypes, self.ple_window_dtype)

    def _warmup_prefill_kernels(self, *args, **kwargs) -> None:
        # Upstream GDN owns only the first two states and unpacks their dtypes
        # as a fixed pair during warmup. Keep the extra PLE declaration visible
        # to vLLM cache allocation while presenting the native pair inside that
        # GDN implementation detail.
        self._gdn_internal_state_view = True
        try:
            super()._warmup_prefill_kernels(*args, **kwargs)
        finally:
            self._gdn_internal_state_view = False

    @property
    def ple_conv_window(self) -> torch.Tensor:
        """The ``[num_blocks, channels, state_len]`` PLE window view."""

        if self.ple_window_shape is None:
            raise RuntimeError("PLE window state is not installed")
        return self.kv_cache[2]


_QWEN4_EXP_GDN_STATE_CLASSES: dict[type, type] = {}


def _install_ple_window_states(model: nn.Module, vllm_config: VllmConfig) -> None:
    """Install and assign the GDN-page-backed state used by every PLE layer."""

    shape = ple_conv_window_state_shape(vllm_config)
    if shape is None:
        return

    gdn_layers = sorted(
        (
            module
            for module in model.modules()
            if isinstance(module, QwenGatedDeltaNetAttention)
        ),
        key=lambda module: int(module.layer_idx),
    )
    ple_layers = sorted(
        (module for module in model.modules() if isinstance(module, Qwen4ExpPLELayer)),
        key=lambda module: int(module.layer_idx),
    )
    if len(ple_layers) > len(gdn_layers):
        raise ValueError(
            "Qwen4Exp cannot host more PLE windows than GDN layers: "
            f"got {len(ple_layers)} PLE layers and {len(gdn_layers)} GDN layers"
        )

    dtype = vllm_config.model_config.dtype
    for layer in gdn_layers:
        base_cls = type(layer)
        state_cls = _QWEN4_EXP_GDN_STATE_CLASSES.get(base_cls)
        if state_cls is None:
            state_cls = type(
                f"Qwen4Exp{base_cls.__name__}",
                (_Qwen4ExpGDNStateMixin, base_cls),
                {"__module__": __name__},
            )
            _QWEN4_EXP_GDN_STATE_CLASSES[base_cls] = state_cls
        layer.__class__ = state_cls
        layer.ple_window_shape = shape
        layer.ple_window_dtype = dtype

    for ple_layer, host in zip(ple_layers, gdn_layers):
        ple_layer.state_host = host
        host.hosts_ple_window = True


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


def _ple_skip_substrs(config: object) -> tuple[str, ...]:
    if getattr(config, "ple_layer_ids", ()):
        return ()
    return (".ple.",)


def _requires_raw_input_tokens(config: object) -> bool:
    return bool(getattr(config, "ple_layer_ids", ()))


_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint keeps down and injection projections separate; runtime packs
# them into adjacent logical shards of one MergedColumnParallelLinear.
_HC_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
    }
)


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen4Exp HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        ple_conv_state: torch.Tensor | None,
        ple_state_indices: torch.Tensor | None,
        state_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                ple_conv_state,
                ple_state_indices,
            )

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen4Exp routed experts through vLLM's EPLB protocol."""

    def set_moe_parameters(self, layers: Iterable[nn.Module]) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen4ExpSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "ple_state_indices": 1,
        "state_indices": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        # Bound by Qwen4ExpModelState once the model is built; stays None when
        # the checkpoint has no QSA layers.
        self.qsa_pool_carry: QSAPoolCarry | None = None
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = False
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # hidden state [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        ple_conv_state: torch.Tensor | None = None,
        ple_state_indices: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.qsa_pool_carry is not None:
            # The QSA indexers sit behind an opaque attention op, so they read
            # this step's slot map off the shared carry instead of taking it as
            # an argument. Refreshing it here -- from this call's own inputs --
            # is what keeps it in step with the batch.
            self.qsa_pool_carry.state_indices = state_indices

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                ple_conv_state=ple_conv_state,
                ple_state_indices=ple_state_indices,
                state_indices=state_indices,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                # Deepstack is an external addition to the materialized
                # multi-stream state and therefore terminates delayed combine.
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        # The final mixer consumes the last pending combine and returns both
        # the sampled single stream and the materialized multi-stream state.
        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = (
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ) + _ple_skip_substrs(self.config)
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={substr: None for substr in skip_substrs}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        return loaded


def _verify_supported_config(vllm_config: VllmConfig) -> None:
    """Reject configurations this port does not implement.

    In tree these checks lived in ``MODELS_CONFIG_MAP``; out of tree there is no
    per-architecture config hook, so they run at model construction instead.
    Every one of them would otherwise fail late and obscurely -- or, worse,
    quietly produce wrong output -- rather than at startup.
    """
    import vllm.envs as envs

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        raise NotImplementedError(
            "Qwen4Exp requires the V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1): "
            "its PLE GDN-page window routing, PLE n-gram context and QSA group "
            "accumulator are owned by Qwen4ExpModelState, which only V2 "
            "instantiates"
        )

    compilation_config = vllm_config.compilation_config
    cudagraph_mode = compilation_config.cudagraph_mode
    if cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise NotImplementedError(
            "Qwen4Exp only supports cudagraph_mode=NONE or FULL_DECODE_ONLY; "
            f"got {cudagraph_mode.name}. Prefill and mixed batches must remain eager."
        )
    if (
        cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        and compilation_config.mode != CompilationMode.NONE
    ):
        raise NotImplementedError(
            "Qwen4Exp FULL_DECODE_ONLY requires compilation mode=0 "
            f"(NONE), got {compilation_config.mode}."
        )

    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Qwen4Exp does not support pipeline parallelism")
    if getattr(parallel_config, "enable_eplb", False):
        raise NotImplementedError("Qwen4Exp does not support EPLB")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "Qwen4Exp does not support MTP or speculative decoding"
        )
    if getattr(parallel_config, "enable_dbo", False) or (
        getattr(parallel_config, "ubatch_size", 1) > 1
    ):
        # PLE and QSA both roll per-request state forward once per forward
        # pass, which microbatching would run twice on the same slots.
        raise NotImplementedError(
            "Qwen4Exp does not support dual-batch overlap or microbatching"
        )
    if getattr(vllm_config.cache_config, "cpu_offload_gb", 0):
        raise NotImplementedError("Qwen4Exp does not support CPU offload")
    _verify_prefix_caching_support(vllm_config)


def _verify_prefix_caching_support(vllm_config: VllmConfig) -> None:
    """Reject the prefix-caching combinations whose state cannot be restored.

    A cache hit hands the model a request that starts mid-sequence, so every
    piece of per-request state must be reconstructible from the cached blocks.
    The GDN state is (vLLM's ``mamba_cache_mode="align"`` keeps a block-aligned
    checkpoint), the PLE window shares that checkpoint, and the QSA group
    accumulator is empty at every cache-hit boundary. The cases below remain
    unsupported.
    """
    cache_config = vllm_config.cache_config
    if not getattr(cache_config, "enable_prefix_caching", False):
        return

    if vllm_config.kv_transfer_config is not None:
        # QSA declares supports_kv_connector() == False: the compressed-K side
        # cache is not part of what a connector transfers.
        raise NotImplementedError(
            "Qwen4Exp does not support prefix caching together with a KV connector"
        )
    if getattr(cache_config, "prefix_match_unit", None) is not None:
        # Sub-block matches can resume without a matching PLE window snapshot
        # and can also drop a partial group sum from the QSA accumulator.
        raise NotImplementedError("Qwen4Exp does not support --prefix-match-unit")


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.requires_raw_input_tokens = _requires_raw_input_tokens(config)
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        _verify_supported_config(vllm_config)
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        _install_ple_window_states(self.model, vllm_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the PLE inputs built by
        # Qwen4ExpModelState.prepare_inputs reach Qwen4ExpModel.
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        # In tree, Qwen3.5 copies the checkpoint's mamba_ssm_dtype onto
        # cache_config from MODELS_CONFIG_MAP. Out of tree there is no such
        # hook, and letting "auto" stand would silently drop the GDN recurrent
        # state from the float32 this checkpoint asks for to the model dtype.
        # An explicit --mamba-ssm-cache-dtype still wins, as it does upstream.
        ssm_cache_dtype = vllm_config.cache_config.mamba_ssm_cache_dtype
        if ssm_cache_dtype == "auto":
            ssm_cache_dtype = getattr(
                vllm_config.model_config.hf_text_config, "mamba_ssm_dtype", "auto"
            )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        dtypes = cls.get_gdn_mamba_state_dtype_from_config(vllm_config)
        if ple_conv_window_state_shape(vllm_config) is None:
            return dtypes
        return (*dtypes, vllm_config.model_config.dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        shapes = cls.get_gdn_mamba_state_shape_from_config(vllm_config)
        window = ple_conv_window_state_shape(vllm_config)
        return shapes if window is None else (*shapes, window)

    def get_mamba_state_copy_func(
        self,
    ) -> tuple[MambaStateCopyFunc, ...]:
        funcs = MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        if ple_conv_window_state_shape(self.vllm_config) is None:
            return funcs
        return (*funcs, get_temporal_copy_spec)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip = {"mtp.": None}
        skip.update({substr: None for substr in _ple_skip_substrs(self.config)})
        mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_substr=skip)
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen4ExpConfig:
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen4ExpMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen4Exp language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ]
    }

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen4ExpForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        text_config = vllm_config.model_config.hf_text_config
        self.requires_raw_input_tokens = _requires_raw_input_tokens(text_config)
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self.set_moe_parameters(self.language_model.model.layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen4Exp language_model_only does not accept multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            ple_conv_state=kwargs.get("ple_conv_state"),
            ple_state_indices=kwargs.get("ple_state_indices"),
            state_indices=kwargs.get("state_indices"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip = {"mtp.": None}
        skip.update(
            {substr: None for substr in _ple_skip_substrs(self.language_model.config)}
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr=skip,
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    def get_mamba_state_copy_func(
        self,
    ) -> tuple[MambaStateCopyFunc, ...]:
        return self.language_model.get_mamba_state_copy_func()


class Qwen4ExpMTP(nn.Module):
    """MTP is gated off in this BI-V150 port."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Qwen4Exp MTP is disabled on this port. Run without speculative decoding."
        )


def register_model() -> None:
    """Register Qwen4Exp architectures through the official model registry."""
    from vllm import ModelRegistry

    module = "vllm_iluvatar.models.qwen4_exp.model"
    for architecture in (
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    ):
        ModelRegistry.register_model(
            architecture,
            f"{module}:{architecture}",
        )


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
