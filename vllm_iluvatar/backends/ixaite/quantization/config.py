# Source: ixaite_compressed_tensors.py at 47fb804; independent config and scheme construction.
# SPDX-License-Identifier: Apache-2.0
"""Standard W8A8 provider route, separate from vendor checkpoint formats."""

from copy import deepcopy
from functools import cache, cached_property

from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    is_activation_quantization_format,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_iluvatar.backends.ixaite.runtime import (
    load_ixaite_python_op,
    provider_mode,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    _LEGACY_QUANT_METHOD,
    IluQuantConfig,
    _convert_legacy_ilu_deepseek_v4_w4a8,
    _is_legacy_ilu_deepseek_v4_w4a8,
)


# Preserve corrected extension metadata without changing the mainline parser.
class _IxaiteQuantMetadata(IluQuantConfig):
    # Duplicate targets must take both quant arguments and hints from the last group.
    @classmethod
    def _quantization_scheme_map_from_config(cls, config: dict) -> dict:
        target_scheme_map = (
            CompressedTensorsConfig._quantization_scheme_map_from_config(config=config)
        )
        config_groups = config.get("config_groups", dict())
        for target, _ in target_scheme_map.items():
            target_quant_config = None
            # Community parsing lets the last group win for duplicate targets;
            # its extension metadata must come from that same group.
            for _, quant_config in reversed(list(config_groups.items())):
                targets = quant_config.get("targets")
                if target in targets:
                    target_quant_config = quant_config
                    break
            assert target_quant_config is not None
            target_scheme_map[target]["ilu_params"] = dict(
                target_quant_config.get("iluvatar", {})
            )
        return target_scheme_map


# Bind the provider before constructing objects that own weight layout and state.
@cache
def _select_w8a8_linear_scheme_cls(
    strategy: str,
    is_static_input_scheme: bool,
    input_symmetric: bool,
):
    """Bind one W8A8 scheme class for a complete quantization contract."""

    # Check the whole lifecycle before choosing a provider that owns weight layout.
    def load_ixaite_scheme():
        if strategy != QuantizationStrategy.CHANNEL:
            return None, "ixaite W8A8 requires channel-wise weight quantization"
        # Provider-specific objects must not be constructed unless their whole
        # load-to-forward lifecycle is available in this worker.
        reasons: list[str] = []
        for func_name in ("apply", "bind", "warmup"):
            implementation, reason = load_ixaite_python_op("w8a8", func_name)
            if implementation is None:
                return None, f"ixaite W8A8 {func_name} is unavailable: {reason}"
            reasons.append(reason)

        from vllm_iluvatar.backends.ixaite.quantization.w8a8 import (  # noqa: E501
            IxaiteCompressedTensorsW8A8Int8,
        )

        return IxaiteCompressedTensorsW8A8Int8, reasons[0]

    scheme_cls, reason = load_ixaite_scheme()
    if scheme_cls is None:
        raise RuntimeError(f"ixaite W8A8 is unavailable: {reason}")
    return scheme_cls


# A grouped DS4 layout must be resolved before dense W8A8 transforms mutate it.
class IxaiteCompressedTensorsLinearMethod(CompressedTensorsLinearMethod):
    # Ordinary linears retain community processing with their own selected scheme.
    def process_weights_after_loading(self, layer):
        from vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint import (
            maybe_repack_ds4_wo_a_w8a16,
        )

        if not maybe_repack_ds4_wo_a_w8a16(layer):
            return super().process_weights_after_loading(layer)


# Keep standard parsing independent; delegate formats with storage metadata intact.
class IxaiteCompressedTensorsConfig(CompressedTensorsConfig):
    # Keep the plugin's existing device capability requirement.
    @classmethod
    def get_min_capability(cls):
        return IluQuantConfig.get_min_capability()

    # Legacy checkpoints keep upstream conversion, including BMM and MoE metadata.
    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        if _is_legacy_ilu_deepseek_v4_w4a8(config):
            config = _convert_legacy_ilu_deepseek_v4_w4a8(config)
        return super().from_config(config)

    # Delay vendor parsing until a layer actually requires its checkpoint contract.
    @cached_property
    def _ilu_config(self):
        return _IxaiteQuantMetadata.from_config(deepcopy(self.config))

    # vLLM sets packed-module aliases after parsing, so synchronize at delegation.
    def _vendor_config(self):
        self._ilu_config.packed_modules_mapping = self.packed_modules_mapping
        return self._ilu_config

    # Both routes must resolve the same renamed checkpoint targets and exclusions.
    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        self._vendor_config().apply_vllm_mapper(hf_to_vllm_mapper)
        super().apply_vllm_mapper(hf_to_vllm_mapper)

    # Match the community INT8 contracts without interpreting vendor execution hints.
    def _standard_w8a8(self, layer, layer_name):
        parts = self.get_scheme_dict(layer, layer_name)
        if not parts or parts.get("output_activations") is not None:
            return False
        weights, inputs = parts.get("weights"), parts.get("input_activations")
        quant_format = parts.get("format") or self.quant_format
        return (
            weights is not None
            and inputs is not None
            and weights.type == inputs.type == QuantizationType.INT
            and is_activation_quantization_format(quant_format)
            and (
                self._is_static_tensor_w8a8(weights, inputs)
                or self._is_dynamic_token_w8a8(weights, inputs)
            )
        )

    # Community loading and transforms remain in charge of standard W8A8 layers.
    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, (LinearBase, ParallelLMHead)):
            from vllm.model_executor.layers.fused_moe import RoutedExperts

            if isinstance(layer, RoutedExperts):
                return self._moe_method(layer, prefix)
            # Unquantized attention retains its scale-only community lifecycle.
            if not isinstance(layer, RoutedExperts):
                method = super().get_quant_method(layer, prefix)
                if method is None or (
                    type(method) is CompressedTensorsKVCacheMethod
                    and self.kv_cache_scheme is None
                ):
                    return method
            raise RuntimeError(
                f"ixaite quantization is not available for {type(layer).__name__}: {prefix}"
            )
        method = super().get_quant_method(layer, prefix)
        if type(method) is CompressedTensorsLinearMethod:
            return IxaiteCompressedTensorsLinearMethod(self)
        return method

    # Resolve fused projections once, retaining vendor storage metadata without execution.
    def _moe_method(self, layer, prefix):
        config = self._vendor_config()
        config._add_fused_moe_to_target_scheme_map()
        parts = [
            config.get_scheme_dict(layer, prefix + suffix)
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        if any(part != parts[0] for part in parts[1:]):
            raise ValueError(
                "All MoE projections must have the same quantization scheme"
            )
        if parts[0] is None:
            from vllm_iluvatar.backends.ixaite.moe.unquantized import (
                IluUnquantizedFusedMoEMethod,
            )

            return IluUnquantizedFusedMoEMethod(layer.moe_config)
        weight, inputs = parts[0].get("weights"), parts[0].get("input_activations")
        params = parts[0].get("ilu_params", {})
        if params.get("method") is not None:
            raise RuntimeError(
                f"ixaite MoE checkpoint method is unavailable: {params['method']}"
            )
        if (
            weight is not None
            and inputs is not None
            and weight.num_bits == 4
            and inputs.num_bits == 8
        ):
            from vllm_iluvatar.backends.ixaite.moe.w4a8 import (
                IluCompressedTensorsW4A8Int8MoEMethod,
            )

            return IluCompressedTensorsW4A8Int8MoEMethod(
                weight, inputs, layer.moe_config, prefix, params
            )
        if self._is_dynamic_token_w8a8(weight, inputs):
            from vllm_iluvatar.backends.ixaite.moe.w8a8 import (
                IluCompressedTensorsW8A8Int8MoEMethod,
            )

            return IluCompressedTensorsW8A8Int8MoEMethod(
                weight, inputs, layer.moe_config, prefix, params
            )
        raise RuntimeError(f"ixaite MoE quantization contract is unavailable: {prefix}")

    # Select once before weights exist; neither route changes provider in forward.
    def get_scheme(self, layer, layer_name=None):
        if not self._standard_w8a8(layer, layer_name):
            parts = self._vendor_config().get_scheme_dict(layer, layer_name)
            if parts is None:
                return None
            raise RuntimeError(
                f"ixaite does not support this quantization contract: {layer_name}"
            )
        scheme = super().get_scheme(layer, layer_name)
        scheme_cls = _select_w8a8_linear_scheme_cls(
            scheme.strategy, scheme.is_static_input_scheme, scheme.input_symmetric
        )
        return scheme_cls(
            strategy=scheme.strategy,
            is_static_input_scheme=scheme.is_static_input_scheme,
            input_symmetric=scheme.input_symmetric,
        )


# Run after upstream decorators so import order cannot change the chosen route.
def register_ixaite_quantization_config():
    config_cls = (
        IluQuantConfig
        if provider_mode() == "baseline"
        else IxaiteCompressedTensorsConfig
    )
    for name in ("compressed-tensors", _LEGACY_QUANT_METHOD):
        register_quantization_config(name)(config_cls)
    if provider_mode() != "baseline":
        from vllm_iluvatar.backends.ixaite.quantization.minimax_m3_flash import (
            METHOD_NAME as _MINIMAX_FLASH,
        )
        from vllm_iluvatar.backends.ixaite.quantization.minimax_m3_flash import (
            IxaiteMiniMaxM3FlashConfig,
        )

        register_quantization_config(_MINIMAX_FLASH)(IxaiteMiniMaxM3FlashConfig)
