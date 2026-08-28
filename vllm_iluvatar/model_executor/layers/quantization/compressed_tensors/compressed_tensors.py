# SPDX-License-Identifier: Apache-2.0
"""Iluvatar compressed-tensors linear method-selection wrapper.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors.py

Target   : Replace CompressedTensorsConfig with IluQuantConfig
Baseline : main_6e448d0
Why      : Official CompressedTensors standards cannot cover
           distribution specific quantization implementations.
How      : IluQuantConfig extends the CompressedTensorsConfig and
           overrides the get_quant_method and get_scheme methods
           to add Iluvatar specific quantization implementations.
           Legacy ``ilu_deepseek_v4_w4a8`` checkpoints are registered on
           the same class and rewritten to config_groups in ``from_config``.
Scope    : register_quantization_config during file import time.
Safety   : Mainly call super methods, unsupported schemes return the original community scheme unchanged.
Remove   : when community compressed-tensors give an oot-like
           extension point for distribution specific quantization implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int import (  # noqa: E501
    CompressedTensorsW4A8Int,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8,
)

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
    IluCompressedTensorsW4A8Int,
    IluCompressedTensorsW8A8Int8,
)

logger = init_logger("vllm." + __name__)

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]
_ORIGINAL_GET_SCHEME_FROM_PARTS: Callable | None = None

_LEGACY_QUANT_METHOD = "ilu_deepseek_v4_w4a8"
_DEFAULT_W8A8_ILU_PARAMS = {
    "gemm_format": "TN",
    "mixed_strategy": "CHANNEL",
    "gemm_mode": "PERF",
}


def _is_legacy_ilu_deepseek_v4_w4a8(config: dict[str, Any]) -> bool:
    if config.get("quant_method") == _LEGACY_QUANT_METHOD:
        return True
    return "config_groups" not in config and "routed_experts" in config


def _legacy_tensor_args(
    section: dict[str, Any], *, dynamic_default: bool
) -> dict[str, Any]:
    return {
        "num_bits": int(section["num_bits"]),
        "type": section.get("type", "int"),
        "symmetric": bool(section.get("symmetric", True)),
        "strategy": section.get("strategy", "channel"),
        "group_size": section.get("group_size"),
        "dynamic": bool(section.get("dynamic", dynamic_default)),
    }


def _legacy_input_activations(section: dict[str, Any]) -> dict[str, Any] | None:
    act = section.get("input_activations")
    if not act:
        return None
    return _legacy_tensor_args(act, dynamic_default=True)


def _legacy_w8a8_group(targets: list[str], section: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "int-quantized",
        "targets": targets,
        "weights": _legacy_tensor_args(section, dynamic_default=False),
        "input_activations": _legacy_input_activations(section),
        "output_activations": None,
        "iluvatar": dict(_DEFAULT_W8A8_ILU_PARAMS),
    }


def _convert_legacy_ilu_deepseek_v4_w4a8(config: dict[str, Any]) -> dict[str, Any]:
    """Map the removed ``ilu_deepseek_v4_w4a8`` schema onto config_groups."""
    groups: dict[str, Any] = {}

    routed = config.get("routed_experts") or {}
    if routed:
        groups["routed_experts"] = {
            "targets": ["re:.*\\.experts"],
            "weights": _legacy_tensor_args(routed, dynamic_default=False),
            "input_activations": _legacy_input_activations(routed),
            "output_activations": None,
            "iluvatar": {
                "gemm_format": str(routed.get("gemm_format", "TN")),
                "pack_version": int(routed.get("pack_version", 2)),
            },
        }

    shared = config.get("shared_experts") or {}
    if shared:
        groups["shared_experts"] = _legacy_w8a8_group(
            ["re:.*\\.shared_expert(?:s)?(?:\\.|$)"],
            shared,
        )

    dense = config.get("dense_linears") or {}
    if dense and dense.get("targets"):
        groups["dense_linears"] = _legacy_w8a8_group(list(dense["targets"]), dense)

    bmm = config.get("bmm_w8a16_layers") or {}
    if bmm and bmm.get("targets"):
        groups["wo_a_w8a16"] = {
            "targets": list(bmm["targets"]),
            "weights": _legacy_tensor_args(bmm, dynamic_default=False),
            "input_activations": None,
            "output_activations": None,
            "iluvatar": {
                "method": "IluV4W8A16BMMLinearMethod",
                "bmm_w8a16_gemm_format": str(bmm.get("gemm_format", "k16n16")),
            },
        }

    mla_bmm = config.get("mla_bmm_w8a16") or {}
    if mla_bmm and mla_bmm.get("targets"):
        groups["kv_b_w8a16"] = {
            "targets": list(mla_bmm["targets"]),
            "weights": _legacy_tensor_args(mla_bmm, dynamic_default=False),
            "input_activations": None,
            "output_activations": None,
            "iluvatar": {
                "method": "IluGlmKVBProjW8A16Method",
            },
        }

    ignore: list[str] = []
    for name in list(config.get("ignore") or []) + list(
        config.get("bf16_layers") or []
    ):
        if name not in ignore:
            ignore.append(name)

    ignore.append("lm_head")
    return {
        "quant_method": "compressed-tensors",
        "format": "iluvatar-mixed",
        "config_groups": groups,
        "ignore": ignore,
    }


@register_quantization_config(_LEGACY_QUANT_METHOD)
@register_quantization_config("compressed-tensors")
class IluQuantConfig(CompressedTensorsConfig):
    """
    Custom mixed-precision quant config for Iluvatar opt quant / model.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # ixformer w4a8 / w8a8 int8 kernels target sm75+ in practice; keep
        # generous to let ilu Iluvatar / consumer Hopper run.
        return 75

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluQuantConfig:
        if _is_legacy_ilu_deepseek_v4_w4a8(config):
            logger.info_once(
                "ilu_deepseek_v4_w4a8 is deprecated and only used for some key clients, please use compressed-tensors instead"
            )
            logger.info_once(
                "Converting legacy quant_method %r to compressed-tensors",
                _LEGACY_QUANT_METHOD,
            )
            config = _convert_legacy_ilu_deepseek_v4_w4a8(config)
        return super().from_config(config)

    @classmethod
    def _quantization_scheme_map_from_config(
        cls, config: dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        target_scheme_map = super()._quantization_scheme_map_from_config(config=config)
        config_groups = config.get("config_groups", dict())
        for target, _ in target_scheme_map.items():
            target_quant_config = None
            for _, quant_config in config_groups.items():
                targets = quant_config.get("targets")
                if target in targets:
                    target_quant_config = quant_config
                    break
            assert target_quant_config is not None
            target_scheme_map[target]["ilu_params"] = dict(
                target_quant_config.get("iluvatar", {})
            )
        return target_scheme_map

    # ----- Dispatch --------------------------------------------------------
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Lazy imports to avoid heavy dependencies at config-class import time.
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        from vllm_iluvatar.model_executor.layers.quantization import (
            IluvatarQuantRegistry,
        )

        if isinstance(layer, (LinearBase, RoutedExperts)):
            scheme_dict = self.get_scheme_dict(layer=layer, layer_name=prefix)

            if scheme_dict is None:
                return super().get_quant_method(layer=layer, prefix=prefix)
            # Work on a copy: get_quant_method may be called more than once for
            # the same target and must not consume method metadata.
            ilu_params = dict(scheme_dict.get("ilu_params", {}))
            method_name = ilu_params.pop("method", None)
            if method_name is not None:
                method = IluvatarQuantRegistry.get_method(method_name)
                if method is None:
                    raise ValueError(
                        f"Method {method_name!r} is not registered in "
                        "IluvatarQuantRegistry"
                    )
                if isinstance(layer, RoutedExperts):
                    return method(self, layer, ilu_params)
                return method(self, ilu_params)

        return super().get_quant_method(layer=layer, prefix=prefix)

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        output_quant: QuantizationArgs | None = None,
        format: str | None = None,
        layer_name: str | None = None,
        ilu_params: dict[str, Any] | None = None,
    ):
        scheme = super()._get_scheme_from_parts(
            weight_quant=weight_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            format=format,
            layer_name=layer_name,
        )
        if not ilu_params:
            return scheme

        # [ILU override] W8A8 INT8 linear scheme - delta: use Iluvatar kernel
        # layout, padding, and ixformer execution while preserving upstream
        # scheme detection.
        if isinstance(scheme, CompressedTensorsW8A8Int8):
            return IluCompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=not input_quant.dynamic,
                input_symmetric=input_quant.symmetric,
                ilu_params=ilu_params,
            )

        # [ILU override] W4A8 INT linear scheme - delta: use Iluvatar packed
        # W4A8 layout and ixformer execution.
        if isinstance(scheme, CompressedTensorsW4A8Int):
            return IluCompressedTensorsW4A8Int(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                is_static_input_scheme=bool(input_quant and not input_quant.dynamic),
                input_symmetric=input_quant.symmetric,
                ilu_params=ilu_params,
            )

        return scheme

    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> CompressedTensorsScheme | None:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Use the new get_quant_args method to extract QuantizationArgs
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        if not scheme_dict:
            return super().get_scheme(layer, layer_name)

        ilu_params = dict(scheme_dict.get("ilu_params", {}))
        if not ilu_params:
            return super().get_scheme(layer, layer_name)

        # the following code is copied from original code, but we add the ilu_params to the scheme_dict
        weight_quant = None
        input_quant = None
        output_quant = None
        format = None
        if scheme_dict:
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            output_quant = scheme_dict.get("output_activations")
            format = scheme_dict.get("format")

        if weight_quant is None:
            # Falling back to UnquantizedLinearMethod
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant,
                input_quant=input_quant,
                output_quant=output_quant,
                format=format,
                layer_name=layer_name,
                ilu_params=ilu_params,
            )

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
        return scheme
