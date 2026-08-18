# SPDX-License-Identifier: Apache-2.0
"""Iluvatar compressed-tensors linear method-selection wrapper.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors.py

Target   : Replace CompressedTensorsConfig with IluQuantConfig
Baseline : main_61ac3680210
Why      : Official CompressedTensors standards cannot cover
           distribution specific quantization implementations.
How      : IluQuantConfig extends the CompressedTensorsConfig and
           overrides the get_quant_method and get_scheme methods
           to add Iluvatar specific quantization implementations.
Scope    : register_quantization_config during file import time.
Safety   : Mainly call super methods, unsupported schemes return the original community scheme unchanged.
Remove   : when community compressed-tensors give an oot-like
           extension point for distribution specific quantization implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

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

logger = init_logger(__name__)
QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]
_ORIGINAL_GET_SCHEME_FROM_PARTS: Callable | None = None


@register_quantization_config("iluvatar-compressed-tensors")
class IluQuantConfig(CompressedTensorsConfig):
    """
    Custom mixed-precision quant config for Iluvatar opt quant / model.
    """

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        kv_cache_scheme: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        transform_config: dict[str, Any] | None = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
    ) -> None:
        super().__init__(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            kv_cache_scheme=kv_cache_scheme,
            config=config,
            transform_config=transform_config,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
        )

    @classmethod
    def get_name(cls):  # type: ignore[override]
        return "iluvatar-compressed-tensors"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # ixformer w4a8 / w8a8 int8 kernels target sm75+ in practice; keep
        # generous to let ilu Iluvatar / consumer Hopper run.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluQuantConfig:
        if "config_groups" in config:
            grps_without_attn_quant = {}
            for k, v in config["config_groups"].items():
                # e.g. LlamaAttention, Qwen3Attention, etc.
                if len(v["targets"]) == 1 and v["targets"][0].endswith("Attention"):
                    logger.warning(
                        "Skipping CompressedTensors config group for %s. Attention "
                        "quant is coupled with KV-cache quantization in vLLM.",
                        v["targets"][0],
                    )
                    continue
                grps_without_attn_quant[k] = v
            config["config_groups"] = grps_without_attn_quant

        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)

        CompressedTensorsConfig._parse_sparsity_config(config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            config=config,
            transform_config=config.get("transform_config"),
            kv_cache_scheme=config.get("kv_cache_scheme"),
            total_num_heads=config.get("total_num_heads"),
            total_num_kv_heads=config.get("total_num_kv_heads"),
        )

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
        ilu_params = ilu_params or {}
        scheme = super()._get_scheme_from_parts(
            weight_quant=weight_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            format=format,
            layer_name=layer_name,
        )

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
                ilu_params=dict(scheme_dict["ilu_params"]),
            )

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
        return scheme
