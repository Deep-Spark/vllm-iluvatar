# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization configuration and checkpoint/method selection."""

from __future__ import annotations

import copy
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
    IluFp8MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.linear import (
    IluFp8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.config import (
    resolve_ilu_runtime_method,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear.native_to_fp8 import (
    IluFp8PerChannelOnlineLinearMethod,
)


@register_quantization_config("fp8")
class IluFp8Config(Fp8Config):
    """FP8 config whose routed experts and linear layers use Iluvatar kernels.

    ``weight_block_size`` describes routed experts. The optional
    ``none_moe_weight_block_size`` independently selects block scaling for
    ordinary Linear layers; ``null`` selects per-channel scaling.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
        gemm_format: str = "TN",
        none_moe_use_weight_block_size: bool = False,
        none_moe_is_checkpoint_fp8_serialized: bool = False,
    ) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )
        self.gemm_format = gemm_format
        self.none_moe_use_weight_block_size = none_moe_use_weight_block_size
        self.none_moe_is_checkpoint_fp8_serialized = (
            none_moe_is_checkpoint_fp8_serialized
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        # HY4 names the routed-expert storage format ``expert_dtype`` even
        # though the mixed checkpoint's top-level quant_method remains fp8.
        expert_dtype = cls.get_from_keys_or(config, ["expert_dtype"], None)
        if (
            store_dtype is not None
            and expert_dtype is not None
            and store_dtype != expert_dtype
        ):
            raise ValueError("Conflicting store_dtype and expert_dtype in FP8 config")
        if store_dtype is None:
            store_dtype = expert_dtype
        gemm_format = cls.get_from_keys_or(config, ["gemm_format"], "TN")
        gemm_format = gemm_format.upper()
        # for models that not the all weights are fp8, those layer's name are passed with ignored_layers
        # if we want to use fp8 for whose layers, we can do:
        #    1. offline quant those layers and remove those layer's names from ignored_layers,
        #       add none_moe_use_weight_block_size: False to config.json
        #    2. remove those layer's names from ignored_layers and add none_moe_is_checkpoint_fp8_serialized: False and
        #       none_moe_use_weight_block_size: False to config.json to enable online fp8 quantization for those layers
        # default to True so official vllm can use the default behavior.
        none_moe_use_weight_block_size = cls.get_from_keys_or(
            config, ["none_moe_use_weight_block_size"], True
        )
        none_moe_is_checkpoint_fp8_serialized = bool(
            cls.get_from_keys_or(
                config, ["none_moe_is_checkpoint_fp8_serialized"], True
            )
        )
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        if gemm_format != "TN":
            raise ValueError("IluFp8Config only supports tn gemm_format for now")
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
            gemm_format=gemm_format,
            none_moe_use_weight_block_size=none_moe_use_weight_block_size,
            none_moe_is_checkpoint_fp8_serialized=none_moe_is_checkpoint_fp8_serialized,
        )

    def _is_layer_skipped(self, prefix: str) -> bool:
        """Match exact exclusions and parent-module exclusions.

        Step-3.7 VL remaps ``modules_to_not_convert`` onto
        ``language_model.model.*`` while MTP drafts still build under
        ``model.*``. Accept both spellings so shared target quant configs
        still skip BF16 MTP layers (e.g. sliding-attn ``g_proj``).
        """
        prefixes = [prefix]
        if prefix.startswith("model."):
            prefixes.append(f"language_model.{prefix}")
        elif prefix.startswith("language_model.model."):
            prefixes.append(prefix[len("language_model.") :])

        for candidate in prefixes:
            if is_layer_skipped(
                prefix=candidate,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return True
            if any(candidate.startswith(f"{name}.") for name in self.ignored_layers):
                return True
        return False

    def _get_original_method_and_disposition(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                return UnquantizedLinearMethod(), "ignored"

            new_quant = copy.copy(self)
            new_quant.weight_block_size = (
                self.weight_block_size if self.none_moe_use_weight_block_size else None
            )
            if (
                self.none_moe_is_checkpoint_fp8_serialized
                and self.is_checkpoint_fp8_serialized
            ):
                method = IluFp8LinearMethod(new_quant)
            else:
                method = IluFp8PerChannelOnlineLinearMethod(new_quant)
            return method, "quantized"

        if (
            isinstance(layer, RoutedExperts)
            and self.is_checkpoint_fp8_serialized
            and self.store_dtype == "mxfp4"
            and not self._is_layer_skipped(prefix)
        ):
            from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
                IluMxfp4MoEMethod,
            )

            return IluMxfp4MoEMethod(layer.moe_config), "quantized"

        if (
            isinstance(layer, RoutedExperts)
            and self.is_checkpoint_fp8_serialized
            and self.weight_block_size is not None
            and self.store_dtype != "mxfp4"
            and not self._is_layer_skipped(prefix)
        ):
            return IluFp8MoEMethod(self, layer), "quantized"
        return super().get_quant_method(layer, prefix), "quantized"

    # [ILU override] Fp8Config.get_quant_method — delta: first preserve the
    # checkpoint's original dispatch, then optionally overlay a strictly
    # validated additional_config.ilu_quant method.
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        (
            original_method,
            original_disposition,
        ) = self._get_original_method_and_disposition(layer, prefix)
        runtime_method = resolve_ilu_runtime_method(
            base_quant_config=self,
            original_method=original_method,
            original_disposition=original_disposition,
            layer=layer,
            prefix=prefix,
        )
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
            IluMxfp4MoEMethod,
            select_mxfp4_method,
        )

        if isinstance(original_method, IluMxfp4MoEMethod):
            return select_mxfp4_method(layer, original_method, runtime_method)
        return runtime_method if runtime_method is not None else original_method


__all__ = [
    "IluFp8Config",
]
