# SPDX-License-Identifier: Apache-2.0
"""Private shared implementation for runtime Linear W8A8 conversion."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_iluvatar.model_executor.layers.quantization.common.w8a8_linear import (
    IluW8A8Int8LinearMethod,
)

_ALLOWED_OPTIONS = frozenset({"gemm_format", "mixed_strategy", "gemm_mode"})


class _IluRuntimeW8A8Int8LinearMethod(IluW8A8Int8LinearMethod):
    """Validate the explicit runtime conversion and track its load state."""

    # Source ``weights.format`` and checkpoint dtypes this subclass accepts,
    # plus the original dispatch it is allowed to replace.
    source_format: str
    source_dtypes: tuple[str, ...]
    original_disposition: str
    original_method_cls: type

    def __init__(self, context: Any) -> None:
        self.context = context
        super().__init__(context.original_method, **context.options)

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        group = context.group
        weights = group.weights
        source, target = weights.source, weights.target
        where = group.group_id
        if not isinstance(context.layer, LinearBase):
            raise TypeError(f"{where}: runtime W8A8 only supports LinearBase")
        if not isinstance(context.original_method, cls.original_method_cls):
            raise TypeError(
                f"{where}: a {cls.source_format} source requires "
                f"{cls.original_method_cls.__name__}, got "
                f"{type(context.original_method).__name__}"
            )
        if group.require.original_disposition != cls.original_disposition:
            raise ValueError(
                f"{where}: a {cls.source_format} source requires "
                f"original_disposition={cls.original_disposition}"
            )
        # Dense Linears are small enough that the whole-layer stage costs one
        # layer of staging; weight_load would mean reimplementing the loader
        # for no memory win.
        if (
            weights.loading.mode != "transform"
            or weights.loading.stage != "process_weights_after_loading"
        ):
            raise ValueError(
                f"{where}: runtime W8A8 requires "
                "loading=transform@process_weights_after_loading"
            )
        if source.format != cls.source_format or tuple(source.dtypes) != (
            cls.source_dtypes
        ):
            raise ValueError(
                f"{where}: expected a {cls.source_format} source of "
                f"{list(cls.source_dtypes)}"
            )
        if (
            target.format != "int8-channel"
            or target.dtype != "int8"
            or target.strategy != "channel"
            or set(target.extra)
        ):
            raise ValueError(f"{where}: unsupported runtime W8A8 weight target")
        activation = group.input_activations
        if (
            activation is None
            or activation.num_bits != 8
            or activation.type != "int"
            or activation.strategy != "token"
            or not activation.dynamic
            or group.output_activations is not None
        ):
            raise ValueError(
                f"{where}: runtime W8A8 requires dynamic per-token INT8 input "
                "activations and null output_activations"
            )
        unknown = set(group.options) - _ALLOWED_OPTIONS
        if unknown:
            raise ValueError(f"{where}: unsupported options: {sorted(unknown)}")
        # The GLM W8A8 kernels the offline checkpoint selects, pinned so a
        # config cannot quietly pick an untested GEMM for these layers.
        for option, required in (
            ("gemm_format", "TN"),
            ("mixed_strategy", "CHANNEL"),
            ("gemm_mode", "PERF"),
        ):
            if group.options.get(option, required) != required:
                raise ValueError(f"{where}: runtime W8A8 requires {option}={required}")
        return cls(context)

    def create_weights(self, layer: torch.nn.Module, *weight_args, **weight_attrs):
        super().create_weights(layer, *weight_args, **weight_attrs)
        self.context.record_shape_result(transformed=True)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.context.assert_validation_complete()
        return super().apply(layer, x, bias)
