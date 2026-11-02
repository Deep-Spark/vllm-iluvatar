# SPDX-License-Identifier: Apache-2.0
"""Load-time conversion of a Linear to the static GLM per-channel W8A8 path.

Two source contracts reach the same resident weights, because the official
GLM-5.2 FP8 checkpoint stores its Linear layers in two different ways:

* ``fp8-block``: everything the FP8 conversion touched -- attention
  ``fused_qkv_a_proj`` / ``q_b_proj`` / ``o_proj``, the DSA ``indexer.wq_b``,
  the dense MLPs and the shared experts. ``IluFp8Config`` would dispatch these
  to ``IluFp8LinearMethod`` and run a 128x128 block-scaled FP8 GEMM.
* ``native``: the MTP ``eh_proj``, which the checkpoint lists in
  ``modules_to_not_convert`` and ships as BF16, so ``IluFp8Config`` leaves it
  on ``UnquantizedLinearMethod``.

Both end up as INT8 ``[N, K]`` plus an FP32 ``[N, 1]`` per-output-channel
scale consumed by :class:`IluCompressedTensorsW8A8Int8` -- the same scheme the
offline ``GLM-5.2-w4a8-NN`` checkpoint selects for these layers through its
``dense_linears`` / ``shared_experts`` config groups. Converting them matters
beyond the GEMM itself: the decoder's fused RMSNorm+INT8 path and the q_a
norm+quant fusion both arm themselves by checking that their consumer's
``scheme`` is this class, so leaving a single consumer on FP8 silently keeps
those fusions on the split path.

Per-channel scales are computed from this rank's shard. For the column-parallel
and replicated layers that is the whole checkpoint row; for the row-parallel
``o_proj`` / ``down_proj`` it is a K slice, so the scale is finer than the
offline checkpoint's full-K scale. That stays numerically valid -- each rank
dequantizes its own partial product before the reduction -- and unlike W4A8 NN
there is no second scale stage that depends on the INT8 amax being exactly 127.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)
from vllm_iluvatar.model_executor.layers.quantization.fp8 import IluFp8LinearMethod
from vllm_iluvatar.model_executor.layers.quantization.ilu_quant_registry import (
    IluvatarQuantRegistry,
)

_ALLOWED_OPTIONS = frozenset({"gemm_format", "mixed_strategy", "gemm_mode"})


def _quantize_per_channel_int8(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 of an FP32 ``[N, K]`` block."""
    amax = value.abs().amax(dim=1, keepdim=True)
    # An all-zero channel would divide by zero; a scale of 1 keeps it zero.
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    return torch.round(value / scale).clamp(-127, 127).to(torch.int8), scale


def quantize_fp8_block_to_w8a8_int8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    block_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one block-FP8 ``[N, K]`` tensor to INT8 + ``[N, 1]`` scale.

    Dequantize and requantize one source row-block at a time so a large
    projection never needs a full FP32 copy of itself.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("W8A8 conversion requires a 2-D float8_e4m3fn weight")
    n, k = weight.shape
    bn, bk = block_shape
    expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    if tuple(block_scale.shape) != expected or block_scale.dtype != torch.float32:
        raise ValueError(
            f"block scale shape/dtype {tuple(block_scale.shape)}/{block_scale.dtype} "
            f"does not match expected {expected}/torch.float32"
        )

    qweight = torch.empty_like(weight, dtype=torch.int8)
    channel_scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for n0 in range(0, n, bn):
        n1 = min(n0 + bn, n)
        row_scale = block_scale[n0 // bn].repeat_interleave(bk)[:k]
        value = weight[n0:n1].float() * row_scale.unsqueeze(0)
        qweight[n0:n1], channel_scale[n0:n1] = _quantize_per_channel_int8(value)
    return qweight, channel_scale


def quantize_native_to_w8a8_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one unquantized ``[N, K]`` tensor to INT8 + ``[N, 1]`` scale."""
    if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("W8A8 conversion requires a 2-D BF16/FP16 weight")
    return _quantize_per_channel_int8(weight.float())


class _IluRuntimeW8A8Int8LinearMethod(LinearMethodBase):
    """Shared load/compute surface for the two W8A8 source contracts.

    Weights are created by the original method so the checkpoint's own
    ``weight_loader`` keeps driving TP and fused-shard assembly; only the
    resident format changes, in ``process_weights_after_loading``.
    """

    # Source ``weights.format`` and checkpoint dtypes this subclass accepts,
    # plus the original dispatch it is allowed to replace.
    source_format: str
    source_dtypes: tuple[str, ...]
    original_disposition: str
    original_method_cls: type

    def __init__(self, context: Any) -> None:
        self.context = context
        self.original_method = context.original_method
        self.scheme = IluCompressedTensorsW8A8Int8(
            strategy="channel",
            is_static_input_scheme=False,
            input_symmetric=True,
            ilu_params={
                "gemm_format": context.options.get("gemm_format", "TN"),
                "mixed_strategy": context.options.get("mixed_strategy", "CHANNEL"),
                "gemm_mode": context.options.get("gemm_mode", "PERF"),
            },
        )

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
        self.original_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        # UnquantizedLinearMethod does not set this, and the scheme reads it to
        # tell a fused module from a plain one.
        layer.logical_widths = output_partition_sizes
        layer.scheme = self.scheme
        # The scheme pads rather than rejecting shapes, so every selected layer
        # is transformed and none falls back.
        self.context.record_shape_result(transformed=True)

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight, channel_scale = self._quantize(layer)
        replace_parameter(layer, "weight", qweight)
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(channel_scale, requires_grad=False),
        )
        # Dynamic symmetric activations: the scheme reads these by name and
        # takes the no-zero-point path when they are absent.
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.context.assert_validation_complete()
        return self.scheme.apply_weights(layer, x, bias)


@IluvatarQuantRegistry.register_method
class IluFp8ToW8A8Int8LinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    """Replace a serialized block-FP8 Linear with the static GLM W8A8 path."""

    source_format = "fp8-block"
    source_dtypes = ("float8_e4m3fn",)
    original_disposition = "quantized"
    original_method_cls = IluFp8LinearMethod

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self.block_shape = tuple(context.group.weights.source.extra["block_shape"])

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        source = context.group.weights.source
        if set(source.extra) != {"block_shape"} or tuple(
            source.extra["block_shape"]
        ) != (128, 128):
            raise ValueError(
                f"{context.group.group_id}: block-FP8 source requires "
                "block_shape [128, 128]"
            )
        if tuple(context.base_quant_config.weight_block_size or ()) != (128, 128):
            raise ValueError(
                f"{context.group.group_id}: block_shape must match the "
                "checkpoint's weight_block_size"
            )
        return super().from_ilu_runtime_config(context)

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            raise RuntimeError("block-FP8 source scale is missing")
        qweight, channel_scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data, scale.data, self.block_shape
        )
        del layer._parameters["weight_scale_inv"]
        return qweight, channel_scale


@IluvatarQuantRegistry.register_method
class IluNativeToW8A8Int8LinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    """Quantize an unquantized Linear onto the static GLM W8A8 path.

    Used for the MTP ``eh_proj``, which the FP8 checkpoint leaves BF16 while
    the offline W4A8 NN checkpoint quantizes it with the other dense Linears.
    """

    source_format = "native"
    source_dtypes = ("bfloat16",)
    original_disposition = "ignored"
    original_method_cls = UnquantizedLinearMethod

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        if set(context.group.weights.source.extra):
            raise ValueError(
                f"{context.group.group_id}: a native source takes no extra "
                "weights.source fields"
            )
        return super().from_ilu_runtime_config(context)

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_native_to_w8a8_int8(layer.weight.data)
