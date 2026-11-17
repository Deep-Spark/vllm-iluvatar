# SPDX-License-Identifier: Apache-2.0
"""Native-weight to per-channel FP8 online linear conversion for ivcore30."""

from __future__ import annotations

import copy
from functools import wraps
from typing import TYPE_CHECKING, Any

import ixformer.inference.functions as ixfops
import torch
from ixformer.inference.functions import fp8_per_token_dynamic_scaled
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp8.linear import (
    _IXMM_FP8_GEMV_MAX_M,
    _can_fuse_bf16_fp8_gemv,
)
from vllm_iluvatar.model_executor.layers.quantization.common.fp8_input import (
    IluFP8Input,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.config import (
    dtype_name,
)

if TYPE_CHECKING:
    from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import (
        IluFp8Config,
    )

logger = init_logger(__name__)


@IluvatarQuantRegistry.register_method
class IluFp8PerChannelOnlineLinearMethod(OnlineLinearBase):
    """Load floating-point weights and quantize them for ixmm after loading.

    The public name follows vLLM's online per-channel method. ixmm's FP8 linear
    epilogue consumes one weight scale per output channel, so the materialized
    ``[N, K]`` weight is quantized row-wise.
    """

    uses_meta_device: bool = True

    def __init__(
        self,
        quant_config: IluFp8Config,
        *,
        runtime_context: Any | None = None,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.gemm_format = quant_config.gemm_format.upper()
        self._unquantized_method = UnquantizedLinearMethod()
        self._runtime_context = runtime_context
        if runtime_context is None:
            self._source_dtypes = None
            self._fallback_method = self._unquantized_method
        else:
            self._source_dtypes = runtime_context.group.weights.source.dtypes
            self._fallback_method = runtime_context.original_method
            if self._fallback_method is None:
                raise ValueError(
                    f"{runtime_context.prefix}: runtime FP8 fallback requires "
                    "the original quantization method"
                )

    def _get_source_dtypes(self) -> tuple[str, ...] | None:
        return getattr(self, "_source_dtypes", None)

    def _get_fallback_method(self):
        return getattr(self, "_fallback_method", self._unquantized_method)

    def _get_runtime_context(self):
        return getattr(self, "_runtime_context", None)

    @classmethod
    def from_ilu_runtime_config(
        cls, context: Any
    ) -> IluFp8PerChannelOnlineLinearMethod:
        group = context.group
        weights = group.weights
        if not isinstance(context.layer, LinearBase):
            raise TypeError(f"{group.group_id}: online FP8 only supports LinearBase")
        if not isinstance(context.original_method, UnquantizedLinearMethod):
            raise TypeError(
                f"{group.group_id}: online FP8 fallback currently requires an "
                "UnquantizedLinearMethod"
            )
        if group.require.original_disposition != "ignored":
            raise ValueError(
                f"{group.group_id}: online FP8 requires original_disposition=ignored"
            )
        if weights.loading.mode != "transform" or (
            weights.loading.stage != "process_weights_after_loading"
        ):
            raise ValueError(
                f"{group.group_id}: online FP8 requires "
                "loading=transform@process_weights_after_loading"
            )
        if (
            weights.source.format != "native"
            or weights.target.format != "fp8"
            or weights.target.dtype != "float8_e4m3fn"
            or weights.target.strategy != "channel"
        ):
            raise ValueError(
                f"{group.group_id}: unsupported online FP8 weight contract"
            )
        activation = group.input_activations
        if (
            activation is None
            or activation.num_bits != 8
            or activation.type != "float"
            or activation.strategy != "token"
            or not activation.dynamic
            or group.output_activations is not None
        ):
            raise ValueError(
                f"{group.group_id}: online FP8 requires dynamic per-token FP8 "
                "input activations and null output_activations"
            )
        allowed_options = {"gemm_format", "shape_mismatch_policy"}
        unknown_options = set(group.options) - allowed_options
        if unknown_options:
            raise ValueError(
                f"{group.group_id}: unsupported options: {sorted(unknown_options)}"
            )
        shape_policy = group.options.get("shape_mismatch_policy", "fallback_original")
        if shape_policy != "fallback_original":
            raise ValueError(
                f"{group.group_id}: shape_mismatch_policy must be fallback_original"
            )
        gemm_format = group.options.get("gemm_format", "TN")
        if gemm_format != "TN":
            raise ValueError(f"{group.group_id}: only gemm_format=TN is supported")

        quant_config = copy.copy(context.base_quant_config)
        quant_config.weight_block_size = None
        quant_config.gemm_format = gemm_format
        return cls(quant_config, runtime_context=context)

    def _shape_fallback_reason(self, n: int, k: int) -> str | None:
        reasons = []
        if k % 64:
            reasons.append(f"K={k} is not divisible by 64")
        if self.gemm_format == "TN" and n % 32:
            reasons.append(f"N={n} is not divisible by 32 for GEMV")
        return "; ".join(reasons) or None

    @staticmethod
    def _make_staging_parameter(
        param: ModelWeightParameter,
        dtype: torch.dtype,
        weight_loader,
    ) -> ModelWeightParameter:
        staging = ModelWeightParameter(
            data=torch.empty_like(param, dtype=dtype),
            input_dim=param.input_dim,
            output_dim=param.output_dim,
            weight_loader=weight_loader,
        )
        # Preserve loader-specific metadata and the layer-local TP state
        # (notably disable_tp/replicated parameters). The loader itself must
        # remain the real checkpoint loader, not this method's wrapper.
        attributes = param.__dict__.copy()
        attributes["_weight_loader"] = weight_loader
        staging.__dict__.update(attributes)
        return staging

    def _make_online_weight_loader(self, layer: torch.nn.Module, weight_loader):
        @wraps(weight_loader, assigned=("__doc__", "__annotations__"))
        def ilu_fp8_online_loader(
            param: ModelWeightParameter,
            loaded_weight: torch.Tensor,
            *args,
            **kwargs,
        ):
            # The public layerwise loader executes once against meta merely to
            # count copy_ numel, then replays against a materialized parameter.
            if param.is_meta:
                return weight_loader(param, loaded_weight, *args, **kwargs)
            loaded_dtype = dtype_name(loaded_weight.dtype)
            source_dtypes = self._get_source_dtypes()
            if source_dtypes is not None and loaded_dtype not in source_dtypes:
                raise ValueError(
                    f"{getattr(layer, 'prefix', '<unknown>')}: runtime FP8 "
                    f"expected checkpoint dtype in {source_dtypes}, got "
                    f"{loaded_dtype}"
                )

            staging_by_name = layer.__dict__.setdefault("_ilu_fp8_staging", {})
            staging = staging_by_name.get("weight")
            if staging is None:
                staging = self._make_staging_parameter(
                    param, layer.orig_dtype, weight_loader
                )
                staging_by_name["weight"] = staging
            return weight_loader(staging, loaded_weight, *args, **kwargs)

        return ilu_fp8_online_loader

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
        output_size_per_partition = sum(output_partition_sizes)
        param_dtype = dtype_name(params_dtype)
        source_dtypes = self._get_source_dtypes()
        if source_dtypes is not None and param_dtype not in source_dtypes:
            raise ValueError(
                f"{getattr(layer, 'prefix', '<unknown>')}: runtime FP8 expects "
                f"parameter dtype in {source_dtypes}, got {param_dtype}"
            )
        fallback_reason = self._shape_fallback_reason(
            output_size_per_partition, input_size_per_partition
        )
        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is None:
            raise ValueError("online FP8 linear requires a weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        layer._ilu_fp8_online_fallback_reason = fallback_reason

        if fallback_reason:
            # Keep the online layerwise loader's meta-device contract while
            # delegating parameter construction and metadata to the original
            # method.
            with torch.device("meta"):
                self._get_fallback_method().create_weights(
                    layer,
                    input_size_per_partition,
                    output_partition_sizes,
                    input_size,
                    output_size,
                    params_dtype,
                    **extra_weight_attrs,
                )
            initialize_online_processing(layer)
            logger.warning(
                "Online FP8 fallback to %s for %s: shape=[%d, %d], %s",
                params_dtype,
                getattr(layer, "prefix", "<unknown>"),
                output_size_per_partition,
                input_size_per_partition,
                fallback_reason,
            )
            runtime_context = self._get_runtime_context()
            if runtime_context is not None:
                runtime_context.record_shape_result(transformed=False)
            return

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self._make_online_weight_loader(layer, weight_loader),
        )
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)
        if not fallback_reason:
            runtime_context = self._get_runtime_context()
            if runtime_context is not None:
                runtime_context.record_shape_result(transformed=True)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        fallback_reason = getattr(layer, "_ilu_fp8_online_fallback_reason", None)
        if fallback_reason:
            if layer.weight.dtype != layer.orig_dtype:
                raise RuntimeError(
                    f"{getattr(layer, 'prefix', '<unknown>')}: BF16 fallback "
                    f"weight has unexpected dtype {layer.weight.dtype}"
                )
            self._get_fallback_method().process_weights_after_loading(layer)
            layer._already_called_process_weights_after_loading = True
            return

        staging_by_name = layer.__dict__.get("_ilu_fp8_staging")
        staging = (
            staging_by_name.get("weight") if isinstance(staging_by_name, dict) else None
        )
        if staging is None:
            raise RuntimeError(
                f"{getattr(layer, 'prefix', '<unknown>')}: online FP8 staging "
                "weight is missing; the checkpoint loader may have bypassed "
                "weight.weight_loader"
            )
        if not staging.is_cuda or staging.stride(-1) != 1:
            raise RuntimeError(
                f"{getattr(layer, 'prefix', '<unknown>')}: online FP8 staging "
                f"must be CUDA and K-contiguous, got {staging.device=} "
                f"{staging.stride()=}"
            )

        expected_scale_shape = (layer.output_size_per_partition,)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            weight_scale = torch.nn.Parameter(
                torch.empty(
                    expected_scale_shape,
                    dtype=torch.float32,
                    device=layer.weight.device,
                ),
                requires_grad=False,
            )
            layer.register_parameter("weight_scale", weight_scale)
        elif (
            tuple(weight_scale.shape) != expected_scale_shape
            or weight_scale.dtype != torch.float32
            or weight_scale.device != layer.weight.device
        ):
            raise RuntimeError(
                f"{getattr(layer, 'prefix', '<unknown>')}: invalid reusable "
                f"weight_scale shape/dtype/device: {tuple(weight_scale.shape)}, "
                f"{weight_scale.dtype}, {weight_scale.device}"
            )

        weight_ptr = layer.weight.data_ptr()
        fp8_per_token_dynamic_scaled(
            staging,
            fp8_output=layer.weight.data,
            output_scales=layer.weight_scale.data,
        )
        if layer.weight.data_ptr() != weight_ptr:
            raise RuntimeError("online FP8 quantization replaced the weight buffer")

        # This scale is generated from BF16 staging, never loaded from checkpoint.
        layer.weight_scale.ilu_derived_at_load = True
        staging_by_name.pop("weight")
        if not staging_by_name:
            layer.__dict__.pop("_ilu_fp8_staging", None)
        layer._can_use_gemm = True
        layer._already_called_process_weights_after_loading = True

    def supports_prequantized_input(self, layer: torch.nn.Module) -> bool:
        """Whether this initialized layer consumes per-token E4M3 activations."""
        return (
            not getattr(layer, "_ilu_fp8_online_fallback_reason", None)
            and layer.weight.dtype == torch.float8_e4m3fn
            and hasattr(layer, "weight_scale")
            and (not hasattr(layer, "input_is_parallel") or layer.input_is_parallel)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | IluFP8Input,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fallback_reason = getattr(layer, "_ilu_fp8_online_fallback_reason", None)
        runtime_context = self._get_runtime_context()
        if runtime_context is not None:
            runtime_context.assert_validation_complete()
        if isinstance(x, tuple) and not isinstance(x, IluFP8Input):
            raise TypeError("Prequantized linear input must use IluFP8Input")
        if isinstance(x, IluFP8Input) and not self.supports_prequantized_input(layer):
            raise ValueError("This linear layer cannot consume prequantized FP8 input")
        if fallback_reason:
            return self._get_fallback_method().apply(layer, x, bias)
        if layer.weight.dtype != torch.float8_e4m3fn or not hasattr(
            layer, "weight_scale"
        ):
            raise RuntimeError(
                f"{getattr(layer, 'prefix', '<unknown>')}: online FP8 forward "
                "requires a float8_e4m3fn weight and per-channel weight_scale"
            )

        prequantized = isinstance(x, IluFP8Input)
        if prequantized:
            x, scale_x, output_dtype = x
            if (
                x.ndim < 2
                or x.dtype != torch.float8_e4m3fn
                or not x.is_contiguous()
                or not x.is_cuda
                or x.device != layer.weight.device
                or x.shape[-1] != layer.input_size_per_partition
                or scale_x.dtype != torch.float32
                or scale_x.device != x.device
                or not scale_x.is_contiguous()
                or scale_x.shape != (*x.shape[:-1], 1)
                or output_dtype != torch.bfloat16
            ):
                raise ValueError("invalid prequantized FP8 activation contract")
        else:
            output_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])

        output = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            dtype=torch.bfloat16,
            device=x.device,
        )
        bias = None if bias is None else bias.to(torch.bfloat16)

        if not prequantized and _can_fuse_bf16_fp8_gemv(
            x_2d, layer, self.gemm_format, bias
        ):
            ixfops.per_channel_gemv_bf16_fp8_bf16(
                x_2d, layer.weight, layer.weight_scale, output, bias
            )
            return output.view(*x.shape[:-1], layer.output_size_per_partition)

        if prequantized:
            qx = x_2d
            scale_x = scale_x.reshape(-1)
        else:
            qx, scale_x = fp8_per_token_dynamic_scaled(x_2d)
        scale_w = layer.weight_scale
        m = qx.shape[0]
        can_use_gemm = getattr(layer, "_can_use_gemm", False)

        if can_use_gemm and m <= _IXMM_FP8_GEMV_MAX_M:
            ixfops.fp8_per_channel_gemv(
                qx,
                layer.weight,
                scale_x,
                scale_w,
                output,
                bias,
                format=self.gemm_format,
                backend="ixmm",
            )
        else:
            ixfops.fp8_per_channel_mm(
                qx, layer.weight, scale_x, scale_w, output, bias, backend="ixmm"
            )

        result = output.to(output_dtype) if output_dtype != torch.bfloat16 else output
        return result.view(*x.shape[:-1], layer.output_size_per_partition)
