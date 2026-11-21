# SPDX-License-Identifier: Apache-2.0
"""MXFP8 dense linear sources executed as channel-wise W8A8 INT8.

Corresponds to: vllm/model_executor/layers/quantization/modelopt.py

Target   : the ``LinearBase`` branch of ``ModelOptQuantConfigBase.get_quant_method``
           for ``quant_algo="MXFP8"``, i.e. ``build_linear_method`` ->
           ``ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic)``.
Baseline : that method's ``create_weights`` ends in ``select_linear_kernel``,
           which picks an MXFP8 kernel from ``_POSSIBLE_MXFP8_KERNELS[CUDA]``.
           ``IluMxfp8LinearKernel`` is prepended there by
           ``patch_mxfp8_linear_kernel``, whose ``_apply()`` returns early when
           ``not is_ivcore30()``.
Why      : on ivcore11 that leaves only community kernels, none of which run on
           Iluvatar, so an MXFP8 checkpoint cannot load at all -- and ivcore11's
           dense default is W8A8 INT8 regardless (``fp8_linear_backend()``).
How      : keep the checkpoint contract by delegating parameter registration to
           the community ``KMxfp8Static`` scheme, select no kernel, and
           requantize to per-output-channel INT8 at post-load with the existing
           ``quantize_fp8_block_to_w8a8_int8`` -- MXFP8 is a block-FP8 source
           with a ``[1, 32]`` grid and UE8M0 scales, both of which that
           converter already accepts. Execution is the shared
           ``IluCompressedTensorsW8A8Int8`` scheme, exactly as the block-FP8
           source reaches INT8 through ``IluFp8LinearMethod``.
Note     : per-channel scales are computed from this rank's shard, so for the
           row-parallel layers they are finer than a full-K offline scale would
           be. That stays valid -- each rank dequantizes its own partial product
           before the reduction -- and unlike W4A8 NN there is no second scale
           stage depending on the INT8 amax being exactly 127.
Scope    : MXFP8 dense linears on ivcore11, under both registry names
           (``mxfp8`` for MiniMax-style tagging and ``modelopt_mxfp8`` for
           NVIDIA ModelOpt tagging). ivcore30 is untouched and keeps FP8
           execution on ``IluMxfp8LinearKernel``.
Remove   : when an ixformer MXFP8 GEMM exists for ivcore11.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.modelopt import (
    SCHEME_FOR,
    WEIGHT,
    CkptCtx,
    Shapes,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
from vllm.model_executor.utils import is_weights_pre_processed

from vllm_iluvatar.model_executor.layers.quantization.common.fp8_weights import (
    publish_int8_weights,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_int8 import (
    quantize_fp8_block_to_w8a8_int8,
)

logger = init_logger(__name__)

# The checkpoint stores the UE8M0 exponent byte as uint8 (``MXFP8_SCALE_DTYPE``).
# Reinterpreting it as e8m0 is what makes ``.float()`` decode 0x7F as 1.0
# instead of 127.0 -- the silent 1e38 blow-up a plain cast would produce.
_E8M0 = torch.float8_e8m0fnu


@register_weight_loader_v2_supported_method
class IluMxFp8ToW8A8Int8LinearMethod(LinearMethodBase):
    """MXFP8 checkpoint source, per-output-channel INT8 execution.

    Parameter registration is delegated to the community ``KMxfp8Static``
    scheme so the on-disk contract (fp8-e4m3 ``weight`` plus UE8M0
    ``weight_scale`` on a ``[1, 32]`` grid) and its TP sharding are shared with
    the FP8 path rather than restated here. Nothing selects an MXFP8 kernel:
    the weight is INT8 by the time the first forward runs.
    """

    def __init__(self, quant_config: Any) -> None:
        from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
            IluCompressedTensorsW8A8Int8,
        )

        self.quant_config = quant_config
        self.block_shape = (1, MXFP8_BLOCK_SIZE)
        self.scheme = IluCompressedTensorsW8A8Int8(
            strategy="channel",
            is_static_input_scheme=False,
            input_symmetric=True,
            ilu_params={
                "gemm_format": "TN",
                "mixed_strategy": "CHANNEL",
                "gemm_mode": "PERF",
            },
        )
        logger.info_once("MXFP8 Linear execution backend: w8a8_int8")

    @property
    def supports_pre_processed_weights(self) -> bool:
        # The requant reads the checkpoint's own fp8 values and block scales.
        return False

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
        del input_size, output_size
        if is_weights_pre_processed():
            raise ValueError(
                "MXFP8 to INT8 conversion requires original checkpoint weights"
            )
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        SCHEME_FOR[kMxfp8Static].create_weights(
            layer,
            WEIGHT,
            CkptCtx(),
            Shapes(output_partition_sizes, input_size_per_partition, params_dtype),
            extra_weight_attrs.get("weight_loader"),
        )
        layer.scheme = self.scheme

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data,
            layer.weight_scale.data.view(_E8M0),
            self.block_shape,
        )
        del layer.weight_scale
        publish_int8_weights(layer, weight, scale)
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scheme.apply_weights(layer, x, bias)


def mxfp8_int8_linear_method(
    quant_config: Any, layer: torch.nn.Module, prefix: str
) -> IluMxFp8ToW8A8Int8LinearMethod | None:
    """ivcore11 INT8 dense execution for an MXFP8 source, or ``None`` to defer.

    Shared by the two MXFP8 configs (``mxfp8`` and ``modelopt_mxfp8``), which
    differ only in the registry name they claim. Only ivcore11 is claimed;
    every other architecture returns ``None`` and stays on community dispatch,
    which on ivcore30 is ``ModelOptLinearMethod`` plus ``IluMxfp8LinearKernel``.
    ``fp8_linear_backend()`` is still consulted so an explicit
    ``ilu_compute.linear.fp8_backend`` is validated rather than silently
    ignored; on ivcore11 it can only resolve to ``w8a8_int8``.

    Excluded layers (and the ``ParallelLMHead`` that is not a ``LinearBase``)
    are left to ``super()`` so the unquantized branches -- including HY V4's
    fp32 head -- keep their current method.
    """
    from vllm_iluvatar.config.ilu_compute import fp8_linear_backend
    from vllm_iluvatar.ilu_util.arch import IVCORE11, cuda_arch

    if not isinstance(layer, LinearBase):
        return None
    if cuda_arch() != IVCORE11 or fp8_linear_backend() != "w8a8_int8":
        return None
    if quant_config.is_layer_excluded(prefix):
        return None
    # Community applies the same substring exemption before quantizing.
    if any(
        tag in prefix for tag in ("vision_tower", "vision_model", "vit_large_projector")
    ):
        return None
    return IluMxFp8ToW8A8Int8LinearMethod(quant_config)
