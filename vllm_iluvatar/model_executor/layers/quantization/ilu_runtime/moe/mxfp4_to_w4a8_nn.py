# SPDX-License-Identifier: Apache-2.0
"""Runtime MXFP4 conversion into the existing NN/v1 W4A8 layout."""

from vllm_iluvatar.model_executor.layers.quantization.backends.w4a8.nn_moe import (
    IluW4A8NNMoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8 import (
    _pack_int4,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

from .mxfp4_to_w4a8_tn import _IluMxfp4W4A8Loader, _quantize_mxfp4_to_int4


def quantize_mxfp4_to_w4a8_nn(weight, block_scale):
    """Pack directly into NN/v1 with complete per-channel INT4 scales."""
    q4, channel, factor = _quantize_mxfp4_to_int4(weight, block_scale)
    n, k = q4.shape
    if n % 32:
        raise ValueError("NN packing requires 32-aligned N")
    nn = q4.T.contiguous().reshape(k // 32, 2, 16, n // 32, 2, 16)
    nn = nn.permute(0, 4, 2, 3, 1, 5).contiguous().reshape(k, n)
    return _pack_int4(nn), (channel * factor).T.contiguous()


@IluvatarQuantRegistry.register_method
class IluMxfp4ToW4A8NNMoEMethod(_IluMxfp4W4A8Loader, IluW4A8NNMoEMethod):
    """MXFP4 streaming loader with direct NN/v1 allocation and compute."""

    _format = "NN"
    _pack_version = 1
    _quantize_shard = staticmethod(quantize_mxfp4_to_w4a8_nn)
