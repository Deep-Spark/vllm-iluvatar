# SPDX-License-Identifier: Apache-2.0
"""Per-expert MXFP8 loading into the IxInfer W4A8 weight contract.

MXFP8 is block-FP8 on a ``[1, 32]`` grid: the values are the same
``float8_e4m3fn`` the block-FP8 loader already stages, and ``fp8_block_to_w4a8``
takes the block shape as an argument. The only real difference is the scale
encoding -- UE8M0 exponents rather than a multiplicative float -- so this module
is the decode plus the target contracts, and reuses the block-FP8 machinery for
geometry, TP slicing, paired staging and the conversion itself.

The code multiplier 18 is applied by the IxInfer kernel through the weight
spec, and ``fp8_block_to_w4a8`` leaves it out of the channel scale for the
``ixinfer_*`` formats. Nothing here may fold it in a second time.
"""

import torch

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)

from .block_fp8_loader import BlockFp8ToW4A8Loader

MXFP8_BLOCK_SHAPE = (1, 32)


class MxFp8ToW4A8Loader(BlockFp8ToW4A8Loader):
    """MXFP8 routed experts converted shard by shard to IxInfer W4A8."""

    source_name = "MXFP8"
    # MXFP8 checkpoints name the grid ``weight_scale``, not DeepSeek's
    # ``weight_scale_inv``; both are multiplicative once decoded.
    scale_parameter_name = "weight_scale"
    source_scale_dtypes = (torch.uint8,)
    source_scale_metadata = ("U8",)

    def __init__(self, weight_spec, load_spec):
        if tuple(load_spec.block_shape) != MXFP8_BLOCK_SHAPE:
            raise ValueError(
                f"MXFP8 requires block shape {MXFP8_BLOCK_SHAPE}, "
                f"got {tuple(load_spec.block_shape)}"
            )
        super().__init__(weight_spec, load_spec)

    def _decode_scale(self, scale):
        """UE8M0 -> FP32 by placing the biased exponent into bits 23-30.

        Matches ``_e8m0_to_fp32`` in ``ilu_fp8_block_scaled_mm``, which is not
        imported here to keep this module free of the dense kernel stack. The
        NaN encoding 255 becomes +inf rather than NaN; both are non-finite, so
        the converter still invalidates the channel and ``finalize`` rejects
        the layer.
        """
        return (scale.to(torch.int32) << 23).view(torch.float32)


def make_mxfp8_loader(weight_spec, load_spec):
    contracts = {
        "ixinfer_w4a8_tn_v2": MoEWeightSpec(
            "ixinfer_w4a8_tn_v2", "TN", torch.int8, code_multiplier=18
        ),
        "ixinfer_w4a8_nn_v2": MoEWeightSpec(
            "ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18
        ),
    }
    if contracts.get(weight_spec.format_id) != weight_spec:
        raise ValueError(f"No MXFP8 loader for weight contract {weight_spec}")
    return MxFp8ToW4A8Loader(weight_spec, load_spec)
