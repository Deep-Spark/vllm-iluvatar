# SPDX-License-Identifier: Apache-2.0
"""Per-expert MXFP8 loading into whichever weight contract compute selected.

MXFP8 is block-FP8 on a ``[1, 32]`` grid: the values are the same
``float8_e4m3fn`` the block-FP8 loader already stages, and both converters take
the block shape as an argument. The only real difference is the scale encoding
-- UE8M0 exponents rather than a multiplicative float -- so this module is the
decode plus the target contracts, and reuses the block-FP8 machinery for
geometry, TP slicing, paired staging and the conversion itself.

Which contract applies is the compute backend's answer, not the checkpoint's:
IxInfer W4A8 by default on ivcore11, the ix30 FP4 layout by default on
ivcore30. The table below mirrors ``make_block_fp8_loader``'s, so an explicit
``moe`` mode reaches the same targets from either source.

The code multiplier 18 is applied by the IxInfer kernel through the weight
spec, and ``fp8_block_to_w4a8`` leaves it out of the channel scale for the
``ixinfer_*`` formats. Nothing here may fold it in a second time.
"""

import torch

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)

from .block_fp8_loader import BlockFp8ToFP4FP8Loader, BlockFp8ToW4A8Loader

MXFP8_BLOCK_SHAPE = (1, 32)


class _MxFp8Source:
    """What makes the checkpoint MXFP8 rather than DeepSeek block-FP8.

    Mixed ahead of a block-FP8 target loader, which supplies ``convert_shard``.
    Target selection is per architecture and source decoding is per checkpoint,
    so the two have to compose rather than nest.
    """

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


class MxFp8ToW4A8Loader(_MxFp8Source, BlockFp8ToW4A8Loader):
    """MXFP8 routed experts converted shard by shard to IxInfer W4A8."""


class MxFp8ToFP4FP8Loader(_MxFp8Source, BlockFp8ToFP4FP8Loader):
    """MXFP8 routed experts converted shard by shard to the ix30 FP4 layout."""


def make_mxfp8_loader(weight_spec, load_spec):
    contracts = {
        "ixinfer_w4a8_tn_v2": (
            MxFp8ToW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_tn_v2", "TN", torch.int8, code_multiplier=18),
        ),
        "ixinfer_w4a8_nn_v2": (
            MxFp8ToW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18),
        ),
        "ixmm_w4a8_nn": (
            MxFp8ToW4A8Loader,
            MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8),
        ),
        "ixmm_fp4fp8": (
            MxFp8ToFP4FP8Loader,
            MoEWeightSpec(
                "ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256, alignment=32
            ),
        ),
    }
    entry = contracts.get(weight_spec.format_id)
    if entry is None or entry[1] != weight_spec:
        raise ValueError(f"No MXFP8 loader for weight contract {weight_spec}")
    return entry[0](weight_spec, load_spec)
