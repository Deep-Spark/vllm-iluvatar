# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-expert weights restated as block-FP8, then converted to W4A8.

An NVFP4 tensor *is* a block-FP8 tensor on a ``[1, 16]`` grid, exactly:

* the e2m1 magnitudes -- 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives -- are
  all representable in ``float8_e4m3fn``;
* the per-16 e4m3 block scale is representable in FP32.

The per-expert global scale does not belong in that grid. It is a uniform
factor over the whole expert, so it cancels out of the converter's per-channel
``amax / 127`` normalization and survives only in the channel scale the
converter returns. Folding it into the block scale would instead round the
product to FP32 *before* quantizing, making the INT4 codes depend on that
rounding; multiplying the channel scale afterwards leaves them exact.

So the conversion reuses ``fp8_block_to_w4a8`` unchanged, which keeps the
INT8-then-INT4 normalization, the full-K channel amax and the NN packing in
the one kernel the block-FP8 and MXFP8 sources already share -- and keeps the
code multiplier 18 the kernel's business rather than the scale's. The only
precision cost of this path is the FP4 -> INT4 requantization itself, which is
inherent to the target format.
"""

import torch

NVFP4_GROUP_SIZE = 16
NVFP4_BLOCK_SHAPE = (1, NVFP4_GROUP_SIZE)
# e2m1 code -> magnitude, indexed by the three magnitude bits.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_source_as_block_fp8(weight, weight_scale):
    """Unpack one expert chunk into ``float8_e4m3fn`` values and an FP32 grid.

    Args:
        weight:       ``[E, n, k/2]`` uint8, two e2m1 nibbles packed along k.
        weight_scale: ``[E, n, k/16]`` float8_e4m3fn per-16 block scale.

    Returns:
        ``[E, n, k]`` float8_e4m3fn values and ``[E, n, k/16]`` float32 scales.
        Both conversions are exact; the per-expert global scale is not applied.
    """
    table = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=weight.device)
    # Low nibble is the even k index, high nibble the odd one.
    nibbles = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2)
    values = table[(nibbles & 0x07).long()]
    values = torch.where(nibbles & 0x08 != 0, -values, values)
    return values.to(torch.float8_e4m3fn), weight_scale.float()


def convert_nvfp4_to_w4a8(
    weight, weight_scale, global_scale, out_weight, out_scale, *, weight_format
):
    """Convert one expert chunk of NVFP4 into a W4A8 contract, expert by expert.

    ``out_weight`` / ``out_scale`` are views of the final parameters, shaped as
    the target ``MoEWeightSpec`` requires. The whole ``[2I, K]`` w13 block is
    converted in one call: the channel scale is per output row and the NN
    packing tiles N in groups of 32, so a gate/up split at a multiple of 32
    lands on exactly the same bytes as converting the halves separately.
    """
    from ixformer.inference import functions as ops

    values, scale = nvfp4_source_as_block_fp8(weight, weight_scale)
    for e in range(values.shape[0]):
        ops.fp8_block_to_w4a8(
            values[e],
            scale[e],
            out_weight[e],
            out_scale[e],
            block_shape=NVFP4_BLOCK_SHAPE,
            weight_format=weight_format,
            backend="ixformer_native",
        )
    # The uniform per-expert factor the converter never saw. An invalidated
    # channel stays NaN, which ``process_weights_after_loading`` rejects.
    out_scale.mul_(global_scale.float().reshape(-1, 1, 1))
