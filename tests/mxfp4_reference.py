# SPDX-License-Identifier: Apache-2.0
"""Torch reference for MXFP4 loading tests; never used in production."""

import torch


def mxfp4_channel_int4(weight, scale):
    """Decode E2M1/UE8M0 and quantize with the existing INT8/18 convention."""
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError("Expected packed MXFP4 and UE8M0 bytes")
    n, half_k = weight.shape
    if half_k % 16 or scale.shape != (n, half_k // 16):
        raise ValueError("Invalid MXFP4 block geometry")
    if (scale == 255).any():
        raise ValueError("UE8M0 NaN scale")
    codes = torch.stack((weight & 15, weight >> 4), -1).reshape(n, -1, 32)
    lut = weight.new_tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    values = (lut[codes.long()] * torch.exp2(scale.float() - 127)[..., None]).flatten(1)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite MXFP4 expert")
    channel = values.abs().amax(-1, keepdim=True) / 127
    channel = torch.where(channel == 0, 1.0, channel)
    q8 = (values / channel).round().clamp(-127, 127)
    return (q8 / 18).round().clamp(-8, 7).to(torch.int8), channel.T.contiguous()


def pack_ixinfer(values, layout):
    """Signed [N,K] values to IxInfer v2 TN/NN storage."""
    n, k = values.shape
    if n % 128 or k % 64:
        raise ValueError("Original-weight W4A8 requires aligned expert dimensions")
    if layout == "NN":
        values = (
            values.T.contiguous()
            .reshape(k // 32, 2, 16, n // 32, 2, 16)
            .permute(0, 4, 2, 3, 1, 5)
            .contiguous()
            .reshape(k, n)
        )
    elif layout != "TN":
        raise ValueError(f"Invalid IxInfer layout: {layout}")
    rows, cols = values.shape
    tiles = values.reshape(rows, cols // 32, 32)
    return ((tiles[..., :16] & 15) | (tiles[..., 16:] << 4)).reshape(rows, cols // 2)
