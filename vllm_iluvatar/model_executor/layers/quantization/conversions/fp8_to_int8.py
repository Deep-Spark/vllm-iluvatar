# SPDX-License-Identifier: Apache-2.0
"""FP32 conversion of FP8 source blocks to per-channel INT8 weights."""

import torch


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
    if tuple(block_scale.shape) != expected or block_scale.dtype not in (
        torch.float32,
        torch.float8_e8m0fnu,
    ):
        raise ValueError(
            f"block scale shape/dtype {tuple(block_scale.shape)}/{block_scale.dtype} "
            f"does not match expected {expected} with FP32 or E8M0 scales"
        )

    block_scale = block_scale.float()
    if (
        not torch.isfinite(block_scale).all()
        or not torch.isfinite(weight.float()).all()
    ):
        raise ValueError("block-FP8 source contains non-finite weights or scales")
    qweight = torch.empty_like(weight, dtype=torch.int8)
    channel_scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for n0 in range(0, n, bn):
        n1 = min(n0 + bn, n)
        row_scale = block_scale[n0 // bn].repeat_interleave(bk)[:k]
        value = weight[n0:n1].float() * row_scale.unsqueeze(0)
        qweight[n0:n1], channel_scale[n0:n1] = _quantize_per_channel_int8(value)
    return qweight, channel_scale
