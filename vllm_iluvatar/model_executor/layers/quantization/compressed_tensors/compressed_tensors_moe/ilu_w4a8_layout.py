# SPDX-License-Identifier: Apache-2.0
"""Lossless W4A8 v2 packed-weight layout conversion for loaded MoE experts.

The TN checkpoint stores pairs of INT4 values along K in ``[E, N, K/2]``.
Ixformer's NN kernel stores pairs along N in a swizzled ``[E, K, N/2]``.
Conversion copies the original signed four-bit payloads exactly; it does
not dequantize and requantize the checkpoint weights.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def repack_w4a8_tn_to_nn(
    packed_tn: torch.Tensor,
    *,
    logical_n: int,
    logical_k: int,
    nn_n: int,
    nn_k: int,
) -> torch.Tensor:
    """Return packed NN weights, one expert at a time to cap temporary memory.

    ``packed_tn`` may already contain TN zero-padding. ``logical_n`` and
    ``logical_k`` identify checkpoint payload dimensions; output padding is
    recreated for NN rather than carrying over TN's different alignment.
    The swizzle and 32-element nibble pairing match ixformer's
    ``quant_repack_int4(..., version=2, format="NN")`` exactly.
    """
    if packed_tn.dtype != torch.int8 or packed_tn.ndim != 3:
        raise ValueError("packed_tn must be a [E, N, K/2] torch.int8 tensor")
    if not packed_tn.is_contiguous():
        raise ValueError("packed_tn must be contiguous")
    experts, tn_n, tn_k_half = packed_tn.shape
    tn_k = tn_k_half * 2
    if experts <= 0 or not (0 < logical_n <= tn_n and 0 < logical_k <= tn_k):
        raise ValueError("invalid logical dimensions for packed TN weight")
    if tn_k % 32 or nn_n % 128 or nn_k % 64 or nn_k < 256:
        raise ValueError("packed TN or target NN dimensions violate kernel alignment")
    if logical_n > nn_n or logical_k > nn_k:
        raise ValueError("target NN weight is smaller than logical weight")

    packed_nn = torch.empty(
        (experts, nn_k, nn_n // 2),
        dtype=torch.int8,
        device=packed_tn.device,
    )
    for expert in range(experts):
        # In TN each 16-byte group has K[0:16] in low nibbles and
        # K[16:32] in high nibbles. Keep the raw 4-bit payloads unsigned.
        chunks = packed_tn[expert].view(tn_n, tn_k // 32, 16)
        unpacked = torch.cat(
            (chunks & 0x0F, (chunks >> 4) & 0x0F), dim=-1
        ).reshape(tn_n, tn_k)

        # NN begins with K-major storage, then swizzles each 32x32 tile.
        k_major = torch.zeros((nn_k, nn_n), dtype=torch.int8, device=packed_tn.device)
        k_major[:logical_k, :logical_n].copy_(
            unpacked[:logical_n, :logical_k].transpose(0, 1)
        )
        swizzled = (
            k_major.view(nn_k // 32, 2, 16, nn_n // 32, 2, 16)
            .permute(0, 4, 2, 3, 1, 5)
            .contiguous()
            .view(nn_k, nn_n)
        )
        pairs = swizzled.view(nn_k, nn_n // 32, 32)
        packed_nn[expert].copy_(
            ((pairs[..., :16] & 0x0F) | (pairs[..., 16:] << 4)).reshape(
                nn_k, nn_n // 2
            )
        )

    return packed_nn
