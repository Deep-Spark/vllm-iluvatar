# SPDX-License-Identifier: Apache-2.0
"""INT8 AG packet metadata; packing kernels are owned by ixformer."""

from dataclasses import dataclass

import ixformer.inference.functions as ixfops
import torch


@dataclass(frozen=True)
class RouterPacketLayout:
    hidden: int
    experts: int
    ids_dtype: torch.dtype | None


def pack_router_input(q, scales, logits, input_ids=None):
    packet = ixfops.pack_router_packet(
        q.contiguous(),
        scales.contiguous(),
        logits.contiguous(),
        None if input_ids is None else input_ids.contiguous(),
    )
    return packet, RouterPacketLayout(
        q.shape[1], logits.shape[1], None if input_ids is None else input_ids.dtype
    )


def unpack_router_input(packet, layout, compute_dtype):
    logits, ids = ixfops.unpack_router_packet_metadata(
        packet, layout.hidden, layout.experts, layout.ids_dtype
    )
    if packet.shape[1] % 16 == 0:
        # Expand consumes the activation and FP32 scale in-place in each AG row.
        quantized = (packet.view(torch.int8), layout.hidden, compute_dtype)
    else:
        # Packets without IDs need not have the expand kernel's 16-byte stride.
        q = packet[:, : layout.hidden].view(torch.int8).contiguous()
        scale = packet[:, layout.hidden : layout.hidden + 4].clone().view(torch.float32)
        quantized = (q, scale, compute_dtype)
    return quantized, logits, ids
