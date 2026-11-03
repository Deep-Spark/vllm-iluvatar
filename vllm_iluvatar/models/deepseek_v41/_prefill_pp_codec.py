"""Optional CQ prefill PP transport: symmetric INT8 blocks of 512 elements."""
from math import prod

import torch


def encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Token stays the leading dimension for IntermediateTensors slicing.
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(
        (x.shape[0], prod(x.shape[1:]) // 512), device=x.device, dtype=torch.float32
    )
    torch.ops._ixformer_C.v41_pp_quant_int8(x, out, scales)
    return out, scales


def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.bfloat16)
    torch.ops._ixformer_C.v41_pp_dequant_int8(x, scales, out)
    return out
