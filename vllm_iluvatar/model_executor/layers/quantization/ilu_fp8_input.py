"""Per-token FP8 activations passed between a producer and a linear method."""

from typing import NamedTuple

import torch


class IluFP8Input(NamedTuple):
    """Contiguous CUDA E4M3 [..., K], FP32 [..., 1] dequantization scales.

    The consumer owns bias, output conversion and tensor-parallel reduction.
    Row-parallel consumers must already own the local input shard.
    """

    values: torch.Tensor
    scales: torch.Tensor
    output_dtype: torch.dtype = torch.bfloat16
