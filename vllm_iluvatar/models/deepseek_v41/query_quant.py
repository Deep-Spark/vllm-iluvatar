# SPDX-License-Identifier: Apache-2.0
"""Share one normalized INT8 query between compatible projection consumers.

This adapts the producer/consumer quantization contract from upstream #57679
to the MR dynamic symmetric INT8 linear scheme. Cache formats are unchanged.
"""

from collections.abc import Sequence
from typing import TypeAlias

import ixformer.inference.functions as ixfops
import torch
from torch import nn

from vllm_iluvatar.ilu_util.arch import IVCORE11, cuda_arch
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (
    IluCompressedTensorsW8A8Int8,
)

QueryActivation: TypeAlias = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.dtype]
)


def can_fuse_query_quant(
    linears: Sequence[nn.Module], query_width: int, key_width: int
) -> bool:
    """Require the native vector/reduction limits and every consumer's contract."""
    if (
        not linears
        or key_width != 512
        or not (512 <= query_width <= 2048)
        or query_width % 128
    ):
        return False
    for linear in linears:
        scheme = getattr(linear, "scheme", None)
        if (
            type(scheme) is not IluCompressedTensorsW8A8Int8
            or scheme.is_static_input_scheme
            or not scheme.input_symmetric
        ):
            return False
    return True


def query_inputs_are_compatible(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Keep empty, non-BF16 and unaligned input views on the ordinary path."""
    return (
        q.is_cuda
        and k.is_cuda
        and q.dtype == k.dtype == torch.bfloat16
        and q.ndim == k.ndim == 2
        and q.shape[0] > 0
        and q.stride(1) == k.stride(1) == 1
        and q.stride(0) % 2 == k.stride(0) % 2 == 0
        and q.data_ptr() % 4 == k.data_ptr() % 4 == 0
    )


def fused_query_norm(
    q, k, q_weight, k_weight, eps
) -> tuple[QueryActivation, torch.Tensor]:
    """Normalize Q/K once and retain a shared per-token INT8 query activation."""
    quantized_q = q_scales = None
    rows = q.shape[0]
    if cuda_arch() == IVCORE11 and rows % 64:
        # Projection consumers may issue a complete SME row tile. Keep that
        # tile in the producer's allocation without changing logical M or
        # adding a quantization/copy kernel for either query consumer.
        physical_rows = (rows + 63) // 64 * 64
        quantized_q = torch.empty(
            (physical_rows, q.shape[1]), device=q.device, dtype=torch.int8
        )[:rows]
        q_scales = torch.empty(
            (physical_rows, 1), device=q.device, dtype=torch.float32
        )[:rows]
    normalized_q, normalized_k, quantized, scales = ixfops.rms_norm_qk_dynamic_int8(
        q,
        k,
        q_weight,
        k_weight,
        eps,
        quantized_q=quantized_q,
        q_scales=q_scales,
    )
    return (quantized, scales, normalized_q.dtype), normalized_k
