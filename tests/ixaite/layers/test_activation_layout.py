# SPDX-License-Identifier: Apache-2.0
"""Keep ixaite decode padding checks separate from baseline activation tests."""

from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.custom_op import CustomOp

from vllm_iluvatar.backends.ixaite.layers.activation import _ACT_ROW_ALIGN
from vllm_iluvatar.backends.ixaite.layers.activation import (
    IluSiluAndMul as IxaiteSiluAndMul,
)


class _CudaLikePlatform:
    def is_cuda_alike(self) -> bool:
        return True

    def is_xpu(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("rows", "expect_padded"),
    [(8, True), (_ACT_ROW_ALIGN - 1, True), (_ACT_ROW_ALIGN, False), (64, False)],
)
def test_silu_and_mul_over_allocates_rows_for_short_batches(rows, expect_padded):
    """Short decode batches get storage the down_proj GEMM can read in place."""
    d = 6
    x = torch.randn(rows, 2 * d, dtype=torch.float32)

    with (
        patch(
            "vllm_iluvatar.model_executor.layers.activation.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(CustomOp, "__init__", return_value=None),
    ):
        act = IxaiteSiluAndMul()
    act.op = lambda out, inp: out.copy_(
        torch.nn.functional.silu(inp[..., :d]) * inp[..., d:]
    )

    out = act.forward_cuda(x)

    assert out.shape == (rows, d)
    assert out.is_contiguous()
    torch.testing.assert_close(out, torch.nn.functional.silu(x[:, :d]) * x[:, d:])
    backing_rows = out.untyped_storage().nbytes() // (d * out.element_size())
    assert backing_rows == (_ACT_ROW_ALIGN if expect_padded else rows)


def test_silu_and_mul_keeps_tight_output_for_non_2d_input():
    d = 6
    x = torch.randn(2, 3, 2 * d, dtype=torch.float32)

    with (
        patch(
            "vllm_iluvatar.model_executor.layers.activation.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(CustomOp, "__init__", return_value=None),
    ):
        act = IxaiteSiluAndMul()
    act.op = lambda out, inp: out.copy_(
        torch.nn.functional.silu(inp[..., :d]) * inp[..., d:]
    )

    out = act.forward_cuda(x)

    assert out.shape == (2, 3, d)
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()
