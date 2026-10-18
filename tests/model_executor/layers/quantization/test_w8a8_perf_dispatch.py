# SPDX-License-Identifier: Apache-2.0
"""CPU-side routing tests; numerical GEMM tests belong to iXinfer."""

import pytest
from ixformer.inference.functions.w8a8 import w8a8_gemm, w8a8_gemv

from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (
    IluCompressedTensorsW8A8Int8,
)


@pytest.fixture
def scheme(monkeypatch):
    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    return IluCompressedTensorsW8A8Int8(
        strategy="channel",
        is_static_input_scheme=False,
        input_symmetric=True,
        ilu_params={
            "mixed_strategy": "CHANNEL",
            "gemm_format": "NN",
            "gemm_mode": "PERF",
        },
    )


@pytest.mark.parametrize("n,k,first_gemm_m", [
    (1152, 5120, 17), (2304, 5120, 1), (4608, 5120, 1),
    (5120, 576, 17), (5120, 1152, 1), (5120, 2304, 1),
    (1792, 5120, 1), (4096, 1280, 1),
    (5120, 2048, 17), (5120, 4096, 1), (5120, 8192, 1),
    (16384, 1280, 17), (32768, 1280, 17), (5120, 15360, 1),
    (4096, 4096, 2), (4096, 2048, 2),
    (24576, 4096, 17), (4096, 12288, 1),
])
@pytest.mark.parametrize("m", [1, 2, 16, 17, 24, 32, 33, 64, 128])
def test_mr_w8a8_kernel_selection(scheme, n, k, first_gemm_m, m):
    expected = w8a8_gemm if m >= first_gemm_m else w8a8_gemv
    assert scheme._get_perf_func(n, k, m) is expected
    assert scheme._choose_format(n, k) == "NN"


def test_unlisted_shape_uses_default_dispatch(scheme):
    assert scheme._get_perf_func(1234, 5678, 24) is None


def test_normal_mode_uses_default_dispatch(scheme):
    scheme.perf_mode = "NORMAL"
    assert scheme._get_perf_func(2304, 5120, 1) is None


def test_cq_rules_unchanged(scheme):
    scheme.arch = "ivcore30"
    assert scheme._choose_format(2048, 4096) == "TN"
    assert scheme._get_perf_func(2048, 4096, 16) is w8a8_gemv
    assert scheme._get_perf_func(2048, 4096, 17) is w8a8_gemm

