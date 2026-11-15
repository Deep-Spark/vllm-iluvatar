# SPDX-License-Identifier: Apache-2.0
"""The fused Markov sampler must preserve rejection-sampling inputs."""

import os

import pytest
import torch

from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark_bias_sample import (
    FusedBiasWorkspace,
    fused_bias_gumbel_sample,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("use_workspace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_fused_bias_preserves_sampled_ids_and_rejection_logits(
    monkeypatch, dtype, temperature, use_fp64, use_workspace
):
    from vllm.v1.worker.gpu.sample import gumbel

    from vllm_iluvatar.v1.worker.gpu.spec_decode import sampling_kernels

    library = os.environ.get("VLLM_ILU_DSPARK_FUSED_BIAS_LIBRARY")
    if library and not hasattr(torch.ops._C, "ilu_dspark_bias_sample_v2"):
        torch.ops.load_library(library)
    state_library = os.environ.get("VLLM_ILU_DSPARK_FUSED_BIAS_STATE_LIBRARY")
    if state_library and not hasattr(torch.ops._C, "ilu_dspark_bias_sample_v4"):
        torch.ops.load_library(state_library)
    monkeypatch.setattr(
        gumbel, "_gumbel_sample_kernel", sampling_kernels.gumbel_sample_kernel
    )
    torch.manual_seed(20261113)
    x = torch.randn((3, 5, 8193), device="cuda", dtype=dtype)[:, 2]
    bias = torch.randn((3, 2, 8193), device="cuda", dtype=dtype)[:, 1]
    bias[:, -1] = float("nan")
    mapping = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
    temperatures = torch.full((4,), temperature, device="cuda")
    seeds = torch.arange(4, device="cuda", dtype=torch.int64) + 20261113
    positions = torch.arange(6, device="cuda", dtype=torch.int64)[::2]
    column = torch.tensor(2, device="cuda", dtype=torch.int32)
    old_cache = torch.full((4, 5, 8193), -123, device="cuda", dtype=dtype)
    new_cache = old_cache.clone()
    combined = x + bias
    combined[:, -17:] = -float("inf")
    expected = gumbel.gumbel_sample(
        combined,
        mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=old_cache,
        logits_cache_col=column,
        use_fp64=use_fp64,
    )
    actual = fused_bias_gumbel_sample(
        x,
        bias,
        mapping,
        temperatures,
        seeds,
        positions,
        new_cache,
        column,
        use_fp64=use_fp64,
        num_vocab_padding=17,
    )
    if use_workspace:
        assert hasattr(torch.ops._C, "ilu_dspark_bias_sample_v4"), (
            "Build the native strided-state sampler"
        )
        workspace = FusedBiasWorkspace(4, 8193, x.device, use_fp64)
        output_storage = torch.full((3, 5), -123, device="cuda", dtype=torch.int64)
        predicted_pos = (torch.arange(6, device="cuda", dtype=torch.int64) + 1)[::2]
        new_cache.fill_(-123)
        actual = workspace.sample_into(
            x,
            bias,
            mapping,
            temperatures,
            seeds,
            predicted_pos,
            new_cache,
            column,
            output_storage[:, 2],
            17,
        )
        assert (output_storage[:, [0, 1, 3, 4]] == -123).all()
    assert torch.equal(actual, expected)
    torch.testing.assert_close(new_cache, old_cache, atol=0, rtol=0, equal_nan=True)
