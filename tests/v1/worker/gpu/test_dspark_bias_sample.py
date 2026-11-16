# SPDX-License-Identifier: Apache-2.0
"""The fused Markov sampler must preserve rejection-sampling inputs."""

import importlib.util
from pathlib import Path

import numpy as np
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
    if use_fp64 and torch.ops._C.compiled_cuda_arch() == "ivcore11":
        with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
            fused_bias_gumbel_sample(
                x,
                bias,
                mapping,
                temperatures,
                seeds,
                positions,
                new_cache,
                column,
                use_fp64=True,
                num_vocab_padding=17,
            )
        if use_workspace:
            workspace = FusedBiasWorkspace(4, 8193, x.device, True)
            with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
                workspace.sample_into(
                    x,
                    bias,
                    mapping,
                    temperatures,
                    seeds,
                    positions,
                    new_cache,
                    column,
                    torch.empty(3, dtype=torch.int64, device="cuda"),
                    17,
                )
        assert torch.equal(new_cache, old_cache)
        return
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


@pytest.fixture(scope="module")
def cpu_reference():
    # Reuse the independent integer RNG oracle, not another GPU sampler.
    path = Path(__file__).resolve().parents[1] / "test_sampling_randomness_cuda.py"
    spec = importlib.util.spec_from_file_location("sampling_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._reference


def _native_fused(
    backend,
    x,
    bias,
    mapping,
    temperatures,
    seeds,
    positions,
    cache,
    *,
    fp64=False,
    padding=0,
    columns=None,
):
    rows, vocab = x.shape
    if columns is None:
        columns = torch.zeros((), dtype=torch.int32, device=x.device)
    blocks = (vocab + 1023) // 1024
    if backend == "v2":
        return torch.ops._C.ilu_dspark_bias_sample_v2(
            x,
            bias,
            mapping,
            temperatures,
            seeds,
            positions,
            cache,
            columns,
            fp64,
            padding,
            1024,
            256,
        )
    ids = torch.empty((rows, blocks), dtype=torch.int64, device=x.device)
    values = torch.empty_like(ids, dtype=torch.float64 if fp64 else torch.float32)
    if backend == "v4":
        output = torch.empty(rows, dtype=torch.int64, device=x.device)
        torch.ops._C.ilu_dspark_bias_sample_v4(
            x,
            bias,
            mapping,
            temperatures,
            seeds,
            positions,
            cache,
            columns,
            fp64,
            padding,
            1024,
            256,
            0,
            ids,
            values,
            output,
        )
        return output
    assert backend == "v1"
    torch.ops._C.ilu_fused_bias_gumbel_sample(
        ids,
        ids.stride(0),
        values,
        values.stride(0),
        cache,
        cache.stride(0),
        cache.stride(1),
        columns,
        x,
        x.stride(0),
        mapping,
        seeds,
        positions,
        temperatures,
        vocab,
        1024,
        True,
        True,
        fp64,
        False,
        bias,
        bias.stride(0),
        padding,
    )
    return ids.gather(1, values.argmax(1, keepdim=True)).view(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("backend", ["v1", "v2", "v4"])
@pytest.mark.parametrize("vocab", [32, 64, 129])
def test_each_fused_sampler_has_independent_token_support(
    backend, vocab, cpu_reference
):
    rows = 8192
    x = torch.zeros((rows, vocab), device="cuda")
    bias = torch.zeros_like(x)
    mapping = torch.arange(rows, dtype=torch.int32, device="cuda")
    seeds = torch.arange(rows, dtype=torch.int64, device="cuda")
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    temperatures = torch.ones(rows, device="cuda")
    cache = torch.full((rows, 1, vocab), -123.0, device="cuda")
    actual = (
        _native_fused(backend, x, bias, mapping, temperatures, seeds, positions, cache)
        .cpu()
        .numpy()
    )
    expected = cpu_reference(
        np.zeros((rows, vocab)), np.arange(rows), np.full(rows, 1 << 30)
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(actual)) == vocab
    assert torch.count_nonzero(cache) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("backend", ["v1", "v2", "v4"])
def test_fused_graph_replay_uses_changed_seed_position_and_bias(backend, cpu_reference):
    rows, vocab = 5, 129
    x = torch.zeros((rows, vocab), device="cuda")
    bias = torch.zeros_like(x)
    mapping = torch.arange(rows, dtype=torch.int32, device="cuda")
    seeds = torch.arange(rows, dtype=torch.int64, device="cuda")
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    temperatures = torch.ones(rows, device="cuda")
    cache = torch.empty((rows, 1, vocab), device="cuda")
    columns = torch.zeros((), dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _native_fused(
                backend,
                x,
                bias,
                mapping,
                temperatures,
                seeds,
                positions,
                cache,
                columns=columns,
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _native_fused(
            backend,
            x,
            bias,
            mapping,
            temperatures,
            seeds,
            positions,
            cache,
            columns=columns,
        )
    for step in range(8):
        seed_values = np.arange(rows) + step * 1009
        position_values = np.arange(rows) + step * 17
        bias_cpu = torch.zeros((rows, vocab))
        bias_cpu[:, (step * 13) % vocab] = 0.5
        bias.copy_(bias_cpu)
        seeds.copy_(torch.from_numpy(seed_values))
        positions.copy_(torch.from_numpy(position_values))
        graph.replay()
        expected = cpu_reference(
            bias_cpu.numpy(), seed_values, position_values + (1 << 30)
        )
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
        torch.testing.assert_close(cache[:, 0].cpu(), bias_cpu, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("backend", ["v1", "v2", "v4"])
@pytest.mark.parametrize("double_logits", [False, True])
def test_mr_rejects_fused_fp64_before_writing_cache(backend, double_logits):
    if torch.ops._C.compiled_cuda_arch() != "ivcore11":
        pytest.skip("MR precision contract")
    x = torch.zeros(
        (1, 64), dtype=torch.float64 if double_logits else torch.float32
    ).cuda()
    # Initialize unsupported FP64 data on CPU; this test only exercises rejection.
    cache = torch.full((1, 1, 64), -123, dtype=x.dtype).cuda()
    np.testing.assert_array_equal(cache.cpu().numpy(), -123)
    with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
        _native_fused(
            backend,
            x,
            x,
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.ones(1, device="cuda"),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            cache,
            fp64=not double_logits,
        )
    np.testing.assert_array_equal(cache.cpu().numpy(), -123)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "name", ["ilu_dspark_bias_finish", "ilu_dspark_bias_finish_v4"]
)
def test_mr_rejects_fused_fp64_finish(name):
    if torch.ops._C.compiled_cuda_arch() != "ivcore11":
        pytest.skip("MR precision contract")
    output = torch.full((1,), -123, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="FP64 sampling is not supported"):
        getattr(torch.ops._C, name)(
            torch.ones((1, 2), dtype=torch.float64).cuda(),
            torch.zeros((1, 2), dtype=torch.int64, device="cuda"),
            output,
        )
    assert output.item() == -123
