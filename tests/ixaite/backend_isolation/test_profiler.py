# SPDX-License-Identifier: Apache-2.0
"""A real native graph must fail the same profiler check used for own replay."""

import json

import pytest
import torch

from .ixformer_guard import assert_ixaite_kernel_events, forbid_ixformer_calls


# Empty or CPU-only traces cannot serve as evidence; known vendor kernels must fail.
@pytest.mark.parametrize(
    "events,reason",
    [
        ([], "no GPU"),
        ([{"name": "gdn_chunk", "cat": "cpu_op"}], "no GPU"),
        ([{"name": "void ixformer::act_kernel()", "cat": "kernel"}], "native vendor"),
        ([{"name": "void cuinfer::gemm()", "cat": "kernel"}], "native vendor"),
    ],
)
def test_profiler_rejects_missing_or_native_evidence(events, reason):
    with pytest.raises(AssertionError, match=reason):
        assert_ixaite_kernel_events(events)


# A real native GPU launch proves the checker can recognize prohibited kernels.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
@torch.inference_mode()
def test_native_profiler_positive_control(tmp_path):
    from vllm_iluvatar import _custom_ops as baseline

    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8, 256, device="cuda", dtype=torch.bfloat16)
    baseline.silu_and_mul(out, x)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        baseline.silu_and_mul(out, x)
        torch.cuda.synchronize()
    trace = tmp_path / "native-eager.json"
    profiler.export_chrome_trace(str(trace))
    with pytest.raises(AssertionError, match="native vendor"):
        assert_ixaite_kernel_events(json.loads(trace.read_text())["traceEvents"])


# Profile both actual graph replays, retaining changed-input numerical validation.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
@torch.inference_mode()
def test_native_positive_control_and_own_graph_replay(tmp_path):
    import ixaite

    from vllm_iluvatar import _custom_ops as baseline
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    ixaite.initialize(cuda_arch())
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    native_out = torch.empty(8, 256, device="cuda", dtype=torch.bfloat16)
    baseline.silu_and_mul(native_out, x)
    native_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(native_graph):
        baseline.silu_and_mul(native_out, x)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profiler:
        native_graph.replay()
        torch.cuda.synchronize()
    native_trace = tmp_path / "native.json"
    profiler.export_chrome_trace(str(native_trace))
    native_events = json.loads(native_trace.read_text())["traceEvents"]
    native_kernels = any(event.get("cat") == "kernel" for event in native_events)
    with pytest.raises(
        AssertionError, match="native vendor" if native_kernels else "no GPU"
    ):
        assert_ixaite_kernel_events(native_events)

    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    with forbid_ixformer_calls():
        ixaite.rms_norm(out, x, weight, 1e-6)
        own_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(own_graph):
            ixaite.rms_norm(out, x, weight, 1e-6)
        x.add_(0.5)
        with torch.profiler.profile(activities=activities) as profiler:
            own_graph.replay()
            torch.cuda.synchronize()
        reference = (
            x.float()
            * x.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()
            * weight.float()
        ).bfloat16()
        torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)
    own_trace = tmp_path / "own.json"
    profiler.export_chrome_trace(str(own_trace))
    own_events = json.loads(own_trace.read_text())["traceEvents"]
    if not native_kernels:
        # CoreX 5.1 records GraphLaunch but omits replay GPU kernels; fail closed.
        with pytest.raises(AssertionError, match="no GPU"):
            assert_ixaite_kernel_events(own_events)
        pytest.skip(
            "CoreX profiler omits CUDA Graph replay kernels; strict replay evidence unavailable"
        )
    kernels = assert_ixaite_kernel_events(own_events)
    assert any("rmsnorm" in name for name in kernels), kernels
