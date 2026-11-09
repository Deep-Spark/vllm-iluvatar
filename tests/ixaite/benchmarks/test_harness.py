# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tests.ixaite.benchmarks.core import harness, timing
from tests.ixaite.benchmarks.core.gate import evaluate_suite
from tests.ixaite.benchmarks.core.report import build_report_payload, write_json
from tests.ixaite.benchmarks.core.types import Providers, TimingResult, Workload
from tests.ixaite.benchmarks.suites.attention import AttentionCase, AttentionSuite
from tests.ixaite.benchmarks.suites.gdn_packed_decode import GdnPackedDecodeCase
from tests.ixaite.benchmarks.suites.gdn_spec_delta_update import GdnSpecDeltaUpdateCase


@dataclass
class _Case:
    case_id: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.workloads: list[Workload] = []

    def make_workload(self, device, dtype) -> Workload:
        del device, dtype
        workload = Workload(kwargs={"out": torch.zeros(1)})
        self.workloads.append(workload)
        return workload

    def run(self, fn, workload: Workload) -> None:
        workload.kwargs["out"].copy_(fn())

    def check(self, baseline: Workload, flydsl: Workload) -> None:
        torch.testing.assert_close(
            flydsl.kwargs["out"], baseline.kwargs["out"], rtol=0, atol=0
        )

    def aux_from_ms(self, median_ms, workload):
        del median_ms, workload
        return None


def test_cuda_graph_replay_is_correctness_gated(monkeypatch) -> None:
    case = _Case("decode", ("eager_decode",))
    flydsl_calls = 0

    def flydsl():
        nonlocal flydsl_calls
        flydsl_calls += 1
        return torch.full((1,), 1 if flydsl_calls == 1 else 2)

    providers = Providers(
        baseline_name="baseline",
        baseline=lambda: torch.ones(1),
        flydsl=flydsl,
    )
    calls = 0

    def fake_measure(run, **kwargs):
        nonlocal calls
        del kwargs
        run()
        calls += 1
        return TimingResult(1.0, (1.0,), True, True)

    monkeypatch.setattr(harness, "measure_callable", fake_measure)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    result = harness.bench_case(
        type("Suite", (), {"name": "test", "gate_min_speedup": lambda self: 1.0})(),
        case,
        providers,
        model="test",
        arch="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        warmup=0,
        repeats=1,
        check_only=False,
        cuda_graph=True,
    )

    assert result.gate == "fail"
    assert result.reason.startswith("correctness: cuda_graph replay correctness:")
    assert result.baseline.median_ms == 1.0
    assert result.flydsl.median_ms == 1.0
    assert result.baseline.correct is False
    assert result.flydsl.correct is False


def test_case_calls_per_run_normalizes_reported_latency(monkeypatch) -> None:
    """A case that batches N calls per timed unit still reports per-call ms."""

    case = _Case("decode", ("eager_decode",))
    case.calls_per_run = 4
    seen: list[int] = []

    def fake_measure(run, **kwargs):
        seen.append(kwargs["calls_per_sample"])
        run()
        return TimingResult(8.0 / kwargs["calls_per_sample"], (2.0,), True, True)

    providers = Providers(
        baseline_name="baseline",
        baseline=lambda: torch.ones(1),
        flydsl=lambda: torch.ones(1),
    )
    monkeypatch.setattr(harness, "measure_callable", fake_measure)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    result = harness.bench_case(
        type("Suite", (), {"name": "test", "gate_min_speedup": lambda self: 1.0})(),
        case,
        providers,
        model="test",
        arch="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        warmup=0,
        repeats=1,
        check_only=False,
        cuda_graph=False,
    )

    assert seen == [4, 4]
    assert result.baseline.median_ms == 2.0
    assert result.flydsl.median_ms == 2.0


def test_decode_linear_cases_rotate_weights_across_a_timed_unit() -> None:
    """Decode shapes must not be timed against one resident weight."""

    from tests.ixaite.benchmarks.suites.linear import LinearCase

    decode = LinearCase("down", "mlp_down_proj", T=1, K=12288, N=4096)
    assert decode.calls_per_run > 1

    calls: list[int] = []
    workload = decode.make_workload(torch.device("cpu"), torch.float32)
    assert len({w.data_ptr() for w in workload.kwargs["cold_weights"]}) == (
        decode.calls_per_run - 1
    )

    def fn(x, weight, bias):
        del x, bias
        calls.append(weight.data_ptr())
        return torch.zeros(decode.T, decode.N)

    decode.run(fn, workload)
    assert len(set(calls)) == decode.calls_per_run
    assert calls[-1] == workload.kwargs["weight"].data_ptr()

    prefill = LinearCase("pre", "mlp_down_proj", T=512, K=12288, N=4096)
    assert prefill.calls_per_run == 1


def test_graph_pins_are_retained_when_graph_reset_is_not_confirmed(monkeypatch) -> None:
    calls = []

    class Graph:
        def reset(self):
            calls.append("reset")

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    reset = timing._reset_cuda_graph(
        Graph(),
        device=torch.device("cuda"),
        cleanup=lambda: calls.append("cleanup"),
    )

    assert reset is False
    assert calls == []


def test_graph_pins_release_only_after_successful_reset(monkeypatch) -> None:
    calls = []

    class Graph:
        def reset(self):
            calls.append("reset")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))

    reset = timing._reset_cuda_graph(
        Graph(),
        device=torch.device("cuda"),
        cleanup=lambda: calls.append("cleanup"),
    )

    assert reset is True
    assert calls == ["sync", "reset", "cleanup"]


def test_attention_graph_run_filters_unsupported_prefill(monkeypatch, tmp_path) -> None:
    decode = _Case("decode", ("eager_decode",))
    prefill = _Case("prefill", ("paged_prefill",))

    class Suite:
        name = "attention"

        def resolve_providers(self):
            return Providers(
                baseline_name="baseline",
                baseline=lambda: None,
                flydsl=lambda: None,
            )

        def iter_cases(self, model, case_filter):
            del model, case_filter
            return [prefill, decode]

        @staticmethod
        def supports_case_cuda_graph(case):
            return "eager_decode" in case.tags

    seen: list[str] = []

    def fake_bench_case(suite, case, providers, **kwargs):
        del suite, providers, kwargs
        seen.append(case.case_id)
        return object()

    monkeypatch.setattr(harness, "bench_case", fake_bench_case)

    results = harness.run_suite(
        Suite(),
        model="test",
        case_filter=None,
        arch="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        warmup=0,
        repeats=1,
        check_only=False,
        cuda_graph=True,
    )

    assert seen == ["decode"]
    assert len(results) == 2
    assert results[0].gate == "skip"

    Suite.iter_cases = lambda self, model, case_filter: [prefill]
    results = harness.run_suite(
        Suite(),
        model="test",
        case_filter="prefill",
        arch="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        warmup=0,
        repeats=1,
        check_only=False,
        cuda_graph=True,
    )
    assert len(results) == 1
    assert results[0].gate == "skip"
    assert results[0].baseline.correct is False
    assert results[0].baseline.error == results[0].reason
    assert evaluate_suite(results, check_only=False) == 1

    payload = build_report_payload(results, repo_root=tmp_path, argv=[])
    assert payload["results"][0]["baseline"]["median_ms"] is None
    report_path = tmp_path / "report.json"
    write_json(report_path, payload)
    assert "NaN" not in report_path.read_text()


def test_attention_graph_uses_production_cache_capacity() -> None:
    case = AttentionCase(
        case_id="decode",
        operation="flash_attn_with_kvcache",
        context_len=1169,
        batch=1,
        num_query_heads=6,
        num_kv_heads=1,
        head_dim=256,
        page_size=16,
        graph_capacity=131072,
        tags=("eager_decode",),
    )
    workload = case.make_workload(torch.device("cpu"), torch.bfloat16)

    assert workload.kwargs["block_table"].shape == (1, 8192)
    assert workload.kwargs["k_cache"].shape[0] == 74
    assert workload.kwargs["max_context_len"] == 1169

    case.prepare_cuda_graph(workload)
    q_before_replay = workload.kwargs["q"].clone()
    case.prepare_cuda_graph_replay(workload)

    assert workload.kwargs["use_cuda_graph"] is True
    assert workload.kwargs["max_context_len"] == 131072
    assert workload.kwargs["k_cache"].shape[0] == 8192
    assert workload.kwargs["v_cache"].shape[0] == 8192
    assert workload.kwargs["block_table"][0, -1].item() == 8191
    assert workload.kwargs["cache_seqlens"].item() == 2338
    torch.testing.assert_close(workload.kwargs["q"], -q_before_replay)


def test_attention_graph_replay_is_checked_against_fresh_reference() -> None:
    case = AttentionCase(
        case_id="decode",
        operation="flash_attn_with_kvcache",
        context_len=16,
        batch=1,
        num_query_heads=6,
        num_kv_heads=1,
        head_dim=256,
        page_size=16,
        graph_capacity=16,
        tags=("eager_decode",),
    )
    workload = case.make_workload(torch.device("cpu"), torch.bfloat16)
    workload.kwargs["out"].zero_()
    case.prepare_cuda_graph(workload)
    case.prepare_cuda_graph_replay(workload)
    assert workload.kwargs["cache_seqlens"].item() == 8

    def reference(operation, kwargs):
        assert operation == "flash_attn_with_kvcache"
        kwargs["out"].fill_(1)

    with pytest.raises(AssertionError):
        case.check_cuda_graph_replay(reference, workload)


def test_qwen35_attention_uses_production_graph_capacity() -> None:
    cases = AttentionSuite().iter_cases("qwen35_9b", None)
    decode = next(case for case in cases if "eager_decode" in case.tags)
    assert decode.graph_capacity == 8192


def test_qwen36_27b_vision_attention_is_packed_encoder() -> None:
    cases = AttentionSuite().iter_cases("qwen36_27b_vision", None)
    assert cases
    assert all("packed_encoder" in case.tags for case in cases)
    geometries = {
        (case.num_query_heads, case.head_dim, case.softmax_head_dim) for case in cases
    }
    assert geometries == {(8, 72, 72), (8, 128, 72), (4, 128, 72)}
    workload = cases[0].make_workload(torch.device("cpu"), torch.bfloat16)
    assert workload.kwargs["causal"] is False
    assert "block_table" not in workload.kwargs
    assert "out" in workload.kwargs
    assert workload.kwargs["out"].shape == workload.kwargs["q"].shape
    assert workload.kwargs["softmax_scale"] == pytest.approx(
        (cases[0].softmax_head_dim or cases[0].head_dim) ** -0.5
    )
    other = cases[0].make_workload(torch.device("cpu"), torch.bfloat16)
    workload.kwargs["out"].fill_(1)
    other.kwargs["out"].fill_(1)
    cases[0].check(workload, other)


def test_gdn_packed_decode_graph_replay_restores_ssm_state() -> None:
    case = GdnPackedDecodeCase(
        case_id="decode",
        num_tokens=2,
        num_k_heads=4,
        head_k_dim=128,
        num_v_heads=12,
        head_v_dim=128,
    )
    workload = case.make_workload(torch.device("cpu"), torch.bfloat16)
    original = workload.meta["state0"].clone()
    workload.kwargs["initial_state"].zero_()
    case.prepare_cuda_graph_replay(workload)
    torch.testing.assert_close(workload.kwargs["initial_state"], original)


def test_gdn_spec_delta_update_graph_replay_restores_ssm_state() -> None:
    case = GdnSpecDeltaUpdateCase(
        case_id="spec",
        num_seqs=1,
        tokens_per_seq=2,
        num_k_heads=4,
        head_k_dim=128,
        num_v_heads=12,
        head_v_dim=128,
    )
    workload = case.make_workload(torch.device("cpu"), torch.bfloat16)
    original = workload.meta["state0"].clone()
    workload.args[6].zero_()
    case.prepare_cuda_graph_replay(workload)
    torch.testing.assert_close(workload.args[6], original)
