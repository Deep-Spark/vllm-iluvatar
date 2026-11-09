# SPDX-License-Identifier: Apache-2.0
"""Run one suite case: correctness first, optional timing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import torch

from tests.ixaite.benchmarks.core.gate import evaluate_suite, per_case_gate
from tests.ixaite.benchmarks.core.timing import DEFAULT_STABILITY_PCT, measure_callable
from tests.ixaite.benchmarks.core.types import (
    BenchCase,
    CaseResult,
    Providers,
    Suite,
    TimingResult,
    Workload,
)


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Clone, keeping any spare rows the source's storage carries.

    A row-slice of a larger buffer is a meaningful input shape: a provider may
    read past the logical row count (the ixaite small-batch GEMM reads a fixed
    brick height in place). ``Tensor.clone`` returns tight storage, which would
    silently charge one side of the comparison for a staged copy.
    """
    if tensor.dim() == 2 and tensor.storage_offset() == 0 and tensor.stride(1) == 1:
        cols = tensor.shape[1]
        if tensor.stride(0) == cols:
            rows = tensor.untyped_storage().nbytes() // (cols * tensor.element_size())
            if rows > tensor.shape[0]:
                buffer = torch.empty(
                    (rows, cols), dtype=tensor.dtype, device=tensor.device
                )
                out = buffer[: tensor.shape[0]]
                out.copy_(tensor)
                return out
    return tensor.clone()


def clone_workload(wl: Workload) -> Workload:
    """Deep-copy tensor fields so inplace ops cannot cross-contaminate providers."""

    def _clone(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return _clone_tensor(value)
        if isinstance(value, tuple):
            return tuple(_clone(item) for item in value)
        if isinstance(value, list):
            return [_clone(item) for item in value]
        if isinstance(value, dict):
            return {key: _clone(item) for key, item in value.items()}
        return value

    return Workload(
        args=tuple(_clone(arg) for arg in wl.args),
        kwargs={key: _clone(val) for key, val in wl.kwargs.items()},
        outputs={key: _clone(val) for key, val in wl.outputs.items()},  # type: ignore[misc]
        meta=dict(wl.meta),
    )


def _skipped_timing(*, correct: bool, error: str | None = None) -> TimingResult:
    return TimingResult(
        median_ms=float("nan"),
        samples_ms=(),
        stable=True,
        correct=correct,
        error=error,
    )


def _attach_aux(
    timing: TimingResult, case: BenchCase, wl_meta_source: object
) -> TimingResult:
    if not timing.correct or math.isnan(timing.median_ms):
        return timing
    aux = case.aux_from_ms(timing.median_ms, wl_meta_source)  # type: ignore[arg-type]
    if aux is None:
        return timing
    metric, unit = aux
    return TimingResult(
        median_ms=timing.median_ms,
        samples_ms=timing.samples_ms,
        stable=timing.stable,
        correct=timing.correct,
        error=timing.error,
        aux_metric=metric,
        aux_unit=unit,
    )


def bench_case(
    suite: Suite,
    case: BenchCase,
    providers: Providers,
    *,
    model: str,
    arch: str,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    check_only: bool,
    cuda_graph: bool = False,
    stability_pct: float = DEFAULT_STABILITY_PCT,
    stability_us: float | None = None,
) -> CaseResult:
    """Correctness-gate then optionally time baseline vs FlyDSL."""
    # Correctness stays eager; cuda_graph only affects the timed path.
    use_graph = bool(cuda_graph) and not check_only
    mode = "check_only" if check_only else ("bench_cudagraph" if use_graph else "bench")
    # Same random inputs on both sides; separate clones for inplace outputs.
    wl_b = case.make_workload(device, dtype)
    wl_f = clone_workload(wl_b)

    def run_b() -> None:
        case.run(providers.baseline, wl_b)

    def run_f() -> None:
        case.run(providers.flydsl, wl_f)

    run_b()
    run_f()
    torch.cuda.synchronize(device)

    try:
        case.check(wl_b, wl_f)
        correct = True
        err: str | None = None
    except Exception as error:  # noqa: BLE001 -- surface any check failure
        correct = False
        err = str(error)

    if not correct:
        bad = _skipped_timing(correct=False, error=err)
        return CaseResult(
            suite=suite.name,
            case_id=case.case_id,
            model=model,
            mode=mode,
            baseline_name=providers.baseline_name,
            flydsl_name=providers.flydsl_name,
            dtype=str(dtype).removeprefix("torch."),
            arch=arch,
            baseline=bad,
            flydsl=bad,
            speedup=None,
            gate="fail",
            reason=f"correctness: {err}",
            tags=tuple(case.tags),
            cuda_graph=use_graph,
        )

    if check_only:
        skipped = _skipped_timing(correct=True)
        return CaseResult(
            suite=suite.name,
            case_id=case.case_id,
            model=model,
            mode=mode,
            baseline_name=providers.baseline_name,
            flydsl_name=providers.flydsl_name,
            dtype=str(dtype).removeprefix("torch."),
            arch=arch,
            baseline=skipped,
            flydsl=skipped,
            speedup=None,
            gate="pass",
            reason="check-only ok",
            tags=tuple(case.tags),
            cuda_graph=False,
        )

    if use_graph:
        prepare_cuda_graph = getattr(case, "prepare_cuda_graph", None)
        if callable(prepare_cuda_graph):
            prepare_cuda_graph(wl_b)
            prepare_cuda_graph(wl_f)
    prepare_cuda_graph_replay = getattr(case, "prepare_cuda_graph_replay", None)
    replay_setup_b = (
        (lambda: prepare_cuda_graph_replay(wl_b))
        if use_graph and callable(prepare_cuda_graph_replay)
        else None
    )
    replay_setup_f = (
        (lambda: prepare_cuda_graph_replay(wl_f))
        if use_graph and callable(prepare_cuda_graph_replay)
        else None
    )
    cleanup_cuda_graph = getattr(case, "cleanup_cuda_graph", None)
    graph_cleanup_f = (
        cleanup_cuda_graph
        if use_graph and callable(cleanup_cuda_graph)
        else None
    )

    # A case may run the operator several times per timed unit, e.g. over
    # rotating weights so the measurement cannot lean on a cache the
    # deployment does not have. Timing divides by it and stays per call.
    calls_per_sample = max(1, int(getattr(case, "calls_per_run", 1)))
    tb = _attach_aux(
        measure_callable(
            run_b,
            device=device,
            warmup=warmup,
            repeats=repeats,
            cuda_graph=use_graph,
            cuda_graph_replay_setup=replay_setup_b,
            stability_pct=stability_pct,
            stability_us=stability_us,
            calls_per_sample=calls_per_sample,
        ),
        case,
        wl_b,
    )
    tf = _attach_aux(
        measure_callable(
            run_f,
            device=device,
            warmup=warmup,
            repeats=repeats,
            cuda_graph=use_graph,
            cuda_graph_replay_setup=replay_setup_f,
            cuda_graph_cleanup=graph_cleanup_f,
            stability_pct=stability_pct,
            stability_us=stability_us,
            calls_per_sample=calls_per_sample,
        ),
        case,
        wl_f,
    )
    if use_graph and tb.correct and tf.correct:
        torch.cuda.synchronize(device)
        try:
            validate_replay = getattr(case, "check_cuda_graph_replay", None)
            if callable(validate_replay):
                validate_replay(providers.baseline, wl_b)
                validate_replay(providers.baseline, wl_f)
            case.check(wl_b, wl_f)
        except Exception as error:  # noqa: BLE001 -- surface graph replay failures
            graph_error = f"cuda_graph replay correctness: {error}"
            tb = replace(tb, correct=False, error=graph_error)
            tf = replace(tf, correct=False, error=graph_error)
    speedup = (
        tb.median_ms / tf.median_ms
        if tb.correct and tf.correct and tf.median_ms > 0
        else None
    )
    gate, reason = per_case_gate(
        tb,
        tf,
        min_speedup=suite.gate_min_speedup(),
        check_only=False,
    )
    return CaseResult(
        suite=suite.name,
        case_id=case.case_id,
        model=model,
        mode=mode,
        baseline_name=providers.baseline_name,
        flydsl_name=providers.flydsl_name,
        dtype=str(dtype).removeprefix("torch."),
        arch=arch,
        baseline=tb,
        flydsl=tf,
        speedup=speedup,
        gate=gate,
        reason=reason,
        tags=tuple(case.tags),
        cuda_graph=use_graph,
    )


def run_suite(
    suite: Suite,
    *,
    model: str,
    case_filter: str | None,
    arch: str,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    check_only: bool,
    cuda_graph: bool = False,
    stability_pct: float = DEFAULT_STABILITY_PCT,
    stability_us: float | None = None,
) -> list[CaseResult]:
    """Resolve providers once, then bench every matching case."""
    providers = suite.resolve_providers()
    cases = list(suite.iter_cases(model, case_filter))
    if not cases:
        raise RuntimeError(
            f"suite {suite.name!r} produced no cases for model={model!r} "
            f"filter={case_filter!r}"
        )
    supports_case_cuda_graph = getattr(suite, "supports_case_cuda_graph", None)
    results: list[CaseResult] = []
    for case in cases:
        if (
            cuda_graph
            and callable(supports_case_cuda_graph)
            and not supports_case_cuda_graph(case)
        ):
            skip_reason = "CUDA Graph timing is unsupported for this case"
            skipped = _skipped_timing(correct=False, error=skip_reason)
            results.append(
                CaseResult(
                    suite=suite.name,
                    case_id=case.case_id,
                    model=model,
                    mode="bench_cudagraph",
                    baseline_name=providers.baseline_name,
                    flydsl_name=providers.flydsl_name,
                    dtype=str(dtype).removeprefix("torch."),
                    arch=arch,
                    baseline=skipped,
                    flydsl=skipped,
                    speedup=None,
                    gate="skip",
                    reason=skip_reason,
                    tags=tuple(case.tags),
                    cuda_graph=True,
                )
            )
            continue
        results.append(
            bench_case(
                suite,
                case,
                providers,
                model=model,
                arch=arch,
                device=device,
                dtype=dtype,
                warmup=warmup,
                repeats=repeats,
                check_only=check_only,
                cuda_graph=cuda_graph,
                stability_pct=stability_pct,
                stability_us=stability_us,
            )
        )
    return results


def format_speedup(
    speedup: float | None,
    *,
    baseline_ms: float | None = None,
    flydsl_ms: float | None = None,
) -> str:
    """Human-readable speedup: baseline_ms / flydsl_ms (>1 means FlyDSL faster)."""
    if speedup is None:
        return "-"
    if speedup >= 1.0:
        rel = f"flydsl {speedup:.2f}x faster"
    elif speedup > 0:
        rel = f"flydsl {1.0 / speedup:.2f}x slower"
    else:
        rel = "invalid"
    detail = f"{speedup:.3f}x ({rel})"
    if (
        baseline_ms is not None
        and flydsl_ms is not None
        and not (baseline_ms != baseline_ms or flydsl_ms != flydsl_ms)  # NaN check
    ):
        detail = (
            f"base {baseline_ms:.3f}ms / fly {flydsl_ms:.3f}ms = {speedup:.3f}x ({rel})"
        )
    return detail


def summarize(results: Sequence[CaseResult]) -> str:
    """Return a short human-readable summary table."""
    has_eager = any(item.mode == "bench" for item in results)
    has_graph = any(item.cuda_graph for item in results)
    if has_eager and has_graph:
        timing_note = "timing=eager + cuda_graph.replay (capture not timed)"
    elif has_graph:
        timing_note = "timing=cuda_graph.replay (capture not timed)"
    else:
        timing_note = "timing=eager kernel launch"
    lines = [
        "speedup = median_ms(baseline) / median_ms(flydsl); >1 means FlyDSL is faster",
        timing_note,
        f"{'mode':<16} {'case_id':<44} {'gate':<6} {'vs (baseline/flydsl)':<52} reason",
        "-" * 138,
    ]
    for item in results:
        timed = item.mode != "check_only"
        base_ms = item.baseline.median_ms if timed else None
        fly_ms = item.flydsl.median_ms if timed else None
        vs = format_speedup(item.speedup, baseline_ms=base_ms, flydsl_ms=fly_ms)
        lines.append(
            f"{item.mode:<16} {item.case_id:<44} {item.gate:<6} {vs:<52} {item.reason}"
        )
    passed = sum(1 for item in results if item.gate == "pass")
    failed = sum(1 for item in results if item.gate == "fail")
    skipped = sum(1 for item in results if item.gate == "skip")
    lines.append("-" * 138)
    lines.append(f"pass={passed} fail={failed} skip={skipped} total={len(results)}")
    return "\n".join(lines)


__all__ = ["bench_case", "evaluate_suite", "format_speedup", "run_suite", "summarize"]
