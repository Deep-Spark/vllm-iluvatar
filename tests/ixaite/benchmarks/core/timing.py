# SPDX-License-Identifier: Apache-2.0
"""GPU event timing for cross-provider microbenchmarks."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

import torch

from tests.ixaite.benchmarks.core.types import TimingResult

# Default relative headroom: max(sample) must stay within this percent of median.
DEFAULT_STABILITY_PCT = 3.0


def stability_bound_ms(
    median_ms: float,
    *,
    stability_pct: float = DEFAULT_STABILITY_PCT,
    stability_us: float | None = None,
) -> float:
    """Return the max allowed sample latency in milliseconds.

    Relative slack is ``median * pct / 100``. Absolute slack (when set) is
    ``stability_us / 1000``. The allowed bound is median plus the larger slack
    so short cuda-graph replay samples can use an absolute floor without
    discarding the percentage rule for longer eager kernels.
    """
    if stability_pct < 0:
        raise ValueError(f"stability_pct must be >= 0, got {stability_pct}")
    if stability_us is not None and stability_us < 0:
        raise ValueError(f"stability_us must be >= 0, got {stability_us}")
    if stability_pct == 0 and stability_us is None:
        raise ValueError("set stability_pct > 0 and/or stability_us")

    slack_ms = median_ms * (stability_pct / 100.0) if stability_pct > 0 else 0.0
    if stability_us is not None:
        slack_ms = max(slack_ms, stability_us / 1000.0)
    return median_ms + slack_ms


def samples_are_stable(
    samples_ms: Sequence[float],
    median_ms: float,
    *,
    stability_pct: float = DEFAULT_STABILITY_PCT,
    stability_us: float | None = None,
) -> bool:
    """True when every sample is within the configured stability bound."""
    if not samples_ms:
        return False
    return max(samples_ms) <= stability_bound_ms(
        median_ms,
        stability_pct=stability_pct,
        stability_us=stability_us,
    )


def _capture_cuda_graph(
    run: Callable[[], object],
    *,
    device: torch.device,
    cleanup: Callable[[], None] | None = None,
) -> torch.cuda.CUDAGraph:
    """Capture ``run`` into a CUDA Graph on ``device`` (caller must warm first)."""
    graph = torch.cuda.CUDAGraph()
    try:
        # Side-stream capture matches the usual PyTorch CUDA Graph recipe and keeps
        # the default stream free for later event timing of graph.replay().
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            # Some production entry points keep stream-local staging buffers and
            # require one prewarm on the exact stream that will be captured.
            run()
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                run()
        torch.cuda.current_stream(device).wait_stream(stream)
    except Exception:
        _reset_cuda_graph(graph, device=device, cleanup=cleanup)
        raise
    return graph


def _reset_cuda_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    cleanup: Callable[[], None] | None,
) -> bool:
    """Reset a graph before releasing its pinned addresses.

    If synchronization or reset fails, retain the pins: leaking until process
    teardown is safer than releasing storage still referenced by a live graph.
    """
    try:
        torch.cuda.synchronize(device)
        graph.reset()
    except Exception:  # noqa: BLE001 -- CUDA may already be in an error state
        return False
    if cleanup is not None:
        cleanup()
    return True


def measure_callable(
    run: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    cuda_graph: bool = False,
    cuda_graph_replay_setup: Callable[[], None] | None = None,
    cuda_graph_cleanup: Callable[[], None] | None = None,
    stability_pct: float = DEFAULT_STABILITY_PCT,
    stability_us: float | None = None,
    calls_per_sample: int = 1,
) -> TimingResult:
    """Warm up, then return median latency and stability for ``run``.

    When ``cuda_graph`` is true, eager warmup + capture happen first; timed
    samples measure ``graph.replay()`` only (not capture / JIT).

    ``calls_per_sample`` is how many operator calls one ``run`` performs, for
    cases that batch several calls into a timed unit. Samples are divided by
    it, so the reported median and the stability verdict stay per call.

    Stability: ``max(samples) <= median + max(median*pct/100, us/1000)`` when
    both knobs are active; see ``stability_bound_ms``.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if calls_per_sample < 1:
        raise ValueError(f"calls_per_sample must be >= 1, got {calls_per_sample}")

    graph: torch.cuda.CUDAGraph | None = None

    def release_graph() -> None:
        nonlocal graph
        if graph is None:
            return
        captured_graph = graph
        graph = None
        _reset_cuda_graph(
            captured_graph,
            device=device,
            cleanup=cuda_graph_cleanup,
        )

    def collect_samples(timed: Callable[[], object]) -> list[float]:
        samples: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            timed()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) / calls_per_sample)
        return samples

    if cuda_graph:
        # Eager warmup so JIT / allocator settle before capture.
        for _ in range(max(warmup, 3)):
            run()
        torch.cuda.synchronize(device)
        try:
            graph = _capture_cuda_graph(
                run,
                device=device,
                cleanup=cuda_graph_cleanup,
            )
        except Exception as error:  # noqa: BLE001 -- surface capture failures
            return TimingResult(
                median_ms=float("nan"),
                samples_ms=(),
                stable=False,
                correct=False,
                error=f"cuda_graph capture failed: {error}",
            )
        try:
            timed: Callable[[], object] = graph.replay
            for _ in range(warmup):
                timed()
            torch.cuda.synchronize(device)
            samples = collect_samples(timed)
            if cuda_graph_replay_setup is not None:
                try:
                    cuda_graph_replay_setup()
                    timed()
                    torch.cuda.synchronize(device)
                except Exception as error:  # noqa: BLE001 -- surface replay failures
                    return TimingResult(
                        median_ms=float("nan"),
                        samples_ms=(),
                        stable=False,
                        correct=False,
                        error=f"cuda_graph replay validation failed: {error}",
                    )
        finally:
            release_graph()
    else:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize(device)
        samples = collect_samples(run)

    median_ms = statistics.median(samples)
    stable = samples_are_stable(
        samples,
        median_ms,
        stability_pct=stability_pct,
        stability_us=stability_us,
    )
    return TimingResult(
        median_ms=median_ms,
        samples_ms=tuple(samples),
        stable=stable,
        correct=True,
    )
