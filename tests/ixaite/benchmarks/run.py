# SPDX-License-Identifier: Apache-2.0
"""CLI entry: ``python -m tests.ixaite.benchmarks`` from the vllm-iluvatar repo root."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Trigger suite registration.
import tests.ixaite.benchmarks.suites  # noqa: F401
from tests.ixaite.benchmarks.core import registry
from tests.ixaite.benchmarks.core.harness import evaluate_suite, run_suite, summarize
from tests.ixaite.benchmarks.core.model_catalog import list_models, load_model
from tests.ixaite.benchmarks.core.report import (
    build_report_payload,
    write_csv,
    write_json,
)
from tests.ixaite.benchmarks.core.timing import DEFAULT_STABILITY_PCT

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUT = Path(__file__).resolve().parent / "reports"

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.ixaite.benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Cross-provider microbenchmarks: FlyDSL (ixaite) vs baseline "
            "(ixformer / IXF).\n"
            "\n"
            "Correctness always runs first (eager). Timing then measures median "
            "latency for\n"
            "each selected mode; speedup = median_ms(baseline) / median_ms(flydsl) "
            "(>1 means\n"
            "FlyDSL is faster)."
        ),
        epilog=(
            "examples:\n"
            "  # list suites\n"
            "  python -m tests.ixaite.benchmarks --list\n"
            "\n"
            "  # correctness only (no timing)\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b "
            "--check-only --gate\n"
            "\n"
            "  # default: report BOTH eager and CUDA Graph replay, then gate\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --gate\n"
            "\n"
            "  # only one timing mode\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --timing eager --gate\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --timing cuda-graph --gate\n"
            "\n"
            "  # stability: default max within 3% of median; optional absolute us floor\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --stability-pct 5 --gate\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --stability-us 5 --gate\n"
            "\n"
            "  # filter cases / skip writing reports\n"
            "  python -m tests.ixaite.benchmarks --suite silu_and_mul --case rows=8 --no-write\n"
            "\n"
            "reports:\n"
            f"  JSON + CSV under {_DEFAULT_OUT} (gitignored) unless --no-write.\n"
            "  mode column: bench (eager) or bench_cudagraph (CUDA Graph replay).\n"
        ),
    )

    discover = parser.add_argument_group("discovery")
    discover.add_argument(
        "--list",
        action="store_true",
        help="list registered suites and exit",
    )
    discover.add_argument(
        "--list-models",
        action="store_true",
        help="list repository model profiles and exit",
    )

    select = parser.add_argument_group("suite selection")
    select.add_argument(
        "--suite",
        type=str,
        help="suite to run (required unless --list); e.g. silu_and_mul, linear",
    )
    select.add_argument(
        "--model",
        type=str,
        default="qwen35_9b",
        help="model case catalog key (default: %(default)s)",
    )
    select.add_argument(
        "--case",
        type=str,
        default=None,
        help="substring filter on case_id (e.g. attn_qkv_proj or rows=8)",
    )
    select.add_argument(
        "--dtype",
        choices=tuple(_DTYPES),
        default="bfloat16",
        help="tensor dtype (default: %(default)s)",
    )
    select.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device string (default: %(default)s)",
    )
    select.add_argument(
        "--full-profile",
        action="store_true",
        help="expand linear T set to include the full CUDA Graph capture table",
    )

    timing = parser.add_argument_group("timing")
    timing.add_argument(
        "--check-only",
        action="store_true",
        help="correctness only; skip GPU event timing (ignores --timing)",
    )
    timing.add_argument(
        "--timing",
        choices=("both", "eager", "cuda-graph"),
        default="both",
        help=(
            "timing modes in the report (default: %(default)s). "
            "both=eager+cuda-graph; eager=kernel launch; "
            "cuda-graph=CUDAGraph.replay after warmup+capture "
            "(capture/JIT not timed)"
        ),
    )
    timing.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="untimed warmup iterations per provider before sampling (default: %(default)s)",
    )
    timing.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="timed iterations; median latency is reported (default: %(default)s)",
    )
    timing.add_argument(
        "--stability-pct",
        type=float,
        default=DEFAULT_STABILITY_PCT,
        help=(
            "relative stability: max(sample) must be within this percent of the "
            "median (default: %(default)s). Use 0 with --stability-us for an "
            "absolute-only check"
        ),
    )
    timing.add_argument(
        "--stability-us",
        type=float,
        default=None,
        help=(
            "optional absolute stability floor in microseconds: max(sample) may "
            "exceed the median by up to this many us. When set together with "
            "--stability-pct, the larger of the two slacks is used"
        ),
    )

    gate = parser.add_argument_group("gate")
    gate.add_argument(
        "--gate",
        action="store_true",
        help=(
            "exit 1 if any case fails (correctness / stability / "
            "speedup < suite minimum, usually 1.0)"
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="report directory for JSON/CSV (default: %(default)s)",
    )
    output.add_argument(
        "--no-write",
        action="store_true",
        help="print summary only; do not write JSON/CSV reports",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        for name in registry.list_suites():
            print(name)
        return 0
    if args.list_models:
        for name in list_models():
            print(name)
        return 0

    if not args.suite:
        print(
            "error: --suite is required (or pass --list/--list-models)",
            file=sys.stderr,
        )
        return 2

    try:
        load_model(args.model)
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("error: CUDA / Iluvatar device is required", file=sys.stderr)
        return 2

    suite = registry.get(args.suite)
    if args.full_profile and hasattr(suite, "_full_profile"):
        suite._full_profile = True  # type: ignore[attr-defined]

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    arch = cuda_arch()
    if arch is None:
        print("error: cannot detect Iluvatar GPU architecture", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]

    if args.stability_pct < 0:
        print("error: --stability-pct must be >= 0", file=sys.stderr)
        return 2
    if args.stability_us is not None and args.stability_us < 0:
        print("error: --stability-us must be >= 0", file=sys.stderr)
        return 2
    if args.stability_pct == 0 and args.stability_us is None:
        print(
            "error: set --stability-pct > 0 and/or --stability-us",
            file=sys.stderr,
        )
        return 2

    if args.check_only:
        timing_modes: list[str] = []
    elif args.timing == "eager":
        timing_modes = ["eager"]
    elif args.timing == "cuda-graph":
        timing_modes = ["cuda_graph"]
    else:
        timing_modes = ["eager", "cuda_graph"]

    if "cuda_graph" in timing_modes and not getattr(suite, "supports_cuda_graph", True):
        if args.timing == "cuda-graph":
            print(
                f"error: suite {suite.name!r} does not support CUDA Graph timing",
                file=sys.stderr,
            )
            return 2
        timing_modes.remove("cuda_graph")
        print(f"note: suite {suite.name!r} is eager-only; skipping CUDA Graph timing")

    try:
        if args.check_only:
            results = run_suite(
                suite,
                model=args.model,
                case_filter=args.case,
                arch=arch,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                check_only=True,
                cuda_graph=False,
                stability_pct=args.stability_pct,
                stability_us=args.stability_us,
            )
        else:
            results = []
            for mode in timing_modes:
                results.extend(
                    run_suite(
                        suite,
                        model=args.model,
                        case_filter=args.case,
                        arch=arch,
                        device=device,
                        dtype=dtype,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        check_only=False,
                        cuda_graph=(mode == "cuda_graph"),
                        stability_pct=args.stability_pct,
                        stability_us=args.stability_us,
                    )
                )
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not results or all(result.gate == "skip" for result in results):
        print(
            "error: no selected cases support the requested timing mode",
            file=sys.stderr,
        )
        return 2

    print(summarize(results))

    if not args.no_write:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{args.suite}_{args.model}_{stamp}"
        if args.check_only:
            stem += "_check"
        elif timing_modes:
            stem += "_" + "_".join(
                "cudagraph" if mode == "cuda_graph" else mode for mode in timing_modes
            )
        json_path = args.out / f"{stem}.json"
        csv_path = args.out / f"{stem}.csv"
        payload = build_report_payload(
            results,
            repo_root=_REPO_ROOT,
            argv=sys.argv if argv is None else ["tests.ixaite.benchmarks", *argv],
            extra={
                "suite": args.suite,
                "model": args.model,
                "check_only": args.check_only,
                "timing_modes": timing_modes,
                "gate_requested": args.gate,
                "full_profile": args.full_profile,
                "arch": arch,
                "dtype": args.dtype,
                "stability_pct": args.stability_pct,
                "stability_us": args.stability_us,
            },
        )
        write_json(json_path, payload)
        write_csv(csv_path, results)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")

    if args.gate:
        return evaluate_suite(results, check_only=args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
