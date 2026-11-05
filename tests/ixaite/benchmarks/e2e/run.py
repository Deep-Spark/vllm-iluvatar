# SPDX-License-Identifier: Apache-2.0
"""CLI for repository-contained FlyDSL versus baseline end-to-end benchmarks."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.ixaite.benchmarks.core.model_catalog import e2e_spec, list_e2e_models
from tests.ixaite.benchmarks.e2e.report import (
    build_compare,
    summary,
    write_json,
    write_metrics_csv,
)
from tests.ixaite.benchmarks.e2e.runner import (
    CORRECTNESS_KINDS,
    SUPPORTED_DATASETS,
    VIDEO_GREEDY_EXACT,
    check_greedy_exact,
    command_lines,
    default_python,
    prepare_video_correctness,
    run_provider,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[1] / "e2e-runs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.ixaite.benchmarks.e2e",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Heavy end-to-end serving comparison: baseline (non-FlyDSL) versus "
            "ixaite (FlyDSL).\n"
            "\n"
            "By default this command starts two independent vllm serve processes "
            "sequentially.\n"
            "The runner injects VLLM_ILU_KERNEL_PROVIDER=baseline into the first "
            "process and\n"
            "VLLM_ILU_KERNEL_PROVIDER=ixaite into the second; users do not set "
            "that variable.\n"
            "Service startup defaults to baseline; this benchmark defaults to "
            "comparing both providers.\n"
            "Pass --provider baseline or --provider ixaite to run one phase.\n"
            "Every provider and workload receives its own untimed warmup requests.\n"
            "\n"
            "This is separate from `python -m tests.ixaite.benchmarks`, the L2 per-op "
            "correctness/performance gate."
        ),
        epilog=(
            "examples:\n"
            "  python -m tests.ixaite.benchmarks.e2e --list-models\n"
            "  python -m tests.ixaite.benchmarks.e2e --model qwen35_9b "
            "--model-path /models/Qwen3.5-9B\n"
            "  python -m tests.ixaite.benchmarks.e2e --model qwen35_9b "
            "--model-path /models/Qwen3.5-9B --device 0 --dry-run\n"
            "  python -m tests.ixaite.benchmarks.e2e --model qwen35_9b "
            "--model-path /models/Qwen3.5-9B --provider ixaite\n"
            "  python -m tests.ixaite.benchmarks.e2e --model qwen36_27b_w8a8_tp2 "
            "--model-path /models/Qwen3.6-27B-W8A8 --provider ixaite\n"
            "\n"
            "outputs:\n"
            "  manifest.json records the invocation and environment overrides.\n"
            "  compare.json and metrics.csv contain renderer-neutral metrics.\n"
            "  phases/<provider>/ contains server logs and raw vllm bench JSON.\n"
            "  HTML is intentionally not generated.\n"
        ),
    )
    discovery = parser.add_argument_group("model discovery")
    discovery.add_argument(
        "--list-models",
        action="store_true",
        help="list repository model profiles and exit without starting a server",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        help="repository model profile slug; see --list-models",
    )
    model.add_argument(
        "--model-path",
        type=Path,
        help="local model weights/config directory (never stored in a profile)",
    )
    model.add_argument(
        "--workload",
        action="append",
        dest="workloads",
        help="run only this workload id; repeat to select multiple workloads",
    )

    serving = parser.add_argument_group("serving")
    serving.add_argument(
        "--provider",
        choices=("baseline", "ixaite"),
        default=None,
        help=(
            "run one provider; omit to compare baseline and ixaite in independent "
            "processes (service default: baseline)"
        ),
    )
    serving.add_argument(
        "--python",
        default=default_python(),
        help="Python interpreter used for vllm serve and vllm bench (default: current)",
    )
    serving.add_argument(
        "--host",
        default="127.0.0.1",
        help="local serve host (default: %(default)s)",
    )
    serving.add_argument(
        "--port",
        type=int,
        default=8000,
        help="local serve port reused sequentially by both providers (default: %(default)s)",
    )
    serving.add_argument(
        "--device",
        default=None,
        help=(
            "optional CUDA_VISIBLE_DEVICES value for child processes; when omitted, "
            "the current environment is inherited"
        ),
    )
    serving.add_argument(
        "--startup-timeout",
        type=int,
        default=None,
        help="override profile server readiness timeout in seconds",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run directory (default: tests/ixaite/benchmarks/e2e-runs/<UTC timestamp>-<model>)",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print child commands without starting servers or writing files",
    )
    return parser.parse_args(argv)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _validated_profile(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not args.model:
        raise ValueError("--model is required (or pass --list-models)")
    if args.model_path is None:
        raise ValueError("--model-path is required")
    if not args.model_path.is_dir():
        raise ValueError(f"--model-path is not a directory: {args.model_path}")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")

    profile = e2e_spec(args.model)
    workloads = profile.get("workloads", [])
    correctness = profile.get("correctness")
    if not isinstance(workloads, list):
        raise ValueError(f"model {args.model!r} e2e workloads must be a list")
    if not workloads and correctness is None:
        raise ValueError(
            f"model {args.model!r} has neither e2e workloads nor correctness probe"
        )
    seen_ids: set[str] = set()
    for item in workloads:
        if not isinstance(item, dict):
            raise ValueError(f"model {args.model!r} workload must be an object")
        workload_id = item.get("id")
        if not isinstance(workload_id, str) or not workload_id:
            raise ValueError(
                f"model {args.model!r} workload id must be a non-empty string"
            )
        if workload_id in seen_ids:
            raise ValueError(
                f"model {args.model!r} duplicate workload id {workload_id!r}"
            )
        seen_ids.add(workload_id)
        for key in ("input_len", "output_len", "num_prompts", "max_concurrency"):
            if key not in item:
                raise ValueError(
                    f"model {args.model!r} workload {workload_id!r} missing key {key!r}"
                )
        dataset = item.get("dataset", "random")
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"model {args.model!r} workload {workload_id!r} dataset must be one "
                f"of {sorted(SUPPORTED_DATASETS)}, got {dataset!r}"
            )
        if "mm" in item and not isinstance(item["mm"], dict):
            raise ValueError(
                f"model {args.model!r} workload {workload_id!r} mm must be an object"
            )
    if correctness is not None:
        if not isinstance(correctness, dict):
            raise ValueError(f"model {args.model!r} correctness must be an object")
        kind = correctness.get("kind")
        if kind not in CORRECTNESS_KINDS:
            raise ValueError(
                f"model {args.model!r} correctness kind must be one of "
                f"{sorted(CORRECTNESS_KINDS)}, got {kind!r}"
            )
        prompt = correctness.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                f"model {args.model!r} correctness prompt must be non-empty"
            )
        if kind == VIDEO_GREEDY_EXACT:
            bucket = correctness.get("video_bucket")
            if (
                not isinstance(bucket, list)
                or len(bucket) != 3
                or not all(isinstance(value, int) and value > 0 for value in bucket)
                or bucket[2] <= 1
            ):
                raise ValueError(
                    f"model {args.model!r} correctness video_bucket must be "
                    "[height, width, frames] with frames > 1"
                )
        else:
            # One request cannot disagree with itself, so a probe whose whole
            # job is to catch nondeterminism needs at least two.
            repeats = correctness.get("repeats")
            if not isinstance(repeats, int) or repeats < 2:
                raise ValueError(
                    f"model {args.model!r} correctness repeats must be an int >= 2"
                )
        if correctness.get("temperature") != 0:
            raise ValueError(f"model {args.model!r} correctness temperature must be 0")
        for key in ("max_tokens", "request_timeout_s"):
            if not isinstance(correctness.get(key), int) or correctness[key] <= 0:
                raise ValueError(
                    f"model {args.model!r} correctness {key} must be a positive int"
                )
        if not isinstance(correctness.get("random_seed"), int):
            raise ValueError(
                f"model {args.model!r} correctness random_seed must be an int"
            )
    if args.workloads:
        selected = set(args.workloads)
        known = {str(item["id"]) for item in workloads}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(
                f"unknown workload(s) for {args.model}: {unknown}; known: {sorted(known)}"
            )
        workloads = [item for item in workloads if item["id"] in selected]
    return profile, workloads


# Omission keeps comparison orchestration outside the two backend values.
def _providers(value: str | None) -> tuple[str, ...]:
    return ("baseline", "ixaite") if value is None else (value,)


def _default_out(model: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUT_ROOT / f"{stamp}-{model}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_models:
        for model in list_e2e_models():
            print(model)
        return 0

    try:
        profile, workloads = _validated_profile(args)
        providers = _providers(args.provider)
        output_dir = args.out or _default_out(args.model)
        serve_args = profile.get("serve_args", [])
        profile_env = profile.get("env", {})
        warmup_requests = int(profile.get("warmup_requests", 0))
        startup_timeout_s = (
            args.startup_timeout
            if args.startup_timeout is not None
            else int(profile.get("startup_timeout_s", 1800))
        )
        if warmup_requests < 0:
            raise ValueError("profile warmup_requests must be >= 0")
        if startup_timeout_s <= 0:
            raise ValueError("startup timeout must be > 0")
        if not isinstance(serve_args, list) or not all(
            isinstance(value, str) for value in serve_args
        ):
            raise ValueError("profile serve_args must be a list of strings")
        if not isinstance(profile_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in profile_env.items()
        ):
            raise ValueError("profile env must map strings to strings")

        dry_commands = command_lines(
            providers=providers,
            python=args.python,
            model=args.model,
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            serve_args=serve_args,
            warmup_requests=warmup_requests,
            workloads=workloads,
            output_dir=output_dir,
        )
        if args.dry_run:
            for label, command in dry_commands:
                provider = label.split()[0]
                print(f"{label} ({'='.join(['VLLM_ILU_KERNEL_PROVIDER', provider])})")
                print(f"  {shlex.join(command)}")
            correctness = profile.get("correctness")
            if correctness is not None:
                for provider in providers:
                    if correctness["kind"] == VIDEO_GREEDY_EXACT:
                        detail = f"shared random-mm video {correctness['video_bucket']}"
                    else:
                        detail = (
                            f"{correctness['repeats']}x on the graph server, "
                            "then the same on a --enforce-eager server"
                        )
                    print(
                        f"{provider} greedy POST "
                        f"http://{args.host}:{args.port}/v1/chat/completions "
                        f"({detail})"
                    )
            return 0

        if not _port_is_available(args.host, args.port):
            raise ValueError(f"{args.host}:{args.port} is already in use")

        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv if argv is None else ["tests.ixaite.benchmarks.e2e", *argv],
            "git_sha": _git_sha(),
            "host": platform.node(),
            "platform": platform.platform(),
            "model": {"slug": args.model, "path": str(args.model_path)},
            "providers": list(providers),
            "device": args.device or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "port": args.port,
            "warmup_requests": warmup_requests,
            "workloads": [item["id"] for item in workloads],
            "profile_env": profile_env,
        }
        write_json(output_dir / "manifest.json", manifest)

        correctness = profile.get("correctness")
        prepared_correctness = correctness
        if correctness is not None and correctness["kind"] == VIDEO_GREEDY_EXACT:
            prepared_correctness = prepare_video_correctness(correctness, output_dir)
        phases = {}
        for provider in providers:
            print(f"running {provider}: starting vllm serve")
            phases[provider] = run_provider(
                provider=provider,
                python=args.python,
                model=args.model,
                model_path=args.model_path,
                host=args.host,
                port=args.port,
                serve_args=serve_args,
                profile_env=profile_env,
                device=args.device,
                startup_timeout_s=startup_timeout_s,
                warmup_requests=warmup_requests,
                workloads=workloads,
                correctness=prepared_correctness,
                output_dir=output_dir,
            )

        correctness_result = check_greedy_exact(phases)
        compare = build_compare(
            model=args.model,
            model_path=args.model_path,
            phases=phases,
        )
        compare["correctness"] = correctness_result
        write_json(output_dir / "compare.json", compare)
        write_metrics_csv(output_dir / "metrics.csv", compare)
        summary_text = summary(compare)
        (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
        print(summary_text, end="")
        print(f"wrote {output_dir}")
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
