# SPDX-License-Identifier: Apache-2.0
"""Serve lifecycle and workload execution for end-to-end benchmarks."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_TEXT_DATASET = "random"
_MM_DATASET = "random-mm"
# Both synthesize their prompts in-process, so no phase needs an external
# dataset on disk. random-mm additionally attaches generated images, which is
# the only way a serve forward reaches the vision tower.
SUPPORTED_DATASETS = frozenset({_TEXT_DATASET, _MM_DATASET})

# The completions endpoint drops the sampled images, so an mm workload only
# reaches the vision tower over the chat endpoint.
_BACKENDS: dict[str, str] = {_TEXT_DATASET: "vllm", _MM_DATASET: "openai-chat"}

_MM_WORKLOAD_FLAGS: tuple[tuple[str, str], ...] = (
    ("base_items_per_request", "--random-mm-base-items-per-request"),
    ("limit_mm_per_prompt", "--random-mm-limit-mm-per-prompt"),
    ("bucket_config", "--random-mm-bucket-config"),
)

_PROVIDER_ENV = "VLLM_ILU_KERNEL_PROVIDER"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Correctness probes. ``video_greedy_exact`` compares the two providers to each
# other and needs the vision tower, so it only says anything on a profile that
# has one. ``tp_graph_vs_eager`` compares a provider to itself and needs only
# text, which is what lets it run on a tensor-parallel profile.
VIDEO_GREEDY_EXACT = "video_greedy_exact"
TP_GRAPH_VS_EAGER = "tp_graph_vs_eager"
CORRECTNESS_KINDS = frozenset({VIDEO_GREEDY_EXACT, TP_GRAPH_VS_EAGER})

# --enforce-eager overrides cudagraph_mode to NONE whatever the profile's
# --compilation-config asked for, so appending it is enough to get a reference
# that runs the collective on every forward instead of replaying it.
_EAGER_FLAG = "--enforce-eager"


def build_server_command(
    *,
    python: str,
    model_path: Path,
    model: str,
    host: str,
    port: int,
    serve_args: Sequence[str],
) -> list[str]:
    """Build the vLLM serve subprocess command."""
    for forbidden in ("--host", "--port"):
        if forbidden in serve_args:
            raise ValueError(f"profile serve_args must not override {forbidden}")
    return [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        model,
        *serve_args,
    ]


def workload_dataset(workload: dict[str, Any]) -> str:
    """Return the validated dataset name for one workload."""
    dataset = str(workload.get("dataset", _TEXT_DATASET))
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"workload dataset must be one of {sorted(SUPPORTED_DATASETS)}, "
            f"got {dataset!r}"
        )
    return dataset


def dataset_args(workload: dict[str, Any]) -> list[str]:
    """Return the dataset-selection args for one workload.

    Workloads default to the text ``random`` dataset. A ``random-mm`` workload
    reads its per-modality knobs from the workload's ``mm`` object; anything
    left out keeps the vLLM default.
    """
    dataset = workload_dataset(workload)
    args = ["--dataset-name", dataset]
    if dataset != _MM_DATASET:
        return args
    mm = workload.get("mm") or {}
    for key, flag in _MM_WORKLOAD_FLAGS:
        value = mm.get(key)
        if value is not None:
            args += [flag, str(value)]
    return args


def build_benchmark_command(
    *,
    python: str,
    model: str,
    tokenizer_path: Path,
    base_url: str,
    workload: dict[str, Any],
    warmup_requests: int,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    """Build one vLLM serving benchmark subprocess command."""
    dataset = workload_dataset(workload)
    command = [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        _BACKENDS[dataset],
        "--base-url",
        base_url,
        "--model",
        model,
        "--tokenizer",
        str(tokenizer_path),
        *dataset_args(workload),
        "--input-len",
        str(workload["input_len"]),
        "--output-len",
        str(workload["output_len"]),
        "--num-prompts",
        str(workload["num_prompts"]),
        "--max-concurrency",
        str(workload["max_concurrency"]),
        "--num-warmups",
        str(warmup_requests),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--disable-tqdm",
    ]
    # openai-chat still defaults to /v1/completions in vLLM bench serve.
    # Completions drop sampled images; the vision tower only runs on chat.
    if dataset == _MM_DATASET:
        command += ["--endpoint", _CHAT_COMPLETIONS_PATH]
    return command


def _wait_until_ready(
    process: subprocess.Popen[str], base_url: str, timeout_s: int
) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"{base_url}/v1/models"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vllm serve exited before ready (exit {return_code})")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError(f"vllm serve was not ready after {timeout_s}s: {url}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


@contextlib.contextmanager
def _serve(
    *,
    command: Sequence[str],
    env: dict[str, str],
    log_path: Path,
    base_url: str,
    startup_timeout_s: int,
):
    """Hold one vllm serve up for the duration of the block."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_until_ready(process, base_url, startup_timeout_s)
            yield
        finally:
            _stop_server(process)


def prepare_video_correctness(
    correctness: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Generate one deterministic random-mm clip shared by both phases."""
    from vllm.benchmarks.datasets import RandomMultiModalDataset

    bucket = tuple(int(value) for value in correctness["video_bucket"])
    height, width, num_frames = bucket
    dataset = RandomMultiModalDataset(random_seed=int(correctness["random_seed"]))
    video = dataset.generate_synthetic_video(width, height, num_frames)
    video_bytes = video["bytes"]
    video_path = output_dir / "correctness-video.mp4"
    video_path.write_bytes(video_bytes)
    return {
        **correctness,
        "video_bucket": list(bucket),
        "video_path": str(video_path),
        "video_sha256": hashlib.sha256(video_bytes).hexdigest(),
    }


def _message_content(correctness: dict[str, Any]) -> Any:
    """Chat content for one probe: the prompt, plus its clip where there is one."""
    if correctness["kind"] == VIDEO_GREEDY_EXACT:
        video_bytes = Path(correctness["video_path"]).read_bytes()
        video_url = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(
            "ascii"
        )
        return [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": correctness["prompt"]},
        ]
    return correctness["prompt"]


def _greedy_request_payload(
    *, model: str, correctness: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _message_content(correctness)}],
        "temperature": correctness["temperature"],
        "max_tokens": correctness["max_tokens"],
        "seed": correctness["random_seed"],
    }


def _post_greedy(*, base_url: str, payload: dict[str, Any], timeout_s: int) -> str:
    """Send one greedy request and return its output text."""
    request = urllib.request.Request(
        f"{base_url}{_CHAT_COMPLETIONS_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"greedy request failed with HTTP {error.code}: {body}"
        ) from error

    try:
        output_text = raw["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError(f"greedy response has no output text: {raw}") from error
    if not isinstance(output_text, str) or not output_text:
        raise RuntimeError("greedy response output text is empty")
    return output_text


def _run_greedy_correctness(
    *,
    base_url: str,
    model: str,
    correctness: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    """Send the profile's greedy request, repeated as the profile asks.

    Repeats are what make a replay-unsafe collective visible without a second
    server: identical serial requests at temperature 0 must give identical
    text, so a barrier that races the peer's payload shows up as divergence
    between repeats. Requests go out one at a time on purpose -- concurrent
    ones would land in different batch shapes, which changes reduction order
    for legitimate reasons.
    """
    payload = _greedy_request_payload(model=model, correctness=correctness)
    timeout_s = int(correctness["request_timeout_s"])
    outputs = [
        _post_greedy(base_url=base_url, payload=payload, timeout_s=timeout_s)
        for _ in range(int(correctness.get("repeats", 1)))
    ]

    result = {
        "kind": correctness["kind"],
        "prompt": correctness["prompt"],
        "temperature": correctness["temperature"],
        "max_tokens": correctness["max_tokens"],
        "random_seed": correctness["random_seed"],
        "outputs": outputs,
        "output_text": outputs[0],
        "provider_notes": correctness.get("provider_notes", {}),
    }
    if correctness["kind"] == VIDEO_GREEDY_EXACT:
        result["video_bucket"] = correctness["video_bucket"]
        result["video_sha256"] = correctness["video_sha256"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _check_video_greedy_exact(
    phases: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Require baseline and ixaite to emit the same greedy output."""
    if "baseline" not in phases or "ixaite" not in phases:
        return None
    baseline = phases["baseline"]["correctness"]
    ixaite = phases["ixaite"]["correctness"]
    if baseline["video_sha256"] != ixaite["video_sha256"]:
        raise RuntimeError("baseline and ixaite did not use the same video clip")
    if baseline["output_text"] != ixaite["output_text"]:
        raise RuntimeError(
            "greedy video output mismatch: "
            f"baseline={baseline['output_text']!r}, "
            f"ixaite={ixaite['output_text']!r}"
        )
    return {
        "kind": baseline["kind"],
        "matched": True,
        "video_sha256": baseline["video_sha256"],
        "output_text": baseline["output_text"],
    }


def _check_tp_graph_vs_eager(
    phases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require each provider to agree with itself, then with its eager self.

    Both halves name a single provider, which is the point: comparing the two
    providers to each other reports a mismatch without saying which one is
    wrong, and the non-FlyDSL side is the one called "baseline".
    """
    providers = {}
    for provider, phase in phases.items():
        result = phase["correctness"]
        outputs = result["outputs"]
        distinct = sorted(set(outputs))
        if len(distinct) > 1:
            raise RuntimeError(
                f"{provider} graph replay is not deterministic across "
                f"{len(outputs)} identical greedy requests, which a "
                f"replay-safe collective would be: {distinct}"
            )
        reference = result.get("eager_outputs")
        if reference is not None and sorted(set(reference)) != distinct:
            raise RuntimeError(
                f"{provider} graph output disagrees with its own eager output. "
                "A replay-unsafe collective is the likely cause; a decode batch "
                "padded to a captured size is the other one, and it would move "
                f"both providers. graph={distinct}, eager={sorted(set(reference))}"
            )
        providers[provider] = {
            "output_text": outputs[0],
            "repeats": len(outputs),
            "eager_compared": reference is not None,
        }
    return {
        "kind": TP_GRAPH_VS_EAGER,
        "matched": True,
        "providers": providers,
    }


def check_greedy_exact(phases: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Run whichever correctness probe the phases carry."""
    kinds = {
        phase["correctness"]["kind"]
        for phase in phases.values()
        if phase.get("correctness") is not None
    }
    if not kinds:
        return None
    if len(kinds) > 1:
        raise RuntimeError(f"phases disagree on the correctness kind: {sorted(kinds)}")
    if any(phase.get("correctness") is None for phase in phases.values()):
        raise RuntimeError("greedy correctness result is missing from one provider")
    if kinds == {TP_GRAPH_VS_EAGER}:
        return _check_tp_graph_vs_eager(phases)
    return _check_video_greedy_exact(phases)


# Give every service its own fixed backend while preserving user log settings.
def _phase_env(
    *, provider: str, profile_env: dict[str, str], device: str | None
) -> dict[str, str]:
    if provider not in {"baseline", "ixaite"}:
        raise ValueError(f"provider must be baseline|ixaite, got {provider!r}")
    env = os.environ.copy()
    env.update(profile_env)
    env[_PROVIDER_ENV] = provider
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env


def check_workload_result(workload: dict[str, Any], result: dict[str, Any]) -> None:
    """Reject truncated streams even when vllm bench reports HTTP success.

    These workloads use fixed output lengths and --ignore-eos, so every request
    must finish its complete token budget before the result can be accepted.
    """
    expected = {
        "completed": int(workload["num_prompts"]),
        "failed": 0,
        "total_output_tokens": int(workload["num_prompts"])
        * int(workload["output_len"]),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"workload {workload['id']!r}: {key}={result.get(key)!r}, "
                f"expected {value}; inspect the client and server logs"
            )


def run_provider(
    *,
    provider: str,
    python: str,
    model: str,
    model_path: Path,
    host: str,
    port: int,
    serve_args: Sequence[str],
    profile_env: dict[str, str],
    device: str | None,
    startup_timeout_s: int,
    warmup_requests: int,
    workloads: Sequence[dict[str, Any]],
    correctness: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Start one provider server, run all workloads, then stop it."""
    phase_dir = output_dir / "phases" / provider
    phase_dir.mkdir(parents=True, exist_ok=True)
    server_log = phase_dir / "server.log"
    base_url = f"http://{host}:{port}"
    server_command = build_server_command(
        python=python,
        model_path=model_path,
        model=model,
        host=host,
        port=port,
        serve_args=serve_args,
    )
    env = _phase_env(provider=provider, profile_env=profile_env, device=device)

    results: dict[str, Any] = {}
    correctness_result = None
    with _serve(
        command=server_command,
        env=env,
        log_path=server_log,
        base_url=base_url,
        startup_timeout_s=startup_timeout_s,
    ):
        for workload in workloads:
            workload_id = str(workload["id"])
            result_filename = f"{workload_id}.json"
            client_log = phase_dir / f"{workload_id}.log"
            command = build_benchmark_command(
                python=python,
                model=model,
                tokenizer_path=model_path,
                base_url=base_url,
                workload=workload,
                warmup_requests=warmup_requests,
                result_dir=phase_dir,
                result_filename=result_filename,
            )
            with client_log.open("w", encoding="utf-8") as client_output:
                try:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=client_output,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"{provider} workload {workload_id!r} failed; see {client_log}"
                    ) from error
            raw_path = phase_dir / result_filename
            results[workload_id] = json.loads(raw_path.read_text(encoding="utf-8"))
            check_workload_result(workload, results[workload_id])
        if correctness is not None:
            correctness_result = _run_greedy_correctness(
                base_url=base_url,
                model=model,
                correctness=correctness,
                result_path=phase_dir / "greedy.json",
            )

    if correctness_result is not None and correctness["kind"] == TP_GRAPH_VS_EAGER:
        # A second server for the same provider, cudagraphs off. The point is
        # to reduce on every forward instead of replaying a captured launch,
        # so the comparison is against a collective this provider is known to
        # run correctly rather than against the other provider.
        eager_log = phase_dir / "server-eager.log"
        with _serve(
            command=build_server_command(
                python=python,
                model_path=model_path,
                model=model,
                host=host,
                port=port,
                serve_args=[*serve_args, _EAGER_FLAG],
            ),
            env=env,
            log_path=eager_log,
            base_url=base_url,
            startup_timeout_s=startup_timeout_s,
        ):
            eager = _run_greedy_correctness(
                base_url=base_url,
                model=model,
                correctness=correctness,
                result_path=phase_dir / "greedy-eager.json",
            )
        correctness_result["eager_outputs"] = eager["outputs"]

    return {
        "provider": provider,
        "server_command": server_command,
        "workloads": results,
        "correctness": correctness_result,
    }


def command_lines(
    *,
    providers: Sequence[str],
    python: str,
    model: str,
    model_path: Path,
    host: str,
    port: int,
    serve_args: Sequence[str],
    warmup_requests: int,
    workloads: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[tuple[str, list[str]]]:
    """Return all commands a dry run would execute."""
    lines: list[tuple[str, list[str]]] = []
    base_url = f"http://{host}:{port}"
    for provider in providers:
        lines.append(
            (
                f"{provider} serve",
                build_server_command(
                    python=python,
                    model_path=model_path,
                    model=model,
                    host=host,
                    port=port,
                    serve_args=serve_args,
                ),
            )
        )
        for workload in workloads:
            workload_id = str(workload["id"])
            lines.append(
                (
                    f"{provider} {workload_id}",
                    build_benchmark_command(
                        python=python,
                        model=model,
                        tokenizer_path=model_path,
                        base_url=base_url,
                        workload=workload,
                        warmup_requests=warmup_requests,
                        result_dir=output_dir / "phases" / provider,
                        result_filename=f"{workload_id}.json",
                    ),
                )
            )
    return lines


def default_python() -> str:
    """Return the interpreter running the e2e orchestrator."""
    return sys.executable
