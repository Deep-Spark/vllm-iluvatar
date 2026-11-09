# SPDX-License-Identifier: Apache-2.0
"""Run opt-in TP2 mixed-serving diagnostics and replay real FA/GDN inputs.

Performance is measured separately by ``python -m tests.ixaite.benchmarks.e2e`` using the
uninstrumented chunked-prefill profile. This command records correctness and
execution evidence, never performance claims from instrumented requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from tests.ixaite.benchmarks.core.model_catalog import e2e_spec
from tests.ixaite.benchmarks.e2e.runner import (
    _phase_env,
    _stop_server,
    _wait_until_ready,
    build_server_command,
)

MODEL = "qwen36_27b_w8a8_tp2_chunked_prefill"


# Incomplete final JSONL lines can be observed while a worker appends a step.
def read_steps(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# Use a streaming response to know the short prompt actually completed prefill.
def request_tokens(base_url, tokens, output_len, started):
    payload = dict(
        model=MODEL,
        prompt=tokens,
        max_tokens=output_len,
        temperature=0,
        seed=0,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    text = []
    usage = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                text.append(choice.get("text", ""))
                started.set()
            usage = event.get("usage") or usage
    if usage is None or usage["completion_tokens"] != output_len:
        raise RuntimeError(f"incomplete request: {usage}")
    return dict(prompt_tokens=tokens, output="".join(text), usage=usage)


# Fail unless the real scheduler recorded both overlap and multi-step prefill.
def check_steps(rows):
    mixed = [r["step"] for r in rows if r["prefills"] and r["decodes"]]
    if not mixed:
        raise AssertionError("no actual mixed decode/prefill step recorded")
    grouped = {}
    for row in rows:
        for req in row["requests"]:
            grouped.setdefault(req["request_id"], []).append((row["step"], req))
    long_requests = []
    for request_id, steps in grouped.items():
        if steps[0][1]["prompt_tokens"] <= 512:
            continue
        prefill = [r for _, r in steps if r["computed"] < r["prompt_tokens"]]
        if len(prefill) < 2 or not any(r["computed"] > 0 for r in prefill):
            raise AssertionError(f"{request_id} did not continue a prefill")
        if not any(r["computed"] >= r["prompt_tokens"] for _, r in steps):
            raise AssertionError(f"{request_id} never entered decode")
        long_requests.append(
            dict(
                request_id=request_id,
                chunks=len(prefill),
                prompt_tokens=prefill[0]["prompt_tokens"],
            )
        )
    if not long_requests:
        raise AssertionError("no long requests recorded")
    return dict(mixed_steps=mixed, long_requests=long_requests)


# Capture real serving inputs on each rank; worker selection is test-only.
def run_phase(args, provider, prompts):
    spec = e2e_spec(MODEL)
    phase = args.out / provider
    phase.mkdir(parents=True, exist_ok=False)
    serve_args = [
        *spec["serve_args"],
        "--worker-cls",
        "tests.ixaite.backend_isolation.chunked_worker.ChunkedPrefillWorker",
    ]
    command = build_server_command(
        python=sys.executable,
        model_path=args.model_path,
        model=MODEL,
        host="127.0.0.1",
        port=args.port,
        serve_args=serve_args,
    )
    env = _phase_env(provider=provider, profile_env=spec["env"], device=args.device)
    env["SWCOMP3266_AUDIT_DIR"] = str(phase)
    base_url = f"http://127.0.0.1:{args.port}"
    import ixaite

    versions = {}
    for name, directory in (
        ("plugin", Path(__file__).resolve().parents[3]),
        ("ixaite", Path(ixaite.__file__).resolve().parents[2]),
    ):
        sha = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(["git", "-C", str(directory), "diff", "HEAD"])
        versions[name] = dict(
            sha=sha, tracked_diff_sha256=hashlib.sha256(diff).hexdigest()
        )
    communication = {
        name: os.environ[name]
        for name in (
            "VLLM_DISABLE_PYNCCL",
            "VLLM_FORCE_NCCL_COMM",
            "GLOO_SOCKET_IFNAME",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
        )
        if name in os.environ
    }
    (phase / "manifest.json").write_text(
        json.dumps(
            dict(
                command=command,
                versions=versions,
                communication=communication,
                profile=spec,
                device=args.device,
            ),
            indent=2,
        )
    )
    outputs = []
    with (phase / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_until_ready(process, base_url, spec["startup_timeout_s"])
            for prompt in prompts:
                started = threading.Event()
                before = len(read_steps(phase / "rank0/steps.jsonl"))
                with ThreadPoolExecutor(max_workers=2) as executor:
                    short = executor.submit(
                        request_tokens, base_url, prompt[:128], 1024, started
                    )
                    deadline = time.monotonic() + 300
                    # First token alone proves prefill completion, so also wait
                    # for a recorded decode step before admitting the long input.
                    while not (
                        started.is_set()
                        and any(
                            r["decodes"]
                            for r in read_steps(phase / "rank0/steps.jsonl")[before:]
                        )
                    ):
                        if short.done():
                            short.result()
                            raise RuntimeError("short request completed before overlap")
                        if time.monotonic() > deadline:
                            raise TimeoutError("short request did not reach decode")
                        time.sleep(0.02)
                    long = executor.submit(
                        request_tokens, base_url, prompt, 128, threading.Event()
                    )
                    outputs.extend([short.result(), long.result()])
        finally:
            _stop_server(process)
    (phase / "requests.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2)
    )
    ranks = {
        str(rank): check_steps(read_steps(phase / f"rank{rank}/steps.jsonl"))
        for rank in (0, 1)
    }
    return dict(command=command, ranks=ranks)


# Use the baseline FLA math on compact states, preserving value-major layout.
@torch.inference_mode()
def reference_gdn(q, k, v, g, beta, state, mask, cu, scale):
    from vllm_iluvatar.third_party.flash_linear_attention.ops.chunk import (
        chunk_gated_delta_rule,
    )

    initial = state.clone()
    initial[~mask.bool()] = 0
    out, final = chunk_gated_delta_rule(
        *(tensor.unsqueeze(0) for tensor in (q, k, v, g, beta)),
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        scale=scale,
    )
    state.copy_(final)
    return out.squeeze(0)


# Reuse the baseline ABI on the exact captured tensors; every saved step is checked.
def replay(directory, *, fail_fast=True):
    import ixaite
    import ixformer.contrib.vllm_flash_attn as fa
    import torch

    ixaite.initialize("ivcore11")
    results = []
    for path in sorted(directory.glob("*/rank*/*.pt")):
        data = torch.load(path, weights_only=True)
        provider = path.parent.parent.name
        errors = []

        def compare(actual, expected, tolerance, errors=errors):
            try:
                torch.testing.assert_close(
                    actual.cpu(), expected, rtol=tolerance, atol=tolerance
                )
                assert torch.isfinite(actual).all()
            except AssertionError as error:
                if fail_fast:
                    raise
                errors.append(str(error))

        if path.stem.endswith("gdn"):
            q, k, v, g, beta, state, mask, cu = [
                data[n].cuda()
                for n in ("q", "k", "v", "g", "beta", "state", "mask", "cu")
            ]
            out = reference_gdn(q, k, v, g, beta, state, mask, cu, data["scale"])
            compare(out, data["out"], 2e-2)
            compare(state, data["final_state"], 3e-2)
            state_error = float((state.cpu() - data["final_state"]).abs().max())
        else:
            kwargs = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in data["kwargs"].items()
            }
            out = torch.empty_like(kwargs["q"])
            kwargs["out"] = out
            fn = (
                fa.flash_attn_varlen_func
                if provider == "ixaite"
                else ixaite.flash_attn_varlen_func
            )
            fn(**kwargs)
            compare(out, data["out"], 3e-2)
            state_error = None
        results.append(
            dict(
                file=str(path.relative_to(directory)),
                max_abs_output_error=float((out.cpu() - data["out"]).abs().max()),
                max_abs_state_error=state_error,
                passed=not errors,
                errors=errors,
            )
        )
    if not results:
        raise AssertionError("no captured inputs to replay")
    return results


# Inspect kernels launched inside the annotated actual-provider call, excluding
# diagnostic copies and the independent offline reference replay.
def check_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [
        e
        for e in events
        if e.get("name") in ("SWCOMP3266_GDN_PRODUCTION", "SWCOMP3266_FA_PRODUCTION")
    ]
    correlations = set()
    forbidden = []
    for scope in scopes:
        for e in events:
            if e.get("tid") == scope.get("tid") and scope["ts"] <= e.get(
                "ts", -1
            ) < scope["ts"] + scope.get("dur", 0):
                if e.get("name", "").startswith(
                    ("aten::index_select", "aten::index_copy", "aten::_to_copy")
                ):
                    # State is rank 4; output_indices legitimately scatter the
                    # rank-3 output. Shape recording distinguishes the two.
                    dims = e.get("args", {}).get("Input Dims", [])
                    if not dims or len(dims[0]) == 4:
                        forbidden.append(e["name"])
                if "correlation" in e.get("args", {}):
                    correlations.add(e["args"]["correlation"])
    kernels = sorted(
        {
            e["name"]
            for e in events
            if e.get("cat") == "kernel"
            and e.get("args", {}).get("correlation") in correlations
        }
    )
    if not scopes or not kernels:
        raise AssertionError(f"no attributable kernel events in {path}")
    if forbidden:
        raise AssertionError(f"state gather/scatter in production: {forbidden}")
    if not any("gdn_" in n for n in kernels) or not any(
        "flash_attn" in n for n in kernels
    ):
        raise AssertionError(f"expected ixaite GDN/FA symbols absent: {kernels}")
    if any("merge_attn" in n or "cuinfer" in n or "ixformer" in n for n in kernels):
        raise AssertionError(f"unexpected baseline/cascade kernel: {kernels}")
    return kernels


# Keep paths/devices in the invocation rather than hard-coding local deployment.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--port", type=int, default=18367)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.replay_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        paragraph = tokenizer.encode(
            "The following passage explains how plants use sunlight, water and carbon dioxide to grow. Summarize the process clearly. ",
            add_special_tokens=False,
        )
        prompts = [(paragraph * (n // len(paragraph) + 1))[:n] for n in (2048, 2305)]
        phases = {
            provider: run_phase(args, provider, prompts)
            for provider in ("baseline", "ixaite")
        }
        (args.out / "schedule-summary.json").write_text(json.dumps(phases, indent=2))
    results = replay(args.out, fail_fast=False)
    (args.out / "replay.json").write_text(json.dumps(results, indent=2))
    traces = {
        str(p.relative_to(args.out)): check_trace(p)
        for p in args.out.glob("ixaite/rank*/mixed-trace.json")
    }
    if len(traces) != 2:
        raise AssertionError("missing TP rank profiler trace")
    (args.out / "kernel-summary.json").write_text(json.dumps(traces, indent=2))
    failures = [r for r in results if not r["passed"]]
    if failures:
        raise AssertionError(
            f"{len(failures)} captured comparisons failed; see replay.json"
        )
    print(
        f"passed {len(results)} captured FA/GDN comparisons; both ranks recorded mixed steps and ixaite kernels"
    )


if __name__ == "__main__":
    main()
