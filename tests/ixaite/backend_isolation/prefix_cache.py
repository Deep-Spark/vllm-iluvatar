# SPDX-License-Identifier: Apache-2.0
"""Run real prefix-cache requests, then replay captured all-mode states against baseline math."""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from tests.ixaite.benchmarks.core.model_catalog import e2e_spec
from tests.ixaite.benchmarks.e2e.runner import (
    _phase_env,
    _stop_server,
    _wait_until_ready,
    build_server_command,
)

MODEL = "qwen36_27b_w8a8_tp2_prefix_cache"


# Batch prompts in a single request so the scheduler can observe their common prefix.
def request(base, prompts, output_len):
    payload = dict(
        model=MODEL,
        prompt=prompts,
        max_tokens=output_len,
        temperature=0,
        seed=0,
        ignore_eos=True,
        logprobs=1,
    )
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


# Isolate server lifetime and inherited provider settings for each comparison phase.
def run(args, provider, mode, prompts):
    phase = args.out / f"{provider}-{mode}"
    phase.mkdir(parents=True, exist_ok=False)
    spec = e2e_spec("qwen36_27b_w8a8_tp2_chunked_prefill")
    serve = list(spec["serve_args"])
    serve.remove("--no-enable-prefix-caching")
    if mode == "align":
        # The baseline's physical Mamba block exceeds 512 tokens. Align mode
        # requires a serving budget large enough to schedule a complete block.
        serve[serve.index("--max-num-batched-tokens") + 1] = "2048"
    if mode == "none":
        serve += ["--no-enable-prefix-caching"]
    else:
        serve += ["--enable-prefix-caching", "--mamba-cache-mode", mode]
    serve += [
        "--no-disable-cascade-attn",
        "--max-num-seqs",
        "32",
        "--worker-cls",
        "tests.ixaite.backend_isolation.prefix_worker.PrefixCacheWorker",
    ]
    command = build_server_command(
        python=sys.executable,
        model_path=args.model_path,
        model=MODEL,
        host="127.0.0.1",
        port=args.port,
        serve_args=serve,
    )
    env = _phase_env(provider=provider, profile_env=spec["env"], device=args.device)
    env["SWCOMP3265_AUDIT_DIR"] = str(phase)
    import ixaite

    versions = {}
    for name, root in (
        ("plugin", Path(__file__).resolve().parents[3]),
        ("ixaite", Path(ixaite.__file__).resolve().parents[2]),
    ):
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        (phase / f"{name}.patch").write_bytes(diff)
        files = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ]
            )
            .decode()
            .split("\0")
        )
        versions[name] = dict(
            sha=sha,
            diff_sha256=hashlib.sha256(diff).hexdigest(),
            added_files={
                f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                for f in files
                if f and (root / f).is_file()
            },
        )
    (phase / "manifest.json").write_text(
        json.dumps(
            dict(command=command, provider=provider, mode=mode, versions=versions),
            indent=2,
        )
    )
    base = f"http://127.0.0.1:{args.port}"
    with (phase / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_until_ready(process, base, 3600)
            warm = request(base, [prompts[0][:-17]], 1)
            outputs = request(base, prompts, args.output_len)
            (phase / "responses.json").write_text(
                json.dumps(dict(warm=warm, batch=outputs), indent=2)
            )
        finally:
            _stop_server(process)
    return phase


# Replay each saved boundary using the independent baseline FLA path.
def replay_gdn(path):
    import torch

    from tests.ixaite.backend_isolation.chunked_prefill import reference_gdn

    data = torch.load(path, weights_only=True)
    state = data["state"].cuda()
    initial = state.clone()
    reference = state.clone()
    outputs = torch.empty_like(data["out"], device="cuda")
    q, k, v, g, beta = (data[name].cuda() for name in ("q", "k", "v", "g", "beta"))
    for seq in range(len(data["computed"])):
        lo, hi = map(int, data["cu"][seq : seq + 2])
        computed = int(data["computed"][seq])
        block = data["block_size"]
        source = int(data["source"][seq])
        if computed > 0 and computed % block == 0:
            torch.testing.assert_close(
                data["final_state"][source], data["state"][source], rtol=0, atol=0
            )
        h = initial[source : source + 1].clone()
        has = bool(data["mask"][seq])
        cursor = lo
        while cursor < hi:
            count = min(block - computed % block, hi - cursor)
            bounds = torch.tensor([0, count], dtype=torch.int32)
            out = reference_gdn(
                *(tensor[cursor : cursor + count] for tensor in (q, k, v, g, beta)),
                h,
                torch.tensor([has], device="cuda", dtype=torch.int32),
                bounds.cuda(),
                q.shape[-1] ** -0.5,
            )
            outputs[cursor : cursor + count] = out
            computed += count
            cursor += count
            has = True
            reference[int(data["table"][seq, (computed - 1) // block])] = h[0]
    torch.testing.assert_close(outputs.cpu(), data["out"], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        reference.cpu(), data["final_state"], atol=3e-2, rtol=3e-2
    )
    return dict(file=str(path), passed=True)


# Cascades must be present on both ranks and agree with baseline on identical captured KV.
def replay(phase):
    import torch

    from tests.ixaite.backend_isolation.chunked_prefill import read_steps
    from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import cascade_attention

    reports = []
    for rank in range(2):
        directory = phase / f"rank{rank}"
        rows = read_steps(directory / "steps.jsonl")
        hits = [r for row in rows for r in row["new"] if r["computed"] > 0]
        if len(hits) < 8:
            raise AssertionError(
                f"rank{rank}: fewer than eight real cache hits: {hits}"
            )
        for path in sorted(directory.glob("*-gdn.pt")):
            reports.append(replay_gdn(path))
        data = torch.load(directory / "cascade.pt", weights_only=True)
        kw = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in data["kwargs"].items()
        }
        out = torch.empty_like(data["out"], device="cuda")
        cascade_attention(
            out, data["query"].cuda(), data["key"].cuda(), data["value"].cuda(), **kw
        )
        torch.testing.assert_close(out.cpu(), data["out"], atol=3e-2, rtol=3e-2)
        trace = json.loads((directory / "cascade-trace.json").read_text())
        kernels = [e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel"]
        if not any("merge_attn_states" in name for name in kernels) or not any(
            "flash_attn" in name for name in kernels
        ):
            raise AssertionError(f"rank{rank}: missing ixaite cascade kernels")
        if any("cuinfer" in name or "ixformer" in name for name in kernels):
            raise AssertionError(f"rank{rank}: cascade fallback")
        gdn_trace = json.loads((directory / "gdn-trace.json").read_text())
        gdn_kernels = [
            e["name"] for e in gdn_trace["traceEvents"] if e.get("cat") == "kernel"
        ]
        if not any("gdn_wy_fwd_h" in name and "_apc" in name for name in gdn_kernels):
            raise AssertionError(f"rank{rank}: missing ixaite APC state kernel")
        if any("cuinfer" in name or "ixformer" in name for name in gdn_kernels):
            raise AssertionError(f"rank{rank}: GDN fallback")
        reports.append(
            dict(
                rank=rank,
                hits=len(hits),
                cascade=True,
                kernels=kernels,
                gdn_kernels=gdn_kernels,
            )
        )
    (phase / "replay.json").write_text(json.dumps(reports, indent=2))


# Deployment paths stay in command-line arguments; all is never relabeled align.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--port", type=int, default=18368)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--phase",
        choices=["ixaite-all", "baseline-none", "baseline-align"],
        default="ixaite-all",
    )
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    provider, mode = args.phase.split("-")
    if not args.replay_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        paragraph = tokenizer.encode(
            "Explain how plants use sunlight and water to grow. ",
            add_special_tokens=False,
        )
        prefix = (paragraph * (2048 // len(paragraph) + 1))[:2048]
        tails = [
            tokenizer.encode(
                f" Summarize the process for student {i}. ", add_special_tokens=False
            )
            for i in range(8)
        ]
        prompts = [prefix + (tail * 4)[:17] for tail in tails]
        (args.out / "prompts.json").write_text(json.dumps(prompts))
        run(args, provider, mode, prompts)
    if mode == "all":
        replay(args.out / args.phase)


if __name__ == "__main__":
    main()
