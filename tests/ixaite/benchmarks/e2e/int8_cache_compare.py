# SPDX-License-Identifier: Apache-2.0
"""Compare native cache vs shared cache while keeping Attention on baseline."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from tests.ixaite.benchmarks.e2e.runner import (
    _serve,
    build_benchmark_command,
    build_server_command,
    check_workload_result,
)

PHASES = (
    "calibrate",
    "KINT8-baseline",
    "KINT8-shared",
    "KVINT8-baseline",
    "KVINT8-shared",
)
WORKLOADS = (
    dict(
        id="bs1_128_64", input_len=128, output_len=64, num_prompts=12, max_concurrency=1
    ),
    dict(
        id="bs8_128_64", input_len=128, output_len=64, num_prompts=24, max_concurrency=8
    ),
    dict(
        id="bs1_1024_64",
        input_len=1024,
        output_len=64,
        num_prompts=12,
        max_concurrency=1,
    ),
)
MODEL_NAME = "int8-cache-compare"


def _complete(base_url, prompt, tokens):
    """Keep greedy token IDs so cache/layout errors cannot hide behind HTTP 200."""
    payload = dict(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=tokens,
        temperature=0,
        seed=3910,
        ignore_eos=True,
        return_token_ids=True,
    )
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    for choice in result["choices"]:
        assert len(choice["token_ids"]) == tokens, result
        assert choice["finish_reason"] == "length", result
    return result


def run_phase(args):
    """Reuse the repository serve/benchmark lifecycle, with an explicit test worker."""
    import torch

    phase_dir = args.output / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    scale_path = args.output / "v_scale.pt"
    calibration = args.phase == "calibrate"
    shared = args.phase.endswith("-shared")
    mode = "" if calibration else args.phase.split("-")[0]
    if calibration and scale_path.exists():
        raise ValueError(f"calibration file already exists: {scale_path}")
    if mode == "KVINT8" and not scale_path.is_file():
        raise ValueError("run the calibrate phase before KVINT8")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=args.device,
        VLLM_ILU_KERNEL_PROVIDER="baseline",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ILU_FLASH_ATTN_QUANT_TYPE=mode,
        VLLM_ILU_FLASH_ATTN_V_SCALE_PATH=str(scale_path)
        if calibration or mode == "KVINT8"
        else "",
    )
    env.pop("VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH", None)
    env.pop("VLLM_ILU_IXAITE_OP_ALLOWLIST", None)
    serve_args = [
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "1024",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--compilation-config",
        '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}',
    ]
    if calibration or args.eager:
        serve_args.append("--enforce-eager")
    if shared:
        serve_args += [
            "--worker-cls",
            "tests.ixaite.benchmarks.e2e.int8_cache_bridge.CacheReadWorker",
        ]
    command = build_server_command(
        python=sys.executable,
        model_path=args.model_path,
        model=MODEL_NAME,
        host="127.0.0.1",
        port=args.port,
        serve_args=serve_args,
    )
    manifest = dict(
        phase=args.phase,
        command=command,
        torch=torch.__version__,
        model_path=str(args.model_path),
        gpu=args.device,
        tensor_parallel_size=args.tp,
        cache="ixaite independent writer" if shared else "native writer/fused decode",
        attention="baseline ixformer Q quantization + prefill + decode",
        cuda_graph=not (calibration or args.eager),
        env={
            k: env[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_ILU_KERNEL_PROVIDER",
                "VLLM_ILU_FLASH_ATTN_QUANT_TYPE",
                "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH",
            )
        },
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    (phase_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base_url = f"http://127.0.0.1:{args.port}"
    measurements = {}
    with _serve(
        command=command,
        env=env,
        log_path=phase_dir / "server.log",
        base_url=base_url,
        startup_timeout_s=900,
    ):
        if calibration:
            response = _complete(
                base_url,
                [
                    "Explain what a key-value cache stores during language model generation.",
                    "Solve this step by step: a train travels 240 km in 3 hours. What is its average speed?",
                    "Write a short Python function that adds two numbers and explain it.",
                ],
                128,
            )
            (phase_dir / "calibration_responses.json").write_text(
                json.dumps(response, indent=2) + "\n"
            )
        else:
            probes = [
                _complete(base_url, prompt, 32)
                for prompt in (
                    "User: What is 2 + 2?\nAssistant:",
                    "User: Explain why KV cache is useful.\nAssistant:",
                    "User: What is 2 + 2?\nAssistant:",
                )
            ]
            assert (
                probes[0]["choices"][0]["token_ids"]
                == probes[2]["choices"][0]["token_ids"]
            )
            (phase_dir / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")
            for workload in WORKLOADS:
                for repeat in range(2):
                    label = f"{workload['id']}-r{repeat}"
                    filename = label + ".json"
                    client = build_benchmark_command(
                        python=sys.executable,
                        model=MODEL_NAME,
                        tokenizer_path=args.model_path,
                        base_url=base_url,
                        workload=workload,
                        warmup_requests=2,
                        result_dir=phase_dir,
                        result_filename=filename,
                    )
                    with (phase_dir / (label + ".log")).open("w") as output:
                        subprocess.run(
                            client,
                            env=env,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    raw = json.loads((phase_dir / filename).read_text())
                    check_workload_result(workload, raw)
                    measurements[label] = raw
                    print(
                        json.dumps(
                            {
                                "phase": args.phase,
                                "workload": label,
                                "output_tok_s": raw["output_throughput"],
                                "mean_ttft_ms": raw["mean_ttft_ms"],
                                "mean_tpot_ms": raw["mean_tpot_ms"],
                            }
                        ),
                        flush=True,
                    )
    if calibration:
        scales = torch.load(scale_path, map_location="cpu", weights_only=True)
        config = json.loads((args.model_path / "config.json").read_text())
        assert set(scales) == {str(i) for i in range(config["num_hidden_layers"])}
        expected = (
            config["num_key_value_heads"],
            config["hidden_size"] // config["num_attention_heads"],
        )
        for scale in scales.values():
            assert tuple(scale.shape) == expected
            assert torch.isfinite(scale).all() and (scale > 0).all()
        measurements = dict(
            layers=len(scales),
            shape=expected,
            v_scale_sha256=hashlib.sha256(scale_path.read_bytes()).hexdigest(),
        )
    (phase_dir / "result.json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "status": "PASS"}), flush=True)


def main():
    """Run one resumable phase; all phases use the same fixed cache capacity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=38107)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.tp <= 0 or len(args.device.split(",")) != args.tp:
        parser.error("--device must name one GPU per TP rank")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    args.model_path, args.output = args.model_path.resolve(), args.output.resolve()
    run_phase(args)


if __name__ == "__main__":
    main()
