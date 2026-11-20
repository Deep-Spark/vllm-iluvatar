# SPDX-License-Identifier: Apache-2.0
"""Run the prefill launcher up to the vLLM executable and inspect the deployment."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "glm5next"


@pytest.fixture
def launch(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "compressed-tensors"}})
    )
    vllm = tmp_path / "vllm"
    vllm.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps([os.environ['VLLM_PP_LAYER_PARTITION'], sys.argv[1:]]))\n"
    )
    vllm.chmod(0o755)

    def run(platform, devices, **overrides):
        env = {
            "PATH": str(tmp_path) + os.pathsep + os.defpath,
            "MODEL_PATH": str(tmp_path),
            "DEVICES": devices,
            "PYTHON_BIN": sys.executable,
            **overrides,
        }
        return subprocess.run(
            ["bash", EXAMPLES / f"run_{platform}_prefill.sh", "--serve"],
            capture_output=True,
            text=True,
            env=env,
        )

    return run


@pytest.mark.parametrize(
    ("platform", "devices", "overrides", "partition", "budget"),
    [
        ("cq", "0,1,2,3", {}, "13,12,11,9", 2304),
        ("cq", "6,7,4,5", {"PP_SIZE": "4"}, "13,12,11,9", 2304),
        ("cq", "0,1,2,3,6,7,4,5", {}, "7,6,6,6,6,5,5,4", 2304),
        ("mr", "0,1,2,3,4,5,6,7", {}, "7,6,6,6,5,6,5,4", 2048),
        (
            "cq",
            "0,1,2,3",
            {"VLLM_PP_LAYER_PARTITION": "12,12,12,9"},
            "12,12,12,9",
            2304,
        ),
    ],
)
def test_prefill_presets(launch, platform, devices, overrides, partition, budget):
    result = launch(platform, devices, **overrides)
    assert result.returncode == 0, result.stderr
    actual, args = json.loads(result.stdout.splitlines()[-1])
    assert actual == partition
    assert args[args.index("--pipeline-parallel-size") + 1] == str(
        len(partition.split(","))
    )
    assert args[args.index("--max-num-batched-tokens") + 1] == str(budget)
    assert args[args.index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
    assert (
        json.loads(args[args.index("--speculative-config") + 1])[
            "num_speculative_tokens"
        ]
        == 1
    )
    if platform == "cq":
        assert "--quantization" not in args
    else:
        assert args[args.index("--quantization") + 1] == "compressed-tensors"


def test_cq_launchers_load_original_weights_by_default_and_honor_override(
    launch, tmp_path
):
    result = launch("cq", "0,1,2,3,6,7,4,5", QUANTIZATION="fp8")
    assert result.returncode == 0, result.stderr
    _, args = json.loads(result.stdout.splitlines()[-1])
    assert args[args.index("--quantization") + 1] == "fp8"

    base_env = {
        "PATH": os.defpath,
        "MODEL_PATH": str(tmp_path),
        "DEVICES": "0,1,2,3,6,7,4,5",
    }
    decode_default = subprocess.run(
        ["bash", EXAMPLES / "run_cq_decode.sh", "--print-command"],
        capture_output=True,
        text=True,
        env=base_env,
    )
    assert decode_default.returncode == 0, decode_default.stderr
    assert "--quantization" not in decode_default.stdout.split()

    decode_override = subprocess.run(
        ["bash", EXAMPLES / "run_cq_decode.sh", "--print-command"],
        capture_output=True,
        text=True,
        env={**base_env, "QUANTIZATION": "ilu_f4f8"},
    )
    assert decode_override.returncode == 0, decode_override.stderr
    decode_tokens = decode_override.stdout.split()
    assert decode_tokens[decode_tokens.index("--quantization") + 1] == "ilu_f4f8"


@pytest.mark.parametrize(
    ("devices", "overrides", "message"),
    [
        ("0,1,1,3", {}, "distinct"),
        ("0,1,2,3", {"PP_SIZE": "8"}, "DEVICES"),
        ("0,1,2,3", {"VLLM_PP_LAYER_PARTITION": "12,12,12,8"}, "totaling 45"),
        ("0,1,2,3", {"VLLM_PP_LAYER_PARTITION": "1,14,15,15"}, "target MLA"),
    ],
)
def test_invalid_topology_stops_before_vllm(launch, devices, overrides, message):
    result = launch("cq", devices, PYTHONOPTIMIZE="1", **overrides)
    assert result.returncode != 0
    assert message in result.stderr
