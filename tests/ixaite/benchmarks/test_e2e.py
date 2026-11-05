# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path

import pytest

from tests.ixaite.benchmarks.e2e.report import build_compare, write_metrics_csv
from tests.ixaite.benchmarks.e2e.run import main
from tests.ixaite.benchmarks.e2e.runner import (
    _greedy_request_payload,
    _phase_env,
    build_benchmark_command,
    check_greedy_exact,
)


def test_dry_run_prints_both_provider_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert (
        main(
            [
                "--model",
                "qwen35_9b",
                "--model-path",
                str(model_path),
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "VLLM_ILU_KERNEL_PROVIDER=baseline" in output
    assert "VLLM_ILU_KERNEL_PROVIDER=ixaite" in output
    assert "--num-warmups 10" in output
    assert not (tmp_path / "out").exists()


# The CLI names only actual backends; comparison is the omitted option.
@pytest.mark.parametrize("provider", ["auto", "both"])
def test_cli_rejects_old_provider_values(provider):
    with pytest.raises(SystemExit) as error:
        main(["--provider", provider])
    assert error.value.code == 2


# Exercise the real runner boundary: each phase receives a separate service env.
@pytest.mark.parametrize(
    "provider,expected",
    [
        (None, ["baseline", "ixaite"]),
        ("baseline", ["baseline"]),
        ("ixaite", ["ixaite"]),
    ],
)
def test_cli_launches_independent_provider_services(
    tmp_path, monkeypatch, provider, expected
):
    import tests.ixaite.benchmarks.e2e.run as cli
    import tests.ixaite.benchmarks.e2e.runner as runner

    model_path = tmp_path / "model"
    model_path.mkdir()
    workload = {
        "id": "probe",
        "input_len": 1,
        "output_len": 1,
        "num_prompts": 1,
        "max_concurrency": 1,
    }
    monkeypatch.setattr(cli, "e2e_spec", lambda model: {"workloads": [workload]})
    services = []
    monkeypatch.setattr(cli, "_port_is_available", lambda *_: True)
    monkeypatch.setattr(cli, "_git_sha", lambda: "test")

    # Keep HTTP and GPU work out of this test while retaining phase orchestration.
    @contextlib.contextmanager
    def serve(**kwargs):
        services.append(kwargs)
        yield

    # Match bench's result-file interface so the full runner validates outputs.
    def benchmark(command, **kwargs):
        result_dir = Path(command[command.index("--result-dir") + 1])
        filename = command[command.index("--result-filename") + 1]
        (result_dir / filename).write_text(
            json.dumps({"completed": 1, "failed": 0, "total_output_tokens": 1})
        )

    monkeypatch.setattr(runner, "_serve", serve)
    monkeypatch.setattr(runner.subprocess, "run", benchmark)
    arguments = [
        "--model",
        "probe",
        "--model-path",
        str(model_path),
        "--out",
        str(tmp_path / "out"),
    ]
    if provider is not None:
        arguments.extend(["--provider", provider])
    assert main(arguments) == 0
    assert [
        service["env"]["VLLM_ILU_KERNEL_PROVIDER"] for service in services
    ] == expected
    assert len({id(service["env"]) for service in services}) == len(expected)
    assert set(
        json.loads((tmp_path / "out" / "compare.json").read_text())["phases"]
    ) == set(expected)


def test_video_dry_run_has_greedy_probe_without_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert (
        main(
            [
                "--model",
                "qwen35_9b_video",
                "--model-path",
                str(model_path),
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count("vllm.entrypoints.cli.main serve") == 2
    assert "baseline greedy POST" in output
    assert "ixaite greedy POST" in output
    assert "shared random-mm video [448, 448, 4]" in output
    assert " bench serve " not in output
    assert not (tmp_path / "out").exists()


def test_tp2_dry_run_prints_graph_then_eager_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert (
        main(
            [
                "--model",
                "qwen35_9b_tp2",
                "--model-path",
                str(model_path),
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "--tensor-parallel-size 2" in output
    assert "4x on the graph server" in output
    assert "--enforce-eager" in output
    assert "shared random-mm video" not in output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("model", "tp"),
    (
        ("qwen36_27b_w8a8_tp2", "2"),
        ("qwen36_27b_w8a8_tp4", "4"),
        ("qwen36_27b", "4"),
    ),
)
def test_qwen36_27b_dry_run_prints_workloads_and_graph_eager_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model: str, tp: str
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert (
        main(
            [
                "--model",
                model,
                "--model-path",
                str(model_path),
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert f"--tensor-parallel-size {tp}" in output
    assert "4x on the graph server" in output
    assert "--enforce-eager" in output
    assert "VLLM_ILU_KERNEL_PROVIDER=ixaite" in output
    assert not (tmp_path / "out").exists()


def test_greedy_video_payload_uses_the_prepared_clip(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"same-video")
    payload = _greedy_request_payload(
        model="qwen35_9b_video",
        correctness={
            "kind": "video_greedy_exact",
            "video_path": str(video_path),
            "prompt": "Describe this video in one sentence.",
            "temperature": 0,
            "max_tokens": 32,
            "random_seed": 0,
        },
    )

    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "video_url"
    assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert content[1] == {
        "type": "text",
        "text": "Describe this video in one sentence.",
    }
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 32


def test_greedy_exact_accepts_equal_outputs() -> None:
    probe = {
        "kind": "video_greedy_exact",
        "video_sha256": "same",
        "output_text": "A short synthetic video.",
    }
    result = check_greedy_exact(
        {
            "baseline": {"correctness": probe},
            "ixaite": {"correctness": probe.copy()},
        }
    )
    assert result == {
        "kind": "video_greedy_exact",
        "matched": True,
        "video_sha256": "same",
        "output_text": "A short synthetic video.",
    }


def test_greedy_exact_rejects_different_outputs() -> None:
    with pytest.raises(RuntimeError, match="greedy video output mismatch"):
        check_greedy_exact(
            {
                "baseline": {
                    "correctness": {
                        "kind": "video_greedy_exact",
                        "video_sha256": "same",
                        "output_text": "baseline",
                    }
                },
                "ixaite": {
                    "correctness": {
                        "kind": "video_greedy_exact",
                        "video_sha256": "same",
                        "output_text": "ixaite",
                    }
                },
            }
        )


def _tp_probe(outputs: list[str], eager: list[str] | None = None) -> dict[str, object]:
    probe: dict[str, object] = {
        "kind": "tp_graph_vs_eager",
        "outputs": outputs,
        "output_text": outputs[0],
    }
    if eager is not None:
        probe["eager_outputs"] = eager
    return probe


def test_tp_probe_accepts_a_provider_that_agrees_with_itself() -> None:
    result = check_greedy_exact(
        {
            "baseline": {"correctness": _tp_probe(["one, two"] * 3, ["one, two"])},
            "ixaite": {"correctness": _tp_probe(["one, two"] * 3, ["one, two"])},
        }
    )
    assert result["matched"] is True
    assert result["providers"]["ixaite"] == {
        "output_text": "one, two",
        "repeats": 3,
        "eager_compared": True,
    }


def test_tp_probe_rejects_nondeterministic_replay_by_provider_name() -> None:
    # The provider has to be named. Comparing the two providers to each other
    # is what this probe replaces, precisely because it cannot say which side
    # is wrong.
    with pytest.raises(RuntimeError, match="baseline graph replay is not"):
        check_greedy_exact(
            {
                "baseline": {"correctness": _tp_probe(["one, two", "one, three"])},
                "ixaite": {"correctness": _tp_probe(["one, two", "one, two"])},
            }
        )


def test_tp_probe_rejects_graph_disagreeing_with_its_own_eager_run() -> None:
    with pytest.raises(RuntimeError, match="ixaite graph output disagrees"):
        check_greedy_exact(
            {"ixaite": {"correctness": _tp_probe(["graph"] * 2, ["eager"])}}
        )


def test_tp_probe_runs_without_an_eager_reference() -> None:
    result = check_greedy_exact({"ixaite": {"correctness": _tp_probe(["same"] * 2)}})
    assert result["providers"]["ixaite"]["eager_compared"] is False


def test_correctness_kinds_may_not_be_mixed_across_phases() -> None:
    with pytest.raises(RuntimeError, match="disagree on the correctness kind"):
        check_greedy_exact(
            {
                "baseline": {
                    "correctness": {
                        "kind": "video_greedy_exact",
                        "video_sha256": "same",
                        "output_text": "text",
                    }
                },
                "ixaite": {"correctness": _tp_probe(["text"] * 2)},
            }
        )


def test_benchmark_command_uses_vllm_warmup_and_json_output(tmp_path: Path) -> None:
    command = build_benchmark_command(
        python="/venv/bin/python",
        model="qwen35_9b",
        tokenizer_path=Path("/models/Qwen3.5-9B"),
        base_url="http://127.0.0.1:8000",
        workload={
            "id": "decode",
            "input_len": 128,
            "output_len": 128,
            "num_prompts": 40,
            "max_concurrency": 1,
        },
        warmup_requests=10,
        result_dir=tmp_path,
        result_filename="decode.json",
    )
    assert command[:4] == [
        "/venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
    ]
    assert command[command.index("--num-warmups") + 1] == "10"
    assert command[command.index("--tokenizer") + 1] == "/models/Qwen3.5-9B"
    assert "--save-result" in command
    assert command[command.index("--result-filename") + 1] == "decode.json"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert command[command.index("--backend") + 1] == "vllm"


def test_benchmark_command_uses_openai_chat_for_random_mm(tmp_path: Path) -> None:
    command = build_benchmark_command(
        python="/venv/bin/python",
        model="qwen35_9b_vision",
        tokenizer_path=Path("/models/Qwen3.5-9B"),
        base_url="http://127.0.0.1:8000",
        workload={
            "id": "bs1_img896_128",
            "dataset": "random-mm",
            "input_len": 128,
            "output_len": 128,
            "num_prompts": 20,
            "max_concurrency": 1,
            "mm": {
                "base_items_per_request": 1,
                "limit_mm_per_prompt": '{"image": 1, "video": 0}',
                "bucket_config": "{(896, 896, 1): 1.0}",
            },
        },
        warmup_requests=10,
        result_dir=tmp_path,
        result_filename="bs1_img896_128.json",
    )
    assert command[command.index("--backend") + 1] == "openai-chat"
    assert command[command.index("--endpoint") + 1] == "/v1/chat/completions"
    assert command[command.index("--dataset-name") + 1] == "random-mm"
    assert command[command.index("--random-mm-bucket-config") + 1] == (
        "{(896, 896, 1): 1.0}"
    )


def test_compare_report_speedups_and_csv(tmp_path: Path) -> None:
    def phase(output_throughput: float, ttft_ms: float) -> dict[str, object]:
        return {
            "workloads": {
                "decode": {
                    "completed": 4,
                    "failed": 0,
                    "duration": 1.0,
                    "output_throughput": output_throughput,
                    "median_ttft_ms": ttft_ms,
                }
            },
        }

    compare = build_compare(
        model="qwen35_9b",
        model_path=Path("/models/Qwen3.5-9B"),
        phases={
            "baseline": phase(100.0, 20.0),
            "ixaite": phase(110.0, 10.0),
        },
    )
    assert compare["schema_version"] == 2
    assert "routes" not in compare["phases"]["ixaite"]
    assert compare["speedups"]["decode"]["output_throughput"] == 1.1
    assert compare["speedups"]["decode"]["median_ttft_ms"] == 2.0

    csv_path = tmp_path / "metrics.csv"
    write_metrics_csv(csv_path, compare)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["provider"] for row in rows} == {"baseline", "ixaite"}
    output_rows = [row for row in rows if row["metric"] == "output_throughput"]
    assert {row["speedup"] for row in output_rows} == {"1.1"}


def test_single_provider_report_keeps_metrics_without_speedups() -> None:
    phase = {
        "workloads": {
            "decode": {
                "completed": 4,
                "failed": 0,
                "duration": 1.0,
                "output_throughput": 100.0,
            }
        },
    }

    # A single provider phase records raw metrics;
    # no baseline exists from which a speedup could be calculated.
    compare = build_compare(
        model="qwen36_27b_w8a8_tp2",
        model_path=Path("/models/Qwen3.6-27B-W8A8"),
        phases={"ixaite": phase},
    )

    assert set(compare["phases"]) == {"ixaite"}
    assert compare["phases"]["ixaite"]["workloads"]["decode"]["metrics"]
    assert compare["speedups"] == {}


# Benchmark launch does not change logging or hide conflicting legacy controls.
@pytest.mark.parametrize("inherited", [None, "WARNING", "DEBUG"])
def test_phase_env_preserves_log_level_and_legacy_controls(monkeypatch, inherited):
    if inherited is None:
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    else:
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", inherited)
    monkeypatch.setenv("VLLM_ILU_IXAITE_OP_ALLOWLIST", "linear")
    env = _phase_env(provider="ixaite", profile_env={}, device="2")
    assert env.get("VLLM_LOGGING_LEVEL") == inherited
    assert env["VLLM_ILU_IXAITE_OP_ALLOWLIST"] == "linear"
    assert env["VLLM_ILU_KERNEL_PROVIDER"] == "ixaite"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


# Invalid programmatic callers cannot bypass the same two-value contract.
@pytest.mark.parametrize("provider", ["auto", "both"])
def test_phase_env_rejects_old_provider_values(provider):
    with pytest.raises(ValueError, match="baseline\\|ixaite"):
        _phase_env(provider=provider, profile_env={}, device=None)
