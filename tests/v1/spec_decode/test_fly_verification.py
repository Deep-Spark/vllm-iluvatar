"""FLy integrates ixformer verification with vLLM's ordinary recovery stage."""

import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize(
    "value,enabled", [(None, False), ("0", False), ("1", True), ("true", True)]
)
def test_fly_gate_is_registered(monkeypatch, value, enabled):
    import vllm.envs as community_envs

    from vllm_iluvatar import envs

    key = "VLLM_ILU_ENABLE_FLY_REJECTION"
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    envs.apply()
    assert community_envs.environment_variables[key]() is enabled
    assert envs.fly_rejection_enabled() is enabled


def test_v1_runner_cannot_silently_ignore_fly():
    from types import SimpleNamespace

    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(rejection_sample_method="fly"),
        use_v2_model_runner=False,
    )
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        IluPlatform.check_and_update_config(cfg)


def test_standard_sampler_import_does_not_load_fly_verifier():
    code = """
import sys
from types import SimpleNamespace
from vllm_iluvatar.ilu_patch.patch_fly import register
register()
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
assert RejectionSampler.__module__.endswith('.fly_sampler')
spec = SimpleNamespace(rejection_sample_method='standard', num_speculative_tokens=3,
                       enable_adaptive_verification=False)
sampler = RejectionSampler(None, spec, 'cpu')
assert sampler.fly_window_size == 0
assert 'vllm_iluvatar.v1.worker.gpu.spec_decode.fly_rejection' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "VLLM_ILU_ENABLE_FLY_REJECTION": "1"},
        check=True,
    )


@pytest.mark.parametrize("floor,expected", [(0.01, 3), (0.9, 0)])
def test_greedy_recovery_after_fly_window(floor, expected, cuda_device):
    # vLLM workers install process-wide import hooks on startup. Exercise a fresh worker interpreter.
    subprocess.run(
        [sys.executable, __file__, str(floor), str(expected), cuda_device], check=True
    )


def run_worker_case(floor, expected, device):
    from vllm.platforms import current_platform

    assert current_platform.device_type == "cuda"
    from vllm_iluvatar.v1.worker.gpu.spec_decode.fly_rejection import rejection_sample

    # Draft 2 is not target argmax. Two following native accepts may confirm it.
    target = torch.full((4, 17), -1.0, device=device)
    target[:, 1] = 1.5
    draft = torch.zeros((1, 3, 17), device=device)
    tokens = torch.tensor([0, 2, 1, 1], device=device)
    idx = torch.zeros(1, device=device, dtype=torch.int32)
    rows = torch.arange(4, device=device, dtype=torch.int32)
    sampled, n = rejection_sample(
        target,
        draft,
        tokens,
        torch.tensor([0, 4], device=device, dtype=torch.int32),
        rows,
        idx,
        torch.zeros_like(rows),
        rows,
        torch.zeros(1, device=device),
        torch.tensor([7], device=device),
        3,
        fly_window_size=2,
        fly_entropy_threshold=0.1,
        fly_min_target_probability=floor,
    )
    # The vLLM recovery stage appends one target/bonus token.
    assert n.item() == expected + 1
    assert sampled[0, : n.item()].tolist() == ([2, 1, 1, 1] if expected == 3 else [1])


if __name__ == "__main__":
    run_worker_case(float(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
