# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VLLM_ILU_CUDA_MEM_FRACTION env parsing and worker capping."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.worker.gpu_worker import Worker

import vllm_iluvatar.envs as ilu_envs
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker


def test_cuda_mem_fraction_env_unset_and_empty(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_CUDA_MEM_FRACTION", raising=False)
    assert ilu_envs.VLLM_ILU_CUDA_MEM_FRACTION is None
    assert not ilu_envs.is_set("VLLM_ILU_CUDA_MEM_FRACTION")

    monkeypatch.setenv("VLLM_ILU_CUDA_MEM_FRACTION", "   ")
    assert ilu_envs.VLLM_ILU_CUDA_MEM_FRACTION is None
    assert ilu_envs.is_set("VLLM_ILU_CUDA_MEM_FRACTION")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.95", 0.95), ("0.955", 0.955), ("1", 1.0), (" 0.85 ", 0.85)],
)
def test_cuda_mem_fraction_env_valid(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_ILU_CUDA_MEM_FRACTION", raw)
    assert ilu_envs.VLLM_ILU_CUDA_MEM_FRACTION == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.05", "nan", "inf", "abc"])
def test_cuda_mem_fraction_env_invalid(monkeypatch, raw):
    monkeypatch.setenv("VLLM_ILU_CUDA_MEM_FRACTION", raw)
    with pytest.raises(ValueError, match="VLLM_ILU_CUDA_MEM_FRACTION"):
        _ = ilu_envs.VLLM_ILU_CUDA_MEM_FRACTION


def test_ilu_worker_init_device_applies_memory_fraction(monkeypatch):
    ilu_envs.apply()
    calls: list[tuple[float, torch.device]] = []

    def fake_super_init(self):
        self.device = torch.device("cuda:3")

    monkeypatch.setattr(Worker, "init_device", fake_super_init)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda frac, dev: calls.append((frac, dev)),
    )

    worker = IluWorker.__new__(IluWorker)

    monkeypatch.delenv("VLLM_ILU_CUDA_MEM_FRACTION", raising=False)
    worker.init_device()
    assert calls == []

    monkeypatch.setenv("VLLM_ILU_CUDA_MEM_FRACTION", "0.95")
    worker.init_device()
    assert calls == [(0.95, torch.device("cuda:3"))]
