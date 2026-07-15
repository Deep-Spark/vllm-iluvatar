# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar Worker dummy-batch patch."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_gpu_worker


def test_execute_dummy_batch_passes_profile_seq_lens(monkeypatch):
    """After _apply(), execute_dummy_batch must forward profile_seq_lens=1."""
    calls: list[dict] = []

    class FakeModelRunner:
        uniform_decode_query_len = 4

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    class FakeWorker:
        def __init__(self):
            self.model_runner = FakeModelRunner()

        def execute_dummy_batch(self) -> None:
            num_tokens = getattr(
                self.model_runner, "uniform_decode_query_len", 1
            )
            self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker_module.Worker = FakeWorker

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker")
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module
    )

    patch_gpu_worker._apply()

    worker = FakeWorker()
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 4,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }


def test_execute_dummy_batch_uses_fallback_query_len(monkeypatch):
    """When model_runner lacks uniform_decode_query_len, default to 1."""
    calls: list[dict] = []

    class BareModelRunner:
        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    class FakeWorker:
        def __init__(self):
            self.model_runner = BareModelRunner()

        def execute_dummy_batch(self) -> None:
            num_tokens = getattr(
                self.model_runner, "uniform_decode_query_len", 1
            )
            self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker_module.Worker = FakeWorker

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker")
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module
    )

    patch_gpu_worker._apply()

    worker = FakeWorker()
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 1,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }
