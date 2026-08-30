# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import pytest

from vllm_iluvatar.ilu_util.onchip_check import (
    OnchipChecker,
    OnchipCheckError,
    OnchipResult,
    OnchipStatus,
)


def _result(status: OnchipStatus) -> OnchipResult:
    return OnchipResult(
        status=status,
        errors=(1, 0, 0, 0) if status is OnchipStatus.FAULT else (0, 0, 0, 0),
        first_level=0,
        first_level_name="SLB(shared)",
        first_block=0,
        first_thread=0,
        first_word=0,
        first_expected=1,
        first_got=0,
        first_phase=0,
        coverage_incomplete=status is OnchipStatus.INCOMPLETE,
        error="cuda failed" if status is OnchipStatus.CUDA_ERROR else "",
    )


class _FakeNative:
    def __init__(self):
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        return {"status": OnchipStatus.OK, "errors": [0, 0, 0, 0]}

    def config(self):
        return {}

    def free(self):
        self.calls.append("free")


@pytest.fixture
def configured_env(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_ONCHIP_CHECK", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK", 0, raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_ONCHIP_CHECK_KMSG", False, raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES", 2, raising=False)
    monkeypatch.setattr(
        envs, "VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS", 0.0, raising=False
    )
    monkeypatch.setattr(
        envs,
        "VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL",
        2,
        raising=False,
    )


def test_disabled_start_does_not_load_native(monkeypatch):
    monkeypatch.setattr(OnchipChecker, "enabled", staticmethod(lambda: False))
    checker = OnchipChecker()

    checker.start(0)

    assert checker._native is None


def test_start_retries_incomplete(configured_env, monkeypatch):
    checker = OnchipChecker()
    native = _FakeNative()
    checker._native = native
    results = iter(
        [
            _result(OnchipStatus.INCOMPLETE),
            _result(OnchipStatus.INCOMPLETE),
            _result(OnchipStatus.OK),
        ]
    )
    monkeypatch.setattr(checker, "_run", lambda: next(results))

    checker.start(3)

    assert checker._initialized
    assert native.calls[0] == (
        "init",
        {"device": 3, "level_mask": 0, "kmsg": False},
    )


def test_start_fails_after_incomplete_retries(configured_env, monkeypatch):
    checker = OnchipChecker()
    checker._native = _FakeNative()
    monkeypatch.setattr(checker, "_run", lambda: _result(OnchipStatus.INCOMPLETE))

    with pytest.raises(OnchipCheckError, match="after 3 attempt"):
        checker.start(0)


def test_start_fault_fails_without_retry(configured_env, monkeypatch):
    checker = OnchipChecker()
    native = _FakeNative()
    checker._native = native
    calls = 0

    def run_fault():
        nonlocal calls
        calls += 1
        return _result(OnchipStatus.FAULT)

    monkeypatch.setattr(checker, "_run", run_fault)

    with pytest.raises(OnchipCheckError, match="Startup on-chip check failed"):
        checker.start(0)

    assert calls == 1
    assert native.calls[-1] == "free"


def test_fault_before_batch_raises(configured_env, monkeypatch):
    checker = OnchipChecker()
    checker._initialized = True
    checker._device_index = 1
    monkeypatch.setattr(checker, "_run", lambda: _result(OnchipStatus.FAULT))

    with pytest.raises(OnchipCheckError, match="failed before batch"):
        checker.before_batch()


def test_incomplete_before_batch_continues_and_rate_limits(
    configured_env, monkeypatch, caplog
):
    checker = OnchipChecker()
    checker._initialized = True
    checker._device_index = 1
    monkeypatch.setattr(checker, "_run", lambda: _result(OnchipStatus.INCOMPLETE))

    with caplog.at_level(logging.WARNING):
        checker.before_batch()
        checker.before_batch()
        checker.before_batch()

    messages = [
        record.message
        for record in caplog.records
        if "continuing this batch" in record.message
    ]
    assert len(messages) == 2


def test_free_is_idempotent(configured_env):
    checker = OnchipChecker()
    native = _FakeNative()
    checker._native = native
    checker._initialized = True

    checker.free()
    checker.free()

    assert native.calls == ["free"]
