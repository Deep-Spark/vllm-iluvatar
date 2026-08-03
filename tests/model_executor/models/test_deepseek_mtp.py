# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar DeepSeek MTP OOT registration."""

from __future__ import annotations

from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.interfaces import SupportsPP

from vllm_iluvatar.model_executor.models import deepseek_mtp


def test_deepseek_mtp_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    deepseek_mtp.register_model()

    assert calls == [
        (
            "DeepSeekMTPModel",
            "vllm_iluvatar.model_executor.models.deepseek_mtp:IluDeepSeekMTP",
        )
    ]


def test_ilu_deepseek_mtp_subclasses_community_and_supports_pp():
    assert issubclass(deepseek_mtp.IluDeepSeekMTP, DeepSeekMTP)
    assert SupportsPP in deepseek_mtp.IluDeepSeekMTP.__mro__
    assert deepseek_mtp.IluDeepSeekMTP.supports_pp is True
