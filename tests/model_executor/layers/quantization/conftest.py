# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the quantization tests."""

from __future__ import annotations

import pytest

from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime import (
    config as ilu_runtime_config,
)


@pytest.fixture(autouse=True)
def allow_ilu_quant_on_any_arch(monkeypatch):
    """Let ``additional_config.ilu_quant`` parse and dispatch on CPU/ivcore11.

    The runtime path is ivcore30-only and rejects other architectures when the
    payload is resolved. Most tests here exercise parsing and method selection,
    not the device, so they bypass that gate; tests of the gate itself restore
    ``capabilities.require_ivcore30``.
    """
    monkeypatch.setattr(ilu_runtime_config, "require_ivcore30", lambda feature: None)
