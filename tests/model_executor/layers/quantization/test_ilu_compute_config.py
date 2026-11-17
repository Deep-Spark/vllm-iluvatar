# SPDX-License-Identifier: Apache-2.0
"""Execution preferences must not confuse architecture defaults with overrides."""

from types import SimpleNamespace

import pytest

from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.ilu_util import arch


@pytest.mark.parametrize(
    "device,setting,expected",
    [
        ("ivcore11", "auto", "w8a8_int8"),
        ("ivcore30", "auto", "fp8"),
        ("ivcore30", "w8a8_int8", "w8a8_int8"),
        ("ivcore11", "w8a8_int8", "w8a8_int8"),
    ],
)
def test_linear_selection(monkeypatch, device, setting, expected):
    monkeypatch.setattr(arch, "cuda_arch", lambda: device)
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": setting}}}
        ),
    )
    assert ilu_compute.fp8_linear_backend() == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"linear": {"fp8_backend": "int9"}},
        {"linear": {"backend": "fp8"}},
        {"schema_version": True},
    ],
)
def test_invalid_compute_config(monkeypatch, raw):
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(additional_config={"ilu_compute": raw}),
    )
    with pytest.raises(ValueError):
        ilu_compute.fp8_linear_backend()


def test_explicit_fp8_rejects_ivcore11(monkeypatch):
    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": "fp8"}}}
        ),
    )
    with pytest.raises(ValueError, match="not supported"):
        ilu_compute.fp8_linear_backend()
