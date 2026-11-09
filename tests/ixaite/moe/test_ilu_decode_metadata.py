# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Iluvatar MoE decode-vs-capture dispatch helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_iluvatar.backends.ixaite.moe.decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)


@pytest.mark.parametrize(
    ("attn_metadata", "expected"),
    [
        (None, False),
        ({}, False),
        (SimpleNamespace(num_prefills=0, num_decodes=2), True),
        (SimpleNamespace(num_prefills=1, num_decodes=0), False),
        (SimpleNamespace(num_prefills=0, num_decodes=0, num_spec_decodes=3), True),
    ],
)
def test_is_only_decode_eager(monkeypatch, attn_metadata, expected):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.moe.decode_metadata._stream_is_capturing",
        lambda: False,
    )
    assert _is_only_decode(attn_metadata) is expected


def test_is_only_decode_none_metadata_during_capture_uses_gemv(monkeypatch):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.moe.decode_metadata._stream_is_capturing",
        lambda: True,
    )
    assert _is_only_decode(None) is True
    assert _is_only_decode({}) is True


def test_is_only_decode_capture_fallback_disabled_keeps_general_path(monkeypatch):
    """Capture-safe GEMM callers capture the general path.

    With ``capture_fallback=False`` an empty-metadata capture dummy run must
    not be routed to the decode GEMV path: its shape-keyed persistent
    workspaces would be baked into every piecewise capture size.
    """
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.moe.decode_metadata._stream_is_capturing",
        lambda: True,
    )
    assert _is_only_decode(None, capture_fallback=False) is False
    assert _is_only_decode({}, capture_fallback=False) is False
    # Reported batch counters still win over the fallback flag.
    decode = SimpleNamespace(num_prefills=0, num_decodes=2)
    assert _is_only_decode(decode, capture_fallback=False) is True


def test_is_only_decode_does_not_override_prefill_counters_while_capturing(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.moe.decode_metadata._stream_is_capturing",
        lambda: True,
    )
    prefill = SimpleNamespace(num_prefills=4, num_decodes=0)
    assert _is_only_decode(prefill) is False
    with pytest.raises(RuntimeError, match="not CUDA-graph safe"):
        _ensure_cuda_graph_safe_dispatch(False, prefill)
