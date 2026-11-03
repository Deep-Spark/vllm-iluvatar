# SPDX-License-Identifier: Apache-2.0
"""Keep anonymous Engram fallbacks out of bulk host registration."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.engram import EngramEmbedding


def embedding(regions):
    return SimpleNamespace(
        weight=torch.ones((4, 8), dtype=torch.uint8),
        scale=torch.ones((4, 1), dtype=torch.float32),
        _regions=regions,
        _dev_weight=None,
        _dev_scale=None,
        _uva_failed=False,
    )


@pytest.mark.parametrize("anonymous_plane", [0, 1])
def test_anonymous_fallback_never_registers_host_pages(monkeypatch, anonymous_plane):
    regions = [object(), object()]
    regions[anonymous_plane] = None
    module = embedding(regions)
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "engram_register_host", lambda tensor: calls.append(tensor)
    )
    assert EngramEmbedding.ensure_device_view(module) is False
    assert calls == []
    assert module._uva_failed
    assert module._dev_weight is module._dev_scale is None
    # Repeated lookup stays on the existing bounded CPU gather fallback.
    assert EngramEmbedding.ensure_device_view(module) is False
    assert calls == []


def test_shared_locked_planes_keep_zero_copy_registration(monkeypatch):
    module = embedding([object(), object()])
    views = [object(), object()]
    calls = []

    def register(tensor):
        calls.append(tensor)
        return views[len(calls) - 1]

    monkeypatch.setattr(torch.ops._C, "engram_register_host", register)
    assert EngramEmbedding.ensure_device_view(module) is True
    assert module._dev_weight is views[0] and module._dev_scale is views[1]
    assert len(calls) == 2
    assert EngramEmbedding.ensure_device_view(module) is True
    assert len(calls) == 2
