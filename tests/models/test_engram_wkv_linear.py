"""Check opt-in WKV backend selection and fallback without requiring a GPU."""
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import engram


@pytest.mark.parametrize('enabled,m,dtype,use_ixinfer', [
    (None, 16, torch.bfloat16, False),
    ('0', 16, torch.bfloat16, False),
    ('1', 0, torch.bfloat16, False),
    ('1', 1, torch.bfloat16, True),
    ('1', 16, torch.bfloat16, True),
    ('1', 32, torch.bfloat16, True),
    ('1', 33, torch.bfloat16, False),
    ('1', 16, torch.float32, False),
])
def test_wkv_backend(monkeypatch, enabled, m, dtype, use_ixinfer):
    if enabled is None:
        monkeypatch.delenv('VLLM_ILU_ENGRAM_LINEAR', raising=False)
    else:
        monkeypatch.setenv('VLLM_ILU_ENGRAM_LINEAR', enabled)
    rows = torch.arange(m * 8, dtype=torch.float32).reshape(m, 2, 4).to(dtype)
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype)
    calls = []
    def linear(x, w, *, backend):
        calls.append(backend)
        assert w is weight
        return x @ w.t()
    monkeypatch.setattr(engram.ixfops, 'linear', linear)
    layer = SimpleNamespace(wkv_scale=None, wkv_weight=weight,
                            _gate=lambda hidden, kv: kv)
    result = engram.IluEngram.apply(layer, torch.empty(m, 3), rows)
    assert calls == (['ixinfer'] if use_ixinfer else [])
    torch.testing.assert_close(result, rows.flatten(-2) @ weight.t())
