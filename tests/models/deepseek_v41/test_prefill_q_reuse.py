"""Fused Q buffers must follow views of the current batch, never stale pointers."""
from types import SimpleNamespace

import pytest
import torch

import vllm_iluvatar.models.deepseek_v41.attention as attention


@pytest.fixture
def reuse(monkeypatch):
    monkeypatch.setattr(attention, "_PREFILL_FUSED_Q", True)
    obj = attention.IluDeepseekV41Attention.__new__(attention.IluDeepseekV41Attention)
    torch.nn.Module.__init__(obj)
    q = torch.zeros(256, 64, 512, dtype=torch.bfloat16)
    obj._q_fp8_pt = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    obj._q_scale_pt = torch.empty(256)
    obj._q_quant_workspace_pt = torch.empty(256)
    obj.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty(1, 128, 512, dtype=torch.float8_e4m3fn)
    )
    obj.scale, obj.kv_scale = 512**-0.5, 1.0
    obj.attn_sink = torch.zeros(64)
    obj._tile_window = lambda x: x
    obj._prefill_q_source = (q.data_ptr(), 256, q.stride(0) * q.element_size())
    calls = []

    def quant(query, out, scale, workspace, **kwargs):
        calls.append(("quant", out.data_ptr(), scale.data_ptr()))

    def sparse(out, scale, *args, **kwargs):
        calls.append(("attention", out.data_ptr(), scale.data_ptr()))

    monkeypatch.setattr(attention.ixfops, "quant_mla_q_fp8_pt", quant)
    monkeypatch.setattr(attention.ixfops, "sparse_mla_fp8_pt", sparse)
    return obj, q, calls


@pytest.mark.parametrize("begin,end", [(0, 256), (0, 64), (37, 142), (192, 256)])
def test_current_batch_slices(reuse, begin, end):
    obj, q, calls = reuse
    obj._sparse_fp8_pt_mla(q[begin:end], None, None, None, None, None, None)
    assert calls == [
        ("attention", obj._q_fp8_pt[begin:end].data_ptr(),
         obj._q_scale_pt[begin:end].data_ptr())
    ]


@pytest.mark.parametrize("reason", ["unrelated", "stale", "disabled"])
def test_requantizes_when_reuse_is_invalid(reuse, monkeypatch, reason):
    obj, q, calls = reuse
    if reason == "unrelated":
        q = q.clone()
    elif reason == "stale":
        obj._fused_qnorm_rope_kv_insert(q, None, None, None)
        assert obj._prefill_q_source is None
    else:
        monkeypatch.setattr(attention, "_PREFILL_FUSED_Q", False)
    obj._sparse_fp8_pt_mla(q[:64], None, None, None, None, None, None)
    assert [call[0] for call in calls] == ["quant", "attention"]
