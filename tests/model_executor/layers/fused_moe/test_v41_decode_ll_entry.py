"""Keep the 0.30 DS4 FFN connected to the opt-in decode LL implementation."""
from types import SimpleNamespace
import pytest
import torch
from vllm_iluvatar.models.deepseek_v4.fp8_model import IluDeepseekV4Fp8MoE
from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep

@pytest.mark.parametrize('enabled,decode', [(False, True), (True, False), (True, True)])
def test_ffn_selects_ll_only_for_opted_in_decode(enabled, decode):
    ffn = IluDeepseekV4Fp8MoE.__new__(IluDeepseekV4Fp8MoE)
    torch.nn.Module.__init__(ffn)
    ffn.gate = SimpleNamespace(tid2eid=None)
    ffn._ilu_decode_ll = enabled
    ffn._is_all_ranks_decode_only = lambda: decode
    calls = []
    def normal(**kw):
        calls.append('normal')
        return kw['hidden_states'] + 1
    def low_latency(x, ids):
        calls.append('ll')
        return x + 2
    ffn.experts = normal
    ffn.forward_dp_ep_deep_ep = low_latency
    x = torch.zeros(3, 8)
    expected_ll = enabled and decode
    torch.testing.assert_close(ffn(x), x + (2 if expected_ll else 1))
    assert calls == ['ll' if expected_ll else 'normal']


def test_decode_override_preinitializes_nonzero_draft_layer(monkeypatch):
    monkeypatch.setattr(ep, '_DS4_DECODE_EP_BACKEND', 'deepep_low_latency')
    ffn = ep.IluFusedMoEEP()
    ffn.prefix = 'model.layers.40.ffn'
    ffn.experts = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(
            dp_size=8, use_ep=True, all2all_backend='allgather_reducescatter')),
        _quant_method=SimpleNamespace(supports_internal_mk=True, is_monolithic=False))
    assert ffn._should_use_forward_dp_ep()
    assert ffn._should_preinit_dp_ep_ll_handle()


def test_initialization_runs_after_weight_processing(monkeypatch):
    ffn = ep.IluFusedMoEEP()
    events = []
    quant = SimpleNamespace(process_weights_after_loading=lambda layer: events.append('weights'))
    ffn.experts = SimpleNamespace(
        _quant_method=quant,
        routed_experts=SimpleNamespace(_ensure_moe_quant_config_init=lambda: events.append('quant')),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(all2all_backend='allgather_reducescatter')))
    ffn._should_preinit_dp_ep_ll_handle = lambda: True
    ffn._get_forward_dp_ep_ll_all2all = lambda: events.append('nvsm')
    ffn.init_ep()
    ffn.init_ep()
    assert events == []
    quant.process_weights_after_loading(None)
    assert events == ['weights', 'quant', 'nvsm']
