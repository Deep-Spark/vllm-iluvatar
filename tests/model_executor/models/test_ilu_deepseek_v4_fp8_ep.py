# SPDX-License-Identifier: Apache-2.0
"""FP8 model wiring for converted W4A8 Hybrid EP and MXFP4 decode LL."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp4.mxfp4_moe import (
    IluMxfp4MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.moe.mxfp4_to_w4a8_tn import (
    IluMxfp4ToW4A8TNOnlineMoEMethod,
)
from vllm_iluvatar.models.deepseek_v4 import fp8_model as mod


def make_moe(monkeypatch, kind, hash_layer=True):
    method = object.__new__(
        IluMxfp4MoEMethod if kind == "native" else IluMxfp4ToW4A8TNOnlineMoEMethod
    )
    if kind == "wrapped":
        method = SimpleNamespace(old_quant_method=method)
    runner = Mock(_quant_method=method)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mod, "GateLinear", lambda **kwargs: nn.Module())
    monkeypatch.setattr(mod, "FusedMoEFactory", lambda **kwargs: runner)
    init = Mock()
    monkeypatch.setattr(mod.IluDeepseekV4Fp8MoE, "init_ep", init)
    hf = SimpleNamespace(
        hidden_size=8,
        n_routed_experts=8,
        swiglu_limit=7.0,
        num_hash_layers=3 if hash_layer else 0,
        vocab_size=16,
        num_experts_per_tok=2,
        topk_method="noaux_tc",
        n_shared_experts=None,
        moe_intermediate_size=16,
        norm_topk_prob=True,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            enable_eplb=False,
        ),
    )
    return mod.IluDeepseekV4Fp8MoE(cfg, "model.layers.0.ffn"), runner, init


@pytest.mark.parametrize("kind", ["native", "converted", "wrapped"])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("hash_layer", [False, True])
def test_method_admission_and_forward(monkeypatch, kind, eligible, hash_layer):
    moe, runner, init = make_moe(monkeypatch, kind, hash_layer)
    # Native MXFP4 stays on the community runner unless the decode LL
    # override is enabled. Converted/wrapped W4A8 always install Hybrid EP.
    assert init.call_count == (kind != "native")
    assert moe._ilu_decode_ll is False
    guard = Mock(return_value=eligible)
    hybrid = Mock(return_value=torch.full((2, 8), 7.0))
    monkeypatch.setattr(moe, "_should_use_forward_dp_ep", guard)
    monkeypatch.setattr(moe, "forward_dp_ep", hybrid)
    runner.return_value = torch.full((2, 8), 3.0)
    hidden = torch.arange(16).reshape(2, 8).float()
    ids = torch.tensor([1, 2], dtype=torch.int64) if hash_layer else None
    output = moe(hidden, ids)
    use_hybrid = kind != "native" and eligible
    assert output.shape == hidden.shape
    assert torch.equal(output, torch.full_like(hidden, 7 if use_hybrid else 3))
    assert guard.call_count == (kind != "native")
    assert hybrid.call_count == use_hybrid
    assert runner.call_count == (not use_hybrid)
    if use_hybrid:
        assert hybrid.call_args.args[0] is hidden
        routed_ids = hybrid.call_args.args[1]
    else:
        assert runner.call_args.kwargs["hidden_states"] is hidden
        routed_ids = runner.call_args.kwargs["input_ids"]
    if hash_layer:
        assert routed_ids.dtype == torch.int32
        assert torch.equal(routed_ids, ids)
    else:
        assert routed_ids is None


def test_native_mxfp4_decode_ll_override(monkeypatch):
    monkeypatch.setattr(mod, "_DECODE_EP_BACKEND", "deepep_low_latency")
    monkeypatch.setattr(
        mod.IluDeepseekV4Fp8MoE, "_should_use_forward_dp_ep", lambda self: True
    )
    moe, runner, init = make_moe(monkeypatch, "native", hash_layer=False)
    assert init.call_count == 1
    assert moe._ilu_decode_ll is True
    low_latency = Mock(return_value=torch.full((2, 8), 9.0))
    monkeypatch.setattr(moe, "forward_dp_ep_deep_ep", low_latency)
    monkeypatch.setattr(moe, "_is_all_ranks_decode_only", lambda: True)
    runner.return_value = torch.full((2, 8), 3.0)
    hidden = torch.zeros(2, 8)
    output = moe(hidden, None)
    assert torch.equal(output, torch.full_like(hidden, 9.0))
    assert low_latency.call_count == 1
    runner.assert_not_called()


@pytest.mark.parametrize("kind", ["native", "converted"])
def test_hash_routing_requires_input_ids(monkeypatch, kind):
    moe, runner, _ = make_moe(monkeypatch, kind)
    with pytest.raises(ValueError, match="requires input_ids"):
        moe(torch.zeros(2, 8))
    runner.assert_not_called()
