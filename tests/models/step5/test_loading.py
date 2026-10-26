# SPDX-License-Identifier: Apache-2.0
"""Strict checkpoint parameter audits distinguish derived scales from missing weights."""

import pytest
import torch
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter

from vllm_iluvatar.models.step5.model import _checkpoint_parameter_audit


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_real_kv_placeholders_follow_cache_contract(cache_dtype):
    root = torch.nn.Module()
    root.attn = torch.nn.Module()
    root.attn.kv_cache_dtype = cache_dtype
    root.weight = torch.nn.Parameter(torch.ones(2))
    for name in ("q_scale", "k_scale", "v_scale", "prob_scale"):
        setattr(root.attn, name, KVCacheScaleParameter())
    expected, exemptions = _checkpoint_parameter_audit(root)
    assert "weight" in expected
    scales = {
        f"attn.{name}" for name in ("q_scale", "k_scale", "v_scale", "prob_scale")
    }
    if cache_dtype == "auto":
        assert set(exemptions) == scales and expected == {"weight"}
    else:
        assert not exemptions and expected == scales | {"weight"}


def test_same_scale_names_are_not_a_blanket_exemption():
    root = torch.nn.Module()
    root.k_scale = torch.nn.Parameter(torch.ones(()))
    root.weight_scale = torch.nn.Parameter(torch.ones(2))
    root.fp4_scale = torch.nn.Parameter(torch.ones(2))
    root.fp4_scale.ilu_derived_at_load = True
    expected, exemptions = _checkpoint_parameter_audit(root)
    assert expected == {"k_scale", "weight_scale"} and exemptions == ["fp4_scale"]


@pytest.mark.parametrize(
    "experts,dim,nn_ready,stable_ready",
    [
        (16, 4096, False, True),
        (32, 4096, True, True),
        (32, 96, True, False),
        (17, 96, False, False),
    ],
)
def test_router_weight_preparation_respects_both_gemm_contracts(
    experts, dim, nn_ready, stable_ready
):
    from types import SimpleNamespace

    from vllm_iluvatar.models.step5.native_layers import Step5GateLinear

    layer = SimpleNamespace(
        is_unquantized=True,
        bias=None,
        weight=torch.ones((experts, dim), dtype=torch.float32),
    )
    Step5GateLinear.prepare_native_weights(layer)
    assert (layer._prepared_router_weight is not None) == nn_ready
    assert (layer._prepared_router_weight_nt is not None) == stable_ready
    assert layer._native_weights_ready
