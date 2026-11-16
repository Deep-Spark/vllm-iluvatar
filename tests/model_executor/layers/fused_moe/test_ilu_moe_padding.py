# SPDX-License-Identifier: Apache-2.0
"""Skip-padding masks stay aligned with DP-gathered router rows."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_moe_padding import (
    _gathered_router_padding,
)


def test_gathered_padding_is_cached_and_restored(monkeypatch):
    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_moe_padding

    monkeypatch.setattr(ilu_moe_padding.envs, "VLLM_MOE_SKIP_PADDING", True)
    local_mask = torch.tensor([False, True])
    ctx = SimpleNamespace(is_padding=local_mask)
    gathers = []

    def gather(value, **kwargs):
        gathers.append(value.clone())
        return torch.cat([value, torch.tensor([0, 1, 0], dtype=torch.int8)])

    group = SimpleNamespace(rank_in_group=0, all_gatherv=gather)
    sizes = [2, 3]
    with _gathered_router_padding(ctx, group, sizes):
        assert ctx.is_padding.tolist() == [False, True, False, True, False]
        with _gathered_router_padding(ctx, group, sizes):
            assert ctx.is_padding.tolist() == [False, True, False, True, False]
    assert ctx.is_padding is local_mask
    assert len(gathers) == 1


def test_fused_ep_prefill_routes_with_global_padding(monkeypatch):
    from tests.model_executor.layers.fused_moe.test_ilu_moe_runner import (
        _make_stub_runner,
    )
    from vllm_iluvatar import _custom_ops
    from vllm_iluvatar.model_executor.layers.fused_moe.runner import ilu_moe_runner

    runner = _make_stub_runner(has_shared_experts=False)
    runner._fse_fuse_gate = False
    runner.gate = lambda value: (value, None)
    local_mask = torch.tensor([False, True])
    ctx = SimpleNamespace(is_padding=local_mask)
    mask_gathers = []

    def gather(value, **kwargs):
        if value.dtype == torch.int8:
            mask_gathers.append(value.clone())
            return torch.cat([value, torch.tensor([0, 1, 0], dtype=torch.int8)])
        return torch.cat([value, torch.ones(3, 2)])

    group = SimpleNamespace(rank_in_group=0, all_gatherv=gather)
    monkeypatch.setattr(ilu_moe_runner, "get_dp_group", lambda: group)
    monkeypatch.setattr(ilu_moe_runner, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(ilu_moe_runner.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(runner, "_ilu_dp_chunk_sizes", lambda: [2, 3])
    monkeypatch.setattr(runner, "_sequence_parallel_context", nullcontext)

    def apply(hidden_states, **kwargs):
        weights = torch.ones(5, 1)
        ids = torch.zeros(5, 1, dtype=torch.int32)
        _custom_ops._mask_topk_padding(weights, ids, ctx.is_padding)
        assert ids[:, 0].tolist() == [0, -1, 0, -1, 0]
        return None, hidden_states

    monkeypatch.setattr(runner, "_apply_quant_method", apply)
    for _ in range(2):
        _, output = runner._forward_impl_fused_ep_combine(
            torch.ones(2, 2),
            torch.ones(2, 2),
            None,
            None,
        )
        assert output.shape == (5, 2)
        assert ctx.is_padding is local_mask
    assert len(mask_gathers) == 1


def test_naive_ep_prefill_restores_local_mask(monkeypatch):
    from vllm_iluvatar import _custom_ops
    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep_mod
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
        IluFusedMoEEP,
    )

    local_mask = torch.tensor([False, True])
    ctx = SimpleNamespace(
        is_padding=local_mask,
        dp_metadata=SimpleNamespace(
            sp_local_sizes=lambda *_args, **_kwargs: nullcontext(),
            get_chunk_sizes_across_dp_rank=lambda: [2, 3],
        ),
    )
    mask_gathers = []

    def gather(value, **kwargs):
        if isinstance(value, list):
            return [torch.cat([value[0], torch.ones(3, 2)])]
        mask_gathers.append(value.clone())
        return torch.cat([value, torch.tensor([0, 1, 0], dtype=torch.int8)])

    group = SimpleNamespace(rank_in_group=0, all_gatherv=gather)
    monkeypatch.setattr(ep_mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(ep_mod, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(ep_mod.envs, "VLLM_MOE_SKIP_PADDING", True)

    class RoutingChecked(Exception):
        pass

    def select_experts(hidden_states, **kwargs):
        weights = torch.ones(hidden_states.shape[0], 1)
        ids = torch.zeros_like(weights, dtype=torch.int32)
        _custom_ops._mask_topk_padding(weights, ids, ctx.is_padding)
        assert ids[:, 0].tolist() == [0, -1, 0, -1, 0]
        raise RoutingChecked

    ep = IluFusedMoEEP()
    ep.experts = SimpleNamespace(
        moe_config=SimpleNamespace(pcp_size=1, sp_size=1, is_sequence_parallel=False),
        _quant_method=SimpleNamespace(
            skip_forward_padding=True,
            topk_indices_dtype=torch.int32,
        ),
        shared_experts=None,
        routed_experts=SimpleNamespace(_ensure_moe_quant_config_init=lambda: None),
        apply_routed_input_transform=lambda value: (value, None),
        gate=lambda value: (value, None),
        router=SimpleNamespace(select_experts=select_experts),
    )
    for _ in range(2):
        with pytest.raises(RoutingChecked):
            ep.forward_dp_ep_naive(torch.ones(2, 2))
        assert ctx.is_padding is local_mask
    assert len(mask_gathers) == 1
    assert mask_gathers[0].tolist() == [0, 1]
