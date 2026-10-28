# SPDX-License-Identifier: Apache-2.0
"""Collective selection/order checks using real DPMetadata and mocked groups."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.forward_context import DPMetadata

from vllm_iluvatar.model_executor.layers.fused_moe.runner import ilu_moe_runner as mod

# Per-sequence-parallel-shard token cap for the gather-before-gate path; mirrors
# IluGateLinear.gather_before_gate_max_tokens so the boundary cases stay honest.
MAX_TOKENS = 32


def runner(sp=2):
    gate = Mock()
    # Eligibility is a model property carried on the gate, not a shape sniff.
    gate.gather_before_gate = True
    gate.gather_before_gate_max_tokens = MAX_TOKENS
    obj = SimpleNamespace(
        gate=gate,
        _gather_before_gate_enabled=True,
        do_naive_dispatch_combine=True,
        _kernel_handles_own_dispatch=False,
        _fse_fuse_gate=False,
        moe_config=SimpleNamespace(
            sp_size=sp,
            is_sequence_parallel=sp > 1,
            pcp_size=1,
            moe_parallel_config=SimpleNamespace(
                all2all_backend="allgather_reducescatter"
            ),
        ),
    )
    for name in ("_dispatch_before_gate", "_gather_before_gate", "_forward_impl"):
        setattr(obj, name, MethodType(getattr(mod.IluMoERunner, name), obj))
    return obj


@pytest.mark.parametrize("sp", [1, 2, 4])
@pytest.mark.parametrize(
    "pattern", ["uniform", "idle", "mixed", "empty", "boundary", "over"]
)
def test_all_ranks_choose_same_path(monkeypatch, sp, pattern):
    counts = {
        "uniform": [16] * 8,
        "idle": [16, 0, 8, 0, 0, 0, 0, 0],
        "mixed": [16, 2048, 0, 0, 0, 0, 0, 0],
        "empty": [0] * 8,
        "boundary": [MAX_TOKENS * sp] + [0] * 7,
        "over": [MAX_TOKENS * sp + 1] + [0] * 7,
    }[pattern]
    meta = DPMetadata(torch.tensor(counts, dtype=torch.int32))
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(dp_metadata=meta)
    )
    monkeypatch.setattr(
        mod,
        "get_current_vllm_config",
        Mock(side_effect=AssertionError("not a forward-time API")),
    )
    decisions = []
    for _ in counts:
        for _ in range(sp):
            decisions.append(runner(sp)._dispatch_before_gate())
    assert decisions == [pattern in ("uniform", "idle", "boundary")] * (8 * sp)


@pytest.mark.parametrize(
    "case",
    [
        "no_property",
        "switch_off",
        "no_gate",
        "dp1",
        "internal",
        "backend",
        "pcp",
        "fused",
    ],
)
def test_static_rejections_do_not_read_forward_metadata(monkeypatch, case):
    obj = runner()
    if case == "no_property":
        obj.gate.gather_before_gate = False
    elif case == "switch_off":
        obj._gather_before_gate_enabled = False
    elif case == "no_gate":
        obj.gate = None
    elif case == "dp1":
        obj.do_naive_dispatch_combine = False
    elif case == "internal":
        obj._kernel_handles_own_dispatch = True
    elif case == "backend":
        obj.moe_config.moe_parallel_config.all2all_backend = "deepep_low_latency"
    elif case == "pcp":
        obj.moe_config.pcp_size = 2
    else:
        obj._fse_fuse_gate = True
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        Mock(side_effect=AssertionError("unexpected metadata access")),
    )
    assert not obj._dispatch_before_gate()


def test_missing_dp_metadata_falls_back(monkeypatch):
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None)
    )
    assert not runner()._dispatch_before_gate()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, True),  # unset -> default on
        ("1", True),
        ("0", False),
        ("true", True),
        ("yes", True),
        ("off", False),  # unrecognised -> not truthy
    ],
)
def test_gather_before_gate_env_switch(monkeypatch, raw, expected):
    from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS

    key = "VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE"
    if raw is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, raw)
    assert _ILU_CANONICAL_ONLY_ENV_VARS[key]() is expected


@pytest.mark.parametrize("sp", [1, 2])
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_forward_order_and_local_shared_input(monkeypatch, sp, with_ids, idle, mixed):
    obj = runner(sp)
    counts = [16 * sp, 0, 2048 if mixed else 18, 2]
    meta = DPMetadata(torch.tensor(counts, dtype=torch.int32))
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(dp_metadata=meta)
    )
    sizes = [(n + sp - 1) // sp for n in counts for _ in range(sp)]
    chunks = [
        torch.full((n, 4096), float(i), dtype=torch.bfloat16)
        for i, n in enumerate(sizes)
    ]
    id_chunks = [torch.full((n,), i, dtype=torch.int64) for i, n in enumerate(sizes)]
    rank = sp if idle else 0
    x = chunks[rank]
    ids = id_chunks[rank] if with_ids else None
    shared = torch.randn_like(x)
    gathered_x = torch.cat(chunks)
    gathered_ids = torch.cat(id_chunks) if with_ids else None
    events = []
    obj.routed_experts = SimpleNamespace(_ensure_moe_quant_config_init=lambda: None)
    obj._sequence_parallel_context = lambda: meta.sp_local_sizes(sp, use_ep=True)

    def gate(hidden):
        events.append("gate")
        torch.testing.assert_close(hidden, x if mixed else gathered_x)
        return hidden[:, :288], None

    obj.gate.side_effect = gate

    def all_gatherv(tensors, *, dim, sizes):
        events.append("hidden_ag")
        assert meta.local_sizes is not None and sizes == meta.local_sizes and dim == 0
        assert tensors[0] is x
        assert len(tensors) == (2 if with_ids else 1)
        if with_ids:
            assert tensors[1] is ids
        return [gathered_x, gathered_ids] if with_ids else [gathered_x]

    selected = SimpleNamespace(all_gatherv=all_gatherv, rank_in_group=rank)
    wrong = Mock(side_effect=AssertionError("wrong communication group"))
    monkeypatch.setattr(mod, "get_ep_group", (lambda: selected) if sp > 1 else wrong)
    monkeypatch.setattr(mod, "get_dp_group", (lambda: selected) if sp == 1 else wrong)

    def old_dispatch(hidden, logits, token_ids):
        events.append("old_dispatch")
        assert hidden is x and token_ids is ids
        return gathered_x, gathered_x[:, :288], gathered_ids

    obj._maybe_dispatch = old_dispatch

    def quant(**kwargs):
        events.append("quant")
        assert kwargs["hidden_states"] is gathered_x
        assert kwargs["shared_experts_input"] is shared
        assert kwargs["input_ids"] is gathered_ids
        torch.testing.assert_close(kwargs["router_logits"], gathered_x[:, :288])
        return shared, gathered_x

    obj._apply_quant_method = quant

    def combine(s, h):
        events.append("combine")
        assert s is shared and h is gathered_x and meta.local_sizes is not None
        return shared, x

    obj._maybe_combine = combine
    result = obj._forward_impl(x, torch.empty(x.shape[0], 288), shared, ids)
    assert result[0] is shared and result[1] is x
    assert events == (
        ["gate", "old_dispatch", "quant", "combine"]
        if mixed
        else ["hidden_ag", "gate", "quant", "combine"]
    )
    assert meta.local_sizes is None
