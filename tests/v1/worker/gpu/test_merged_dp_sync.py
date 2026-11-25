"""Differential tests against the exact upstream DP agreement protocol."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu import dp_utils as upstream
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc

from vllm_iluvatar.v1.worker.dp_utils import get_last_all_ranks_decode_only
from vllm_iluvatar.v1.worker.gpu import dp_utils
from vllm_iluvatar.v1.worker.gpu import merged_dp_sync as merged


@pytest.mark.parametrize(
    "scenario",
    ["decode", "mixed_vote", "mixed_query", "eager", "empty", "skip", "varlen"],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_agreement_matches_upstream(monkeypatch, scenario, rank):
    # Columns represent two different ranks; preserve metadata agreement even
    # when decode votes differ, one rank requests eager, or all ranks are idle.
    data = torch.tensor(
        [
            [6, 12],
            [Mode.FULL.value, Mode.FULL.value],
            [6, 6],
            [-1, -1],
            [0, 0],
            [1, 2],
            [1, 1],
            [0, 0],
        ],
        dtype=torch.int32,
    )
    if scenario == "mixed_vote":
        data[6, 1] = 0
    if scenario == "mixed_query":
        data[2, 1] = 3
    if scenario == "eager":
        data[1, 1] = Mode.NONE.value
    if scenario == "empty":
        data[0] = 0
    if scenario == "varlen":
        data[3] = torch.tensor([6, 8])
    calls = []

    def reduce(tensor, group):
        calls.append(tuple(tensor.shape))
        assert torch.equal(tensor[:, rank], data[: len(tensor), rank])
        tensor.copy_(data[: len(tensor)])

    original = dp_utils._ORIGINAL_SYNC or upstream.sync_cudagraph_and_dp_padding
    monkeypatch.setattr(merged.dist, "all_reduce", reduce)
    monkeypatch.setattr(
        merged, "get_dp_group", lambda: SimpleNamespace(cpu_group="test")
    )
    monkeypatch.setattr(
        upstream, "get_dp_group", lambda: SimpleNamespace(cpu_group="test")
    )
    for module in (merged, upstream):
        monkeypatch.setattr(
            module, "should_skip_dp_coordination", lambda: scenario == "skip"
        )

    def dispatch(nreq, ntok, uniform, **kwargs):
        return Desc(
            cg_mode=Mode.FULL if uniform else Mode.NONE,
            num_tokens=ntok,
            num_reqs=nreq,
            uniform_token_count=uniform,
        )

    manager = SimpleNamespace(dispatch=dispatch)
    args = dict(
        cudagraph_manager=manager,
        desired_batch_desc=Desc(
            cg_mode=Mode(int(data[1, rank])),
            num_tokens=int(data[0, rank]),
            num_reqs=int(data[5, rank]),
        ),
        num_tokens=int(data[0, rank]),
        num_reqs=int(data[5, rank]),
        uniform_token_count=int(data[2, rank]),
        dp_size=2,
        dp_rank=rank,
        max_query_len=None if data[3, rank] == -1 else int(data[3, rank]),
        uniform_decode=bool(data[6, rank]),
    )
    before = original(**args)
    after = merged.sync_cudagraph_and_dp_padding(**args)
    assert before[0] == after[0]
    if before[1] is None:
        assert after[1] is None
    else:
        for name in before[1].__dataclass_fields__:
            a, b = getattr(before[1], name), getattr(after[1], name)
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    expected = (
        bool(data[6, rank]) if scenario == "skip" else bool(torch.all(data[6] == 1))
    )
    assert get_last_all_ranks_decode_only() == expected
    assert calls == ([] if scenario == "skip" else [(6, 2), (8, 2)])


def test_all_idle_dummy_short_circuits_to_zero_tokens(monkeypatch):
    from vllm_iluvatar.v1.worker.idle_dummy_state import (
        idle_dummy_batch_scope,
        was_idle_dummy_short_circuited,
    )

    def reduce(tensor, group):
        # Simulate both DP ranks being in runtime idle_dummy_batch_scope.
        tensor[:, 0] = tensor[:, 1]

    monkeypatch.setattr(merged.dist, "all_reduce", reduce)
    monkeypatch.setattr(
        merged, "get_dp_group", lambda: SimpleNamespace(cpu_group="test")
    )
    monkeypatch.setattr(merged, "should_skip_dp_coordination", lambda: False)
    manager = SimpleNamespace(
        dispatch=lambda *a, **k: pytest.fail("dispatch should not run when all idle")
    )
    with idle_dummy_batch_scope():
        desc, sync = merged.sync_cudagraph_and_dp_padding(
            cudagraph_manager=manager,
            desired_batch_desc=Desc(cg_mode=Mode.FULL, num_tokens=6, num_reqs=1),
            num_tokens=6,
            num_reqs=1,
            uniform_token_count=6,
            dp_size=2,
            dp_rank=1,
            uniform_decode=True,
        )
        assert was_idle_dummy_short_circuited() is True
    assert desc == Desc(cg_mode=Mode.NONE, num_tokens=0, num_reqs=0)
    assert sync is None
    assert get_last_all_ranks_decode_only() is False


def test_mixed_real_and_idle_dummy_runs_normal_agreement(monkeypatch):
    from vllm_iluvatar.v1.worker.idle_dummy_state import (
        idle_dummy_batch_scope,
        was_idle_dummy_short_circuited,
    )

    def reduce(tensor, group):
        # Rank 0 is running a real decode batch (row 7 == 0); rank 1 is idle dummy (row 7 == 1).
        tensor[:, 0] = torch.tensor(
            [6, Mode.FULL.value, 6, -1, 0, 1, 1, 0], dtype=torch.int32
        )

    monkeypatch.setattr(merged.dist, "all_reduce", reduce)
    monkeypatch.setattr(
        merged, "get_dp_group", lambda: SimpleNamespace(cpu_group="test")
    )
    monkeypatch.setattr(merged, "should_skip_dp_coordination", lambda: False)
    manager = SimpleNamespace(
        dispatch=lambda nreq, ntok, uniform, **k: Desc(
            cg_mode=Mode.FULL,
            num_tokens=ntok,
            num_reqs=nreq,
            uniform_token_count=uniform,
        )
    )
    with idle_dummy_batch_scope():
        desc, sync = merged.sync_cudagraph_and_dp_padding(
            cudagraph_manager=manager,
            desired_batch_desc=Desc(cg_mode=Mode.FULL, num_tokens=6, num_reqs=1),
            num_tokens=6,
            num_reqs=1,
            uniform_token_count=6,
            dp_size=2,
            dp_rank=1,
            uniform_decode=True,
        )
        assert was_idle_dummy_short_circuited() is False
    assert desc.num_tokens == 6
    assert desc.cg_mode == Mode.FULL
    assert sync is not None
    assert get_last_all_ranks_decode_only() is True


def test_signature():
    assert inspect.signature(merged.sync_cudagraph_and_dp_padding) == inspect.signature(
        upstream.sync_cudagraph_and_dp_padding
    )
