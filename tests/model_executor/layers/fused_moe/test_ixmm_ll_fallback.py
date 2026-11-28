# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep
from vllm_iluvatar.model_executor.layers.fused_moe import ixmm_ll_decode as decode


@pytest.mark.parametrize("rank", [0, 1])
def test_missing_interface_on_one_rank_falls_back_before_shmem_init(monkeypatch, rank):
    group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=object())
    monkeypatch.setenv("VLLM_ILU_V41_IXMM_LL_DECODE", "1")
    monkeypatch.setenv("VLLM_ILU_IXMM_LL_LIBRARY", "routed.so")
    runtime = Mock(uid_size=64, sizes={5: (128, 256)}, tp_workspace_bytes=0)
    reason = "undefined symbol: ixmm_shmem_uid_size"
    loader = Mock(
        return_value=runtime, side_effect=AttributeError(reason) if rank else None
    )
    monkeypatch.setattr(decode, "IxmmLL", loader)
    monkeypatch.setattr(decode, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    votes = [
        (True, None, (runtime.uid_size, runtime.sizes, 1, 0)),
        (True, reason, None),
    ]

    def gather(output, vote, *, group):
        assert vote == votes[rank]
        output[:] = votes

    monkeypatch.setattr(decode.dist, "all_gather_object", gather)
    warning = Mock()
    monkeypatch.setattr(decode.logger, "warning", warning)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(device="cuda"), _ilu_ixmm_ll_token_capacities=(5,)
    )
    for _ in range(2):
        decode.prepare_layer(layer, 18, group)
        assert not hasattr(layer, "_ilu_ixmm_ll")
    loader.assert_called_once_with("routed.so", (5,))
    runtime.initialize.assert_not_called()
    assert warning.call_count == int(rank == 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_empty_graph_plan_still_participates_in_vote(monkeypatch, rank):
    group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=object())
    runtime = Mock(uid_size=64, sizes={5: (128, 256)}, tp_workspace_bytes=0)
    loader = Mock(return_value=runtime)
    monkeypatch.setenv("VLLM_ILU_V41_IXMM_LL_DECODE", "1")
    monkeypatch.setattr(decode, "IxmmLL", loader)
    votes = [
        (True, None, (64, runtime.sizes, 1, 0)),
        (True, "no decode graph workspace for this configuration", None),
    ]

    def gather(output, vote, *, group):
        assert vote == votes[rank]
        output[:] = votes

    monkeypatch.setattr(decode.dist, "all_gather_object", gather)
    assert (
        decode.prepare(
            "cuda", group, () if rank else (5,), SimpleNamespace(world_size=1)
        )
        is None
    )
    runtime.initialize.assert_not_called()


def test_missing_optional_library(tmp_path):
    with pytest.raises(OSError):
        decode.IxmmLL(str(tmp_path / "missing.so"), (5,))


@pytest.mark.parametrize(
    "mode,scatter,tp,expected",
    [
        (CUDAGraphMode.NONE, True, 2, ()),
        (CUDAGraphMode.FULL_DECODE_ONLY, True, 1, (3, 6, 9, 15)),
        (CUDAGraphMode.FULL_DECODE_ONLY, True, 2, (2, 3, 5, 8)),
        (CUDAGraphMode.FULL_DECODE_ONLY, False, 2, (3, 6, 9, 15)),
        (CUDAGraphMode.FULL, True, 2, (1, 2, 4, 7, 12)),
    ],
)
def test_workspace_capacities_follow_capture_config(
    monkeypatch, mode, scatter, tp, expected
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[24, 1, 2, 4, 8, 14],
            max_cudagraph_capture_size=24,
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        scheduler_config=SimpleNamespace(max_num_seqs=5),
        num_speculative_tokens=2,
    )
    monkeypatch.setattr(ep, "VLLM_ILU_DEEPEP_USE_SCATTER", scatter)
    monkeypatch.setattr(ep, "VLLM_ILU_LL_FUSED_AG_AR", False)
    assert ep.decode_graph_token_capacities(config, tp) == expected


@pytest.mark.parametrize(
    "tokens,expected", [(0, None), (1, 2), (2, 2), (3, 5), (6, None)]
)
def test_capacity_lookup(tokens, expected):
    runtime = object.__new__(decode.IxmmLL)
    runtime.capacities = (2, 5)
    assert runtime.capacity_for(tokens) == expected


@pytest.mark.parametrize(
    "chunks,rank", [([3, 2], 0), ([3, 2], 1), ([1, 0], 1), ([2, 2], 1), (None, 0)]
)
def test_tp2_routed_and_shared_outputs(monkeypatch, chunks, rank):
    # Known, distinct input rows detect padding leakage, wrong gather order,
    # and duplicate reduction of already-complete routed expert output.
    rows = sum(chunks) if chunks is not None else 4
    full = torch.arange(1, rows + 1, dtype=torch.float32)[:, None]
    start = sum(chunks[:rank]) if chunks else 0
    hidden = full[start : start + chunks[rank]] if chunks else full
    capacity = 5

    def send(x, ids, stream):
        assert x.shape == (capacity, 1)
        assert torch.all(ids[hidden.shape[0] :] == -1)
        return x

    runtime = SimpleNamespace(
        dispatch_send=send,
        dispatch_recv=Mock(),
        ops=SimpleNamespace(tp_join=object()),
        tp_workspace=object(),
    )
    group = Mock(world_size=2, rank_in_group=rank)

    def fused(
        op,
        local,
        shared,
        counts,
        size,
        routed_scale,
        shared_scale,
        out,
        team,
        stream,
        workspace,
    ):
        assert op is runtime.ops.tp_join and workspace is runtime.tp_workspace
        assert list(counts) == chunks and size == 2 and team == 0
        torch.testing.assert_close(local, hidden)
        torch.testing.assert_close(shared, full)
        out.copy_(full * routed_scale + shared * shared_scale + 7)

    join = Mock(side_effect=fused)
    monkeypatch.setattr(decode, "_call", join)
    group.all_reduce.side_effect = lambda x: x + 7
    monkeypatch.setattr(
        decode.torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1)
    )

    def experts(x, layer, ids, weights, limit):
        assert torch.all(weights[hidden.shape[0] :] == 0)
        return x.clone()

    monkeypatch.setattr(decode, "experts", experts)
    actual = decode.forward(
        runtime,
        capacity,
        hidden,
        torch.zeros_like(hidden),
        torch.ones_like(hidden),
        SimpleNamespace(swiglu_limit=0),
        lambda x: x,
        full,
        2,
        chunks,
        group,
    )
    torch.testing.assert_close(actual, full * 3 + 7)
    assert join.call_count == int(chunks is not None)
    group.all_gather.assert_not_called()
    assert group.all_reduce.call_count == int(chunks is None)


@pytest.mark.parametrize(
    "ep_rank,tp_ranks,valid",
    [(8, [8, 9], True), (9, [8, 9], True), (8, [9, 8], False), (8, [8, 10], False)],
)
def test_tp_workspace_uses_ep_order_and_max_total_rows(
    monkeypatch, ep_rank, tp_ranks, valid
):
    ep_group = SimpleNamespace(rank_in_group=ep_rank, ranks=list(range(16)))
    tp_group = SimpleNamespace(world_size=2, ranks=tp_ranks)
    monkeypatch.setenv("VLLM_ILU_DEEPEP_USE_SCATTER", "1")
    monkeypatch.delenv(decode._WORLD_PAIRS, raising=False)
    runtime = object.__new__(decode.IxmmLL)
    runtime.capacities = (3, 5)
    runtime._bind = Mock()
    runtime.ops = Mock()
    size = Mock(return_value=4096)
    monkeypatch.setattr(decode, "_workspace_size", size)
    if valid:
        runtime.plan_tp_join(ep_group, tp_group)
        size.assert_called_once_with(
            runtime.ops.tp_workspace_size, 10, decode.HIDDEN, 2, True
        )
        assert runtime.tp_workspace_bytes == 4096
    else:
        with pytest.raises(ValueError, match="contiguous TP groups"):
            runtime.plan_tp_join(ep_group, tp_group)
        size.assert_not_called()
