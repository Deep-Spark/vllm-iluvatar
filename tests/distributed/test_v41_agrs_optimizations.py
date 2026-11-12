# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as moe


@pytest.mark.parametrize(
    ("enabled", "sizes", "ep_ranks", "tp", "expect_rs"),
    [
        (True, [2, 2], [0, 1], 1, True),
        (False, [2, 2], [0, 1], 1, False),
        (True, [1, 2], [0, 1], 1, False),
        (True, [2, 2], [1, 0], 1, False),
        (True, [2, 2], [0, 1, 2, 3], 2, False),
    ],
)
def test_tp1_reduce_scatter_route(monkeypatch, enabled, sizes, ep_ranks, tp, expect_rs):
    monkeypatch.setenv("VLLM_ILU_FUSE_EP_COMBINE", "1")
    monkeypatch.setenv("VLLM_ILU_MOE_EP_RS_TP_AG", "0")
    monkeypatch.setattr(moe, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    ep = SimpleNamespace(
        ranks=ep_ranks,
        device_communicator=SimpleNamespace(ixfd_group=object()),
    )
    monkeypatch.setattr(moe, "get_ep_group", lambda: ep)
    all_reduce = Mock()
    monkeypatch.setattr(moe.ixfd, "all_reduce", all_reduce)
    group = SimpleNamespace(
        ranks=[0, 1],
        rank_in_group=1,
        reduce_scatterv=Mock(side_effect=lambda x, **kw: x[sizes[0] :]),
    )
    values = torch.zeros(sum(sizes), 3, dtype=torch.bfloat16)
    shared = torch.ones(sizes[1], 3, dtype=values.dtype)

    got = moe.IluFusedMoEEP._combine_agrs_output(
        values, shared, group, sizes, sequence_parallel=False, use_tp1_rs=enabled
    )

    torch.testing.assert_close(got, shared)
    assert bool(group.reduce_scatterv.call_count) == expect_rs
    assert bool(all_reduce.call_count) != expect_rs
    torch.testing.assert_close(values[: sizes[0]], torch.zeros_like(values[: sizes[0]]))
