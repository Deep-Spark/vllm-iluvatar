# SPDX-License-Identifier: Apache-2.0
"""Per-hop narrowing of the DeepSeek-V4.1 pipeline payload.

The plan is computed independently on every rank, so the invariant that makes
it safe is purely local arithmetic: rank r's ``sends_*`` must equal rank r+1's
``recvs_*``. If it ever does not, the receiver allocates a key the sender never
puts on the wire (or the other way round) and the step deadlocks.
"""

import pytest

from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan

# DeepSeek-V4.1-Flash.
V41 = dict(
    num_layers=40,
    kv_sources=(2, 8, 14, 20),
    index_sources=(2, 8, 14, 20, 24, 28, 32, 36),
    compress_ratios=tuple([0, 0] + [2] * 18 + [1] * 20),
    candidate_source=20,
)


def plans(world: int) -> list:
    return [build_rank_plan(rank=r, world=world, **V41) for r in range(world)]


@pytest.mark.parametrize("world", [1, 2, 4, 8, 10, 16, 20, 40])
def test_send_matches_next_recv(world: int) -> None:
    ranks = plans(world)
    for r in range(world - 1):
        here, nxt = ranks[r], ranks[r + 1]
        assert here.sends_kv == nxt.recvs_kv, f"kv mismatch on hop {r}->{r + 1}"
        assert here.sends_topk == nxt.recvs_topk, f"topk mismatch on hop {r}->{r + 1}"
        assert here.sends_candidates == nxt.recvs_candidates, (
            f"candidates mismatch on hop {r}->{r + 1}"
        )
    last = ranks[-1]
    assert not (last.sends_kv or last.sends_topk or last.sends_candidates)


@pytest.mark.parametrize("world", [2, 4, 8, 10, 16, 20, 40])
def test_every_stranded_rank_is_fed(world: int) -> None:
    """A rank with a forwarder must receive the records it replicates."""
    for plan in plans(world):
        if plan.forwarder is not None:
            assert plan.recvs_kv
        if plan.forwarder_needs_topk:
            assert plan.recvs_topk


@pytest.mark.parametrize("world", [2, 4, 8, 10, 16, 20, 40])
def test_candidates_reach_every_remote_reader(world: int) -> None:
    """Ranks scoring against a shortlist published elsewhere must receive it,
    and the hops in between must relay it."""
    ranks = plans(world)
    publisher = next(
        r for r, p in enumerate(ranks) if p.start_layer <= 20 < p.end_layer
    )
    readers = [
        r
        for r, p in enumerate(ranks)
        if r != publisher
        and any(p.start_layer <= s < p.end_layer for s in (24, 28, 32, 36))
    ]
    for reader in readers:
        assert reader > publisher, "the shortlist never travels backwards"
        assert ranks[reader].recvs_candidates
        for hop in range(publisher, reader):
            assert ranks[hop].sends_candidates, (
                f"hop {hop}->{hop + 1} drops a shortlist rank {reader} needs"
            )


def test_narrowing_actually_drops_hops_on_pp16() -> None:
    """The point of the change: on the default PP16 cut most hops carry no
    shortlist and no KV replica."""
    ranks = plans(16)
    assert sum(p.sends_candidates for p in ranks) == 5
    assert sum(p.sends_kv for p in ranks) == 11
    # Global flags stay on so the replica buffers are still allocated.
    assert all(p.transfers_kv and p.transfers_candidates for p in ranks)


def test_single_rank_carries_nothing() -> None:
    plan = build_rank_plan(rank=0, world=1, **V41)
    assert not (plan.sends_kv or plan.sends_topk or plan.sends_candidates)
    assert not (plan.recvs_kv or plan.recvs_topk or plan.recvs_candidates)
