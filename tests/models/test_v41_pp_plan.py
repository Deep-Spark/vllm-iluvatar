# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 pipeline cut plan, over every pipeline size that divides work.

The plan's whole job is an invariant: after it runs, no layer reads a cache
that lives on another rank. Most of what follows checks exactly that, over the
real V4.1 topology and over synthetic ones chosen to hit the cases the
four-way, layers-0-20 shape never could -- a group spanning three ranks, and a
cut that strands the top-k selection while leaving the KV group intact.
"""

from __future__ import annotations

import pytest

from vllm_iluvatar.models.deepseek_v41.pp_plan import (
    build_rank_plan,
    local_kv_source,
)

# DeepSeek-V4.1-Flash, from the shipped config.
V41_LAYERS = 40
V41_KV_SOURCES = (2, 8, 14, 20)
V41_INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
V41_RATIOS = tuple([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])


def _real_source(layer_id: int, sources: tuple[int, ...]) -> int | None:
    eligible = [s for s in sources if s <= layer_id]
    return max(eligible) if eligible else None


def _plans(
    world: int,
    layers=V41_LAYERS,
    kv=V41_KV_SOURCES,
    index=V41_INDEX_SOURCES,
    ratios=V41_RATIOS,
):
    return [
        build_rank_plan(layers, kv, index, ratios, rank, world) for rank in range(world)
    ]


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_layer_reads_a_local_cache(world: int):
    """The invariant. A resolved owner below the rank's start would be a
    read into another rank's memory."""
    for plan in _plans(world):
        for layer_id in range(plan.start_layer, plan.end_layer):
            if V41_RATIOS[layer_id] == 0:
                continue  # pure sliding window, reads no group
            owner = local_kv_source(
                layer_id, _real_source(layer_id, V41_KV_SOURCES), plan
            )
            assert owner is not None
            assert plan.start_layer <= owner <= layer_id, (
                f"world={world} layer {layer_id} resolved to owner {owner}, "
                f"outside its rank's [{plan.start_layer}, {plan.end_layer})"
            )


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8])
def test_the_forwarder_is_the_first_stranded_layer(world: int):
    for plan in _plans(world):
        stranded = [
            layer_id
            for layer_id in range(plan.start_layer, plan.end_layer)
            if V41_RATIOS[layer_id] > 0
            and (_real_source(layer_id, V41_KV_SOURCES) or 0) < plan.start_layer
        ]
        if not stranded:
            assert plan.forwarder is None
            continue
        assert plan.forwarder == min(stranded)
        assert plan.forwarded_kv_source == _real_source(plan.forwarder, V41_KV_SOURCES)
        # A layer that is its own source can never be a forwarder: it would
        # have to own two caches under one prefix.
        assert plan.forwarder not in V41_KV_SOURCES


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8])
def test_each_hop_agrees_on_what_it_carries(world: int):
    """Each rank derives its hops alone, and the pipeline breaks if the two
    sides of one hop disagree: a key the sender omits would reach the receiver
    holding the previous step's rows, and one it adds has nowhere to land."""
    plans = _plans(world)
    for sender, receiver in zip(plans, plans[1:]):
        assert sender.sends_kv == receiver.recvs_kv
        assert sender.sends_topk == receiver.recvs_topk
        assert sender.sends_candidates == receiver.recvs_candidates
    # Nothing leaves the last rank and nothing arrives at the first.
    assert not (plans[-1].sends_kv or plans[-1].sends_topk)
    assert not (plans[0].recvs_kv or plans[0].recvs_topk)


@pytest.mark.parametrize("world", [2, 3, 4, 5, 6, 7, 8])
def test_a_rank_only_sends_records_it_can_produce(world: int):
    """The outgoing buffer is filled by a local source or by the relay, so
    sending without either would ship whatever the buffer happened to hold."""
    for plan in _plans(world):
        if not plan.sends_kv:
            continue
        owns_source = any(
            plan.start_layer <= s < plan.end_layer for s in V41_KV_SOURCES
        )
        assert owns_source or plan.recvs_kv
    # Same for the selection: it is whatever the rank's last index source
    # wrote, or what it took over.
    for plan in _plans(world):
        if not plan.sends_topk:
            continue
        owns_source = any(
            plan.start_layer <= s < plan.end_layer for s in V41_INDEX_SOURCES
        )
        assert owns_source or plan.recvs_topk


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8])
def test_rank_zero_never_forwards(world: int):
    plan = build_rank_plan(
        V41_LAYERS, V41_KV_SOURCES, V41_INDEX_SOURCES, V41_RATIOS, 0, world
    )
    assert plan.forwarder is None
    assert plan.start_layer == 0


def test_single_rank_carries_nothing():
    plan = build_rank_plan(
        V41_LAYERS, V41_KV_SOURCES, V41_INDEX_SOURCES, V41_RATIOS, 0, 1
    )
    assert plan == build_rank_plan(
        V41_LAYERS, V41_KV_SOURCES, V41_INDEX_SOURCES, V41_RATIOS, 0, 1
    )
    assert plan.forwarder is None
    assert not (plan.recvs_kv or plan.recvs_topk or plan.recvs_candidates)
    assert not (plan.sends_kv or plan.sends_topk or plan.sends_candidates)
    assert (plan.start_layer, plan.end_layer) == (0, V41_LAYERS)


def test_topk_can_be_stranded_while_the_kv_group_is_not():
    """A cut inside an index group but not a KV group.

    Index sources are denser than KV sources, so a rank can start with a local
    KV situation that still needs the previous rank's selection. The rank
    starting at layer 30 reads KV from source 20 and top-k from source 28, two
    different layers, both remote -- the case the KV-only plan missed.
    """
    plans = _plans(4)
    assert [(p.start_layer, p.end_layer) for p in plans] == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
    ]
    # Layer 20 is both a KV and an index source, so rank 2 strands nothing.
    assert plans[2].forwarder is None
    # Rank 3 strands both, and they come from different remote layers.
    assert plans[3].forwarder == 30
    assert plans[3].forwarded_kv_source == 20
    assert plans[3].forwarder_needs_topk
    assert plans[3].recvs_kv and plans[3].recvs_topk
    assert plans[2].sends_kv and plans[2].sends_topk
    # Rank 2 starts on a source of both kinds, so the hop into it carries
    # neither even though the hop out of it carries both.
    assert not (plans[2].recvs_kv or plans[2].recvs_topk)
    assert not (plans[1].sends_kv or plans[1].sends_topk)


def test_a_group_spanning_three_ranks_puts_a_forwarder_on_each():
    """One source, two downstream ranks. The middle one has no local source,
    which is what makes it a relay rather than just a consumer."""
    layers, kv, index = 12, (0,), (0,)
    ratios = tuple([1] * 12)
    plans = [build_rank_plan(layers, kv, index, ratios, r, 3) for r in range(3)]
    assert [(p.start_layer, p.end_layer) for p in plans] == [(0, 4), (4, 8), (8, 12)]
    assert plans[0].forwarder is None
    assert plans[1].forwarder == 4 and plans[1].forwarded_kv_source == 0
    assert plans[2].forwarder == 8 and plans[2].forwarded_kv_source == 0
    # Rank 1 owns no source, so nothing on it would refill the outgoing buffer:
    # it has to send what it received, which is what makes it a relay.
    assert not any(4 <= s < 8 for s in kv)
    assert plans[1].recvs_kv and plans[1].sends_kv


def test_leading_sliding_window_layers_do_not_strand_a_group():
    """Layers 0 and 1 have compress_ratio 0 and read no group at all, so a cut
    that leaves only those on a rank needs no transfer."""
    layers, kv, index = 6, (2,), (2,)
    ratios = (0, 0, 1, 1, 1, 1)
    plans = [build_rank_plan(layers, kv, index, ratios, r, 3) for r in range(3)]
    assert [(p.start_layer, p.end_layer) for p in plans] == [(0, 2), (2, 4), (4, 6)]
    # Rank 1 starts exactly on the source.
    assert plans[1].forwarder is None
    # Rank 2 is stranded.
    assert plans[2].forwarder == 4 and plans[2].forwarded_kv_source == 2


@pytest.mark.parametrize("partition", ["10,10,10,10", "5,5,5,5,5,5,5,5", "20,2,2,16"])
def test_truncation_rejects_a_rank_left_with_no_layer(partition, monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    world = len(partition.split(","))
    for rank in range(world):
        with pytest.raises(ValueError, match="no layer"):
            build_rank_plan(
                V41_LAYERS,
                V41_KV_SOURCES,
                V41_INDEX_SOURCES,
                V41_RATIOS,
                rank,
                world,
                last_layer=20,
            )


@pytest.mark.parametrize("partition", ["5,5,5,25", "3,3,3,3,2,3,2,21", "20,20"])
def test_truncation_accepts_a_cut_where_every_rank_keeps_a_layer(
    partition, monkeypatch
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    world = len(partition.split(","))
    for rank in range(world):
        plan = build_rank_plan(
            V41_LAYERS,
            V41_KV_SOURCES,
            V41_INDEX_SOURCES,
            V41_RATIOS,
            rank,
            world,
            last_layer=20,
        )
        assert plan.start_layer < plan.end_layer <= 21


def test_local_kv_source_rejects_a_plan_that_forgot_a_forwarder():
    plan = build_rank_plan(
        V41_LAYERS, V41_KV_SOURCES, V41_INDEX_SOURCES, V41_RATIOS, 0, 1
    )
    # A source below the rank's start with no forwarder is an inconsistent
    # plan, not something to paper over with a remote read.
    broken = plan.__class__(
        start_layer=10,
        end_layer=20,
        forwarder=None,
        forwarded_kv_source=None,
        forwarder_needs_topk=False,
        recvs_kv=True,
    )
    with pytest.raises(AssertionError):
        local_kv_source(12, 8, broken)
