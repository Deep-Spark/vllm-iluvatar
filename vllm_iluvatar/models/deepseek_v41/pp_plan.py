"""Where a DeepSeek-V4.1 pipeline cut breaks a KV group, and what to do about it.

V4.1 produces compressed KV and index keys on a handful of layers only --
``kv_source_layer_ids`` for the compressed KV and the indexer K cache,
``index_source_layer_ids`` (a superset) for the top-k selection. Every other
layer reads whatever the nearest source below it published. A source and its
consumers therefore form a *group*, and a group is a pipeline hazard: vLLM
gives each rank its own KV caches, so a consumer stranded on a later rank has
nothing to read.

The fix is a replica. The source writes its records a second time into a dense
``[num_tokens, width]`` buffer, that buffer rides the pipeline alongside the
hidden states, and the first affected layer on the next rank -- the
*forwarder* -- writes it into a cache of its own using its own slot mapping.
Step after step the replica accumulates into a full copy of the source's cache,
and the consumers on that rank read the forwarder instead of the real source.

Three things here go beyond the shape that only had to serve a four-way cut of
layers 0-20:

* **Any pipeline size.** Bounds come from ``get_pp_indices``, the same helper
  ``make_layers`` uses, so the plan cannot disagree with the layer list and
  ``VLLM_PP_LAYER_PARTITION`` keeps working.

* **Groups spanning three or more ranks.** A middle rank has no local source to
  fill its outgoing buffer, so its forwarder relays: it copies what it received
  into the outgoing buffer after writing its own cache. Any local source later
  on that rank overwrites the relay, which is what the next rank wants.

* **Index groups.** ``index_source_layer_ids`` is denser than the KV sources
  (2, 8, 14, 20, 24, 28, 32, 36 against 2, 8, 14, 20), so a cut can strand the
  top-k selection while leaving the KV group intact -- a rank starting at layer
  30 reads KV from source 20 and top-k from source 28, two different remote
  layers. Because the top-k buffer is per-rank and rewritten by every index
  source, a rank does not need a designated top-k sender: whatever the buffer
  holds when the rank finishes is by construction the latest selection, either
  its own last index source's or the one it relayed.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.utils import get_pp_indices


@dataclass(frozen=True)
class RankKvPlan:
    """What one pipeline rank has to do about the groups its cut breaks."""

    start_layer: int
    end_layer: int
    # Local layer owning the replica of a group that started on an earlier
    # rank, or None when this rank's cut broke nothing.
    forwarder: int | None
    # The real source that replica stands in for. Only for logging and asserts;
    # the forwarder is addressed by its own layer id everywhere else.
    forwarded_kv_source: int | None
    # The forwarder also has to take the top-k selection over, because this
    # rank's first compressed layer has no local index source above it.
    forwarder_needs_topk: bool
    # Set on every rank when *any* rank needs a transfer. Only decides whether
    # the dense replica buffers are allocated at all; which keys actually ride
    # a given hop is the ``sends_*`` / ``recvs_*`` pair below.
    transfers_kv: bool
    transfers_topk: bool
    # The two-level candidate shortlist is published by a single layer and read
    # by the index sources above it. A cut between them strands the readers the
    # same way a cut inside a kv group strands its consumers, except there is
    # nothing to replicate: the shortlist is a plain per-rank buffer, so it
    # rides the pipeline and every rank copies it in on arrival.
    transfers_candidates: bool = False
    # Per-hop key set. ``recvs_*`` is what rank-1 puts on the wire for us and
    # therefore what ``make_empty_intermediate_tensors`` has to allocate;
    # ``sends_*`` is what we have to put on the wire for rank+1, and equals the
    # next rank's ``recvs_*`` by construction.
    #
    # These exist because the payload is large and most of it is dead weight on
    # most hops. Per token a hop carries 40960 B of hc hidden states, 8192 B of
    # candidates and 2048 B of top-k; sending candidates on every hop rather
    # than only between the publisher and its last reader is ~10% of the total
    # bytes on a PP16 cut. Narrowing is safe because the receiver allocates
    # from the *sender's* metadata (`GroupCoordinator.irecv_tensor_dict`), so
    # the key set is free to differ per hop.
    recvs_kv: bool = False
    recvs_topk: bool = False
    recvs_candidates: bool = False
    sends_kv: bool = False
    sends_topk: bool = False
    sends_candidates: bool = False


def _group_source(layer_id: int, sources: tuple[int, ...]) -> int | None:
    """The source layer whose group ``layer_id`` belongs to."""
    eligible = [source for source in sources if source <= layer_id]
    return max(eligible) if eligible else None


def _first_compressed_layer(
    start: int, end: int, compress_ratios: tuple[int, ...]
) -> int | None:
    """First layer of the range that reads a compressed cache at all.

    Usually ``start``. Layers with ``compress_ratio == 0`` are pure sliding
    window: they read no group and cannot strand one.
    """
    for layer_id in range(start, end):
        if layer_id < len(compress_ratios) and compress_ratios[layer_id] > 0:
            return layer_id
    return None


def _rank_needs(
    start: int,
    end: int,
    kv_sources: tuple[int, ...],
    index_sources: tuple[int, ...],
    compress_ratios: tuple[int, ...],
) -> tuple[int | None, int | None, bool]:
    """``(forwarder, forwarded_kv_source, needs_topk)`` for one rank."""
    layer_id = _first_compressed_layer(start, end, compress_ratios)
    if layer_id is None:
        return None, None, False
    kv_source = _group_source(layer_id, kv_sources)
    if kv_source is None or kv_source >= start:
        # The group starts on this rank, so nothing is stranded. The top-k
        # cannot be stranded either: the KV sources are a subset of the index
        # sources, so a local KV source means a local index source too.
        return None, None, False
    index_source = _group_source(layer_id, index_sources)
    needs_topk = index_source is None or index_source < start
    return layer_id, kv_source, needs_topk


def _rank_reads_remote_candidates(
    start: int,
    end: int,
    index_sources: tuple[int, ...],
    candidate_source: int,
) -> bool:
    """Whether this rank scores against a shortlist published elsewhere."""
    if candidate_source < 0 or start <= candidate_source < end:
        return False
    return any(
        start <= source < end and source > candidate_source for source in index_sources
    )


def build_rank_plan(
    num_layers: int,
    kv_sources: tuple[int, ...],
    index_sources: tuple[int, ...],
    compress_ratios: tuple[int, ...],
    rank: int,
    world: int,
    candidate_source: int = -1,
    last_layer: int | None = None,
) -> RankKvPlan:
    """Plan this rank's share of the cut, and what each hop has to carry.

    Every rank walks every other rank's bounds, so the global ``transfers_*``
    flags and the per-hop key sets come out consistent without any
    communication: rank r's ``sends_*`` is computed from rank r+1's needs and
    is by construction equal to rank r+1's ``recvs_*``.

    ``last_layer`` truncates the model after that layer (a PD prefiller stops
    at the last KV source). The cut itself still comes from ``get_pp_indices``
    over the full depth, so ``VLLM_PP_LAYER_PARTITION`` keeps validating
    against the checkpoint; only the per-rank bounds are clipped, which can
    leave a trailing rank empty.
    """
    depth = num_layers if last_layer is None else min(num_layers, last_layer + 1)

    def clip(of_rank: int) -> tuple[int, int]:
        start, end = get_pp_indices(num_layers, of_rank, world)
        return min(start, depth), min(end, depth)

    rank_bounds = [clip(r) for r in range(world)]
    start, end = rank_bounds[rank]
    if world == 1:
        return RankKvPlan(start, end, None, None, False, False, False)

    # Rank 0 starts every group it reads, so it is never stranded.
    needs = [(None, None, False)] + [
        _rank_needs(s, e, kv_sources, index_sources, compress_ratios)
        for s, e in rank_bounds[1:]
    ]
    reads_candidates = [
        _rank_reads_remote_candidates(s, e, index_sources, candidate_source)
        for s, e in rank_bounds
    ]

    transfers_kv = any(n[0] is not None for n in needs)
    transfers_topk = any(n[2] for n in needs)
    transfers_candidates = any(reads_candidates[1:])

    # The shortlist only has to travel from the rank that publishes it to the
    # last rank that scores against it; before the publisher the local buffer
    # holds nothing anyone may read.
    candidate_rank = next(
        (r for r, (s, e) in enumerate(rank_bounds) if s <= candidate_source < e),
        None,
    )

    def recvs_candidates(r: int) -> bool:
        if candidate_rank is None or r <= candidate_rank:
            return False
        return any(reads_candidates[r:])

    forwarder, kv_source, needs_topk = needs[rank]
    nxt = rank + 1
    return RankKvPlan(
        start_layer=start,
        end_layer=end,
        forwarder=forwarder,
        forwarded_kv_source=kv_source,
        forwarder_needs_topk=needs_topk,
        transfers_kv=transfers_kv,
        transfers_topk=transfers_topk,
        transfers_candidates=transfers_candidates,
        recvs_kv=forwarder is not None,
        recvs_topk=needs_topk,
        recvs_candidates=recvs_candidates(rank),
        sends_kv=nxt < world and needs[nxt][0] is not None,
        sends_topk=nxt < world and needs[nxt][2],
        sends_candidates=nxt < world and recvs_candidates(nxt),
    )


def local_kv_source(
    layer_id: int, real_source: int | None, plan: RankKvPlan
) -> int | None:
    """Cache owner ``layer_id`` actually reads on this rank.

    A source on this rank is read directly. A source on an earlier rank is
    read through the forwarder's replica. There is at most one forwarder per
    rank, and it sits at or before every layer whose source is remote -- a
    layer past a local source has a local source by definition -- so no search
    is needed.
    """
    if real_source is None or real_source >= plan.start_layer:
        return real_source
    assert plan.forwarder is not None, (
        f"layer {layer_id} reads kv source {real_source} from an earlier "
        f"pipeline rank, but rank bounds [{plan.start_layer}, "
        f"{plan.end_layer}) planned no forwarder"
    )
    return plan.forwarder
