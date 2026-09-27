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
    # Set on every rank when *any* rank needs a transfer: the pipeline's
    # intermediate tensors have to carry the same keys everywhere, so the
    # decision to carry them is global even though the need is local.
    transfers_kv: bool
    transfers_topk: bool
    # The two-level candidate shortlist is published by a single layer and read
    # by the index sources above it. A cut between them strands the readers the
    # same way a cut inside a kv group strands its consumers, except there is
    # nothing to replicate: the shortlist is a plain per-rank buffer, so it
    # rides the pipeline and every rank copies it in on arrival.
    transfers_candidates: bool = False


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
        start <= source < end and source > candidate_source
        for source in index_sources
    )


def build_rank_plan(
    num_layers: int,
    kv_sources: tuple[int, ...],
    index_sources: tuple[int, ...],
    compress_ratios: tuple[int, ...],
    rank: int,
    world: int,
    candidate_source: int = -1,
) -> RankKvPlan:
    """Plan this rank's share of the cut, and whether the pipeline carries KV.

    Every rank walks every other rank's bounds, so the global ``transfers_*``
    flags come out identical without any communication.
    """
    start, end = get_pp_indices(num_layers, rank, world)
    if world == 1:
        return RankKvPlan(start, end, None, None, False, False, False)

    transfers_kv = False
    transfers_topk = False
    transfers_candidates = False
    for other in range(1, world):
        other_start, other_end = get_pp_indices(num_layers, other, world)
        forwarder, _, needs_topk = _rank_needs(
            other_start, other_end, kv_sources, index_sources, compress_ratios
        )
        if forwarder is not None:
            transfers_kv = True
            transfers_topk = transfers_topk or needs_topk
        transfers_candidates = transfers_candidates or _rank_reads_remote_candidates(
            other_start, other_end, index_sources, candidate_source
        )

    forwarder, kv_source, needs_topk = (
        _rank_needs(start, end, kv_sources, index_sources, compress_ratios)
        if rank > 0
        else (None, None, False)
    )
    return RankKvPlan(
        start_layer=start,
        end_layer=end,
        forwarder=forwarder,
        forwarded_kv_source=kv_source,
        forwarder_needs_topk=needs_topk,
        transfers_kv=transfers_kv,
        transfers_topk=transfers_topk,
        transfers_candidates=transfers_candidates,
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
