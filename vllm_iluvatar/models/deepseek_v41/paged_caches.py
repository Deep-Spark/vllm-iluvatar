"""V4.1's four caches, declared as vLLM-managed paged layers so requests can batch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4.compressor import CompressorBackend
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_iluvatar.models.deepseek_v41.rope import apply_rope
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

# 16 rather than upstream's 4/8: those are V4 C4A/C128A layouts, and 8 is
# read by the metadata builder as the C128 configuration.
BLOCK_SIZE = 16

# V41_CHECK_CACHE=1 runs each cache kernel's reference chain -- which is the
# torch expression chain the kernel replaced -- alongside it and reports any
# disagreement. The unit tests already compare the two, but only on shapes the
# tests thought to build; this establishes the same equality on whatever the
# model actually asks for, which is what licenses deleting the torch path.
_CHECK_CACHE = os.environ.get("V41_CHECK_CACHE", "0").strip() == "1"
_CHECK_STATS: dict[str, list[int]] = {}

# A DSpark verify step submits ``1 + num_speculative_tokens`` queries per request.
# With the threshold left at 1 those requests count as prefills, so ``_attend``
# sends them to ``prefill_attn``, which materialises each request's whole plane
# once per layer: at concurrency 4 that is 305 gathers and 162 index-plane builds
# per step, 5.1 ms and 1.7 ms of GPU time, for a batch of 24 tokens. The paged
# decode path needs none of it, and already handles multi-token blocks -- the
# draft's own non-causal block goes through the same index builders. Every other
# sparse MLA backend in vLLM declares this; V4.1's builder was the one that did
# not.
#
# Measured at concurrency 4 with a 5-token draft: 3247 -> 2677 kernels and
# 55.34 -> 52.77 ms of GPU work per step, 89.30 -> 81.83 ms per step, and GSM8K
# 500 questions from 5.9 to 4.9 minutes at an unchanged 489/500. Set to 0 to fall
# back to the prefill path.
_SPEC_AS_DECODE = os.environ.get("V41_SPEC_AS_DECODE", "1").strip() == "1"


def _check_report(kind: str, ok: bool, elems: int, detail: str = "") -> None:
    st = _CHECK_STATS.setdefault(kind, [0, 0, 0])
    st[0] += 1
    st[1] += elems
    if not ok:
        st[2] += 1
        print(f"[CHECK_CACHE] {kind} 不一致: {detail}", flush=True)
    total = sum(s[0] for s in _CHECK_STATS.values())
    if total % 500 == 0:
        rows = [
            f"{k}: {v[0]} 次 / {v[1]} 元素 / 不一致 {v[2]}"
            for k, v in sorted(_CHECK_STATS.items())
        ]
        print("[CHECK_CACHE] " + " || ".join(rows), flush=True)


__all__ = [
    "BLOCK_SIZE",
    "V41AttentionMetadata",
    "V41CompressKVCache",
    "V41CompressorBackend",
    "V41CompressorStateCache",
    "V41IndexKeyCache",
    "V41MetadataBuilder",
    "V41WindowKVCache",
    "gather_group_rows",
    "gather_window_rows",
    "group_slots_from_slot_mapping",
    "write_cache",
]


def write_cache(cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
    """Scatter rows into paged cache. Invalid slots (``<0`` or OOB) are skipped.

    The kernel skips invalid rows outright. Torch could not: selecting a subset
    needs the count on the host, and ``nonzero`` shipping it back drained the
    pipeline once per call — 48 times per prefill forward, which profiled as the
    whole of the GPU's idle time. (Boolean ``tensor[mask]`` was not an option
    either: that path is ``THPVariable_getitem`` → ``as_strided`` /
    ``ViewInfo::chain``, which segfaulted after decode WKV.) So the torch version
    aimed invalid rows at row 0 and handed that row its own contents back,
    which only held because row 0 belongs to vLLM's null block. Skipping drops
    that assumption along with fourteen launches per call.
    """
    if slots.numel() == 0 or rows.numel() == 0:
        return
    flat_rows = rows.reshape(slots.numel(), -1)
    flat_slots = slots.reshape(-1).to(torch.int32)

    if _CHECK_CACHE:
        # Cloning the cache to run the torch chain against is not an option here
        # -- it is hundreds of megabytes and this is called 48 times a forward --
        # so check the two properties that distinguish the kernel instead: every
        # valid slot holds its row afterwards, and row 0 is untouched. Row 0 is
        # the interesting one because the torch chain deliberately rewrote it on
        # every invalid slot, and dropping that is the behaviour change. That no
        # *other* row is clobbered is what the unit tests cover, on caches small
        # enough to compare whole.
        flat = cache.reshape(-1, flat_rows.size(-1))
        n_rows = flat.size(0)
        valid = (flat_slots >= 0) & (flat_slots < n_rows)
        touched = torch.masked_select(flat_slots, valid).to(torch.int64)
        row0_before = flat[0].clone() if not bool((touched == 0).any()) else None

        ixfops.v41_write_cache(cache, flat_rows, flat_slots)

        want_rows = flat_rows[torch.masked_select(
            torch.arange(flat_slots.numel(), device=flat.device), valid
        )]
        ok = torch.equal(flat.index_select(0, touched), want_rows)
        if row0_before is not None and not torch.equal(flat[0], row0_before):
            ok = False
        _check_report(
            "write",
            ok,
            int(touched.numel()),
            f"rows={tuple(flat_rows.shape)} 有效={int(touched.numel())} "
            f"无效={int(flat_slots.numel() - touched.numel())}",
        )
        return

    ixfops.v41_write_cache(cache, flat_rows, flat_slots)


def compress_rope_qdq_insert_by_token(
    cache: torch.Tensor,
    x: torch.Tensor,
    md: "V41AttentionMetadata",
    ratio: int,
    qdq_block: int,
    qdq_fmt: str,
    cos_sin: torch.Tensor,
) -> None:
    """``compress_rope_qdq_insert`` for rows indexed by token, not by group.

    The fused kernel packs its rows in closing order, which costs it a prefix
    count over the batch -- and a graph-safe compressor cannot hand it that
    packing, because how many groups close in a step is device data. So rotate
    and quantize every row and let a ``-1`` slot drop the ones carrying nothing:
    three launches instead of one, over ``ratio`` times as many rows, against the
    dozen launches per request that the fused compressor removes.

    Unlike ``compress_rope_qdq_insert``, this mutates ``x``. Nothing reads the
    latent afterwards, but ``publish_keys`` derives the index keys from the
    pre-RoPE rows, so it has to run first.

    The width is ``x``'s own row count, not ``num_actual_tokens``: that one moves
    from step to step, so slicing by it would bake a capture-time batch into the
    graph. Padding needs no trimming here anyway -- vLLM already wrote ``-1`` into
    its ``slot_mapping``, which ``compressed_token_slots`` carries through.
    """
    n = x.size(0)
    apply_rope(x.unsqueeze(0), group_rope_positions(md.positions[:n], ratio), cos_sin)
    ixfops.ds4_act_qdq(x, qdq_block, qdq_fmt)
    write_cache(cache, x, compressed_token_slots(md, ratio)[:n])


def group_rope_positions(positions: torch.Tensor, ratio: int) -> torch.Tensor:
    """First raw position of the ratio group that ends at each token."""
    if ratio == 1:
        return positions.to(torch.int32)
    return (positions // ratio * ratio).to(torch.int32)


def compressed_token_slots(md: V41AttentionMetadata, ratio: int) -> torch.Tensor:
    """One compressed-row slot per token: the group closing at that position.

    The per-group addressing ``group_slots_from_slot_mapping`` derives, but
    indexed by token and with ``-1`` wherever there is nothing to write -- an
    open group, graph padding (vLLM already put ``-1`` in ``slot_mapping``), or a
    null page. That is what lets a caller hand ``write_cache`` a full
    ``[num_tokens, width]`` buffer: the rows it must not publish address slot
    ``-1`` and are skipped, so no row count ever depends on device data.

    The compressor's state slots cannot stand in for these: every KV cache group
    gets its own vLLM block table, so the physical pages of the state,
    compressed-KV and index-key caches differ.
    """
    if ratio == 1:
        # One compressed row per token, in this cache's own page geometry, which
        # is exactly what vLLM's slot mapping already holds.
        return md.slot_mapping
    positions = md.positions
    storage = md.block_size // ratio
    groups = positions // ratio
    block_table = md.block_table
    n_blocks = block_table.size(1)
    blk_idx = groups // storage
    req = md.token_to_req_indices.to(torch.int64).clamp(max=block_table.size(0) - 1)
    blocks = block_table[req, blk_idx.clamp(max=n_blocks - 1)].to(torch.int64)
    slots = blocks * storage + groups % storage
    ok = (
        (blk_idx < n_blocks)
        & (blocks > 0)
        & ((positions + 1) % ratio == 0)
        & (md.slot_mapping >= 0)
    )
    return torch.where(ok, slots, slots.new_full((), -1))


def group_slots_from_slot_mapping(
    md: V41AttentionMetadata, ratio: int, num_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compressed-domain slot and group start position for every group this
    batch closes, derived from the framework's own per-token ``slot_mapping``.

    vLLM already publishes, per cache layer, a slot for each token in the batch:

        slot_mapping[t] = block_table[req, pos // bs] * bs + pos % bs

    A compressed cache holds one row per ``ratio`` tokens, so its rows are
    addressed in the group domain:

        group_slot = block_table[req, g // (bs // ratio)] * (bs // ratio)
                     + g % (bs // ratio)

    The two pick the *same* physical block -- for a token at ``pos`` closing
    group ``g = pos // ratio``, ``g // (bs // ratio) == pos // bs`` -- so the
    physical block number can be recovered from the published slot as
    ``slot_mapping[t] // bs``, and the offset within it as
    ``(slot_mapping[t] % bs) // ratio``. That makes the group slot a closed form
    of ``slot_mapping``: no block-table gather, no per-request loop, and no host
    round trip to build group ids.

    A group is closed by its last token, ``(pos + 1) % ratio == 0``, and the set of
    groups reached that way is exactly the set a request closes in this step: for
    ``start_pos = q * ratio + t`` with ``0 <= t < ratio``, the smallest group whose
    last token is at or after ``start_pos`` is ``ceil((start_pos + 1) / ratio) - 1``,
    which is ``q``.

    ``num_groups`` is how many groups the caller expects, which it already knows
    on the host from its own group spans. Taking it as an argument keeps this
    function free of device-to-host syncs: boolean mask indexing would have to
    count the selected tokens on the device and read that count back before it
    could size the result, so instead every token computes a slot and a running
    cumsum sends it to its compacted position, with non-boundary tokens dumped
    into one scratch entry past the end.

    Returns ``(slots, group_start_positions)``, both ordered by batch token
    index, which is request-major and ascending within a request.
    """
    n = md.num_actual_tokens
    sm = md.slot_mapping[:n].to(torch.int64)
    pos = md.positions[:n].to(torch.int64)
    closes = ((pos + 1) % ratio == 0) & (sm >= 0)

    rows_per_block = md.block_size // ratio
    all_slots = (sm // md.block_size) * rows_per_block + (sm % md.block_size) // ratio
    all_pos = pos - (ratio - 1)

    dest = torch.where(closes, closes.cumsum(0) - 1, torch.full_like(sm, num_groups))
    out_slots = sm.new_zeros(num_groups + 1).scatter_(0, dest, all_slots)
    out_pos = sm.new_zeros(num_groups + 1).scatter_(0, dest, all_pos)
    return (
        out_slots[:num_groups].to(torch.int32),
        out_pos[:num_groups].to(torch.int32),
    )


def compress_rope_qdq_insert(
    cache: torch.Tensor,
    x: torch.Tensor,
    md: "V41AttentionMetadata",
    ratio: int,
    qdq_block: int,
    qdq_fmt: str,
    cos_sin: torch.Tensor,
) -> None:
    """Publish this batch's compressed rows: RoPE, qdq, and the cache write.

    One kernel for what was four calls -- derive the slots, rotate, quantize,
    scatter -- of which the slot derivation alone cost a cumsum and two scatters
    over the whole batch. See ``ixformer``'s ``v41_compress_rope_qdq_insert``;
    it works the same way ``group_slots_from_slot_mapping`` documents, just
    without materialising the slots.

    ``x`` is left untouched, where the chain rotated and quantized it in place.
    No caller reads it afterwards.
    """
    n = md.num_actual_tokens
    ixfops.v41_compress_rope_qdq_insert(
        cache, x, md.slot_mapping[:n], md.positions[:n], cos_sin,
        md.block_size, ratio, qdq_block, qdq_fmt,
    )

    if _CHECK_CACHE:
        # Against the production chain rather than the torch reference: the unit
        # tests already pin the kernel to the reference, so what is worth
        # establishing here is that the chain the server was running and the
        # kernel that replaced it agree on the shapes the model really produces.
        # Imported here rather than at module scope because rope.py is a leaf and
        # this is the only thing in the file that needs it.
        from vllm_iluvatar.models.deepseek_v41.rope import apply_rope

        slots, pos = group_slots_from_slot_mapping(md, ratio, x.shape[0])
        shadow = x.clone()
        apply_rope(shadow.unsqueeze(0), pos, cos_sin)
        ixfops.ds4_act_qdq(shadow, qdq_block, qdq_fmt)
        flat = cache.reshape(-1, x.size(-1))
        valid = (slots >= 0) & (slots < flat.size(0))
        idx = torch.masked_select(slots, valid).to(torch.int64)
        want = shadow[valid]
        got = flat.index_select(0, idx)
        ok = bool(torch.equal(got, want))
        _check_report(
            "compress_rope_qdq_insert", ok, int(want.numel()),
            f"rows={idx.numel()} width={x.size(-1)} fmt={qdq_fmt} ratio={ratio}",
        )


def gather_window_rows(
    cache: torch.Tensor,
    md: "V41AttentionMetadata",
    req: int,
    first_position: int,
    count: int,
) -> torch.Tensor:
    """Rows of a window cache, whose rows are one per raw token, for the
    positions ``first_position + [0, count)`` of one request.

    Resolving the block table and gathering through it is one kernel: see
    ``ixformer``'s ``v41_gather_by_positions``. Splitting it into an address
    array and a gather, as this used to be, cost about twenty-eight aten
    launches per call and an ``arange`` to feed it, once per cache per layer per
    step, which is why the range is passed as two integers.
    """
    return _gather(cache, md.block_table[req], first_position, count, md.block_size)


def gather_group_rows(
    cache: torch.Tensor,
    md: "V41AttentionMetadata",
    req: int,
    first_group: int,
    num_groups: int,
    ratio: int,
) -> torch.Tensor:
    """Rows of a compressed cache, which holds one row per ``ratio`` tokens, for
    the groups ``first_group + [0, num_groups)`` of one request."""
    return _gather(
        cache, md.block_table[req], first_group, num_groups, md.block_size // ratio
    )


def _gather(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    first: int,
    count: int,
    storage_block_size: int,
) -> torch.Tensor:
    got = ixfops.v41_gather_by_positions(
        cache, block_table_row, first, count, storage_block_size
    )
    if _CHECK_CACHE:
        want = ixfops.ref_v41_gather_by_positions(
            cache, block_table_row, first, count, storage_block_size
        )
        _check_report(
            "gather",
            got.shape == want.shape and torch.equal(got, want),
            got.numel(),
            f"out={tuple(got.shape)} first={first} count={count} "
            f"bs={storage_block_size} blocks={block_table_row.numel()}",
        )
    return got


@dataclass
class V41AttentionMetadata:
    # written by every cache
    block_table: torch.Tensor        # [num_reqs, max_blocks_per_req] int32
    slot_mapping: torch.Tensor       # [num_tokens] int64, -1 = padding
    block_size: int

    # int64 because ``ds4_compressor_save_partial_states`` contracts for it; vLLM's own positions
    # tensor is int64 already, so this is a reference rather than a cast.
    positions: torch.Tensor          # [num_tokens] int64

    # who owns which tokens
    query_start_loc: torch.Tensor        # [num_reqs + 1] int32
    query_start_loc_cpu: torch.Tensor    # same, on CPU
    seq_lens: torch.Tensor               # [num_reqs] int32, total length incl. prefix
    seq_lens_cpu: torch.Tensor           # same, on CPU
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int

    # vLLM sorts decodes before prefills, so a batch splits at these boundaries
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    token_to_req_indices: torch.Tensor   # [num_tokens] int32

    # A DSpark draft block attends forwards as well as back, which changes the
    # SWA window from per-token to per-request and widens the rows it needs
    # (``build_noncausal_swa_decode_indices``). ``causal`` is what the community
    # speculator sets on the draft's ``CommonAttentionMetadata``; the width is
    # constant, so the builder computes it once.
    causal: bool = True
    noncausal_index_width: int = 0


class V41MetadataBuilder(AttentionMetadataBuilder):

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = self.kv_cache_spec.block_size
        # ``split_decodes_and_prefills`` assumes decodes are already at the front.
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=_SPEC_AS_DECODE
        )
        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.token_arange = torch.arange(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        spec_config = self.vllm_config.speculative_config
        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        # A draft token sees the window before its block plus the whole block, so
        # at most ``window_size`` + the block length in keys. The block is one
        # token longer than ``num_speculative_tokens`` in the layout DFlash picks
        # when it drafts the bonus token too. Padded to 128 like V4's SWA
        # builder, whose decode kernel pads the q-head count.
        self.noncausal_index_width = (
            cdiv(
                self.vllm_config.model_config.hf_config.window_size
                + spec_config.num_speculative_tokens
                + 1,
                128,
            )
            * 128
            if self.is_dspark
            else 0
        )

    def _derive_positions(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        token_to_req: torch.Tensor,
    ) -> torch.Tensor:
        """Absolute positions from the query layout alone.

        A DSpark draft block arrives without positions: the community speculator
        builds its draft metadata through ``build_attn_metadata``, whose
        ``positions`` argument is optional and which it never fills. They are
        fully determined anyway -- a request whose query spans ``[qs, qe)`` inside
        a sequence of length ``seq_len`` starts that span at ``seq_len - (qe -
        qs)`` -- and this is the same arithmetic ``ds4_compute_swa_decode_indices_
        and_lens`` does internally to place each token in its window.

        Only the draft reaches this path; the model runner always supplies
        positions for the target, so a mismatch here cannot mask a bug there.
        """
        num_tokens = common_attn_metadata.num_actual_tokens
        req = token_to_req[:num_tokens].long()
        query_start_loc = common_attn_metadata.query_start_loc
        starts = query_start_loc[:-1].index_select(0, req)
        query_lens = query_start_loc[1:].index_select(0, req) - starts
        prefix_lens = common_attn_metadata.seq_lens.index_select(0, req) - query_lens
        offsets = self.token_arange[:num_tokens] - starts
        return (prefix_lens + offsets).to(torch.int64)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> V41AttentionMetadata:
        # The split has to use the same threshold the scheduler reordered by, or
        # the two disagree: with ``supports_spec_as_decode`` the threshold is
        # ``1 + num_speculative_tokens``, and hardcoding 1 here sends every
        # DSpark verify step down the prefill path regardless.
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
        )
        token_to_req = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        positions = common_attn_metadata.positions
        if positions is None:
            positions = self._derive_positions(common_attn_metadata, token_to_req)

        causal = common_attn_metadata.causal
        assert causal or self.is_dspark, (
            "non-causal deepseek_v41 SWA is only defined for a DSpark draft "
            "block, but causal=False arrived without DSpark configured"
        )

        return V41AttentionMetadata(
            # Keep ``-1`` sentinels. Clamping them to 0 aliases every hole onto block 0.
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            positions=positions[: common_attn_metadata.num_actual_tokens],
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            # The property, not ``seq_lens.cpu()``: the runner has already filled
            # the host copy from its own bookkeeping, so reading it costs nothing,
            # while a fresh ``.cpu()`` is a blocking transfer that makes the CPU
            # wait out every kernel queued so far -- once per cache group per step.
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            token_to_req_indices=token_to_req,
            causal=causal,
            noncausal_index_width=self.noncausal_index_width,
        )


class V41CompressorBackend(CompressorBackend):

    @staticmethod
    def get_name() -> str:
        return "V41CompressorBackend"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 512, 1024]

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return V41MetadataBuilder


class _V41PagedCache(torch.nn.Module, AttentionLayerBase):

    def __init__(self, *, head_size: int, dtype: torch.dtype, block_size: int,
                 prefix: str) -> None:
        super().__init__()
        self.head_size = head_size
        self.dtype = dtype
        self.block_size = block_size
        self.prefix = prefix
        # vLLM fills this in during cache allocation; the layer only holds the handle.
        self.kv_cache = torch.tensor([])

        if block_size == 8:
            raise ValueError(
                "block_size 8 selects upstream's C128 path in CompressorMetadataBuilder; "
                f"pick another size for V4.1 (layer {prefix})"
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self):  # noqa: D102 - a cache layer never runs; vLLM only needs the hooks
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return V41CompressorBackend


class V41WindowKVCache(_V41PagedCache):
    def __init__(self, *, head_size: int, dtype: torch.dtype, window_size: int,
                 block_size: int, prefix: str) -> None:
        super().__init__(head_size=head_size, dtype=dtype, block_size=block_size, prefix=prefix)
        self.window_size = window_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            sliding_window=self.window_size,
            alignment=512,
            # One row per raw token: the window is not compressed.
            compress_ratio=1,
        )


class V41CompressKVCache(_V41PagedCache):

    def __init__(self, *, head_size: int, dtype: torch.dtype, compress_ratio: int,
                 block_size: int, prefix: str) -> None:
        super().__init__(head_size=head_size, dtype=dtype, block_size=block_size, prefix=prefix)
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        if block_size % compress_ratio:
            raise ValueError(
                f"block_size {block_size} must be a multiple of compress_ratio "
                f"{compress_ratio}, or a page would hold a fractional row (layer {prefix})"
            )
        self.compress_ratio = compress_ratio

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )


class V41IndexKeyCache(V41CompressKVCache):
    """Indexer keys: same page geometry as compressed KV, ``index_head_dim`` rows."""


class V41CompressorStateCache(_V41PagedCache):

    def __init__(self, *, state_dim: int, compress_ratio: int, block_size: int,
                 prefix: str) -> None:
        super().__init__(head_size=state_dim, dtype=torch.float32, block_size=block_size,
                         prefix=prefix)
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        self.state_dim = state_dim
        self.compress_ratio = compress_ratio
        self.sliding_window = compress_ratio

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=512,
            compress_ratio=1,
        )
