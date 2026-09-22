"""Decode-side sparse MLA, plus the paged caches and attention metadata it reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4.compressor import CompressorBackend
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

# 16 rather than upstream's 4/8: those are V4 C4A/C128A layouts, and 8 is
# read by the metadata builder as the C128 configuration.
BLOCK_SIZE = 16


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
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
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


class V41PagedCache(torch.nn.Module, AttentionLayerBase):

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


class V41WindowKVCache(V41PagedCache):
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
    ixfops.v41_write_cache(cache, flat_rows, flat_slots)


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
    return ixfops.v41_gather_by_positions(
        cache, block_table_row, first, count, storage_block_size
    )


def as_flashmla_cache(cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 4:
        if cache.size(-2) != 1:
            raise ValueError(
                f"paged decode cache expected h_kv=1 at dim -2, got {tuple(cache.shape)}"
            )
        return cache
    if cache.dim() == 3:
        return cache.unsqueeze(-2)
    raise ValueError(
        f"paged decode cache must be 3D or 4D, got {tuple(cache.shape)}"
    )


def cache_slot_limit(cache: torch.Tensor) -> int:
    """Rows a paged cache exposes to v3's dense slot walk.

    Cache is ``[num_blocks, page_block_size, 1, D]``. A slot ``>= num_blocks *
    page_block_size`` (or negative) is an OOB load in ``cuInferDSV4Attention``
    when ``need_gather_indices=false``.
    """
    return int(cache.size(0) * cache.size(1))


def _swa_indices_cuda(
    md: V41AttentionMetadata,
    nd: int,
    window_size: int,
    device: torch.device,
    slot_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # pad_walk_tail makes the kernel write every row in [0, nd), including
    # padding tokens, so neither buffer needs a sentinel prefill.
    swa_indices = torch.empty((nd, window_size), dtype=torch.int32, device=device)
    swa_lens = torch.empty(nd, dtype=torch.int32, device=device)
    valid = md.slot_mapping[:nd] >= 0 if md.slot_mapping.numel() >= nd else torch.ones(
        nd, dtype=torch.bool, device=device
    )
    ixfops.ds4_compute_swa_decode_indices_and_lens(
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        window_size=window_size,
        query_start_loc=md.query_start_loc,
        seq_lens=md.seq_lens,
        token_to_req_indices=md.token_to_req_indices,
        is_valid_token=valid,
        block_table=md.block_table,
        block_size=md.block_size,
        num_decode_tokens=nd,
        slot_limit=slot_limit,
        pad_walk_tail=True,
    )
    # The kernel drops page 0 and out-of-cache slots, compacts the row, writes
    # the packed count and pads the walked tail, so the row is already in the
    # shape v3 wants.
    return swa_indices, swa_lens


def _noncausal_swa_indices_cuda(
    md: V41AttentionMetadata,
    nd: int,
    window_size: int,
    device: torch.device,
    slot_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_swa_indices_cuda`` for a DSpark draft block.

    Rows are ``window_size + n_predict`` wide rather than ``window_size``: every
    token of the request reads the whole block as well as the context before it.
    """
    width = md.noncausal_index_width
    # pad_walk_tail writes every row in [0, nd), padding tokens included, so
    # neither buffer needs a sentinel prefill.
    swa_indices = torch.empty((nd, width), dtype=torch.int32, device=device)
    swa_lens = torch.empty(nd, dtype=torch.int32, device=device)
    valid = md.slot_mapping[:nd] >= 0 if md.slot_mapping.numel() >= nd else torch.ones(
        nd, dtype=torch.bool, device=device
    )
    ixfops.ds4_compute_dspark_noncausal_swa_indices_and_lens(
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        window_size=window_size,
        index_width=width,
        query_start_loc=md.query_start_loc,
        seq_lens=md.seq_lens,
        token_to_req_indices=md.token_to_req_indices,
        is_valid_token=valid,
        block_table=md.block_table,
        block_size=md.block_size,
        num_decode_tokens=nd,
        slot_limit=slot_limit,
        pad_walk_tail=True,
    )
    return swa_indices, swa_lens


def _extra_indices_cuda(
    cache_md: V41AttentionMetadata,
    local_idxs: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    token_to_req_indices: torch.Tensor,
    is_valid: torch.Tensor,
    slot_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    storage_bs = cache_md.block_size // ratio
    # Same contract as the SWA builder: page 0 and out-of-cache slots dropped,
    # row compacted, extra_lens holding the packed count, walked tail padded.
    return ixfops.ds4_compute_global_topk_indices_and_lens(
        local_idxs.contiguous(),
        token_to_req_indices.contiguous(),
        cache_md.block_table.contiguous(),
        storage_bs,
        is_valid,
        positions,
        ratio,
        slot_limit,
        True,
    )



def decode_swa_indices(
    md: V41AttentionMetadata,
    positions: torch.Tensor,
    window_size: int,
    slot_limit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global window slots and per-token lengths for this step's decode rows.

    A DSpark draft block attends forwards as well as back, so its window is a
    property of the request rather than of the querying token; ``causal`` is what
    picks between the two builders.
    """
    nd = int(positions.size(0))
    if not md.causal:
        return _noncausal_swa_indices_cuda(
            md, nd, window_size, positions.device, slot_limit
        )
    return _swa_indices_cuda(md, nd, window_size, positions.device, slot_limit)



def decode_extra_indices(
    cache_md: V41AttentionMetadata,
    local_idxs: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    token_to_req_indices: torch.Tensor,
    is_valid: torch.Tensor | None = None,
    slot_limit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global slots for the groups the indexer selected, in the compressed domain."""
    nd = local_idxs.size(0)
    valid = (
        is_valid
        if is_valid is not None
        else torch.ones(nd, dtype=torch.bool, device=local_idxs.device)
    )
    return _extra_indices_cuda(
        cache_md,
        local_idxs,
        positions,
        ratio,
        token_to_req_indices[:nd],
        valid,
        slot_limit,
    )


def decode_sparse_mla(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_md: V41AttentionMetadata,
    positions: torch.Tensor,
    window_size: int,
    attn_sink: torch.Tensor,
    scale: float,
    compress_ratio: int = 0,
    compress_cache: torch.Tensor | None = None,
    compress_md: V41AttentionMetadata | None = None,
    group_idxs: torch.Tensor | None = None,
    group_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged decode attention. ``q`` is ``(nd, H, 512)``; returns the same shape.

    Inverse RoPE stays outside this call (``forward`` still does
    ``apply_rope(..., inverse=True)``).
    """
    del group_lens
    window_cache = as_flashmla_cache(window_cache)
    # The builders take the cache size and hand back rows already in v3's
    # shape; clamping, packing and walk-padding here cost three more passes
    # over every row, per layer, for a result the kernel can produce as it
    # emits.
    swa_idx, swa_len = decode_swa_indices(
        window_md,
        positions,
        window_size,
        slot_limit=cache_slot_limit(window_cache),
    )

    extra_cache = extra_idx = extra_len = None
    if (
        compress_ratio > 0
        and compress_cache is not None
        and compress_md is not None
        and group_idxs is not None
        and group_idxs.size(-1) > 0
    ):
        extra_cache = as_flashmla_cache(compress_cache)
        extra_idx, extra_len = decode_extra_indices(
            compress_md,
            group_idxs,
            positions,
            compress_ratio,
            compress_md.token_to_req_indices,
            is_valid=(
                compress_md.slot_mapping[: q.size(0)] >= 0
                if compress_md.slot_mapping.numel() >= q.size(0)
                else None
            ),
            slot_limit=cache_slot_limit(extra_cache),
        )
        extra_idx = extra_idx.unsqueeze(1).contiguous()

    out = ixfops.ds4_flash_mla_with_kvcache(
        q=q.unsqueeze(1).contiguous(),
        k_cache=as_flashmla_cache(window_cache),
        indices=swa_idx.unsqueeze(1).contiguous(),
        topk_length=swa_len.contiguous(),
        sm_scale=scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices=extra_idx,
        extra_topk_length=extra_len,
        version=3,
    )
    return out.squeeze(1)
