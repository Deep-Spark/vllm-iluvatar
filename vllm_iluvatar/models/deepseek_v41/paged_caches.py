"""V4.1's four caches, declared as vLLM-managed paged layers so requests can batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

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
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

# 16 rather than upstream's 4/8: those are V4 C4A/C128A layouts, and 8 is
# read by the metadata builder as the C128 configuration.
BLOCK_SIZE = 16

__all__ = [
    "BLOCK_SIZE",
    "V41AttentionMetadata",
    "V41CompressKVCache",
    "V41CompressorBackend",
    "V41CompressorStateCache",
    "V41IndexKeyCache",
    "V41MetadataBuilder",
    "V41WindowKVCache",
    "gather_rows",
    "group_slots",
    "slots_for_positions",
    "window_slots",
    "write_cache",
]


def slots_for_positions(
    block_table_row: torch.Tensor,
    positions: torch.Tensor,
    storage_block_size: int,
) -> torch.Tensor:
    pos = positions.to(torch.int64)
    n_blocks = block_table_row.size(0)
    if n_blocks == 0 or pos.numel() == 0:
        return pos.new_full(pos.shape, -1, dtype=torch.int32)
    blk_idx = pos // storage_block_size
    in_range = (blk_idx >= 0) & (blk_idx < n_blocks)
    safe_idx = blk_idx.clamp(0, n_blocks - 1)
    blocks = block_table_row.to(torch.int64).index_select(0, safe_idx)
    slots = blocks * storage_block_size + pos % storage_block_size
    # Block 0 is vLLM's null placeholder. Unused / SWA-evicted columns are 0,
    # not -1; treating 0 as a real page writes the shared null block.
    ok = in_range & (blocks > 0)
    return torch.where(ok, slots, slots.new_full((), -1)).to(torch.int32)


def write_cache(cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
    """Scatter rows into paged cache. Invalid slots (``<0`` or OOB) are skipped.

    Do not use boolean ``tensor[mask]`` here: that path is ``THPVariable_getitem``
    → ``as_strided`` / ``ViewInfo::chain``, which segfaulted after decode WKV.

    ``nonzero`` sidesteps that but has to ship its output size back to the host,
    draining the pipeline once per call — 48 times per prefill forward, which
    profiled as the whole of the GPU's idle time. So instead of selecting the
    valid rows, invalid ones are aimed at row 0 and handed that row's own current
    contents: whichever duplicate index wins writes it back unchanged. Row 0
    belongs to vLLM's null block (``BlockPool`` pops it off the free queue and
    never hands it out), so no valid slot can name it.
    """
    if slots.numel() == 0 or rows.numel() == 0:
        return
    flat = cache.reshape(-1, rows.size(-1))
    n_rows = int(flat.size(0))
    if n_rows == 0:
        return
    idx = slots.reshape(-1).to(torch.int64)
    src = rows.reshape(idx.size(0), -1)
    valid = (idx >= 0) & (idx < n_rows)
    safe_idx = torch.where(valid, idx, idx.new_zeros(()))
    safe_src = torch.where(valid.unsqueeze(-1), src, flat[0].unsqueeze(0))
    flat.index_copy_(0, safe_idx, safe_src)


def gather_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    flat = cache.reshape(-1, cache.size(-1))
    n_rows = int(flat.size(0))
    idx = slots.reshape(-1).to(torch.int64)
    if n_rows == 0:
        return flat.new_zeros(idx.size(0), flat.size(-1)).view(*slots.shape, -1)
    valid = (idx >= 0) & (idx < n_rows)
    gathered = flat.index_select(0, idx.clamp(0, n_rows - 1))
    out = torch.where(valid.unsqueeze(1), gathered, gathered.new_zeros(()))
    return out.view(*slots.shape, flat.size(-1)) if slots.ndim > 1 else out


def window_slots(
    md: "V41AttentionMetadata", req: int, positions: torch.Tensor
) -> torch.Tensor:
    """Global slot ids for a window cache, whose rows are one per raw token."""
    return slots_for_positions(md.block_table[req], positions, md.block_size)


def group_slots(
    md: "V41AttentionMetadata", req: int, groups: torch.Tensor, ratio: int
) -> torch.Tensor:
    return slots_for_positions(md.block_table[req], groups, md.block_size // ratio)


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


class V41MetadataBuilder(AttentionMetadataBuilder):

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = self.kv_cache_spec.block_size
        # ``split_decodes_and_prefills`` assumes decodes are already at the front.
        self._init_reorder_batch_threshold(1)
        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> V41AttentionMetadata:
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=1
        )
        positions = common_attn_metadata.positions
        if positions is None:
            raise RuntimeError(
                "deepseek_v41 paged attention needs CommonAttentionMetadata.positions"
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
            seq_lens_cpu=common_attn_metadata.seq_lens.cpu(),
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            token_to_req_indices=common_attn_metadata.token_to_req_indices(
                self.token_to_req_indices
            ),
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
