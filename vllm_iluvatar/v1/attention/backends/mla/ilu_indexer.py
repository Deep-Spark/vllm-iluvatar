# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout, MLAAttentionSpec


def indexer_kv_dtype() -> str:
    """Record the lightning indexer keeps its index keys in.

    ``VLLM_ILU_V41_INDEXER_KV_DTYPE``; unset means ``fp8``. Read live so a test
    can point the process at another record before the model is built.
    """
    dtype = os.environ.get("VLLM_ILU_V41_INDEXER_KV_DTYPE", "fp8").strip().lower()
    if dtype not in ("bf16", "fp8", "int8"):
        raise ValueError(
            "VLLM_ILU_V41_INDEXER_KV_DTYPE must be bf16, fp8, or int8, "
            f"got {dtype!r}"
        )
    return dtype


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        # DualChunkSwap can leave a PCP rank with only empty segments. There
        # are no local logits to compute; do not construct a zero-step range.
        if chunk_m == 0:
            continue
        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class IluDeepseekV32IndexerBackend(AttentionBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
        return [1 if current_platform.is_rocm() else 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["IluDeepseekV32IndexerMetadataBuilder"]:
        return IluDeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


class IluKpoolTailBackend(IluDeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_name() -> str:
        return "ILU_KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type["IluKpoolTailMetadataBuilder"]:  # type: ignore[override]
        return IluKpoolTailMetadataBuilder


@dataclass
class IluDeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    valid_lens: torch.Tensor
    cu_seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    max_context_len: int
    max_q_len: int  # Maximum query length for dsa_indexer_mqa_logits_with_kvcache
    max_kv_len: int  # Maximum key-value length for dsa_indexer_mqa_logits_with_kvcache
    skip_kv_gather: bool = False


@dataclass
class IluDeepseekV32IndexerPrefillMetadata:
    chunks: list[IluDeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class IluDeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    # schedule_metadata: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    valid_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_context_len: int
    max_q_len: int
    max_kv_len: int
    per_req_decode_lens: torch.Tensor | None = None
    decode_is_uniform: bool = True
    write_max_decode_len: int = 0


@dataclass
class IluDeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: IluDeepSeekV32IndexerDecodeMetadata | None = None
    prefill: IluDeepseekV32IndexerPrefillMetadata | None = None
    num_reqs: int = None
    num_actual_tokens: int = None  # Number of tokens excluding padding.
    max_query_len: int = None
    positions: torch.Tensor | None = None
    pcp_world_size: int = 1
    block_table: torch.Tensor | None = None
    token_to_req_indices: torch.Tensor | None = None
    query_start_loc: torch.Tensor = None
    head_dim: int = None
    num_states: int | None = None


class IluKpoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build only the circular slot mapping needed by the storage-only tail."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        self.decode_threshold = 1 + self.num_speculative_tokens
        self.slot_mapping_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluDeepseekV32IndexerMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            slot_mapping_buffer = self.slot_mapping_buffer[
                : slot_mapping.numel()
            ].view_as(slot_mapping)
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=slot_mapping_buffer,
            )
        return IluDeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            positions=common_attn_metadata.positions[
                : common_attn_metadata.num_actual_tokens
            ]
            if common_attn_metadata.positions is not None
            else None,
            query_start_loc=common_attn_metadata.query_start_loc[
                : common_attn_metadata.num_reqs + 1
            ],
            block_table=common_attn_metadata.block_table_tensor,
        )


# TODO (zyongye) optimize this, this is now vibe coded
def kv_spans_from_batches(
    start_seq_loc: torch.Tensor,
    seq_len_per_batch: torch.Tensor,
    device: torch.device,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of
                     selected tokens per batch.
            Example: [0, 2, 4, 7] ->
                     batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B],
                         **uncompressed** sequence length per batch.
                         Example: [5, 9, 4].
      compress_ratio: cache compression ratio. ``cu_seqlen_ks/ke`` are
                     emitted in compressed-domain so the indexer kernels
                     (dsa_indexer_mqa_logits_with_kvcache /
                     dsa_update_topk_indices) see a logits matrix of width
                     == compressed kv len. ``compress_ratio == 1`` keeps the
                     legacy V3.2 token-domain semantics.

    Returns:
      start_tensor: 1D int32 tensor [N]. Compressed KV start offset of each
                    token's batch.
      end_location: 1D int32 tensor [N]. Exclusive compressed end =
                    start + (uncompressed_pos + 1) // compress_ratio,
                    i.e. the per-query-token causal valid count in the
                    compressed K-cache. ``end_location - start_tensor`` is
                    what ``dsa_update_topk_indices`` reads as ``kv_lens``.

    Assumes the selected tokens within a batch are the **last** `counts[b]`
    positions of that batch's sequence (standard prefill chunked layout).
    """
    q = start_seq_loc.to(dtype=torch.long)
    L_uncomp = seq_len_per_batch.to(dtype=torch.long)
    assert q.dim() == 1 and L_uncomp.dim() == 1
    assert q.numel() == L_uncomp.numel() + 1, "start_seq_loc must have length B+1"

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]  # [B]
    N = int(q[-1].item())  # total selected tokens
    B = L_uncomp.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )

    # Compressed seq_len per batch — what dsa_indexer_mqa_logits_with_kvcache
    # sees in cu_seqlens_kv / kv_cache.shape[1] domain.
    L_comp = L_uncomp // compress_ratio if compress_ratio > 1 else L_uncomp

    # KV start offsets per batch in the concatenated *compressed* KV cache.
    kv_starts_per_batch = torch.cumsum(L_comp, dim=0) - L_comp  # [B]

    batch_id = torch.repeat_interleave(torch.arange(B), counts)  # [N]
    start_tensor = kv_starts_per_batch[batch_id]  # [N]

    L_uncomp_expand = torch.repeat_interleave(L_uncomp, counts)  # [N]
    m_expand = torch.repeat_interleave(counts, counts)  # [N]
    # 1-based position within the selected (last) block (uncompressed domain).
    pos_within = (
        torch.arange(N, dtype=torch.long) - torch.repeat_interleave(q[:-1], counts) + 1
    )

    # Compressed valid causal end count per token:
    #   uncompressed_abs_pos + 1 = L_uncomp - m + pos_within
    #   compressed_valid_count   = (uncompressed_abs_pos + 1) // compress_ratio
    if compress_ratio > 1:
        local_pos = (L_uncomp_expand - m_expand + pos_within) // compress_ratio
    else:
        local_pos = L_uncomp_expand - m_expand + pos_within
    end_location = start_tensor + local_pos  # exclusive end

    # Pin before uploading: a plain ``.to(device)`` from pageable memory is a
    # blocking copy, and in the decode steady state this runs while the CUDA
    # graph of the previous step is still executing, so it stalls the host
    # until the GPU drains instead of just enqueuing the transfer.
    return (
        start_tensor.int().pin_memory().to(device, non_blocking=True),
        end_location.int().pin_memory().to(device, non_blocking=True),
    )


def kv_spans_from_batches_device(
    cu_seqlens_q: torch.Tensor,
    seq_len_per_batch: torch.Tensor,
    token_arange_p1: torch.Tensor,
    num_tokens: int,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side ``kv_spans_from_batches`` for the decode path.

    Same math as the host version, but every input already lives on the GPU.
    The host one needs ``seq_lens`` on the CPU, and the spec-decode proposer
    invalidates ``CommonAttentionMetadata._seq_lens_cpu`` after each draft step
    (it mutates ``seq_lens`` on device), so reading ``seq_lens_cpu`` there costs
    a blocking D2H per draft step -- the same stall ``cu_seqlens_q`` and
    ``cu_seqlens_kv`` above were already moved off the host to avoid.

    Batches here are small enough that every op is launch-bound, and this runs
    once per model forward plus once per draft step, so the expression is
    arranged to keep the op count down rather than to read like the host
    version: ``pos_within``'s ``-q[b]`` cancels the one inside ``counts[b]``,
    which removes ``counts`` and one gather, and ``batch_id`` comes from a
    searchsorted over ``cu_seqlens_q`` instead of a repeat_interleave, which
    would internally redo the prefix sum ``cu_seqlens_q`` already is.

    Args:
      cu_seqlens_q: int32 [B+1] on device, query start loc relative to the
                    first decode token.
      seq_len_per_batch: int32 [B] on device, **uncompressed** sequence length.
      token_arange_p1: cached int32 ``arange(1, ...)``, at least ``num_tokens``
                    long. 1-based so the ``+1`` of the within-block position
                    comes for free; doubles as the searchsorted key.
      num_tokens: N, the unpadded decode token count. Known on the host from
                  ``query_start_loc_cpu``, so no sync is needed to size the
                  slice below.
    """
    num_reqs = seq_len_per_batch.numel()
    if num_tokens == 0:
        empty = seq_len_per_batch.new_empty(0, dtype=torch.int32)
        return empty, empty.clone()

    q_end = cu_seqlens_q[1 : num_reqs + 1]
    positions = token_arange_p1[:num_tokens]

    seq_lens_comp = (
        seq_len_per_batch // compress_ratio if compress_ratio > 1 else seq_len_per_batch
    )
    kv_starts_per_batch = torch.cumsum(seq_lens_comp, dim=0, dtype=torch.int32)
    kv_starts_per_batch -= seq_lens_comp

    # Token i belongs to the first request whose exclusive end exceeds i, i.e.
    # the first ``q_end`` at or above the 1-based position. Requests
    # contributing no tokens leave duplicates in ``q_end`` and are skipped.
    batch_id = torch.searchsorted(q_end, positions)
    start_tensor = kv_starts_per_batch[batch_id]

    # 1-based position within the selected (last) block, uncompressed domain.
    local_pos = (seq_len_per_batch - q_end)[batch_id] + positions
    if compress_ratio > 1:
        local_pos //= compress_ratio

    return start_tensor, start_tensor + local_pos


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


def get_max_decode_context_len(vllm_config: VllmConfig, compress_ratio: int) -> int:
    return cdiv(vllm_config.model_config.max_model_len, compress_ratio)


class IluDeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1
    natively_supported_next_n: list[int] = [1, 2]
    requires_block_table_width = True
    # TODO (matt): integrate kernel with next_n = 4 support

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, block_table_width: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcp_world_size = (
            self.vllm_config.parallel_config.prefill_context_parallel_size
        )
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens
        self.use_flattening = next_n not in self.natively_supported_next_n

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        # KV compression. Default 1 for V3.2 (no compression); read from spec
        # for V4 (compress_ratio == 4 on the indexer cache).
        self.compress_ratio = 1
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            assert isinstance(self.kv_cache_spec.tokens_per_state, int)
            self.compress_ratio = self.kv_cache_spec.tokens_per_state
        if self.pcp_world_size > 1 and self.compress_ratio != 1:
            raise NotImplementedError(
                "Iluvatar PCP requires an uncompressed indexer cache"
            )
        self.max_decode_context_len = get_max_decode_context_len(
            self.vllm_config, self.compress_ratio
        )

        # Pre-allocate compressed slot_mapping buffer for CUDA graph address
        # stability when this group needs compression (V4 indexer cache).
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
        self.v41_req_id_per_token_buffer = torch.empty(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.per_req_decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        if not self.use_flattening and next_n > 1:
            # Native MTP: 2D buffer for per-token seq_lens.
            # Flattening path is never used, so no expanded_seq_lens_buffer.
            self.decode_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_seqs, next_n),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Flattening or no MTP: 1D buffer for expanded per-token seq_lens.
            self.decode_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
        self.decode_position_offsets_buffer = torch.arange(
            scheduler_config.max_num_seqs * next_n,
            dtype=torch.int32,
            device=self.device,
        )
        # 1-based arange for the device-side cu_seqlen_ks/ke build, kept int32
        # so that whole computation stays in one dtype.
        self.decode_token_arange_p1_buffer = torch.arange(
            1,
            scheduler_config.max_num_batched_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = (
            block_table_width
            if block_table_width is not None
            else cdiv(
                self.vllm_config.model_config.max_model_len,
                self.kv_cache_spec.block_size
                * self.vllm_config.parallel_config.decode_context_parallel_size,
            )
        )
        self.cu_seqlens_q_buffer = torch.empty(
            (scheduler_config.max_num_seqs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self.cu_seqlens_kv_buffer = torch.empty(
            (scheduler_config.max_num_seqs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        # One ks/ke span per decode query token. With MTP the verify step
        # schedules up to ``num_decodes * next_n`` decode tokens, so size these
        # by max_num_batched_tokens (>= num_decode_tokens) rather than
        # max_num_seqs (which only fits single-token decode).
        self.cu_seqlen_ks_buffer = torch.empty(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.cu_seqlen_ke_buffer = torch.empty(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.valid_lens_buffer = torch.empty(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_compressed_seq_lens_buffer = torch.empty(
            (scheduler_config.max_num_seqs,),
            dtype=torch.int32,
            device=self.device,
        )

        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.indexer_decode_block_table_buffer: torch.Tensor | None = None

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

    def build_one_prefill_chunk(
        self,
        req_slice: slice,
        query_slice: slice,
        query_start_loc_cpu,
        seq_lens_cpu,
        block_table,
        skip_kv_gather: bool = False,
    ) -> IluDeepseekV32IndexerPrefillChunkMetadata:
        # ``seq_lens_cpu`` is in **uncompressed** token domain. For V4
        # (compress_ratio > 1) the indexer kernels operate over the
        # compressed K cache, so the per-chunk metadata fed to
        # ``dsa_indexer_mqa_logits_with_kvcache`` /
        # ``dsa_update_topk_indices`` needs to be compressed.
        compress_ratio = self.compress_ratio
        if compress_ratio > 1:
            seq_lens_cpu_chunk_comp = seq_lens_cpu[req_slice] // compress_ratio
        else:
            seq_lens_cpu_chunk_comp = seq_lens_cpu[req_slice]

        prefill_query_start_loc = (
            query_start_loc_cpu[req_slice.start : req_slice.stop + 1]
            - query_start_loc_cpu[req_slice.start]
        )
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            prefill_query_start_loc,
            seq_lens_cpu[req_slice],
            self.device,
            compress_ratio=compress_ratio,
        )
        token_start = query_start_loc_cpu[req_slice.start].item()
        total_seq_lens = seq_lens_cpu_chunk_comp.sum()
        num_reqs = req_slice.stop - req_slice.start
        seq_idx = torch.arange(0, num_reqs, dtype=torch.int32)
        # token_to_seq is currently unused by the indexer kernels but kept on
        # the dataclass for parity with V3.2; populate it in compressed-domain.
        token_to_seq = torch.repeat_interleave(seq_idx, seq_lens_cpu_chunk_comp).to(
            self.device
        )
        assert total_seq_lens <= self.max_prefill_buffer_size
        cu_seq_lens = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    seq_lens_cpu_chunk_comp.cumsum(dim=0),
                ]
            )
            .to(torch.int32)
            .to(self.device)
        )
        max_context_len = int(seq_lens_cpu_chunk_comp.max().item())
        # cu_seqlens_q / max_q_len must match the sliced q rows passed to the
        # indexer kernel (q[token_start:token_end]). Request-level packing
        # already scopes prefill_query_start_loc to req_slice; when
        # split_indexer_prefill_chunks further sub-chunks on the query dim
        # (query_slice), rebuild them for that sub-slice.
        full_q = int(prefill_query_start_loc[-1].item())
        sub_m = int(query_slice.stop - query_slice.start)
        if query_slice.start == 0 and query_slice.stop == full_q:
            cu_seqlens_q = prefill_query_start_loc.to(torch.int32).to(self.device)
            max_q_len = (
                (prefill_query_start_loc[1:] - prefill_query_start_loc[:-1])
                .max()
                .item()
            )
        else:
            # M-subchunk of a single overflowing request: one contiguous
            # query span, so cu_seqlens_q is simply [0, sub_m].
            cu_seqlens_q = torch.tensor(
                [0, sub_m], dtype=torch.int32, device=self.device
            )
            max_q_len = sub_m

        chunk_ks = cu_seqlen_ks[query_slice]
        chunk_ke = cu_seqlen_ke[query_slice]
        # Prefill is not CUDA-graph captured (FULL_DECODE_ONLY). Still produce
        # valid_lens here so forward never subtracts.
        valid_lens = chunk_ke - chunk_ks

        return IluDeepseekV32IndexerPrefillChunkMetadata(
            cu_seqlen_ks=chunk_ks,
            cu_seqlen_ke=chunk_ke,
            valid_lens=valid_lens,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            total_seq_lens=total_seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            block_table=block_table[req_slice],
            token_start=token_start + query_slice.start,
            token_end=token_start + query_slice.stop,
            num_reqs=num_reqs,
            max_context_len=max_context_len,
            max_q_len=max_q_len,
            max_kv_len=max_context_len,
            skip_kv_gather=skip_kv_gather,
        )

    def build_decode_metadata(
        self, common_attn_metadata, num_decodes, decode_lens, num_decode_tokens
    ):
        # Per-request query lengths in the (uncompressed) token domain. With
        # MTP the verify step schedules ``next_n = 1 + num_speculative_tokens``
        # query tokens per request, so each request emits multiple decode
        # tokens. We mirror ``build_one_prefill_chunk``: build per-query-token
        # ``cu_seqlen_ks/ke`` via ``kv_spans_from_batches_device`` and a per-request
        # ``cu_seqlens_q`` so the (shared with prefill) kernel
        # ``dsa_indexer_mqa_logits_with_kvcache`` handles >1 query per request.
        #
        # ``decode_qsl_cpu`` is the per-request query start loc relative to the
        # first decode token (decodes are at the front of the batch). For plain
        # single-token decode this is [0, 1, 2, ..., R], for uniform MTP it is
        # [0, next_n, 2*next_n, ...]. CUDA-graph FULL_DECODE_ONLY pads the
        # request dimension up to the captured size; padded rows carry
        # query_len=0 (query_start_loc_cpu is flat-filled for them), so the
        # real requests form a uniform prefix and the trailing entries repeat
        # the last real offset. We allow 0 and only forbid negative lengths.
        #
        # ``num_decode_tokens`` is the **padded** decode token count (==
        # common_attn_metadata.num_actual_tokens under CUDA graph, which the
        # consumer uses to slice q/weights/logits). ``num_real_decode_tokens``
        # is the unpadded count (decode_qsl_cpu[-1]); the gap [real, padded) is
        # CUDA-graph padding rows that must be no-ops downstream.
        decode_qsl_cpu = (
            common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            - common_attn_metadata.query_start_loc_cpu[0]
        )
        decode_lens_cpu = torch.diff(decode_qsl_cpu)
        assert decode_lens_cpu.min().item() >= 0, (
            "Negative decode query length in dsa_indexer backend"
        )
        num_real_decode_tokens = int(decode_qsl_cpu[-1].item())
        assert num_real_decode_tokens <= num_decode_tokens, (
            f"real decode tokens {num_real_decode_tokens} exceed padded "
            f"num_decode_tokens {num_decode_tokens}"
        )
        # Max query length per real request: 1 for plain decode, next_n for MTP
        # (trailing zero-length padding rows never raise the max).
        max_q_len = int(decode_lens_cpu.max().item())
        # ``requires_padding`` flags a *genuinely ragged* batch that needs
        # pack/unpack (e.g. variable accepted spec tokens). CUDA-graph trailing
        # zero-length padding rows are NOT ragged -- the real (non-zero) query
        # lengths are uniform and the padding is handled as no-ops by the
        # mqa-logits kernel (empty batch -> early return) and the topk kernel
        # (cur_len<=0 -> writes -1). So only flag when the real lengths differ.
        nonzero_lens = decode_lens_cpu[decode_lens_cpu > 0]
        requires_padding = bool(
            nonzero_lens.numel() > 0
            and (nonzero_lens.max() != nonzero_lens.min()).item()
        )

        # These host scalars are fixed at CUDA graph capture time. Actual
        # per-request lengths remain device-driven through cu_seqlens_kv.
        max_context_len = self.max_decode_context_len
        max_kv_len = max_context_len

        # Fill cu_seqlens_q buffer in-place (address-stable for CUDA graph).
        # Per-request query offsets, e.g. plain decode -> [0, 1, ..., num_decodes],
        # uniform MTP (next_n=2) -> [0, 2, 4, ..., 2*num_decodes].
        # Derived from the device-side ``query_start_loc`` rather than the host
        # one: the host tensor is not pinned, so uploading it blocks until the
        # previous step's CUDA graph has drained. This is the first GPU-touching
        # statement of a decode step, so that stall costs a whole forward pass
        # of host/device overlap.
        qsl_gpu = common_attn_metadata.query_start_loc
        torch.sub(
            qsl_gpu[: num_decodes + 1],
            qsl_gpu[0],
            out=self.cu_seqlens_q_buffer[: num_decodes + 1],
        )
        cu_seqlens_q = self.cu_seqlens_q_buffer[: num_decodes + 1]

        # ``kv_spans_from_batches_device`` treats the selected tokens of a
        # request as the *last* ``decode_lens[b]`` positions, emitting one
        # causal ``[ks, ke)`` span per query token in the compressed domain.
        # For MTP this is exactly the per-token causal structure: token j of
        # request b attends to ``L_b - next_n + j + 1`` KV positions.
        # Built device-side: the host variant needs ``seq_lens_cpu``, which the
        # spec-decode proposer invalidates every draft step, so it would cost a
        # blocking D2H per step.
        cu_seqlen_ks_tmp, cu_seqlen_ke_tmp = kv_spans_from_batches_device(
            cu_seqlens_q,
            common_attn_metadata.seq_lens[:num_decodes],
            self.decode_token_arange_p1_buffer,
            num_real_decode_tokens,
            compress_ratio=self.compress_ratio,
        )

        # Copy into persistent buffers for CUDA graph address stability. There
        # is one ks/ke entry per **real** decode query token; the topk kernel
        # only reads ``ke - ks`` (the per-token causal length), so the absolute
        # offsets need not match the per-request logits layout. CUDA-graph
        # padding rows [real, padded) get zero-length spans (ks == ke == 0) so
        # ``dsa_update_topk_indices`` writes -1 for them without touching the
        # (uninitialised) padded logits rows. Slicing to the padded length
        # keeps ``cu_seqlen_ks.shape[0] == logits.shape[0]`` (the consumer
        # builds logits from ``q[:num_decode_tokens]``).
        assert cu_seqlen_ks_tmp.shape[0] == num_real_decode_tokens
        self.cu_seqlen_ks_buffer[:num_real_decode_tokens] = cu_seqlen_ks_tmp
        self.cu_seqlen_ke_buffer[:num_real_decode_tokens] = cu_seqlen_ke_tmp
        self.cu_seqlen_ks_buffer[num_real_decode_tokens:num_decode_tokens] = 0
        self.cu_seqlen_ke_buffer[num_real_decode_tokens:num_decode_tokens] = 0
        cu_seqlen_ks = self.cu_seqlen_ks_buffer[:num_decode_tokens]
        cu_seqlen_ke = self.cu_seqlen_ke_buffer[:num_decode_tokens]
        torch.sub(
            cu_seqlen_ke,
            cu_seqlen_ks,
            out=self.valid_lens_buffer[:num_decode_tokens],
        )
        valid_lens = self.valid_lens_buffer[:num_decode_tokens]

        # Fill compressed seq_lens buffer in-place for address stability.
        # ``decode_metadata.seq_lens`` is consumed by topk-update / paged-mqa
        # kernels for the per-token causal masks. Those kernels
        # (dsa_indexer_mqa_logits_with_kvcache / dsa_update_topk_indices) and the
        # K-cache itself live in the **compressed** domain when the indexer
        # cache uses compression (V4: compress_ratio == 4), while vLLM tracks
        # the token domain.
        if self.compress_ratio > 1:
            torch.div(
                common_attn_metadata.seq_lens[:num_decodes],
                self.compress_ratio,
                rounding_mode="floor",
                out=self.decode_compressed_seq_lens_buffer[:num_decodes],
            )
            decode_seq_lens = self.decode_compressed_seq_lens_buffer[:num_decodes]
        else:
            decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]

        # cu_seqlens_kv: prefix sum of compressed seq_lens, with leading 0.
        # Summing the device-side compressed lengths computed just above keeps
        # this off the host: the equivalent host tensor is not pinned, so
        # uploading it would block until the previous CUDA graph drained.
        self.cu_seqlens_kv_buffer[:1].zero_()
        torch.cumsum(
            decode_seq_lens,
            dim=0,
            dtype=torch.int32,
            out=self.cu_seqlens_kv_buffer[1 : num_decodes + 1],
        )
        cu_seqlens_kv = self.cu_seqlens_kv_buffer[: num_decodes + 1]

        decode_metadata = IluDeepSeekV32IndexerDecodeMetadata(
            block_table=common_attn_metadata.block_table_tensor[:num_decodes, ...],
            seq_lens=decode_seq_lens,
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            valid_lens=valid_lens,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q=cu_seqlens_q,
            max_context_len=max_context_len,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            per_req_decode_lens=self.per_req_decode_lens_buffer[:num_decodes],
            decode_is_uniform=(int(decode_lens_cpu.min().item()) == max_q_len),
            write_max_decode_len=max_q_len,
            # schedule_metadata=self.scheduler_metadata_buffer,
        )
        return decode_metadata

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, next_n) for native MTP.
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets
                    + self.decode_position_offsets_buffer[:actual_expanded]
                    + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 2
                # (B, next_n): token j attends to L - next_n + j + 1 KV tokens
                self.decode_seq_lens_buffer[:num_decodes] = (
                    seq_lens.unsqueeze(1) - next_n + 1 + self.offsets_buffer
                )
                seq_lens = self.decode_seq_lens_buffer[:num_decodes]
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluDeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        # When the indexer cache is compressed (V4 ``compress_ratio > 1``)
        # ``common_attn_metadata.slot_mapping`` is the **uncompressed** slot
        # mapping prepared by the model runner. Override it with a compressed
        # version so the ``DeepseekCompressor`` (and any downstream consumer)
        # writes/reads at compressed-domain offsets matching the K-cache
        # layout.  CUDA-graph stability is preserved by reusing
        # ``self.compressed_slot_mapping_buffer``.
        indexer_block_table = common_attn_metadata.block_table_tensor
        if self.compress_ratio > 1:
            kernel_block_size = self.kernel_block_size
            if (
                kernel_block_size is not None
                and self.kv_cache_spec.block_size != kernel_block_size
                and self.kv_cache_spec.block_size % kernel_block_size == 0
            ):
                factor = self.kv_cache_spec.block_size // kernel_block_size
                indexer_block_table = (
                    common_attn_metadata.block_table_tensor[:, ::factor] // factor
                ).contiguous()
            padded_num_tokens = common_attn_metadata.slot_mapping.shape[0]
            get_compressed_slot_mapping(
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.slot_mapping,
                common_attn_metadata.query_start_loc,
                common_attn_metadata.seq_lens,
                indexer_block_table.clamp(min=0),
                self.kv_cache_spec.num_states,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            # Expose the full CUDA-graph padded length (padding rows stay -1).
            slot_mapping = self.compressed_slot_mapping_buffer[:padded_num_tokens]
        else:
            slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
                treat_short_extends_as_decodes=self.pcp_world_size == 1,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            # Logits width is compressed-domain for V4; budget M*N with
            # uncompressed N over-splits and inflates kernel launch count.
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )
            chunks = [
                self.build_one_prefill_chunk(
                    req_slice,
                    query_slice,
                    query_start_loc_cpu,
                    seq_lens_cpu,
                    indexer_block_table,
                    skip_kv_gather=query_slice.start > 0,
                )
                for req_slice, query_slice in chunk_specs
            ]
            prefill_metadata = IluDeepseekV32IndexerPrefillMetadata(
                chunks=chunks,
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            self.per_req_decode_lens_buffer[:num_decodes].copy_(
                self.decode_lens_buffer[:num_decodes]
            )
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len == next_n

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )
            decode_metadata = self.build_decode_metadata(
                common_attn_metadata, num_decodes, decode_lens, num_decode_tokens
            )
            if indexer_block_table is not common_attn_metadata.block_table_tensor:
                rows, cols = indexer_block_table[:num_decodes].shape
                if self.indexer_decode_block_table_buffer is None:
                    self.indexer_decode_block_table_buffer = torch.zeros(
                        (self.vllm_config.scheduler_config.max_num_batched_tokens, cols),
                        dtype=torch.int32,
                        device=self.device,
                    )
                self.indexer_decode_block_table_buffer[:rows, :cols].copy_(
                    indexer_block_table[:num_decodes]
                )
                decode_metadata.block_table = self.indexer_decode_block_table_buffer[
                    :rows, :cols
                ]

        attn_metadata = IluDeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=128,
            block_table=common_attn_metadata.block_table_tensor,
            token_to_req_indices=common_attn_metadata.token_to_req_indices(
                self.v41_req_id_per_token_buffer
            ),
            num_states=int(self.kv_cache_spec.num_states),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
            pcp_world_size=self.pcp_world_size,
        )

        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> IluDeepseekV32IndexerMetadata:
        # Under FULL_DECODE_ONLY CUDA graph the decode ``max_context_len`` /
        # ``max_kv_len`` are Python scalars baked into the graph at capture:
        # they size the indexer logits width and the KV loop bound for
        # ``dsa_indexer_mqa_logits_with_kvcache`` and DO NOT update on replay.
        # ``build_decode_metadata`` derives them from ``seq_lens.max()``, so the
        # captured value tracks whatever dummy seq_len ``_dummy_run`` used. With
        # the community ``seq_lens = max_query_len`` (tiny), the baked bound
        # truncates the top-k candidate window on every real replay -> the
        # sparse MLA only attends to a short recent window -> accuracy collapse
        #
        # Force a fixed, capture-safe upper bound equal to the compressed
        # paged-KV capacity (max_model_len // compress_ratio). This exactly
        # reproduces the known-good ``seq_lens = max_model_len`` capture value
        # but locally, so ``_dummy_run`` may keep the community small seq_len.
        # It only widens the (padded) logits matrix; the per-token causal length
        # fed to ``dsa_update_topk_indices`` is ``cu_seqlen_ke - cu_seqlen_ks``
        # (refreshed each step from real seq_lens), so unused columns never
        # enter the top-k selection. Replay-time ``build`` still runs to refresh
        # the address-stable GPU buffers; its scalar is ignored by the graph.
        attn_metadata = self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )
        if attn_metadata.decode is not None:
            max_context_len = self.vllm_config.model_config.max_model_len
            if self.compress_ratio > 1:
                max_context_len = max_context_len // self.compress_ratio
            attn_metadata.decode.max_context_len = max_context_len
            attn_metadata.decode.max_kv_len = max_context_len
        return attn_metadata


# ---------------------------------------------------------------------------
# DeepSeek-V4 (ilu W4A8) indexer backend
# ---------------------------------------------------------------------------
# V4 indexer 与 V3.2 共享 metadata / builder / impl，仅 backend 名称与
# kernel block size 不同。这里通过派生在文件末尾追加，避免触碰原 V3.2 路径。


class IluDeepseekV4IndexerBackend(IluDeepseekV32IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_V4_INDEXER"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # DeepSeek-V4 packs the indexer pages beside the MLA latent pages inside
        # each block, so the layer dim must sit inside the block dim.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
        # V4 indexer KV-cache block_size。与主 MLA / SWA 一致，
        # 走 NV upstream 的 KV cache 分配路径（indexer 与 main MLA 共用一个
        # KVCacheGroup，kernel_block_size 必须相同）。
        # ixformer dsa_indexer_mqa_logits_with_kvcache 原生 CUDA kernel 只支持
        # block_size in {16, 32, 64}；block_size=256 会触发 wrapper 内的
        # ref fallback（pure-PyTorch ref_dsa_indexer_mqa_logits_with_kvcache），
        # 功能等价但性能更低，后续可在 ixformer 层做 native 256 支持。
        return [256]


class _StorageBlockSizeSpec:
    """Read-only proxy exposing physical indexer block size to build()."""

    def __init__(self, spec: Any) -> None:
        self._spec = spec

    @property
    def block_size(self) -> int:
        storage_block_size = self._spec.storage_block_size
        if storage_block_size is None:
            return int(self._spec.block_size)
        return int(storage_block_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._spec, name)


def make_storage_block_size_build(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Use the physical compressed page size while building index metadata."""

    @wraps(orig)
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ):
        if self.compress_ratio <= 1:
            return orig(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )

        original_spec = self.kv_cache_spec
        self.kv_cache_spec = _StorageBlockSizeSpec(original_spec)
        try:
            return orig(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )
        finally:
            self.kv_cache_spec = original_spec

    return build


class IluDeepseekV41IndexerBackend(IluDeepseekV4IndexerBackend):
    """V4.1 indexer backend with independent BF16/FP8/INT8 cache selection."""

    @staticmethod
    def get_builder_cls():
        return IluDeepseekV41IndexerMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
        # ixinfer INT8 accepts a page of 64 or 256 index states. A kernel block
        # is counted in tokens, so an uncompressed INT8 layer stays at 64.
        # ``get_kv_cache_spec`` gives that layer its own 64-token page: v0.31
        # puts every MLA spec that shares ``block_size`` into one KV group, and
        # ``select_common_block_size`` then demands one kernel block for the
        # ratio-1 and ratio-2 indexers together, which no single size can be.
        dtype = indexer_kv_dtype()
        if dtype == "fp8":
            return [128]
        if dtype == "int8":
            return [64]
        return [64, 128]


class IluDeepseekV41IndexerRatio2Backend(IluDeepseekV41IndexerBackend):
    """Index keys compressed 2:1.

    INT8 mqa-logits only accepts 64 or 256 index states per page. A kernel
    block is counted in tokens, so a 2:1 layer needs 128 tokens to hold the
    same 64 states as an uncompressed one.
    """

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
        return [128]


def indexer_backend_cls(compress_ratio: int) -> type[AttentionBackend]:
    """Page INT8 ratio-2 index keys at 128 tokens (64 states).

    FP8 and BF16 stay on ``IluDeepseekV41IndexerBackend``. The ratio-1 INT8
    spec uses a 64-token block so it is not packed into the 128-token group.
    """
    if indexer_kv_dtype() == "int8" and compress_ratio == 2:
        return IluDeepseekV41IndexerRatio2Backend
    return IluDeepseekV41IndexerBackend


class IluDeepseekV41IndexerMetadataBuilder(IluDeepseekV32IndexerMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize every capture-visible scratch buffer, including padded
        # request/token rows. This is V4.1-only; shared V4/GLM stay unchanged.
        for name in (
            "v41_req_id_per_token_buffer",
            "cu_seqlens_q_buffer",
            "cu_seqlens_kv_buffer",
            "cu_seqlen_ks_buffer",
            "cu_seqlen_ke_buffer",
            "valid_lens_buffer",
            "decode_compressed_seq_lens_buffer",
            "scheduler_metadata_buffer",
        ):
            getattr(self, name).zero_()
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer.fill_(-1)
