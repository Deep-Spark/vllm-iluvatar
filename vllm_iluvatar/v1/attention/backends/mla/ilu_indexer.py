# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

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
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_kv_cache_shard_count



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

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class IluDeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
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
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class IluDeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    max_context_len: int
    max_q_len: int  # Maximum query length for dsa_indexer_mqa_logits_with_blocks
    max_kv_len: int  # Maximum key-value length for dsa_indexer_mqa_logits_with_blocks
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
    cu_seqlens_kv: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_context_len: int
    max_q_len: int
    max_kv_len: int


@dataclass
class IluDeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: IluDeepSeekV32IndexerDecodeMetadata | None = None
    prefill: IluDeepseekV32IndexerPrefillMetadata | None = None


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
                     (dsa_indexer_mqa_logits_with_blocks /
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

    # Compressed seq_len per batch — what dsa_indexer_mqa_logits_with_blocks
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

    return start_tensor.int().to(device), end_location.int().to(device)


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


class IluDeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1
    natively_supported_next_n: list[int] = [1, 2]
    # TODO (matt): integrate kernel with next_n = 4 support

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate compressed slot_mapping buffer for CUDA graph address
        # stability when this group needs compression (V4 indexer cache).
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
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
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_kv_cache_shard_count(),
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
        # ``dsa_indexer_mqa_logits_with_blocks`` /
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

        return IluDeepseekV32IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks[query_slice],
            cu_seqlen_ke=cu_seqlen_ke[query_slice],
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
        # ``cu_seqlen_ks/ke`` via ``kv_spans_from_batches`` and a per-request
        # ``cu_seqlens_q`` so the (shared with prefill) kernel
        # ``dsa_indexer_mqa_logits_with_blocks`` handles >1 query per request.
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

        # Calculate decode metadata parameters in **compressed** domain when
        # the indexer cache uses compression (V4: compress_ratio == 4).
        # ``seq_lens_decode_uncomp`` is what vLLM tracks in token domain;
        # the indexer kernels (dsa_indexer_mqa_logits_with_blocks /
        # dsa_update_topk_indices) and the K-cache itself live in the
        # compressed domain.
        compress_ratio = self.compress_ratio
        seq_lens_decode_uncomp = common_attn_metadata.seq_lens_cpu[:num_decodes]
        if compress_ratio > 1:
            seq_lens_decode = seq_lens_decode_uncomp // compress_ratio
        else:
            seq_lens_decode = seq_lens_decode_uncomp
        max_context_len = int(seq_lens_decode.max().item())
        max_kv_len = max_context_len

        # Fill cu_seqlens_q buffer in-place (address-stable for CUDA graph).
        # Per-request query offsets, e.g. plain decode -> [0, 1, ..., num_decodes],
        # uniform MTP (next_n=2) -> [0, 2, 4, ..., 2*num_decodes].
        self.cu_seqlens_q_buffer[: num_decodes + 1] = decode_qsl_cpu.to(
            dtype=torch.int32, device=self.device
        )
        cu_seqlens_q = self.cu_seqlens_q_buffer[: num_decodes + 1]

        # ``kv_spans_from_batches`` treats the selected tokens of a request as
        # the *last* ``decode_lens[b]`` positions of its sequence, emitting one
        # causal ``[ks, ke)`` span per query token in the compressed domain.
        # For MTP this is exactly the per-token causal structure: token j of
        # request b attends to ``L_b - next_n + j + 1`` KV positions.
        cu_seqlen_ks_tmp, cu_seqlen_ke_tmp = kv_spans_from_batches(
            decode_qsl_cpu,
            seq_lens_decode_uncomp,
            self.device,
            compress_ratio=compress_ratio,
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

        # cu_seqlens_kv: prefix sum of compressed seq_lens, with leading 0.
        self.cu_seqlens_kv_buffer[0] = 0
        seq_lens_decode_gpu = seq_lens_decode.to(dtype=torch.int32, device=self.device)
        self.cu_seqlens_kv_buffer[1 : num_decodes + 1] = torch.cumsum(
            seq_lens_decode_gpu, dim=0
        )
        cu_seqlens_kv = self.cu_seqlens_kv_buffer[: num_decodes + 1]

        # Fill compressed seq_lens buffer in-place for address stability.
        # ``decode_metadata.seq_lens`` is consumed by topk-update / paged-mqa
        # kernels for the per-token causal masks.
        if compress_ratio > 1:
            self.decode_compressed_seq_lens_buffer[:num_decodes] = (
                common_attn_metadata.seq_lens[:num_decodes] // compress_ratio
            )
            decode_seq_lens = self.decode_compressed_seq_lens_buffer[:num_decodes]
        else:
            decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]

        decode_metadata = IluDeepSeekV32IndexerDecodeMetadata(
            block_table=common_attn_metadata.block_table_tensor[:num_decodes, ...],
            seq_lens=decode_seq_lens,
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q=cu_seqlens_q,
            max_context_len=max_context_len,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
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
        if self.compress_ratio > 1:
            padded_num_tokens = common_attn_metadata.slot_mapping.shape[0]
            get_compressed_slot_mapping(
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.query_start_loc,
                common_attn_metadata.seq_lens,
                common_attn_metadata.block_table_tensor.clamp(min=0),
                int(self.kv_cache_spec.storage_block_size),
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
                    common_attn_metadata.block_table_tensor,
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

        attn_metadata = IluDeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=128,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> IluDeepseekV32IndexerMetadata:
        # Under FULL_DECODE_ONLY CUDA graph the decode ``max_context_len`` /
        # ``max_kv_len`` are Python scalars baked into the graph at capture:
        # they size the indexer logits width and the KV loop bound for
        # ``dsa_indexer_mqa_logits_with_blocks`` and DO NOT update on replay.
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
    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # V4 indexer KV-cache block_size。与主 MLA / SWA 一致，
        # 走 NV upstream 的 KV cache 分配路径（indexer 与 main MLA 共用一个
        # KVCacheGroup，kernel_block_size 必须相同）。
        # ixformer dsa_indexer_mqa_logits_with_blocks 原生 CUDA kernel 只支持
        # block_size in {16, 32, 64}；block_size=256 会触发 wrapper 内的
        # ref fallback（pure-PyTorch ref_dsa_indexer_mqa_logits_with_blocks），
        # 功能等价但性能更低，后续可在 ixformer 层做 native 256 支持。
        return [256]
