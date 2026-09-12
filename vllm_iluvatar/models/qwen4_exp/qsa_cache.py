# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged side-cache ownership and metadata for Qwen4Exp QSA.

Each QSA layer keeps one compressed key per completed group, owned here as an
``MLAAttentionSpec`` whose ``compress_ratio`` gives it one physical row per
``compress_ratio`` logical tokens, so its block table follows the main
KV-cache lifecycle. The partial sum of the group still being filled is *not*
cache: it is a per-request fp32 buffer owned by ``Qwen4ExpModelState``, which
is why raw index keys need no paged storage of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from ixformer.inference.functions.qwen38 import qwen_qsa_build_metadata
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    split_indexer_prefill_chunks,
)


def build_qsa_metadata(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del request_capacity
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    # vLLM builds and caches this once per step for every cache group, so the
    # operator takes it as an input rather than recomputing it.
    token_to_req = common_attn_metadata.token_to_req_indices(token_to_req_buffer)[
        :num_tokens
    ]
    logical_positions, slot_mapping = qwen_qsa_build_metadata(
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        token_to_req,
        common_attn_metadata.block_table_tensor,
        common_attn_metadata.slot_mapping,
        logical_positions_buffer,
        slot_mapping_buffer,
        num_tokens,
        num_mapped_tokens,
        storage_block_size,
        compress_ratio,
        k_work_metadata_buffer,
        PAD_SLOT_ID,
        backend="ixformer_native",
    )
    return token_to_req, logical_positions, slot_mapping


@dataclass
class QSAIndexerPrefillChunk:
    token_start: int
    token_end: int
    cu_seqlens_q: torch.Tensor
    cu_seq_lens: torch.Tensor
    valid_lens: torch.Tensor
    block_table: torch.Tensor
    max_q_len: int
    max_kv_len: int
    max_context_len: int


@dataclass
class QSAIndexerPrefillMetadata:
    chunks: list[QSAIndexerPrefillChunk]


@dataclass
class QSAIndexerDecodeMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    valid_lens: torch.Tensor
    block_table: torch.Tensor
    max_q_len: int
    max_kv_len: int
    max_context_len: int


@dataclass
class QSAForwardMetadata(AttentionMetadata):
    """Common per-forward metadata for one QSA side cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    k_work_metadata: torch.Tensor
    num_actual_tokens: int
    storage_block_size: int
    compress_ratio: int
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    prefill: QSAIndexerPrefillMetadata | None = None
    decode: QSAIndexerDecodeMetadata | None = None


class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            compress_ratio = kv_cache_spec.compress_ratio
            assert isinstance(
                compress_ratio, int
            ), "QSA compression requires an integer compress_ratio"
            self.compress_ratio = compress_ratio
            if compress_ratio > 1:
                self.storage_block_size = kv_cache_spec.block_size // compress_ratio
            else:
                self.storage_block_size = kv_cache_spec.block_size
        else:
            self.compress_ratio = 1
            self.storage_block_size = kv_cache_spec.block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        max_requests = vllm_config.scheduler_config.max_num_seqs
        self.request_capacity = max_requests
        if self.compress_ratio != 1:
            max_k_work = (
                max_tokens + (self.compress_ratio - 1) * max_requests
            ) // self.compress_ratio
            self.k_work_metadata_buffer = torch.empty(
                max_k_work, 2, dtype=torch.int32, device=device
            )
        else:
            self.k_work_metadata_buffer = torch.empty(
                0, 2, dtype=torch.int32, device=device
            )
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.max_num_seqs = max_requests
        self.max_num_batched_tokens = max_tokens
        self.decode_block_table_buffer = torch.zeros(
            (
                max_requests,
                kv_cache_spec.max_num_blocks_per_req(vllm_config, self.max_model_len),
            ),
            dtype=torch.int32,
            device=device,
        )
        self._pin_decode_context = False
        self.decode_cu_seqlens_q_buffer = torch.empty(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self.decode_cu_seqlens_kv_buffer = torch.empty(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self.decode_valid_lens_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        # Prefill chunks hold references until the step ends, so each chunk
        # needs its own cu row. Pools grow once, never torch.zeros+H2D.
        self.prefill_valid_lens_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self._prefill_cu_q_pool: list[torch.Tensor] = []
        self._prefill_cu_kv_pool: list[torch.Tensor] = []
        self.has_indexer_buffers = True

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> QSAForwardMetadata:
        self._pin_decode_context = True
        try:
            return self.build(0, common_attn_metadata)
        finally:
            self._pin_decode_context = False

    def _host_seq_lens(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> torch.Tensor:
        """CPU seq_lens already owned by vLLM. Never ``.to("cpu")`` a GPU copy."""
        cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if cpu is None:
            cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if cpu is None:
            raise RuntimeError("QSA metadata requires host seq_lens")
        return cpu

    def _prefill_cu_pair(
        self, chunk_i: int, num_reqs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Persistent ``[max_num_seqs+1]`` rows; grow the pool once per new chunk."""
        cap = max(int(getattr(self, "max_num_seqs", num_reqs)), num_reqs) + 1
        q_pool = getattr(self, "_prefill_cu_q_pool", None)
        if q_pool is None:
            self._prefill_cu_q_pool = []
            self._prefill_cu_kv_pool = []
            q_pool = self._prefill_cu_q_pool
        while len(q_pool) <= chunk_i:
            q_pool.append(torch.empty(cap, dtype=torch.int32, device=device))
            self._prefill_cu_kv_pool.append(
                torch.empty(cap, dtype=torch.int32, device=device)
            )
        return (
            q_pool[chunk_i][: num_reqs + 1],
            self._prefill_cu_kv_pool[chunk_i][: num_reqs + 1],
        )

    def _fill_valid_lens(
        self,
        out: torch.Tensor,
        logical_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        token_to_req: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Write vis in-place. Graph-pad ``pos < 0`` becomes 0; no ``live.any()``."""
        row_seq = seq_lens.index_select(0, token_to_req.long())
        vis = torch.minimum(
            torch.div(logical_positions + 1, compress_ratio, rounding_mode="floor"),
            torch.div(row_seq, compress_ratio, rounding_mode="floor"),
        )
        out.copy_(vis.to(dtype=torch.int32))
        return out

    def _prefill_vis_view(
        self, token_start: int, token_end: int, device: torch.device
    ) -> torch.Tensor:
        buf = getattr(self, "prefill_valid_lens_buffer", None)
        need = token_end
        if buf is None or buf.numel() < need or buf.device != device:
            size = max(need, int(getattr(self, "max_num_batched_tokens", need)))
            self.prefill_valid_lens_buffer = torch.empty(
                size, dtype=torch.int32, device=device
            )
            buf = self.prefill_valid_lens_buffer
        return buf[token_start:token_end]

    def _build_prefill_chunks(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        logical_positions: torch.Tensor,
        token_to_req: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> QSAIndexerPrefillMetadata | None:
        if num_prefills <= 0:
            return None
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = self._host_seq_lens(common_attn_metadata)
        prefill_query_lens_cpu = torch.diff(
            query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
        )
        compressed = (
            seq_lens_cpu[num_decodes : num_decodes + num_prefills]
            // self.compress_ratio
        )
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # split_indexer_prefill_chunks counts 4B/elem; QSA logits are bf16.
        workspace_n = max(
            1, self.max_num_seqs * (self.max_model_len // max(self.compress_ratio, 1))
        )
        chunk_specs = split_indexer_prefill_chunks(
            compressed,
            prefill_query_lens_cpu,
            workspace_n,
            max_logits_bytes * 2,
            request_offset=num_decodes,
        )
        chunks: list[QSAIndexerPrefillChunk] = []
        block_table = common_attn_metadata.block_table_tensor
        device = logical_positions.device
        ratio = self.compress_ratio
        qsl_gpu_all = common_attn_metadata.query_start_loc
        seq_gpu = common_attn_metadata.seq_lens
        for chunk_i, (req_slice, query_slice) in enumerate(chunk_specs):
            n_req = req_slice.stop - req_slice.start
            qsl_cpu = (
                query_start_loc_cpu[req_slice.start : req_slice.stop + 1]
                - query_start_loc_cpu[req_slice.start]
            )
            token_start = int(query_start_loc_cpu[req_slice.start].item()) + int(
                query_slice.start
            )
            token_end = int(query_start_loc_cpu[req_slice.start].item()) + int(
                query_slice.stop
            )
            kv_cpu = seq_lens_cpu[req_slice] // ratio
            max_kv = int(kv_cpu.max().item()) if kv_cpu.numel() else 0
            cu_q, cu_kv = self._prefill_cu_pair(chunk_i, n_req, device)
            kv_gpu = torch.div(seq_gpu[req_slice], ratio, rounding_mode="floor")
            cu_kv[0] = 0
            if n_req:
                torch.cumsum(kv_gpu, 0, out=cu_kv[1 : n_req + 1])
            full_q = int(qsl_cpu[-1].item())
            sub_m = int(query_slice.stop - query_slice.start)
            if query_slice.start == 0 and query_slice.stop == full_q:
                qsl_gpu = qsl_gpu_all[req_slice.start : req_slice.stop + 1]
                if qsl_gpu.dtype != torch.int32:
                    qsl_gpu = qsl_gpu.to(dtype=torch.int32)
                cu_q[0] = 0
                if n_req:
                    torch.sub(qsl_gpu[1:], qsl_gpu[:1], out=cu_q[1 : n_req + 1])
                max_q_len = (
                    int((qsl_cpu[1:] - qsl_cpu[:-1]).max().item())
                    if qsl_cpu.numel() > 1
                    else 0
                )
            else:
                cu_q[0] = 0
                cu_q[1] = sub_m
                cu_q = cu_q[:2]
                cu_kv = cu_kv[:2]
                if n_req != 1:
                    # Sub-query slicing is one oversized request.
                    kv0 = kv_gpu[:1]
                    cu_kv[0] = 0
                    if kv0.numel():
                        cu_kv[1] = kv0[0]
                max_q_len = sub_m
            sl = slice(token_start, token_end)
            vis = self._prefill_vis_view(token_start, token_end, device)
            self._fill_valid_lens(
                vis,
                logical_positions[sl],
                seq_gpu,
                token_to_req[sl],
                ratio,
            )
            chunks.append(
                QSAIndexerPrefillChunk(
                    token_start=token_start,
                    token_end=token_end,
                    cu_seqlens_q=cu_q,
                    cu_seq_lens=cu_kv,
                    valid_lens=vis,
                    block_table=block_table[req_slice],
                    max_q_len=max_q_len,
                    max_kv_len=max_kv,
                    max_context_len=max_kv,
                )
            )
        return QSAIndexerPrefillMetadata(chunks=chunks)

    def _build_decode_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        logical_positions: torch.Tensor,
        token_to_req: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> QSAIndexerDecodeMetadata | None:
        if num_decodes <= 0:
            return None
        ratio = self.compress_ratio
        # Capture pads B to max_num_seqs; eager mixed batches keep live decode B
        # so cu.numel() still matches block_table.size(0)+1.
        pad_b = self._pin_decode_context
        b = self.max_num_seqs if pad_b else num_decodes
        q_len = b + 1
        qsl = common_attn_metadata.query_start_loc[: num_decodes + 1]
        if qsl.dtype != torch.int32:
            qsl = qsl.to(dtype=torch.int32)
        cu_q = self.decode_cu_seqlens_q_buffer[:q_len]
        cu_kv = self.decode_cu_seqlens_kv_buffer[:q_len]
        cu_q[: num_decodes + 1].copy_(qsl)
        seq = common_attn_metadata.seq_lens[:num_decodes]
        kv = torch.div(seq, ratio, rounding_mode="floor")
        cu_kv[0] = 0
        if num_decodes:
            torch.cumsum(kv, 0, out=cu_kv[1 : num_decodes + 1])
        if q_len > num_decodes + 1:
            cu_q[num_decodes + 1 :] = cu_q[num_decodes]
            cu_kv[num_decodes + 1 :] = cu_kv[num_decodes]
        live_block_table = common_attn_metadata.block_table_tensor[:num_decodes]
        block_table_buffer = getattr(self, "decode_block_table_buffer", None)
        num_columns = common_attn_metadata.block_table_tensor.shape[1]
        if (
            block_table_buffer is None
            or block_table_buffer.shape[0] < self.max_num_seqs
            or block_table_buffer.shape[1] < num_columns
            or block_table_buffer.device != live_block_table.device
        ):
            raise RuntimeError(
                "QSA decode block-table workspace was not initialized for "
                f"shape ({self.max_num_seqs}, {num_columns})"
            )
        staged_block_table = block_table_buffer[: self.max_num_seqs, :num_columns]
        staged_block_table.zero_()
        staged_block_table[:num_decodes].copy_(live_block_table)
        vis = self.decode_valid_lens_buffer[:num_decode_tokens]
        self._fill_valid_lens(
            vis,
            logical_positions[:num_decode_tokens],
            seq,
            token_to_req[:num_decode_tokens],
            ratio,
        )
        seq_cpu = self._host_seq_lens(common_attn_metadata)[:num_decodes]
        max_kv = int((seq_cpu // ratio).max().item()) if num_decodes else 0
        if pad_b:
            max_kv = self.max_model_len // max(ratio, 1)
        return QSAIndexerDecodeMetadata(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            valid_lens=vis,
            # Capture must bind max-B storage; live builds still stage into the
            # same storage so graph replay sees current rows and a zero tail.
            block_table=(
                staged_block_table[:b]
                if pad_b
                else common_attn_metadata.block_table_tensor[:b]
            ),
            max_q_len=1,
            max_kv_len=max_kv,
            max_context_len=max_kv,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        build_k_work = self.compress_ratio != 1
        k_work_metadata = self.k_work_metadata_buffer
        request_capacity = None
        if build_k_work:
            request_capacity = self.request_capacity
            # Capacity, not live B: capture/replay keep one k_work shape (Q8).
            max_num_work = (
                num_tokens + (self.compress_ratio - 1) * request_capacity
            ) // self.compress_ratio
            k_work_metadata = self.k_work_metadata_buffer[:max_num_work]
        token_to_req, logical_positions, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            k_work_metadata_buffer=k_work_metadata if build_k_work else None,
            request_capacity=request_capacity,
        )
        num_decodes = num_prefills = 0
        num_decode_tokens = num_prefill_tokens = 0
        prefill = None
        decode = None
        if getattr(self, "has_indexer_buffers", False):
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=1,
                    require_uniform=False,
                )
            )
            prefill = self._build_prefill_chunks(
                common_attn_metadata,
                logical_positions,
                token_to_req,
                num_decodes,
                num_prefills,
            )
            decode = self._build_decode_metadata(
                common_attn_metadata,
                logical_positions,
                token_to_req,
                num_decodes,
                num_decode_tokens,
            )
        return QSAForwardMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill,
            decode=decode,
        )


class QSAStateBackend(AttentionBackend):
    """Key-only dummy backend for out-of-band BF16 QSA side-cache operations."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_EXP_QSA_STATE"

    @staticmethod
    def get_impl_cls():
        class _QSAStateImpl:
            supports_pcp = False
            supports_dcp = False

        return _QSAStateImpl

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, num_kv_heads, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[str, ...]:
        return ("NHD", "HND")

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # get_kv_cache_shape puts num_blocks outermost, so a page padded up to
        # the main KV page stays readable as a strided view over the block dim.
        return True


class _QSAStateCache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        *,
        head_size: int,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        prefix: str,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if head_size <= 0:
            raise ValueError("QSA cache head size must be positive")
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "QSA cache block size must be divisible by the compression ratio"
            )
        self.head_size = head_size
        self.dtype = dtype
        self.cache_config = cache_config
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.kv_cache = torch.tensor([])

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def forward(self) -> None: ...

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Adapt the unified [B, H, N, C] view to QSA's [B, N, H, C]."""
        if kv_cache.ndim != 4 or kv_cache.shape[1] != 1:
            raise ValueError("QSA state cache must be [blocks, 1, states, width]")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[3] != self.head_size:
            raise ValueError("QSA state cache does not match its packed BF16 spec")
        super().bind_kv_cache(kv_cache.transpose(1, 2))

    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAStateBackend


def _main_attention_page_bytes(vllm_config: VllmConfig) -> int:
    """Bytes one block of the main QSA KV cache occupies.

    The same quantity ``Platform.update_block_size_for_backend`` pads the Mamba
    pages up to, so every group ends up on one page size.
    """
    model_config = vllm_config.model_config
    return FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    ).page_size_bytes


class QSACompressedKeyCache(_QSAStateCache):
    """Normalized, group-first-RoPE BF16 key at one row per complete group."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        page_size_padded = None
        if getattr(self.cache_config, "enable_prefix_caching", False):
            # This group's page is 1/(2 * compress_ratio) of the main KV page,
            # and that ratio is exact, so unify_kv_cache_spec_page_size would
            # scale this group's block_size up rather than pad its page. Cache
            # hits round down to the LCM of the group block sizes, so the
            # scaled block_size drags the LCM past max_model_len and pins the
            # hit rate at zero. Declaring the padded page keeps block_size
            # equal to every other group's, at the cost of the padding bytes.
            page_size_padded = _main_attention_page_bytes(vllm_config)
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            page_size_padded=page_size_padded,
        )


__all__ = [
    "QSACompressedKeyCache",
    "QSAForwardMetadata",
    "QSAIndexerDecodeMetadata",
    "QSAIndexerPrefillChunk",
    "QSAIndexerPrefillMetadata",
    "QSAMetadataBuilder",
    "QSAStateBackend",
]
