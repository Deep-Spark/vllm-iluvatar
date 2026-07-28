# SPDX-License-Identifier: Apache-2.0
"""Iluvatar sliding-window-attention cache + backend for DeepSeek-V4.

Mirrors :class:`DeepseekV4SWACache` / :class:`DeepseekSparseSWABackend` from
[vllm/v1/attention/backends/mla/sparse_swa.py](sparse_swa.py).

Supports both bf16 and fp8_ds_mla KV cache formats:

- **bf16**: ``(num_blocks, block_size, head_dim)`` with ``dtype=bfloat16``.
  Page size = ``64 * 512 * 2 = 65536`` bytes.
- **fp8_ds_mla**: ``(num_blocks, block_size, 584)`` with ``dtype=uint8``.
  Per-token layout: 448B FP8 NoPE + 64*2B bf16 RoPE + 8B UE8M0 scale.
  Page size = ``64 * 584 = 37376`` bytes.

The metadata builder is subclassed as ``IluDeepseekSparseSWAMetadataBuilder``
to support ``VLLM_ILU_DS4_PREFILL_AS_DECODE``: when enabled, the SWA kernel
range is extended from decode-only to decode+prefill tokens so that each
prefill query token gets its own per-token SWA window indices/lens (reusing
the decode kernel ``ds4_flash_mla_with_kvcache`` for prefill).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
import vllm.envs as envs

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    _is_flashmla_available,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

logger = logging.getLogger(__name__)

_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"


class IluDeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    """SWA cache layer for DeepSeek-V4 in ilu vllm.

    Supports bf16 (contiguous head_dim) and fp8_ds_mla (584-byte packed)
    layouts, selected via ``cache_config.cache_dtype``.
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.block_size = 64

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        cache_dtype = self.cache_config.cache_dtype
        is_fp8 = cache_dtype in ("fp8_ds_mla", "fp8")

        if is_fp8:
            return SlidingWindowMLASpec(
                block_size=self.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                head_size_v=0,
                dtype=torch.uint8,
                sliding_window=self.window_size,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                # Ilu fp8 attention kernels require 64B-aligned block bases.
                # NOTE: the ilu SWA page is the contiguous 584B/token layout with
                # no per-page padding (37376B == 64*584, already 64-aligned), so
                # the KV-cache reshape falls into the plain ``.view()`` branch.
                # ``kv_quant_mode`` MUST be set so ``initialize_kv_cache_tensors``
                # passes ``cache_dtype_str="fp8_ds_mla"`` to ``get_kv_cache_shape``
                # (which then returns the 584-wide shape). Leaving it at the NONE
                # default makes the runner fall back to ``head_size`` (512) and the
                # view fails with a 584-vs-512 page-size mismatch.
                alignment=64,
                kv_quant_mode=get_kv_quant_mode(cache_dtype),
            )
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version=None,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekSparseSWABackend


class IluDeepseekSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """Iluvatar SWA metadata builder using ixformer kernels.

    Community ``DeepseekSparseSWAMetadataBuilder.build`` launches Triton kernels
    for decode/prefill SWA metadata. Ilu Triton currently fails on that decode
    kernel path, so this override keeps the community data layout but delegates
    the affected metadata computation to ixformer.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        decode_swa_indices = self.decode_swa_indices
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            ixfops.ds4_compute_swa_decode_indices_and_lens(
                swa_indices=self.decode_swa_indices,
                swa_lens=self.decode_swa_lens,
                window_size=self.window_size,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                token_to_req_indices=token_to_req_indices,
                is_valid_token=is_valid_token,
                block_table=block_table,
                block_size=self.block_size,
                num_decode_tokens=num_decode_tokens,
            )

        deepseek_v4_fields = self._build_ilu_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            seq_lens_cpu,
            query_start_loc,
            query_start_loc_cpu,
            block_table,
        )

        tile_sched = self.build_tile_scheduler(num_decode_tokens)

        metadata = DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            prefill_swa_indices=(
                self.prefill_swa_indices[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_swa_lens=(
                self.prefill_swa_lens[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            **deepseek_v4_fields,  # type: ignore[arg-type]
        )
        self._populate_prefill_opt_fields(
            metadata,
            num_decodes,
            num_prefills,
            seq_lens,
            query_start_loc,
            query_start_loc_cpu,
            block_table,
        )

        if envs.VLLM_ILU_DS4_PREFILL_AS_DECODE:
            self._populate_prefill_swa_as_decode(metadata)

        return metadata

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        out: dict[str, FlashMLASchedMeta | None] = {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
        }
        if (
            num_decode_tokens == 0
            or current_platform.is_rocm()
            or current_platform.is_xpu()
            or current_platform.is_device_capability_family(120)
        ):
            return out
        # FlashMLA C++ is unavailable on Iluvatar; ilu attention bypasses
        # tile_scheduler_metadata and uses ixformer decode kernels instead.
        if not _is_flashmla_available()[0]:
            return out
        for layer_type in self._layer_types:
            out[layer_type] = get_mla_metadata()[0]
        return out

    def _build_ilu_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        block_table: torch.Tensor,
    ) -> dict[str, torch.Tensor | None | int]:
        del block_table
        result: dict[str, torch.Tensor | int | None] = {}

        if num_prefills == 0:
            return result

        assert seq_lens_cpu is not None
        pfx_gather_lens = torch.empty(
            num_prefills, dtype=torch.int32, device=seq_lens.device
        )
        ixfops.ds4_compute_swa_prefill_gather_lens(
            prefill_gather_lens=pfx_gather_lens,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            window_size=self.window_size,
        )

        result["prefill_seq_lens"] = seq_lens[num_decodes:]
        result["prefill_seq_lens_cpu"] = seq_lens_cpu[num_decodes:]
        result["prefill_gather_lens"] = pfx_gather_lens
        result["prefill_query_lens_cpu"] = (
            query_start_loc_cpu[num_decodes + 1 : num_decodes + num_prefills + 1]
            - query_start_loc_cpu[num_decodes : num_decodes + num_prefills]
        ).to(dtype=torch.int32)
        result["prefill_window_size"] = self.window_size
        result["prefill_max_model_len"] = self.max_model_len
        result["prefill_max_num_batched_tokens"] = self.max_num_batched_tokens

        return result

    def _populate_prefill_opt_fields(
        self,
        metadata: DeepseekSparseSWAMetadata,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        metadata.prefill_cu_seqlens_q = None
        metadata.prefill_cu_seqlens_k = None
        metadata.prefill_max_seqlen_q = 0
        metadata.prefill_max_seqlen_k = 0
        metadata.prefill_swa_block_table = None

        if num_prefills == 0 or not envs.VLLM_ILU_DS4_PREFILL_OPT:
            return

        pfx_sl = num_decodes
        pfx_el = num_decodes + num_prefills

        metadata.prefill_cu_seqlens_q = (
            query_start_loc[pfx_sl : pfx_el + 1] - query_start_loc[pfx_sl]
        ).to(torch.int32)

        pfx_seq_lens = seq_lens[pfx_sl:pfx_el].to(torch.int32)
        cu_seqlens_k = torch.zeros(
            num_prefills + 1, dtype=torch.int32, device=seq_lens.device
        )
        cu_seqlens_k[1:] = pfx_seq_lens.cumsum(0)
        metadata.prefill_cu_seqlens_k = cu_seqlens_k

        qsl_cpu = query_start_loc_cpu[pfx_sl : pfx_el + 1]
        q_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
        metadata.prefill_max_seqlen_q = int(q_lens_cpu.max().item())

        prefill_seq_lens_cpu = seq_lens[pfx_sl:pfx_el].cpu()
        metadata.prefill_max_seqlen_k = int(prefill_seq_lens_cpu.max().item())
        metadata.prefill_swa_block_table = block_table[num_decodes:]

    def _populate_prefill_swa_as_decode(
        self, metadata: DeepseekSparseSWAMetadata
    ) -> None:
        num_prefill_tokens = metadata.num_prefill_tokens
        if num_prefill_tokens == 0:
            return

        num_decode_tokens = metadata.num_decode_tokens
        num_swa_tokens = num_decode_tokens + num_prefill_tokens

        # Extend SWA computation to cover the prefill token range.
        # ds4_compute_swa_decode_indices_and_lens is purely position-based,
        # so extending the processed range yields correct causal SWA windows
        # for prefill tokens too.  The decode portion (0:num_decode_tokens)
        # is recomputed with identical results; the prefill portion
        # (num_decode_tokens:num_swa_tokens) is newly populated.
        self.decode_swa_lens[num_swa_tokens:] = 0
        ixfops.ds4_compute_swa_decode_indices_and_lens(
            swa_indices=self.decode_swa_indices,
            swa_lens=self.decode_swa_lens,
            window_size=self.window_size,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            token_to_req_indices=self.token_to_req_indices,
            is_valid_token=self.is_valid_token,
            block_table=metadata.block_table,
            block_size=self.block_size,
            num_decode_tokens=num_swa_tokens,
        )

        metadata.prefill_swa_indices = (
            self.decode_swa_indices[num_decode_tokens:num_swa_tokens]
        )
        metadata.prefill_swa_lens = (
            self.decode_swa_lens[num_decode_tokens:num_swa_tokens]
        )


class IluDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """SWA backend supporting bf16 and fp8_ds_mla cache."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8_ds_mla", "fp8",
    ]

    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAMetadataBuilder]:
        return IluDeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)
