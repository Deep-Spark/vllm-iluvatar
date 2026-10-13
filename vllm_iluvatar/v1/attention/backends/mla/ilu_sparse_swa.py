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

The metadata builder is subclassed as ``IluDeepseekSparseSWAMetadataBuilder``,
which builds decode and prefill SWA rows in two separate passes of the plugin
kernel: decode rows into ``decode_swa_indices``, prefill rows into
``prefill_swa_indices`` at ``token_offset = num_decode_tokens``. Both are
per-token, so under ``VLLM_ILU_DS4_PREFILL_AS_DECODE`` prefill runs through
the same per-token sparse MLA op decode uses.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

# DeepseekV4 decode layer types, keyed by compress_ratio. Each type has a distinct
# (topk, extra_topk, extra_page_block_size) config, so they cannot share a
# FlashMLA tile-scheduler plan. Within a type, all ~60 DeepseekV4 layers share one
# plan per step because b / s_q / h_q / page_block_sizes / topks are identical.
_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"
# v4.1 ratio-1 / ratio-2 indexer layers: same indexer-path shape as C4A, but
# their compressed page block is block_size // ratio, so each gets its own plan.
# v4.1 builders classify layer types themselves (ratio 0 is SWA-only there); see
# deepseek_v41/sparse_mla.py.
_LAYER_TYPE_C1A = "c1a"
_LAYER_TYPE_C2A = "c2a"

logger = init_logger("vllm." + __name__)


def build_dspark_noncausal_swa_indices(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_size: int,
    window_size: int,
    index_width: int,
    num_decode_tokens: int,
    device: torch.device,
    out_indices: torch.Tensor | None = None,
    out_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token SWA key slots for a DSpark draft block (non-causal).

    Every query token in a request's draft block attends to the same key set:
    the trailing ``window_size`` context keys plus *all* block keys, including
    tokens that follow it. Slots beyond a token's length are ``-1``, and
    padding tokens get length 0.

    The CUDA kernel reads GPU metadata and only the window's page IDs. It
    initializes every output slot, including padding and reused buffers.
    """
    if out_indices is None:
        indices = torch.empty(
            (num_decode_tokens, 1, index_width), dtype=torch.int32, device=device
        )
    else:
        assert out_indices.shape[0] >= num_decode_tokens
        assert out_indices.shape[1:] == (1, index_width)
        indices = out_indices[:num_decode_tokens]

    if out_lens is None:
        lens = torch.empty(num_decode_tokens, dtype=torch.int32, device=device)
    else:
        assert out_lens.shape[0] >= num_decode_tokens
        lens = out_lens[:num_decode_tokens]

    torch.ops._C.v41_dspark_noncausal_swa_indices_and_lens(
        indices, lens, seq_lens, query_start_loc, block_table, is_valid_token,
        block_size, window_size, index_width, num_decode_tokens,
    )

    return indices, lens


class IluDeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    """SWA cache layer for DeepSeek-V4 in ilu vllm.

    Supports bf16 (contiguous head_dim) and fp8_ds_mla (584-byte packed)
    layouts. uint8, or a cache dtype of fp8 / fp8_ds_mla, selects the packed
    record.
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        backend_cls: type[AttentionBackend] | None = None,
        block_size: int | None = 64,
        packed_bytes_per_token: int = 584,
        packed_page_alignment: int = 576,
        bounded_replay: bool = False,
    ) -> None:
        super().__init__()
        self.backend_cls = backend_cls or IluDeepseekSparseSWABackend
        # DeepseekV4.1 SWA bounded replay.
        self.bounded_replay = bounded_replay
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.packed_bytes_per_token = packed_bytes_per_token
        self.packed_page_alignment = packed_page_alignment
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Any multiple of 32; the sparse decode kernels take the page size at
        # runtime. ``None`` means "whatever the cache config settles on", which
        # can only be read after construction: the backend's preferred page is
        # negotiated once the layers exist.
        self._block_size = block_size
        # uint8: fp8_ds_mla UE8M0 paged layout. bfloat16 / float8_e4m3fn:
        # contiguous full-cache layout.
        assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)

    @property
    def block_size(self) -> int:
        if self._block_size is None:
            return self.cache_config.block_size
        return self._block_size

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        cache_dtype = self.cache_config.cache_dtype
        is_fp8 = cache_dtype in ("fp8_ds_mla", "fp8") or self.dtype == torch.uint8

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
                kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
                bounded_replay=self.bounded_replay,
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
            bounded_replay=self.bounded_replay,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls


class IluDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """SWA backend supporting bf16 and fp8_ds_mla cache."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",
    ]

    @classmethod
    def supports_non_causal(cls) -> bool:
        # DSpark draft blocks attend bidirectionally within the query window;
        # the builder encodes that via non-causal SWA indices.
        return True

    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[IluDeepseekSparseSWAMetadataBuilder]:
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


class IluDeepseekSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """Iluvatar SWA metadata builder using plugin kernels.

    Community ``DeepseekSparseSWAMetadataBuilder.build`` launches Triton kernels
    for decode/prefill SWA metadata. Ilu Triton currently fails on that decode
    kernel path, so this override keeps the community data layout but computes
    the affected metadata with the ``v41_swa_*`` plugin ops (see
    ``csrc/plugin/v41_swa_meta.cu``), which follow the community
    Triton kernels including ``token_offset`` and the uncompacted window span.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = self.vllm_config.speculative_config
        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        # The kernel pads the q-head count, so the index width must be a
        # multiple of 128 (matches community sparse_swa.py).
        self.noncausal_index_width = (
            cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
            if self.is_dspark
            else 0
        )
        # Same buffer the community builder writes from the Triton kernel.
        # Allocate here so the Torch helper and the metadata path cannot
        # silently pick different storage (see 441c2f60 accept-rate drop).
        self.decode_swa_indices_noncausal = (
            torch.empty(
                (
                    self.max_num_batched_tokens,
                    1,
                    self.noncausal_index_width,
                ),
                dtype=torch.int32,
                device=self.device,
            )
            if self.is_dspark
            else None
        )
        # FULL cudagraph replay can pad the request dimension beyond the real
        # batch. The community model state supplies replay_start for real
        # requests only, so keep a persistent buffer for the synthetic rows.
        self._padded_replay_start = torch.zeros_like(self.no_replay_start)

    def _pad_replay_start(
        self,
        replay_start: torch.Tensor | None,
        num_reqs: int,
    ) -> torch.Tensor:
        """Match replay_start to the request dimension used by attention metadata."""
        if replay_start is None:
            return self.no_replay_start[:num_reqs]
        num_actual_reqs = replay_start.numel()
        if num_actual_reqs >= num_reqs:
            return replay_start[:num_reqs]

        padded = self._padded_replay_start[:num_reqs]
        padded[num_actual_reqs:].zero_()
        padded[:num_actual_reqs].copy_(replay_start)
        return padded

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> DeepseekSparseSWAMetadata:
        """``replay_start`` ([num_reqs], SWA bounded replay): the position from
        which each request holds window KV. The model state passes it for every
        batch of a replaying group; graph captures pass nothing.
        """
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        torch.ge(slot_mapping, 0, out=is_valid_token)
        replay_start = self._pad_replay_start(replay_start, seq_lens.numel())

        non_causal = not common_attn_metadata.causal
        decode_swa_width = (
            self.noncausal_index_width if non_causal else self.window_size
        )
        decode_swa_indices = self.decode_swa_indices
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
                assert self.is_dspark, (
                    "Non-causal DeepseekV4 SWA is only supported for the DSpark "
                    "speculation mode, but causal=False was set without DSpark."
                )
                decode_swa_indices = self.decode_swa_indices_noncausal
                # No replay_start: decode rows sit above the hit, so a
                # request's replayed window never reaches them.
                build_dspark_noncausal_swa_indices(
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    block_table=block_table,
                    is_valid_token=is_valid_token,
                    block_size=self.block_size,
                    window_size=self.window_size,
                    index_width=self.noncausal_index_width,
                    num_decode_tokens=num_decode_tokens,
                    device=self.device,
                    out_indices=decode_swa_indices,
                    out_lens=self.decode_swa_lens,
                )
            else:
                torch.ops._C.v41_swa_indices_and_lens(
                    decode_swa_indices,
                    self.decode_swa_lens,
                    self.window_size,
                    decode_swa_indices.shape[-1],
                    self.decode_swa_lens,
                    self.decode_swa_lens,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    self.block_size,
                    replay_start,
                    num_tokens=num_decode_tokens,
                    token_offset=0,
                    has_image=False,
                )

        # Vision variant: per-token in-image visibility for prefill tokens.
        # Decode tokens are always past the image spans (spans are prefilled
        # atomically), so the decode path above never needs them.
        prefill_left_visible: torch.Tensor | None = None
        prefill_right_visible: torch.Tensor | None = None
        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if (
            self.max_image_tokens > 0
            and num_prefill_tokens > 0
            and mm_ranges
            and any(mm_ranges.values())
        ):
            raise AssertionError("image visibility not supported for now")
            prefill_left_visible, prefill_right_visible = self._build_image_visibility(
                common_attn_metadata.num_reqs,
                mm_ranges,
                num_decode_tokens,
                num_prefill_tokens,
                seq_lens,
                query_start_loc,
                token_to_req_indices,
            )

        # Prefill SWA indices live in paged coordinates. `token_offset` lets
        # the kernel read is_valid_token / token_to_req_indices at absolute
        # prefill positions while writing output starting at index 0.
        if num_prefill_tokens > 0:
            has_image = prefill_left_visible is not None
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            torch.ops._C.v41_swa_indices_and_lens(
                prefill_swa_indices,
                prefill_swa_lens,
                self.window_size,
                self.prefill_index_width,
                prefill_left_visible if has_image else prefill_swa_lens,
                prefill_right_visible if has_image else prefill_swa_lens,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                self.block_size,
                replay_start,
                num_tokens=num_prefill_tokens,
                token_offset=num_decode_tokens,
                has_image=has_image,
            )

        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            seq_lens_cpu,
            query_start_loc,
            query_start_loc_cpu,
            replay_start,
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
            decode_swa_width=decode_swa_width,
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
            prefill_left_visible=prefill_left_visible,
            prefill_right_visible=prefill_right_visible,
            replay_start=replay_start,
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            # Upper bound on decode-split rows for the kernel's max_q_len
            # hint. common max_query_len bounds every row (scheduled max under
            # adaptive verification), clamped to what the split can admit so a
            # mixed batch's prefill max does not inflate decode scheduling.
            max_decode_query_len=min(
                common_attn_metadata.max_query_len, self.decode_threshold
            ),
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            tile_sched_c1a=tile_sched[_LAYER_TYPE_C1A],
            tile_sched_c2a=tile_sched[_LAYER_TYPE_C2A],
            **deepseek_v4_fields,  # type: ignore[arg-type]
        )
        if envs.VLLM_ILU_DS4_PREFILL_OPT:
            self._populate_prefill_opt_fields(
                metadata,
                num_decodes,
                num_prefills,
                seq_lens,
                query_start_loc,
                query_start_loc_cpu,
                block_table,
                replay_start,
            )

        # DFlash/DSpark proposers assert ``attn_metadata.causal is False`` when
        # drafting a non-causal block. Community ``DeepseekSparseSWAMetadata``
        # has no ``causal`` field (FA/FlashInfer do); expose it so the assert
        # can verify this builder honored ``CommonAttentionMetadata.causal``.
        metadata.causal = common_attn_metadata.causal
        return metadata

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        out: dict[str, FlashMLASchedMeta | None] = {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
            _LAYER_TYPE_C1A: None,
            _LAYER_TYPE_C2A: None,
        }
        if (
            num_decode_tokens == 0
            or current_platform.is_rocm()
            or current_platform.is_xpu()
            or current_platform.is_device_capability_family(120)
        ):
            return out
        return out

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        replay_start: torch.Tensor,
    ) -> dict[str, torch.Tensor | None | int]:
        result: dict[str, torch.Tensor | int | None] = {}

        # --- Prefill query metadata ---
        if num_prefills > 0:
            assert seq_lens_cpu is not None
            # Plain slices, and the compressed prefill kernel on the opt path
            # reads ``prefill_seq_lens``, so these are unconditional.
            result["prefill_seq_lens"] = seq_lens[num_decodes:]
            result["prefill_seq_lens_cpu"] = seq_lens_cpu[num_decodes:]

        # The rest only feeds the workspace-gather prefill path, so neither
        # opt path pays for the gather-lens kernel or the CPU query lens.
        if num_prefills > 0 and not (
            envs.VLLM_ILU_DS4_PREFILL_OPT or envs.VLLM_ILU_DS4_PREFILL_AS_DECODE
        ):
            pfx_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=seq_lens.device
            )
            torch.ops._C.v41_swa_prefill_gather_lens(
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                replay_start,
                num_prefills,
                num_decodes,
                self.window_size,
            )

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
        replay_start: torch.Tensor,
    ) -> None:
        """Fields for ``ds4_swa_attn_fwd``; left ``None`` when the step must
        take the default prefill path instead.

        That kernel reads every request's keys from position 0 through the
        block table and takes no lower bound, so on its own it would attend to
        window pages below ``replay_start`` that bounded replay never writes.
        Its window is relative, though (key position ``k_len - q_len + i``):
        starting a request's key range at its replay start clamps the window
        exactly there. The block table only shifts by whole pages, so a
        request replaying from mid-page sends the step to the default path,
        whose gather lengths carry the bound.
        """
        metadata.prefill_cu_seqlens_q = None
        metadata.prefill_cu_seqlens_k = None
        metadata.prefill_max_seqlen_q = 0
        metadata.prefill_max_seqlen_k = 0
        metadata.prefill_swa_block_table = None

        if num_prefills == 0 or not envs.VLLM_ILU_DS4_PREFILL_OPT:
            return

        pfx_sl = num_decodes
        pfx_el = num_decodes + num_prefills

        pfx_replay = replay_start[pfx_sl:pfx_el]
        pfx_replay_cpu = pfx_replay.cpu()
        if bool((pfx_replay_cpu % self.block_size).any()):
            return

        metadata.prefill_cu_seqlens_q = (
            query_start_loc[pfx_sl : pfx_el + 1] - query_start_loc[pfx_sl]
        ).to(torch.int32)

        pfx_seq_lens = seq_lens[pfx_sl:pfx_el].to(torch.int32)
        swa_block_table = block_table[num_decodes:]
        if bool(pfx_replay_cpu.any()):
            pfx_seq_lens = pfx_seq_lens - pfx_replay.to(torch.int32)
            rows = block_table[pfx_sl:pfx_el]
            width = rows.shape[1]
            cols = torch.arange(width, device=rows.device).unsqueeze(0) + (
                pfx_replay.to(torch.int64) // self.block_size
            ).unsqueeze(1)
            swa_block_table = torch.gather(rows, 1, cols.clamp_(max=width - 1))
        cu_seqlens_k = torch.zeros(
            num_prefills + 1, dtype=torch.int32, device=seq_lens.device
        )
        cu_seqlens_k[1:] = pfx_seq_lens.cumsum(0)
        metadata.prefill_cu_seqlens_k = cu_seqlens_k

        qsl_cpu = query_start_loc_cpu[pfx_sl : pfx_el + 1]
        q_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
        metadata.prefill_max_seqlen_q = int(q_lens_cpu.max().item())

        prefill_seq_lens_cpu = seq_lens[pfx_sl:pfx_el].cpu() - pfx_replay_cpu
        metadata.prefill_max_seqlen_k = int(prefill_seq_lens_cpu.max().item())
        metadata.prefill_swa_block_table = swa_block_table
