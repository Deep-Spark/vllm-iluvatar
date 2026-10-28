# SPDX-License-Identifier: Apache-2.0
"""fp8-only sliding-window-attention (SWA) stack for DeepSeek-V4.

Packed decode and small-prefill batches use a contiguous ``[0:num_actual]``
SWA index buffer. Large causal TP1 prefill additionally consumes request-local
coordinates and gather lengths. FP8 therefore has its own
``IluFp8SWAMetadata`` and a builder that fully overrides ``build()``
(one position-based SWA pass over the full token range) while still subclassing
the community builder to inherit ``__init__`` (CUDA-graph-fixed persistent
buffers + spec-decode ``decode_threshold`` / ``window_size``). Kept model-local
so the shared ``ilu_sparse_swa.py`` stays fp8-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import torch
from vllm.config import get_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec, get_kv_quant_mode

from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekSparseSWABackend,
    IluDeepseekV4SWACache,
    build_dspark_noncausal_swa_indices,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

# Packed decode checks indices.stride(0)%256==0 (storage width), not size(1).
# Widen the backing row to 256 but expose [:window_size] so size(1) stays
# window_size; padding -1 past size(1) must not enter the K loop.
_FP8_SWA_INDEX_ALIGN = 256


@dataclass
class IluFp8SWAMetadata:
    """SWA metadata for packed decode and eager sparse FP8 prefill.

    Decode tensors remain views of persistent buffers for graph replay.
    Request-local prefill fields are optional and used only in eager prefill.
    """

    # Contiguous all-token (decode + prefill) SWA window slots / lengths.
    # Logical last dim is window_size; backing row may be 256-strided.
    all_swa_indices: torch.Tensor  # [num_tokens, 1, window_size] int32
    all_swa_lens: torch.Tensor  # [num_tokens] int32

    # Per-token validity + global request id, used to globalize compressed
    # top-k slots for the cr>1 extra path.
    is_valid_token: torch.Tensor  # [num_tokens] bool
    token_to_req_indices: torch.Tensor  # [num_tokens] int32

    # KV paged insert (fused qnorm + rope + fp8 insert into the SWA cache).
    slot_mapping: torch.Tensor  # [num_tokens] int32
    block_size: int

    # Number of *real* (scheduled) tokens; forward_mqa trims padding to this.
    # Pad rows at full-graph replay are made harmless by ``all_swa_lens`` being
    # zeroed past this point (see build()), not by the slice.
    num_actual_tokens: int = 0

    # Decode/prefill split (batch reordered decodes-first); num_decode_tokens
    # gates the C128A top-k assembly.
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Eager prefill consumes local query/request coordinates and paged caches.
    # Pure decode leaves these tensor fields None (its graph buffers unchanged).
    num_decodes: int = 0
    num_prefills: int = 0
    # Scheduler's CPU bound; selecting a measured short-context crossover must
    # not require reading the GPU seq_lens tensor back to the host.
    max_prefill_seq_len: int = 0
    query_start_loc: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    prefill_seq_lens: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None

    # False for a DSpark draft block (all queries attend bidirectionally within
    # the block). Mirrors the ``causal`` field the W4A8 builder exposes so a
    # DSpark/DFlash proposer can verify the builder honored the request.
    causal: bool = True


class IluDeepseekV4Fp8SWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """FP8 SWA builder for the packed path and request-local prefill gather.

    Reuses the community ``__init__`` write buffer (``size(-1)==window_size``).
    ``build()`` is fully overridden and, when ``window_size`` is not 256-aligned,
    copies into a 256-stride backing buffer then narrows to ``window_size``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        src_width = self.window_size
        aligned = (
            (src_width + _FP8_SWA_INDEX_ALIGN - 1)
            // _FP8_SWA_INDEX_ALIGN
            * _FP8_SWA_INDEX_ALIGN
        )
        if aligned != src_width:
            max_tokens, _, _ = self.decode_swa_indices.shape
            self._packed_swa_indices = torch.full(
                (max_tokens, 1, aligned),
                -1,
                dtype=torch.int32,
                device=self.decode_swa_indices.device,
            )
        else:
            self._packed_swa_indices = None

        # DSpark draft blocks are drafted non-causally: every query in the block
        # shares one key set (trailing window context + all block keys). The
        # causal ixformer kernel cannot express that, so those windows are built
        # in Torch (``build_dspark_noncausal_swa_indices``) into a dedicated
        # buffer whose backing row is 256-aligned (the fp8 decode kernel checks
        # ``indices.stride(0) % 256 == 0``, same constraint as the causal path).
        spec_config = get_current_vllm_config().speculative_config
        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        self.noncausal_index_width = (
            cdiv(self.window_size + self.num_speculative_tokens, _FP8_SWA_INDEX_ALIGN)
            * _FP8_SWA_INDEX_ALIGN
            if self.is_dspark
            else 0
        )
        self._decode_swa_indices_noncausal = (
            torch.full(
                (self.decode_swa_indices.shape[0], 1, self.noncausal_index_width),
                -1,
                dtype=torch.int32,
                device=self.decode_swa_indices.device,
            )
            if self.is_dspark
            else None
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> IluFp8SWAMetadata:
        causal = getattr(common_attn_metadata, "causal", True)

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Decode/prefill boundary (batch reordered decodes-first), using the
        # base-derived ``decode_threshold``.
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )
        num_swa_tokens = num_decode_tokens + num_prefill_tokens

        # The shared prefill combiner uses the full causal SWA window. Until
        # bounded replay is carried through both gather and combine, reject an
        # explicit boundary instead of gathering a shorter, incompatible pool.
        # Pure decode and non-causal DSpark keep their existing behavior.
        if num_prefills > 0 and causal and replay_start is not None:
            raise ValueError(
                "FP8 sparse prefill does not support explicit replay_start"
            )

        # Fixed-address buffers (allocated in the inherited __init__) keep CUDA
        # graph capture valid across replays.
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        if num_swa_tokens > 0:
            # CUDA-graph padding safety: zero ``decode_swa_lens`` past the real
            # token count so padded rows the captured op still visits carry
            # ``topk_length == 0`` (attend nothing); their stale index contents
            # are never read.
            self.decode_swa_lens[num_swa_tokens:] = 0
            if not causal:
                # DSpark draft block: every query shares one non-causal key set
                # (trailing window context + all block keys). Built in Torch;
                # ``ds4_compute_swa_decode_indices_and_lens`` is causal-only.
                assert self.is_dspark, (
                    "Non-causal DeepseekV4 fp8 SWA is only supported for the "
                    "DSpark speculation mode, but causal=False was set without "
                    "DSpark."
                )
                build_dspark_noncausal_swa_indices(
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    block_table=block_table,
                    is_valid_token=is_valid_token,
                    block_size=self.block_size,
                    window_size=self.window_size,
                    index_width=self.noncausal_index_width,
                    num_decode_tokens=num_swa_tokens,
                    device=self.device,
                    out_indices=self._decode_swa_indices_noncausal,
                    out_lens=self.decode_swa_lens,
                )
            else:
                # ``ds4_compute_swa_decode_indices_and_lens`` is purely
                # position-based, so a single pass over the full range yields the
                # correct causal SWA windows for both decode and prefill tokens.
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
                    num_decode_tokens=num_swa_tokens,
                )
                if self._packed_swa_indices is not None:
                    self._packed_swa_indices[
                        :num_swa_tokens, :, : self.window_size
                    ].copy_(self.decode_swa_indices[:num_swa_tokens])

        # Non-causal DSpark exposes the wider Torch-built buffer; the causal path
        # narrows to window_size. Either way stride(0) is 256-aligned.
        if not causal and self.is_dspark:
            all_swa_indices = self._decode_swa_indices_noncausal[
                :num_swa_tokens, :, : self.noncausal_index_width
            ]
        elif self._packed_swa_indices is not None:
            all_swa_indices = self._packed_swa_indices[
                :num_swa_tokens, :, : self.window_size
            ]
        else:
            all_swa_indices = self.decode_swa_indices[:num_swa_tokens]

        prefill_seq_lens = None
        prefill_gather_lens = None
        if num_prefills > 0 and causal:
            prefill_seq_lens = seq_lens[num_decodes : num_decodes + num_prefills]
            prefill_gather_lens = torch.empty(
                num_prefills,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            ixfops.ds4_compute_swa_prefill_gather_lens(
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
                replay_start=self.no_replay_start,
            )

        return IluFp8SWAMetadata(
            all_swa_indices=all_swa_indices,
            all_swa_lens=self.decode_swa_lens[:num_swa_tokens],
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            num_actual_tokens=num_swa_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            max_prefill_seq_len=(
                common_attn_metadata.max_seq_len if num_prefills else 0
            ),
            query_start_loc=query_start_loc if num_prefills else None,
            query_start_loc_cpu=(
                common_attn_metadata.query_start_loc_cpu if num_prefills else None
            ),
            block_table=block_table if num_prefills else None,
            prefill_seq_lens=prefill_seq_lens,
            prefill_gather_lens=prefill_gather_lens,
            causal=causal,
        )


class IluDeepseekV4Fp8SWABackend(IluDeepseekSparseSWABackend):
    """fp8-only SWA backend routing to the unified metadata builder."""

    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_SPARSE_SWA_FP8"

    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAMetadataBuilder]:
        return IluDeepseekV4Fp8SWAMetadataBuilder


class IluDeepseekV4Fp8SWACache(IluDeepseekV4SWACache):
    """fp8-only SWA cache; only ``get_attn_backend`` differs from the base.

    ``get_kv_cache_spec`` is inherited (already returns the 584B uint8
    ``fp8_ds_mla`` spec for the fp8 cache dtype).
    """

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV4Fp8SWABackend

    def get_kv_cache_spec(self, vllm_config):
        if self.dtype != torch.float8_e4m3fn:
            return super().get_kv_cache_spec(vllm_config)
        return SlidingWindowMLASpec(
            bounded_replay=self.bounded_replay,
            block_size=self.block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str="fp8",
            model_version="deepseek_v4",
            alignment=64,
            kv_quant_mode=get_kv_quant_mode("fp8"),
        )
