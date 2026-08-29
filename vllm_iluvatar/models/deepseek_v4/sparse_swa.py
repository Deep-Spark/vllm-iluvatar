# SPDX-License-Identifier: Apache-2.0
"""fp8-only sliding-window-attention (SWA) stack for DeepSeek-V4.

The fp8 attention unifies prefill and decode onto a single
``ds4_sparse_mla_fp8_decode`` call over all tokens, which needs a
contiguous ``[0:num_actual]`` SWA index buffer covering both. So fp8 gets its
own lean ``IluFp8SWAMetadata`` and a builder that fully overrides ``build()``
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
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekSparseSWABackend,
    IluDeepseekV4SWACache,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

# Packed decode checks indices.stride(0)%256==0 (storage width), not size(1).
# Widen the backing row to 256 but expose [:window_size] so size(1) stays
# window_size; padding -1 past size(1) must not enter the K loop.
_FP8_SWA_INDEX_ALIGN = 256


@dataclass
class IluFp8SWAMetadata:
    """Lean SWA metadata for the fp8 unified (single-call) MQA path.

    Only carries what ``fp8_attention.py`` consumes; every tensor is a slice of
    a persistent, fixed-address buffer so CUDA-graph capture / replay stays
    valid.
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


class IluDeepseekV4Fp8SWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """fp8-only SWA metadata builder for the unified (single-call) MQA path.

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluFp8SWAMetadata:
        # The community builder branches on ``causal`` and switches to a
        # non-causal kernel with wider index buffers for DSpark. This builder
        # only implements the causal window, so reject the other mode instead
        # of silently computing the wrong windows.
        if not getattr(common_attn_metadata, "causal", True):
            raise NotImplementedError(
                "IluDeepseekV4Fp8SWAMetadataBuilder only supports causal "
                "attention; the non-causal (DSpark) path is not implemented "
                "for the raw-FP8 DeepSeek-V4 model."
            )

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

        # Fixed-address buffers (allocated in the inherited __init__) keep CUDA
        # graph capture valid across replays.
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        if num_swa_tokens > 0:
            # ``ds4_compute_swa_decode_indices_and_lens`` is purely
            # position-based, so a single pass over the full range yields the
            # correct causal SWA windows for both decode and prefill tokens.
            # CUDA-graph padding safety: zero ``decode_swa_lens`` past the real
            # token count so padded rows the captured op still visits carry
            # ``topk_length == 0`` (attend nothing); their stale index contents
            # are never read.
            self.decode_swa_lens[num_swa_tokens:] = 0
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
                self._packed_swa_indices[:num_swa_tokens, :, : self.window_size].copy_(
                    self.decode_swa_indices[:num_swa_tokens]
                )

        # Narrow to window_size: squeezed 2D view is (T, window_size) with
        # stride(0) 256-aligned.
        all_swa_indices = (
            self._packed_swa_indices[:num_swa_tokens, :, : self.window_size]
            if self._packed_swa_indices is not None
            else self.decode_swa_indices[:num_swa_tokens]
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
