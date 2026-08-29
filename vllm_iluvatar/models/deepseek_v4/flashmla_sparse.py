# SPDX-License-Identifier: Apache-2.0
"""fp8-only FlashMLA-sparse builder/backend for DeepSeek-V4.

The C128A extra top-k that ``forward_mqa`` feeds to
``ds4_sparse_mla_fp8_decode`` is layer-invariant within a forward step (it
depends only on positions / block_table / slot_mapping). This builder does the
prefill globalization + decode/prefill merge **once per step** in ``build()``,
writing a contiguous ``[0:num_actual]`` global-paged top-k into fixed-address
buffers exposed via ``IluFlashMLASparseMetadata.c128a_global_topk_indices`` /
``c128a_topk_lens``, which ``fp8_attention.forward_mqa`` reads directly. Kept
model-local so the shared ``ilu_flashmla_sparse.py`` stays fp8-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import torch

from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluDeepseekV4FlashMLASparseBackend,
    IluFlashMLASparseMetadataBuilder,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

# Shared builder keeps 128-aligned rows (W4A8). Packed decode needs
# indices.stride(0) % 256 == 0; pad only this fp8 merged buffer.
_FP8_C128A_TOPK_ALIGNMENT = 256


class IluDeepseekV4Fp8FlashMLASparseMetadataBuilder(IluFlashMLASparseMetadataBuilder):
    """fp8-only main-MLA sparse builder that merges C128A top-k once per step.

    Reuses the shared builder's ``__init__`` / ``build()`` and only extends
    ``_build_c128a_metadata`` to globalize prefill and merge decode+prefill into
    a single all-token global top-k buffer.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Persistent buffers for the merged all-token global top-k. Backing
        # row is 256-aligned (stride(0)%256==0); expose only the logical
        # width (padding -1 must not enter the K loop).
        if self.compress_ratio == 128:
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            src_width = self.c128a_max_compressed
            self._c128a_logical_width = src_width
            aligned_width = (
                (src_width + _FP8_C128A_TOPK_ALIGNMENT - 1)
                // _FP8_C128A_TOPK_ALIGNMENT
                * _FP8_C128A_TOPK_ALIGNMENT
            )
            self.c128a_global_topk_buffer = torch.full(
                (max_num_batched_tokens, aligned_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_topk_lens_buffer = torch.empty(
                max_num_batched_tokens,
                dtype=torch.int32,
                device=self.device,
            )

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        result = super()._build_c128a_metadata(cm, req_id_per_token)
        if not result:
            return result

        decode_idx = result.get("c128a_global_decode_topk_indices")
        decode_lens = result.get("c128a_decode_topk_lens")
        prefill_local = result.get("c128a_prefill_topk_indices")

        num_decode_tokens = decode_idx.shape[0] if decode_idx is not None else 0
        num_prefill_tokens = prefill_local.shape[0] if prefill_local is not None else 0
        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return result

        # Compressed-domain paged block size (matches super()'s decode build).
        block_size = self.kv_cache_spec.block_size // self.compress_ratio

        # Per-token validity gate (raw slot mapping >=0 real, <0 padding); the
        # globalize kernel skips reads for padding rows.
        is_valid = cm.slot_mapping[:num_total] >= 0

        merged_idx = self.c128a_global_topk_buffer[:num_total]
        merged_lens = self.c128a_topk_lens_buffer[:num_total]

        # CUDA-graph padding safety: a full graph replays at the fixed max size,
        # so the baked op still visits rows past ``num_total``. Zero their
        # ``topk_length`` so they read no slot instead of dereferencing stale
        # global indices from a previous step (OOB risk); len-0 rows are never
        # read, so the index tail needs no zero-fill.
        self.c128a_topk_lens_buffer[num_total:] = 0

        if num_decode_tokens > 0:
            assert decode_idx is not None and decode_lens is not None
            src = decode_idx.reshape(num_decode_tokens, -1)
            merged_idx[:num_decode_tokens, : src.shape[-1]].copy_(src)
            merged_lens[:num_decode_tokens].copy_(decode_lens)

        if num_prefill_tokens > 0:
            assert prefill_local is not None
            pf_idx, pf_lens = ixfops.ds4_compute_global_topk_indices_and_lens(
                prefill_local[:num_prefill_tokens],
                req_id_per_token[num_decode_tokens:num_total],
                cm.block_table_tensor,
                block_size,
                is_valid[num_decode_tokens:num_total],
            )
            merged_idx[num_decode_tokens:num_total, : pf_idx.shape[-1]].copy_(pf_idx)
            merged_lens[num_decode_tokens:num_total].copy_(pf_lens)

        # size(1) stays the logical width; stride(0) is 256-aligned.
        result["c128a_global_topk_indices"] = merged_idx[:, : self._c128a_logical_width]
        result["c128a_topk_lens"] = merged_lens
        return result


class IluDeepseekV4Fp8FlashMLASparseBackend(IluDeepseekV4FlashMLASparseBackend):
    """fp8-only V4 main-MLA sparse backend routing to the merged-topk builder."""

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV4_FP8"

    @staticmethod
    def get_builder_cls() -> type[IluDeepseekV4Fp8FlashMLASparseMetadataBuilder]:
        return IluDeepseekV4Fp8FlashMLASparseMetadataBuilder
