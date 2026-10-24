# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GLM5Next Kpool sparse indexer built on the generic DSA kernels."""

from __future__ import annotations

from enum import Enum

import ixformer.inference.functions as ixfops
import torch
from ixformer.inference.functions.models.glm5next import (
    expand_pools_and_append_tail,
    kpool_compress_insert,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.common.sparse_indexer import (
    _build_decode_scatter_indices,
    _scatter_decode_tokens_by_request,
)
from vllm.models.glm5next.sparse_indexer import SparseAttnIndexerKpool
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
)

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadata,
)


def _fill_short_decode_causal_indices(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor | None,
    num_decode_tokens: int,
    max_seq_len: int,
    topk_tokens: int,
    compact_pool_indices: bool = False,
) -> bool:
    if positions is None or positions.numel() == 0 or max_seq_len > topk_tokens:
        return False
    rows = topk_indices_buffer[:num_decode_tokens]
    pos = positions[:num_decode_tokens]
    throwaway_lens = torch.empty(
        num_decode_tokens, dtype=torch.int32, device=rows.device
    )
    ixfops.dsv41_fill_short_context_topk(
        rows, throwaway_lens, pos, 4 if compact_pool_indices else 1
    )
    return True


class IndexerType(Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    FP4 = "fp4"


def _kpool_mqa_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    valid_lens: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    max_context_len: int,
    indexer_type: IndexerType,
) -> torch.Tensor:
    if indexer_type == IndexerType.FP8:
        return ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
            q,
            cu_seqlens_q,
            cu_seqlens_kv,
            kv_cache,
            block_table,
            weights,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            max_context_len=max_context_len,
            valid_lens=valid_lens,
        )
    elif indexer_type in (IndexerType.BF16, IndexerType.INT8):
        return ixfops.dsa_indexer_mqa_logits_with_kvcache(
            q,
            cu_seqlens_q,
            cu_seqlens_kv,
            kv_cache,
            block_table,
            weights,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            max_context_len=max_context_len,
            valid_lens=valid_lens,
            backend="ixinfer",
        )
    else:
        raise NotImplementedError(f"Unsupported indexer type: {indexer_type}")


def _select_pools_and_expand(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    out_rows: torch.Tensor,
    compact_pool_indices: bool = False,
) -> None:
    select_k = topk_tokens // index_kpool
    num_rows = logits.shape[0]
    pool_topk = (
        out_rows
        if compact_pool_indices
        else torch.empty((num_rows, select_k), dtype=torch.int32, device=logits.device)
    )
    ixfops.dsa_update_topk_indices(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        select_k,
        pool_topk,
    )
    if compact_pool_indices:
        return
    expanded = expand_pools_and_append_tail(
        pool_topk,
        positions,
        index_kpool,
    )
    out_rows[:, : expanded.shape[-1]] = expanded


@eager_break_during_capture
def ilu_sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    topk_backend: str = "auto",
    compact_pool_indices: bool = False,
) -> torch.Tensor:
    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, IluDeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    if use_fp4_cache:
        raise NotImplementedError(
            "Iluvatar GLM5Next kpool indexer does not support the fp4 index cache"
        )

    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if scale_fmt == "ue8m0":
        indexer_type = IndexerType.FP8
    elif scale_fmt == "float32":
        indexer_type = IndexerType.INT8
    elif scale_fmt == "None":
        indexer_type = IndexerType.BF16
    else:
        raise NotImplementedError(f"Unsupported scale format: {scale_fmt}")
    assert q_scale is None, "q_scale is only used by the (unsupported) fp4 cache"

    assert index_kpool > 1, "kpool indexer requires index_kpool > 1"
    if not skip_k_cache_insert:
        if gate_score is not None and compress_ape is not None:
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                assert has_prefill, "prefill tokens are not found"
                prefill_slice = slice(num_decode_tokens, num_tokens)
                if indexer_type in (
                    IndexerType.FP8,
                    IndexerType.INT8,
                    IndexerType.BF16,
                ):
                    kpool_compress_insert(
                        k[prefill_slice],
                        gate_score[prefill_slice],
                        compress_ape,
                        kv_cache,
                        slot_mapping[prefill_slice],
                        index_kpool,
                        head_dim,
                        indexer_type == IndexerType.FP8,
                        indexer_type == IndexerType.INT8,
                    )
                else:
                    raise NotImplementedError(
                        f"kpool compress-write not implemented for {indexer_type}"
                    )
                if tail_kv_cache is not None and tail_prefix is not None:
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, IluDeepseekV32IndexerMetadata)
                        kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            raise AssertionError(
                "kpool indexer requires gate_score and compress_ape are not None"
            )

    if not compact_pool_indices:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    elif num_tokens < hidden_states.shape[0]:
        topk_indices_buffer[num_tokens : hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            if chunk.max_context_len == 0:
                if compact_pool_indices:
                    topk_indices_buffer[chunk.token_start : chunk.token_end] = -1
                continue
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            w_slice = weights[chunk.token_start : chunk.token_end]
            logits = _kpool_mqa_logits(
                q_slice,
                w_slice,
                kv_cache,
                chunk.block_table,
                chunk.cu_seqlens_q,
                chunk.cu_seq_lens,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                chunk.valid_lens,
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
                indexer_type=indexer_type,
            )
            _select_pools_and_expand(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                positions[chunk.token_start : chunk.token_end],
                topk_tokens,
                index_kpool,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
                compact_pool_indices,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None

        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
            assert isinstance(tail_meta, IluDeepseekV32IndexerMetadata)
            if indexer_type in (
                IndexerType.FP8,
                IndexerType.INT8,
                IndexerType.BF16,
            ):
                _kpool_decode_tail_update(
                    kv_cache,
                    tail_kv_cache,
                    tail_meta,
                    k,
                    gate_score,
                    compress_ape,
                    slot_mapping,
                    positions,
                    decode_metadata,
                    attn_metadata_narrowed.num_decodes,
                    num_decode_tokens,
                    index_kpool,
                    head_dim,
                    indexer_type == IndexerType.INT8,
                )
            else:
                raise NotImplementedError(
                    f"kpool decode compress-write not implemented for {indexer_type}"
                )

        if _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
            compact_pool_indices,
        ):
            return topk_indices_buffer

        q_dec = q_quant[:num_decode_tokens]
        w_dec = weights[:num_decode_tokens]
        logits = _kpool_mqa_logits(
            q_dec,
            w_dec,
            kv_cache,
            decode_metadata.block_table,
            decode_metadata.cu_seqlens_q,
            decode_metadata.cu_seqlens_kv,
            decode_metadata.cu_seqlen_ks,
            decode_metadata.cu_seqlen_ke,
            decode_metadata.valid_lens,
            max_q_len=decode_metadata.max_q_len,
            max_kv_len=decode_metadata.max_kv_len,
            max_context_len=decode_metadata.max_context_len,
            indexer_type=indexer_type,
        )
        _select_pools_and_expand(
            logits,
            decode_metadata.cu_seqlen_ks,
            decode_metadata.cu_seqlen_ke,
            positions[:num_decode_tokens],
            topk_tokens,
            index_kpool,
            topk_indices_buffer[:num_decode_tokens],
            compact_pool_indices,
        )

    return topk_indices_buffer


def _kpool_decode_tail_update(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_meta: IluDeepseekV32IndexerMetadata,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    decode_metadata,
    num_requests: int,
    num_decode_tokens: int,
    index_kpool: int,
    head_dim: int,
    store_int8: bool,
) -> None:
    per_req_lens = decode_metadata.per_req_decode_lens
    if per_req_lens is not None:
        use_uniform = (
            decode_metadata.decode_is_uniform
            and num_decode_tokens == num_requests * decode_metadata.write_max_decode_len
        )
        group_lens = per_req_lens
        lmax = decode_metadata.write_max_decode_len
    else:
        use_uniform = not decode_metadata.requires_padding
        group_lens = decode_metadata.decode_lens
        lmax = int(decode_metadata.decode_lens.max().item())

    if use_uniform:
        next_n = num_decode_tokens // num_requests
        shape2 = (num_requests, next_n)
        dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
        dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
        dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
        dec_pos = positions[:num_decode_tokens].view(shape2)
        dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
    else:
        scatter_idx = _build_decode_scatter_indices(
            group_lens, num_requests, num_decode_tokens
        )
        dec_k = _scatter_decode_tokens_by_request(
            k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_gate = _scatter_decode_tokens_by_request(
            gate_score[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_slot = _scatter_decode_tokens_by_request(
            slot_mapping[:num_decode_tokens], -1, num_requests, lmax, scatter_idx
        )
        dec_pos = _scatter_decode_tokens_by_request(
            positions[:num_decode_tokens],
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )
        dec_tail_slot = _scatter_decode_tokens_by_request(
            tail_meta.slot_mapping[:num_decode_tokens],
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )

    kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail_kv_cache,
        dec_tail_slot,
        dec_k,
        dec_gate,
        compress_ape,
        dec_slot,
        dec_pos,
        index_kpool,
        head_dim,
        True,
        store_int8,
    )


@SparseAttnIndexerKpool.register_oot()
class IluSparseAttnIndexerKpool(SparseAttnIndexerKpool):
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_pool_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        CustomOp.__init__(self)
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_pool_len = max_pool_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""

        if use_fp4_cache or tail_cache is None:
            raise NotImplementedError(
                "Iluvatar GLM5Next requires an FP8 index cache and a paged tail cache"
            )

        from vllm.config import get_current_vllm_config_or_none

        _cfg = get_current_vllm_config_or_none()
        _parallel = _cfg.parallel_config if _cfg is not None else None
        if _parallel is not None and (
            _parallel.prefill_context_parallel_size > 1
            or _parallel.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "IluSparseAttnIndexerKpool does not support PCP/DCP."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return ilu_sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            "auto",
            getattr(self, "compact_pool_indices", False),
        )

    forward_native = forward_cuda
