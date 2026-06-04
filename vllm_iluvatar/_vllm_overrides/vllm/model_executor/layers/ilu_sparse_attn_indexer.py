# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import ixformer.inference.functions as ixfops
import torch
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.ilu_indexer import IluDeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


@torch.inference_mode()
def cp_gather_indexer_k_quant_cache(
    kv_cache,  # [num_blocks, block_size, head_dim]
    dst_value,  # [cu_seq_lens[-1], head_dim]
    block_table,  # [batch_size, num_blocks]
    cu_seq_lens,  # [batch_size + 1, ]
    batch_size,
):
    num_blocks, block_size, _ = kv_cache.shape
    head_dim = dst_value.shape[-1]
    kv_cache = kv_cache.view(num_blocks, -1)

    expected_value = []
    # expected_scale = []
    for b in range(batch_size):
        s = cu_seq_lens[b + 1] - cu_seq_lens[b]
        if s == 0:
            continue
        tot = cdiv(s, block_size)
        blocks = block_table[b, :tot]

        value = []
        scale = []
        full_block = torch.arange(tot - 1, device=kv_cache.device, dtype=torch.int32)
        non_remaining_value = kv_cache[
            blocks[full_block], : block_size * head_dim
        ].view(-1, head_dim)
        # non_remaining_scale = kv_cache[blocks[full_block],
        #                                block_size * head_dim:].view(-1, 4)

        remaining = s - (tot - 1) * block_size

        value = torch.cat(
            [
                non_remaining_value,
                kv_cache[blocks[-1], : remaining * head_dim].view(-1, head_dim),
            ],
            dim=0,
        )
        # scale = torch.cat([
        #     non_remaining_scale,
        #     kv_cache[blocks[-1], block_size * head_dim:block_size * head_dim +
        #              remaining * 4].view(-1, 4)
        # ],
        #                   dim=0)

        expected_value.append(value)
        # expected_scale.append(scale)

    gather_value = torch.cat(expected_value, dim=0).view(-1, head_dim)
    # gather_scale = torch.cat(expected_scale, dim=0).view(-1, 4)
    gather_value = gather_value.view(torch.bfloat16)
    # gather_scale = gather_scale.view(torch.float32)
    dst_value.copy_(gather_value)
    # dst_scale.copy_(gather_scale)


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q,
            k,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, IluDeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    ops.indexer_k_cache(k, kv_cache, slot_mapping)

    # topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            logits = ixfops.dsa_indexer_mqa_logits_with_blocks(
                q[chunk.token_start : chunk.token_end],
                chunk.cu_seqlens_q,
                chunk.cu_seq_lens,
                kv_cache,
                chunk.block_table,
                weights[chunk.token_start : chunk.token_end],
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
            )
            ixfops.dsa_update_topk_indices(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # TODO: support speculative decode
        if decode_metadata.requires_padding:
            raise NotImplementedError(
                "Sparse attention indexer does not support requires_padding"
            )

        # Use dsa_indexer_mqa_logits_with_blocks similar to prefill
        logits = ixfops.dsa_indexer_mqa_logits_with_blocks(
            q[:num_decode_tokens],
            decode_metadata.cu_seqlens_q,
            decode_metadata.cu_seqlens_kv,
            kv_cache,
            decode_metadata.block_table,
            weights[:num_decode_tokens],
            max_q_len=decode_metadata.max_q_len,
            max_kv_len=decode_metadata.max_kv_len,
            max_context_len=decode_metadata.max_context_len,
        )

        ixfops.dsa_update_topk_indices(
            logits,
            decode_metadata.cu_seqlen_ks,
            decode_metadata.cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer[:num_decode_tokens],
        )

    return topk_indices_buffer


def sparse_attn_indexer_original(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            k,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, IluDeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    # scale_fmt can be None, but the function expects str
    ops.indexer_k_cache(k, kv_cache, slot_mapping)

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), torch.bfloat16),
        )[0]
        for chunk in prefill_metadata.chunks:
            k = k_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    chunk.num_reqs,
                )

            logits = ops.ref_mqa_logits(
                q[chunk.token_start : chunk.token_end],
                k,
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = logits.topk(min(topk_tokens, logits.shape[-1]), dim=-1)[1]
            topk_indices -= chunk.cu_seqlen_ks[:, None]
            mask_lo = topk_indices >= 0
            mask_hi = (
                topk_indices - (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks)[:, None] < 0
            )
            mask = torch.full_like(
                topk_indices, False, dtype=torch.bool, device=topk_indices.device
            )
            mask = mask_lo & mask_hi
            topk_indices = topk_indices.masked_fill(~mask, -1)
            topk_indices_buffer[
                chunk.token_start : chunk.token_end, : topk_indices.shape[-1]
            ] = topk_indices.to(dtype=torch.int32)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        # kv_cache shape [
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_decode_tokens = pack_seq_triton(q[:num_decode_tokens], decode_lens)
        else:
            padded_q_decode_tokens = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_decode_tokens.shape[0]
        next_n = padded_q_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        logits = ops.ref_paged_mqa_logits(
            padded_q_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            max_model_len=max_model_len,
            clean_logits=False,
        )

        # padded query len
        current_device = padded_q_decode_tokens.device
        padded_num_tokens = batch_size * next_n
        positions = (
            torch.arange(max_model_len, device=current_device)
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(padded_num_tokens, device=current_device) // next_n
        next_n_offset = (
            torch.arange(padded_num_tokens, device=padded_q_decode_tokens.device)
            % next_n
        )
        index_end_pos = (
            decode_metadata.seq_lens[row_indices] - next_n + next_n_offset
        ).unsqueeze(1)
        # index_end_pos: [B * N, 1]
        mask = positions <= index_end_pos
        # mask: [B * N, L]
        logits = logits.masked_fill(~mask, float("-inf"))
        topk_indices = logits.topk(topk_tokens, dim=-1)[1].to(torch.int32)  # [B * N, K]
        # ensure we don't set indices for the top k
        # that is out of range(masked already)
        # this will happen if context length is shorter than K
        topk_indices[topk_indices > index_end_pos] = -1

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
        topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
            topk_indices.to(dtype=torch.int32)
        )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="ilu_sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("ilu_sparse_attn_indexer")
class IluSparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.ilu_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q,
            k,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
