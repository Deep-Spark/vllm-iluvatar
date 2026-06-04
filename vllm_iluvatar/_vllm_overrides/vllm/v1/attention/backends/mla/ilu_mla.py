# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar MLA attention backend, aligned with v0.17.0 TritonMLAImpl.

Prefill: handled by MLACommonImpl using ixformer's flash_attn_varlen_func.
Decode:  uses ixformer's vllm_paged_attention_mla_fused kernel.

Registered as ILU_MLA in cuda.py so that the priority list
in _get_backend_priorities picks it up automatically.
"""

import functools
from typing import TYPE_CHECKING, ClassVar

import ixformer.inference.functions as ixf_ops
import torch
import vllm.envs as envs

# Iluvatar 上 vLLM 默认的 merge_attn_states 会走
# `torch.ops._C.merge_attn_states`，该 op 在 Corex 上没有注册。
# 这里直接用 ixformer 自带的 `merge_attn_states`，签名是
#   (prefix_output, prefix_lse, suffix_output, suffix_lse,
#    output=None, return_lse=False)
# 与默认接口不同，使用时按 ixformer 风格传参。
from ixformer.contrib.vllm_flash_attn import (
    merge_attn_states as ixf_merge_attn_states,  # type: ignore[import]
)
from vllm import _custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType, MultipleOf, SparseMLAAttentionImpl
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

logger = init_logger(__name__)


class IluMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "ILU_MLA"

    @staticmethod
    def get_impl_cls() -> type["IluMLAImpl"]:
        return IluMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class IluMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "IluMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "IluMLAImpl"
            )

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 ILU MLA not yet supported")

    def reorg_kvcache(
        self,
        allgatered_kv_c_normed: torch.Tensor,
        allgatered_k_pe: torch.Tensor,
        padded_local_chunk_seq_lens_lst: list[int],
        local_context_lens_allranks: list[list[int]],
        sum_seq_len: int,
        max_seq_len: int,
        chunk_size: int,
        chunk_idx: int,
        toks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
        e.g.
        allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                                T0_4, T0_5, pad, pad, T1_2, pad, ...]
        -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                    T1_0, T1_1, T1_2, ...]
        Args:
            padded_local_chunk_seq_lens_lst: local chunk context lengths
                under current CP rank.
            local_context_lens_allranks: local context lengths on each CP rank.
            sum_seq_len: the sum of cp_chunk_seq_lens_lst.
            max_seq_len: the max value of cp_chunk_seq_lens_lst.
            chunk_size: the local padded max context chunk from
                chunked_context_metadata building.
            chunk_idx: chunk idx of chunked_prefill.
            toks: the number of tokens for local gather cache.
        """
        kv_c_segments = []
        k_pe_segments = []
        src_token_idx = 0
        max_seq_len_check = 0
        for padded_local_chunk_seq_len, local_context_lens in zip(
            padded_local_chunk_seq_lens_lst, local_context_lens_allranks
        ):
            cur_seq_len = 0
            for rank, local_context_len in enumerate(local_context_lens):
                # Note(qcs): We split the context into multiple chunks,
                # depending on the size of the workspace.
                # local_context in dcp0:   |-----------------|
                # local_context in dcp1:   |--------------|
                # n*padded_local_chunk:    |-----|-----|-----|
                # local_chunk_len in dcp1: |-----|-----|--|
                # so we need update the last chunk length in dcp1.
                local_chunk_len = min(
                    max(0, local_context_len - chunk_idx * chunk_size),
                    padded_local_chunk_seq_len,
                )
                if local_chunk_len != 0:
                    kv_c_segment = allgatered_kv_c_normed[
                        rank * toks
                        + src_token_idx : rank * toks
                        + src_token_idx
                        + local_chunk_len
                    ]
                    k_pe_segment = allgatered_k_pe[
                        rank * toks
                        + src_token_idx : rank * toks
                        + src_token_idx
                        + local_chunk_len
                    ]
                    kv_c_segments.append(kv_c_segment)
                    k_pe_segments.append(k_pe_segment)
                    cur_seq_len += local_chunk_len
            max_seq_len_check = max(max_seq_len_check, cur_seq_len)
            src_token_idx += padded_local_chunk_seq_len
        reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
        reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
        assert reorganized_kv_c_normed.shape[0] == sum_seq_len
        assert reorganized_k_pe.shape[0] == sum_seq_len
        assert max_seq_len_check == max_seq_len
        return reorganized_kv_c_normed, reorganized_k_pe

    def _context_parallel_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        dcp_world_size: int,
    ):
        assert k_scale is None, "DCP not support scaled kvcache now."
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.padded_local_chunk_seq_lens is not None
        assert prefill_metadata.chunked_context.local_context_lens_allranks is not None
        assert prefill_metadata.chunked_context.padded_local_cu_seq_lens is not None
        assert prefill_metadata.chunked_context.cu_seq_lens_lst is not None
        assert prefill_metadata.chunked_context.chunk_size is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.padded_local_cu_seq_lens[
                    i
                ],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
            # workspace
            # |------- N tokens --------|--------- N*dcp_size tokens ----------|
            # |<- use for local_gather ->|<--------- use for allgather -------->|
            allgather_offset = workspace.shape[0] // (dcp_world_size + 1)
            assert allgather_offset * (dcp_world_size + 1) == workspace.shape[0]
            assert toks <= allgather_offset
            local_gathered_kvcache = workspace[:toks]
            cur_allgather_workspace = workspace[
                allgather_offset : allgather_offset * (1 + dcp_world_size)
            ]
            assert toks * dcp_world_size <= cur_allgather_workspace.shape[0]
            cur_allgather_kvcache = cur_allgather_workspace[: toks * dcp_world_size]
            cur_allgather_kvcache.copy_(
                get_dcp_group().all_gather(local_gathered_kvcache, dim=0)
            )
            assert (
                cur_allgather_kvcache.shape[-1]
                == self.kv_lora_rank + self.qk_rope_head_dim
            )
            allgatered_kv_c_normed, allgatered_k_pe = cur_allgather_kvcache.unsqueeze(
                1
            ).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            kv_c_normed, k_pe = self.reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=prefill_metadata.chunked_context.padded_local_chunk_seq_lens[
                    i
                ],
                local_context_lens_allranks=prefill_metadata.chunked_context.local_context_lens_allranks,
                sum_seq_len=prefill_metadata.chunked_context.cu_seq_lens_lst[i][-1],
                max_seq_len=prefill_metadata.chunked_context.max_seq_lens[i],
                chunk_size=prefill_metadata.chunked_context.chunk_size,
                chunk_idx=i,
                toks=toks,
            )

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                attn_metadata.prefill.prefill_backend.run_prefill_context_chunk(
                    chunk_idx=i,
                    q=q,
                    k=k,
                    v=v,
                )
            )
            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output, output_lse = ixf_merge_attn_states(
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                    return_lse=True,
                )
                ixf_merge_attn_states(
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                    return_lse=True,
                )

        return output, output_lse

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        kv_c_and_k_pe_cache_scale: torch.Tensor | None = None,
    ):
        """Iluvatar-specific 覆写 chunked-context prefill 的 KV 收集。

        基类默认走 `ops.gather_and_maybe_dequant_cache`（Iluvatar 上 -> 走
        `torch.ops._C_cache_ops.gather_and_maybe_dequant_cache`），该 op 在
        Corex 上没有注册。这里参照 v0.17.0 适配的做法：
        - INT8_MLA：调用 ixformer 的 `vllm_gather_cache_int8`（即
          `ops.gather_cache_int8`，本身就是 ixformer 实现）。
        - 其他情况：直接调 ixformer 的 `vllm_cp_gather_cache`，绕开
          `_C_cache_ops` 命名空间。Iluvatar 上 KV cache 保持 bf16/fp16，
          不需要反量化，因此不必走 `gather_and_maybe_dequant_cache`。
        其余逻辑（kv_b_proj、concat、chunk attention、merge_lse）保持与
        基类一致，避免和上层的合并/输出契约偏离。
        """
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        use_int8_mla = envs.VLLM_USE_INT8_MLA

        output = None
        output_lse = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            if use_int8_mla:
                ops.gather_cache_int8(
                    src_cache=kv_c_and_k_pe_cache,
                    src_cache_scale=kv_c_and_k_pe_cache_scale,
                    kv_lora_rank=self.kv_lora_rank,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=(prefill_metadata.chunked_context.cu_seq_lens[i]),
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=(prefill_metadata.chunked_context.starts[i]),
                )
            else:
                ixf_ops.vllm_cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=(prefill_metadata.chunked_context.cu_seq_lens[i]),
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=(prefill_metadata.chunked_context.starts[i]),
                )

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                attn_metadata.prefill.prefill_backend.run_prefill_context_chunk(
                    chunk_idx=i,
                    q=q,
                    k=k,
                    v=v,
                )
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output, output_lse = ixf_merge_attn_states(
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                    return_lse=True,
                )

        return output, output_lse

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        # q_nope: (B, N, P), self.W_UK_T: (N, P, L)
        q_t = q_nope.transpose(0, 1)  # (N, B, P)
        ql = torch.bmm(q_t, self.W_UK_T)  # (N, B, L)
        return ql.transpose(0, 1)  # (B, N, L)

    def _v_up_proj(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, N, L), self.W_UV: (N, L, V)
        x_t = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        result = torch.bmm(x_t, self.W_UV)  # (N, B, V)
        return result.transpose(0, 1)  # (B, N, V)

    # ------------------------------------------------------------------
    # forward_full: full prefill+decode path (moved from
    # MLAAttention._forward_impl_ilu)
    # ------------------------------------------------------------------
    def forward_full(
        self,
        layer: "MLAAttention",
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Iluvatar-optimized forward_impl aligned with v0.17.0.

        Key differences from the default path:
        - Rotary embedding is applied inside (not in the wrapper)
        - KV cache write is done inside (not in forward())
        - Prefill uses ixf_ops.mla_rope for fused rope
        - Decode uses ixf_ops.vllm_paged_attention_mla_fused
        """
        if attn_metadata is None:
            _ = torch.empty(
                (
                    layer.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
            output = torch.empty(
                output.shape[0],
                self.v_head_dim * self.num_heads,
                device=q.device,
                dtype=q.dtype,
            )
            return output

        if self.dcp_world_size == -1:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)
        assert not isinstance(
            self, SparseMLAAttentionImpl
        ), "SparseMLAAttentionImpl is not supported with Iluvatar MLA"

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]
        k_pe = k_pe.unsqueeze(1)

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]
        prefill_k = torch.empty_like(prefill_q)

        write_kv_cache = (None, None)
        is_int8_mla = envs.VLLM_USE_INT8_MLA
        if kv_cache.numel() > 0:
            if has_decode:
                decode_q_pe, decode_k_pe = layer.rotary_emb(
                    positions[:num_decode_tokens],
                    decode_q[..., self.qk_nope_head_dim :],
                    k_pe[:num_decode_tokens],
                )
                if is_int8_mla:
                    kc_int8, kc_s, _ = ops.scaled_int8_quant(
                        k_c_normed[:num_decode_tokens]
                    )
                    kpe_int8, kpe_s, _ = ops.scaled_int8_quant(decode_k_pe.contiguous())
                    ops.concat_and_cache_mla_int8(
                        kv_c_int8=kc_int8,
                        kv_c_scale=kc_s[..., 0],
                        k_pe_int8=kpe_int8,
                        k_pe_scale=kpe_s[..., 0].view(-1, kpe_int8.shape[-2]),
                        kv_cache=kv_cache,
                        kv_cache_scale=kv_cache_scale,
                        slot_mapping=(
                            attn_metadata.slot_mapping.flatten()[:num_decode_tokens]
                        ),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=layer._k_scale,
                    )
                elif self.dcp_world_size > 1 or layer.use_spec_decode:
                    ops.concat_and_cache_mla(
                        k_c_normed[:num_decode_tokens],
                        decode_k_pe,
                        kv_cache,
                        attn_metadata.slot_mapping.flatten()[:num_decode_tokens],
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=layer._k_scale,
                    )
                else:
                    write_kv_cache = (k_c_normed[:num_decode_tokens], decode_k_pe)

            if has_prefill:
                ixf_ops.mla_rope(
                    positions[num_decode_tokens:],
                    prefill_q[..., self.qk_nope_head_dim :],
                    prefill_k_pe.squeeze(1),
                    prefill_k[..., self.qk_nope_head_dim :],
                    layer.rotary_emb.cos_sin_cache,
                )
                if is_int8_mla:
                    pkc_int8, pkc_s, _ = ops.scaled_int8_quant(prefill_k_c_normed)
                    pkpe = prefill_k[..., self.qk_nope_head_dim :].contiguous()
                    pkpe_int8, pkpe_s, _ = ops.scaled_int8_quant(pkpe)
                    ops.concat_and_cache_mla_int8(
                        pkc_int8,
                        pkc_s[..., 0],
                        pkpe_int8,
                        pkpe_s[..., 0].view(-1, pkpe_int8.shape[-2]),
                        kv_cache,
                        kv_cache_scale,
                        attn_metadata.slot_mapping.flatten()[num_decode_tokens:],
                        self.kv_cache_dtype,
                        layer._k_scale,
                    )
                else:
                    ops.concat_and_cache_mla(
                        prefill_k_c_normed,
                        prefill_k[..., self.qk_nope_head_dim :],
                        kv_cache,
                        attn_metadata.slot_mapping.flatten()[num_decode_tokens:],
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=layer._k_scale,
                    )

        output = torch.empty(
            output.shape[0],
            self.num_heads,
            self.v_head_dim,
            device=q.device,
            dtype=q.dtype,
        )

        if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            kv_nope = self.kv_b_proj(prefill_k_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            prefill_k[..., : self.qk_nope_head_dim] = k_nope

            has_context = (
                attn_metadata.prefill is not None
                and attn_metadata.prefill.chunked_context is not None
            )

            output_prefill = (
                attn_metadata.prefill.prefill_backend.run_prefill_new_tokens(
                    q=prefill_q,
                    k=prefill_k,
                    v=v,
                    return_softmax_lse=has_context,
                )
            )

            if has_context:
                suffix_output, suffix_lse = output_prefill
                if self.dcp_world_size > 1:
                    context_output, context_lse = (
                        self._context_parallel_compute_prefill_context(
                            prefill_q,
                            kv_cache,
                            attn_metadata,
                            k_scale=None,
                            dcp_world_size=self.dcp_world_size,
                        )
                    )
                else:
                    context_output, context_lse = self._compute_prefill_context(
                        prefill_q,
                        kv_cache,
                        attn_metadata,
                        layer._k_scale,
                        kv_c_and_k_pe_cache_scale=kv_cache_scale,
                    )
                merged = torch.empty_like(suffix_output)
                merged = ixf_merge_attn_states(
                    prefix_output=context_output,
                    prefix_lse=context_lse,
                    suffix_output=suffix_output,
                    suffix_lse=suffix_lse,
                    output=merged,
                )
                output_prefill = merged

            output[num_decode_tokens:] = output_prefill

        if has_decode:
            assert attn_metadata.decode is not None
            mqa_q = decode_q

            attn_out, lse = self.forward_mqa(
                mqa_q[..., : self.qk_nope_head_dim],
                decode_q_pe,
                kv_cache,
                attn_metadata,
                *write_kv_cache,
                kv_c_and_k_pe_cache_scale=kv_cache_scale,
            )
            if self.dcp_world_size > 1:
                assert lse is not None
                attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())
                output[:num_decode_tokens] = self._v_up_proj(attn_out)
            else:
                assert lse is None
                output[:num_decode_tokens] = attn_out

        return output.view(output.shape[0], self.v_head_dim * self.num_heads)

    # ------------------------------------------------------------------
    # setup_projections: ixformer marlin weight repacking (moved from
    # MLAAttention._setup_ilu_marlin_projections)
    # ------------------------------------------------------------------
    def setup_projections(
        self,
        layer: "MLAAttention",
        act_dtype: torch.dtype,
    ) -> bool:
        """Set up ixformer marlin-optimized _k_up_proj/_v_up_proj.

        Returns True if marlin path was used, False to fall back to standard.
        """
        if not envs.VLLM_MLA_CUSTOMIZE:
            return False

        kv_b_proj_module = self.kv_b_proj
        quant_method_name = (
            kv_b_proj_module.quant_method.__class__.__name__
            if kv_b_proj_module.quant_method is not None
            else None
        )

        if hasattr(
            kv_b_proj_module, "scheme"
        ) and kv_b_proj_module.scheme.__class__.__name__ == [
            "CompressedTensorsW8A8Int8",
            "IluCompressedTensorsW8A8Int8",
        ]:
            kv_b_proj_weight = kv_b_proj_module.weight
            kv_b_proj_weight_scale = kv_b_proj_module.weight_scale
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            W_UK, W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            kv_b_proj_weight_scale = kv_b_proj_weight_scale.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                1,
            )
            W_UK_S, W_UV_S = kv_b_proj_weight_scale.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=1
            )

            self.W_UK = ixf_ops.marlin_w8_weight_repack(
                weights=W_UK.permute(1, 2, 0).contiguous(),
                weight_format="int8",
                reformat="k16n16_grouped_n",
            )
            self.W_UK_S = W_UK_S.contiguous()
            self.W_UV = ixf_ops.marlin_w8_weight_repack(
                weights=W_UV.permute(1, 0, 2).contiguous(),
                weight_format="int8",
                reformat="k16n16",
            )
            self.W_UV_S = W_UV_S.contiguous()

            def _v_up_proj_w8a8(impl_self, x, out=None):
                return ixf_ops.marlin_w8a16(
                    x,
                    impl_self.W_UV,
                    impl_self.W_UV_S,
                    group_size=-1,
                    format="k16n16",
                    batch_first=False,
                )

            def _k_up_proj_w8a8(impl_self, x_nope):
                return ixf_ops.marlin_w8a16(
                    x_nope,
                    impl_self.W_UK,
                    impl_self.W_UK_S,
                    group_size=-1,
                    format="k16n16_grouped_n",
                    batch_first=False,
                )

            self._v_up_proj = functools.partial(_v_up_proj_w8a8, self)
            self._k_up_proj = functools.partial(_k_up_proj_w8a8, self)
            return True

        elif quant_method_name == "IluAWQMarlinLinearMethod":

            def split_4bit_matrix(
                qweight, scales, qzeros, num_heads, head_dim_1, head_dim_2
            ):
                assert head_dim_1 % 8 == 0
                assert head_dim_2 % 8 == 0
                qweight = qweight.view(-1, num_heads, (head_dim_1 + head_dim_2) // 8)
                w1, w2 = qweight.split([head_dim_1 // 8, head_dim_2 // 8], dim=-1)
                scales = scales.view(-1, num_heads, head_dim_1 + head_dim_2)
                s1, s2 = scales.split([head_dim_1, head_dim_2], dim=-1)
                qzeros = qzeros.view(-1, num_heads, (head_dim_1 + head_dim_2) // 8)
                z1, z2 = qzeros.split([head_dim_1 // 8, head_dim_2 // 8], dim=-1)
                return (
                    w1.contiguous().view(w1.shape[0], -1),
                    s1.contiguous().view(s1.shape[0], -1),
                    z1.contiguous().view(z1.shape[0], -1),
                    w2.contiguous().view(w2.shape[0], -1),
                    s2.contiguous().view(s2.shape[0], -1),
                    z2.contiguous().view(z2.shape[0], -1),
                )

            (W_UK, W_UK_S, W_UK_Z, W_UV, W_UV_S, W_UV_Z) = split_4bit_matrix(
                kv_b_proj_module.qweight,
                kv_b_proj_module.scales,
                kv_b_proj_module.qzeros,
                self.num_heads,
                self.qk_nope_head_dim,
                self.v_head_dim,
            )

            W_UK = (
                W_UK.reshape(self.kv_lora_rank, self.num_heads, -1)
                .permute(1, 2, 0)
                .contiguous()
            )
            W_UK_S = (
                W_UK_S.reshape(W_UK_S.shape[0], self.num_heads, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            W_UK_Z = (
                W_UK_Z.reshape(W_UK_Z.shape[0], self.num_heads, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            (self.W_UK, self.W_UK_S, self.W_UK_Z) = ixf_ops.marlin_w4_weight_repack(
                weights=W_UK,
                scales=W_UK_S,
                zeros=W_UK_Z,
                weight_format="gptq_grouped_n",
                reformat="k16n32_grouped_n",
            )

            W_UV = (
                W_UV.reshape(self.kv_lora_rank, self.num_heads, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            W_UV_S = (
                W_UV_S.reshape(W_UV_S.shape[0], self.num_heads, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            W_UV_Z = (
                W_UV_Z.reshape(W_UV_Z.shape[0], self.num_heads, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            (self.W_UV, self.W_UV_S, self.W_UV_Z) = ixf_ops.marlin_w4_weight_repack(
                weights=W_UV,
                scales=W_UV_S,
                zeros=W_UV_Z,
                weight_format="awq",
                reformat="k16n32",
            )
            w4_group_size = kv_b_proj_module.quant_method.quant_config.group_size

            def _v_up_proj_w4a16(impl_self, x, out=None):
                return ixf_ops.marlin_w4a16(
                    x,
                    impl_self.W_UV,
                    impl_self.W_UV_S,
                    impl_self.W_UV_Z,
                    group_size=w4_group_size,
                    format="k16n32",
                    batch_first=False,
                )

            def _k_up_proj_w4a16(impl_self, x_nope):
                return ixf_ops.marlin_w4a16(
                    x_nope,
                    impl_self.W_UK,
                    impl_self.W_UK_S,
                    impl_self.W_UK_Z,
                    group_size=w4_group_size,
                    format="k16n32_grouped_n",
                    batch_first=False,
                )

            self._v_up_proj = functools.partial(_v_up_proj_w4a16, self)
            self._k_up_proj = functools.partial(_k_up_proj_w4a16, self)
            return True

        return False

    # ------------------------------------------------------------------
    # forward_mqa: decode kernel
    # ------------------------------------------------------------------
    def forward_mqa(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_c_normed: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        kv_c_and_k_pe_cache_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 ILU MLA not yet supported")

        decode_meta = attn_metadata.decode
        is_int8_mla = envs.VLLM_USE_INT8_MLA

        q_nope = self._k_up_proj(q_nope)
        q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)

        B = q_nope.shape[0]
        max_decode_seq_len = int(decode_meta.seq_lens.max().item())

        if self.dcp_world_size > 1:
            q = torch.cat([q_nope, q_pe], dim=-1)
            q = get_dcp_group().all_gather(q, dim=1)
            o = torch.empty(
                B,
                q.shape[1],
                self.kv_lora_rank,
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            if is_int8_mla:
                q_int8, q_scale = ops.quant_kv(q)
                attn_out, softmax_lse = ixf_ops.vllm_paged_attention_mla_int8(
                    o,
                    q_int8,
                    q_scale,
                    kv_c_and_k_pe_cache,
                    kv_c_and_k_pe_cache_scale,
                    self.scale,
                    decode_meta.block_table,
                    decode_meta.seq_lens,
                    max_decode_seq_len,
                    return_softmax_lse=True,
                )
            else:
                attn_out, softmax_lse = ixf_ops.vllm_paged_attention_mla(
                    output=o,
                    query=q,
                    kv_cache=kv_c_and_k_pe_cache,
                    scale=self.scale,
                    block_tables=decode_meta.block_table,
                    context_lens=decode_meta.seq_lens,
                    max_context_len=max_decode_seq_len,
                    return_softmax_lse=True,
                )
            return attn_out, softmax_lse

        o = torch.empty(
            B,
            self.num_heads,
            self.kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        if is_int8_mla:
            q = torch.cat([q_nope, q_pe], dim=-1)
            q_int8, q_scale = ops.quant_kv(q)
            ixf_ops.vllm_paged_attention_mla_int8(
                o,
                q_int8,
                q_scale,
                kv_c_and_k_pe_cache,
                kv_c_and_k_pe_cache_scale,
                self.scale,
                decode_meta.block_table,
                decode_meta.seq_lens,
                max_decode_seq_len,
                decode_meta.use_cuda_graph,
            )
        else:
            if k_c_normed is None:
                q = torch.cat([q_nope, q_pe.contiguous()], dim=-1)
                ixf_ops.vllm_paged_attention_mla(
                    output=o,
                    query=q,
                    kv_cache=kv_c_and_k_pe_cache,
                    scale=self.scale,
                    block_tables=decode_meta.block_table,
                    context_lens=decode_meta.seq_lens,
                    max_context_len=max_decode_seq_len,
                    use_cuda_graph=decode_meta.use_cuda_graph,
                )
            else:
                ixf_ops.vllm_paged_attention_mla_fused(
                    output=o,
                    q_nope=q_nope.contiguous(),
                    q_pe=q_pe.contiguous(),
                    kv_cache=kv_c_and_k_pe_cache,
                    scale=self.scale,
                    block_tables=decode_meta.block_table,
                    context_lens=decode_meta.seq_lens,
                    max_context_len=max_decode_seq_len,
                    k_c_normed=k_c_normed,
                    k_pe=k_pe,
                    use_cuda_graph=decode_meta.use_cuda_graph,
                )

        return self._v_up_proj(o), None
