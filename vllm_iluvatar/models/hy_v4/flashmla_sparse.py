# SPDX-License-Identifier: Apache-2.0
"""Sink-capable Iluvatar sparse-MLA backend for HY V4.

Corresponds to: vllm/models/hy_v4/nvidia/flashmla_sparse.py (upstream #54160)

Target   : ``vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse``
           (``IluFlashMLASparseBackend`` / ``IluFlashMLASparseImpl``).
Baseline : the Iluvatar sparse-MLA backend does not declare ``supports_sink``
           and never forwards ``attn_sink``, so the community selector rejects
           it for a layer that needs a sink.
Why      : HY V4's per-head learnable attention sink is part of the trained
           architecture; a backend that silently drops it changes the model's
           output. ixformer's ``sparse_flash_attn`` already accepts a
           keyword-only fp32 ``attn_sink`` of shape ``[h_q]``, so only the
           wiring is missing.
How      : subclass rather than patch, exactly as
           ``IluDeepseekV4FlashMLASparseBackend`` does in the same file: HY V4
           binds this backend explicitly through ``MLAAttention(attn_backend=)``
           and every other model keeps the unmodified base classes.
Scope    : HY V4 sparse MLA with ``learnable_sink``, BF16 KV cache only.
           ``fp8_ds_mla`` and the INT8 KV path are rejected: neither
           ``dsa_flash_mla_with_fp8_kvcache`` nor ``sparse_flash_attn_int8``
           takes ``attn_sink``, and a sink applied on only some steps is worse
           than none.
Remove   : when the base Iluvatar backend forwards sinks for every KV dtype.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
    IluFlashMLASparseImpl,
    IluFlashMLASparseMetadata,
)

from .topk_index_cache import HYV4TopkIndexCache


class HYV4FlashMLASparseImpl(IluFlashMLASparseImpl):
    """Iluvatar sparse MLA that forwards HY V4's per-head learnable sink."""

    # ``IluFlashMLASparseImpl.__init__`` does not chain to
    # ``SparseMLACommonImpl.__init__``, so none of the attributes the community
    # ``MLAAttention`` layer reads off ``self.impl`` exist. Every other model on
    # this backend bypasses that layer with its own attention module; HY V4 is
    # the first to go through it, so declare them here with the values that are
    # true on Iluvatar:
    #   - masked-MHA prefill is FA4 on NV SM100 only;
    #   - the ixformer sparse path is all-MQA (``forward_mqa`` serves prefill
    #     and decode alike), so there is no dense MHA prefill to offer.
    masked_mha_available: ClassVar[bool] = False
    supports_dense_mha_prefill: ClassVar[bool] = False

    def __init__(self, *args, **kwargs) -> None:
        # ``MLAAttention`` passes the sink through the impl kwargs as ``sinks``;
        # pop it so the base ``**mla_args`` stays exactly what it expects. The
        # top-k cache and the "this layer shares another layer's indices" flag
        # ride the same channel.
        sinks = kwargs.pop("sinks", None)
        self._topk_cache: HYV4TopkIndexCache | None = kwargs.pop(
            "topk_index_cache", None
        )
        self._reuses_topk_indices: bool = bool(kwargs.pop("reuses_topk_indices", False))
        super().__init__(*args, **kwargs)
        # Resolved eagerly (the base leaves ``dcp_world_size`` on a -1 sentinel
        # it only fills in during forward) because the community layer reads both
        # before the first forward.
        self.dcp_world_size = get_dcp_group().world_size
        self.pcp_world_size = get_pcp_group().world_size
        if self.use_int8_mla:
            raise NotImplementedError(
                "HY V4 learnable sink requires the BF16 KV sparse MLA path; "
                "sparse_flash_attn_int8 has no attn_sink argument. Unset "
                "VLLM_ILU_MLA_ATTN_QUANT_TYPE=INT8 to run HY V4."
            )
        if sinks is not None:
            if sinks.dtype is not torch.float32:
                raise ValueError(
                    "HY V4 attn_sink must be fp32 (ixformer requirement), got "
                    f"{sinks.dtype}."
                )
            if sinks.numel() != self.num_heads:
                raise ValueError(
                    f"HY V4 attn_sink must hold one value per local head "
                    f"({self.num_heads}), got {sinks.numel()}."
                )
        self.attn_sink = sinks

    def forward_mqa(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The base returns a bare tensor despite annotating a 2-tuple; its only
        # in-plugin caller unpacks it as a tensor. Community ``MLAAttention``
        # does ``attn_out, lse = impl.forward_mqa(...)``, so supply the missing
        # ``lse``. None is correct here: the sparse kernels do not return one and
        # it is only consumed under DCP/PCP, which this path does not support.
        return super().forward_mqa(*args, **kwargs), None

    # [ILU override] IluFlashMLASparseImpl._forward_bf16_kv — delta: a "shared"
    # indexer layer attends with the indices the preceding full layer wrote, so
    # the global-slot conversion and the ``topk_length`` reduction are the same
    # tensors for all 57 of them. Compute them once per step and reuse; see
    # ``topk_index_cache``.
    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        cached = (
            self._topk_cache.load(num_tokens)
            if self._topk_cache is not None and self._reuses_topk_indices
            else None
        )
        if cached is None:
            global_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )
            topk_length = (
                (global_indices.view(num_tokens, -1) >= 0).sum(dim=-1).to(torch.int32)
            )
            if self._topk_cache is not None and not self._reuses_topk_indices:
                self._topk_cache.store(num_tokens, global_indices, topk_length)
        else:
            global_indices, topk_length = cached

        return self._bf16_flash_mla_kernel(
            q, kv_c_and_k_pe_cache, global_indices, topk_length
        )

    # [ILU override] IluFlashMLASparseImpl._bf16_flash_mla_kernel — delta: pass
    # ``attn_sink`` to ixformer and keep HY4's real local-head count. The
    # ixinfer facade supports H=8/16/32/64; accuracy for TP1/2/4/8 is covered
    # by the standalone HY4 operator matrix.
    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attn_sink is None:
            return super()._bf16_flash_mla_kernel(q, kv_c_and_k_pe_cache, topk_indices)

        import ixformer.inference.functions as ixf_ops

        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        if topk_length is None:
            idx = topk_indices.reshape(num_tokens, -1)
            topk_length = (idx >= 0).sum(dim=-1).to(torch.int32)
        output = ixf_ops.sparse_flash_attn(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            512,
            topk_length=topk_length,
            attn_sink=self.attn_sink,
        )
        return output


class HYV4FlashMLASparseBackend(IluFlashMLASparseBackend):
    """Sparse MLA backend advertising sink support for HY V4."""

    # Sinks are only wired for the BF16 KV path, so drop the fp8 aliases the
    # base accepts: selecting them would load the sink and then ignore it.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    # ``get_name`` is deliberately inherited ("FLASHMLA_SPARSE"). Community keys
    # several behaviour tables off the backend name -- the sparse-MLA Triton
    # warmup allow-list and ``_DSV32_MASKED_MHA_THRESHOLDS`` -- and a fresh name
    # would silently opt out of both. The masked-MHA entry is harmless here
    # because ``_force_sparse_mqa`` pins ``sparse_mla_force_mqa``, which vetoes
    # every MHA path.

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
