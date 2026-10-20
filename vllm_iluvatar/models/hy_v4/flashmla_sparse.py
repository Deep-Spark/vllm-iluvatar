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
Scope    : HY V4 sparse MLA with ``learnable_sink`` and BF16 or GLM-layout FP8
           KV cache. ``fp8_ds_mla`` is not used: HY V4 packs 512 e4m3 content
           bytes plus a 64-wide BF16 RoPE tail (640 bytes/token).
Remove   : when the base Iluvatar backend forwards sinks for every KV dtype.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
    IluFlashMLASparseImpl,
    IluFlashMLASparseMetadata,
)

from .topk_index_cache import HYV4TopkIndexCache

logger = init_logger(__name__)

_HYV4_DSA_FP8_ROW_BYTES = 640
_HYV4_DSA_FP8_E4M3_MAX = 448.0


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
        self.use_fp8_attn = bool(kwargs.pop("hyv4_use_fp8_attn", False))
        self._fp8_kv_scale_frozen = bool(kwargs.pop("hyv4_fp8_kv_scale_frozen", False))
        self.prefix = str(kwargs.pop("hyv4_fp8_prefix", ""))
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
                "HY V4 learnable sink requires the BF16 or FP8 KV sparse MLA path; "
                "sparse_flash_attn_int8 has no attn_sink argument. Unset "
                "VLLM_ILU_MLA_ATTN_QUANT_TYPE=INT8 to run HY V4."
            )
        self._fp8_q_heads = self.num_heads
        if self.use_fp8_attn:
            if self.num_heads > 64:
                raise ValueError(
                    f"HY V4 FP8 sparse MLA requires <=64 local heads, got "
                    f"{self.num_heads}"
                )
            self._fp8_q_heads = (
                64 if self.num_heads > 32 else max(16, (self.num_heads + 15) // 16 * 16)
            )
        if sinks is not None:
            if sinks.dtype is not torch.float32:
                raise ValueError(
                    "HY V4 attn_sink must be fp32 (ixformer requirement), got "
                    f"{sinks.dtype}."
                )
            expected_sink_heads = (
                self._fp8_q_heads if self.use_fp8_attn else self.num_heads
            )
            if sinks.numel() != expected_sink_heads:
                raise ValueError(
                    f"HY V4 attn_sink must hold one value per local head "
                    f"after padding ({expected_sink_heads}), got {sinks.numel()}."
                )
        self.attn_sink = sinks

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
        global_indices, topk_length = self._global_topk(
            num_tokens, topk_indices, attn_metadata
        )

        return self._bf16_flash_mla_kernel(
            q, kv_c_and_k_pe_cache, global_indices, topk_length
        )

    def _global_topk(
        self,
        num_tokens: int,
        topk_indices: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return global_indices, topk_length

    def _maybe_init_fp8_kv_scale(
        self, kv_c: torch.Tensor, k_scale: torch.Tensor
    ) -> None:
        if self._fp8_kv_scale_frozen or kv_c.numel() == 0:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        amax = float(kv_c.detach().float().abs().amax())
        if not math.isfinite(amax) or amax <= 0.0:
            return
        scale = amax / _HYV4_DSA_FP8_E4M3_MAX
        k_scale.fill_(scale)
        self._fp8_kv_scale_frozen = True
        logger.info(
            "HYV4 DSA fp8 KV scale for %s frozen online = %.8g (amax/448)",
            self.prefix,
            scale,
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if not self.use_fp8_attn:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        if kv_cache.numel() == 0:
            return
        if getattr(self, "_skip_fp8_kv_insert", False):
            return
        import ixformer.inference.functions as ixf_ops

        self._maybe_init_fp8_kv_scale(kv_c_normed, k_scale)
        ixf_ops.glm_dsa_fp8_kv_insert(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping.flatten(),
            k_scale,
            kv_cache.shape[1],
        )

    def _fp8_flash_mla_kernel(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        import ixformer.inference.functions as ixf_ops

        if q_nope.ndim != 3 or q_pe.ndim != 3:
            raise ValueError(
                "HYV4 FP8 sparse MLA requires flat token layout "
                f"[tokens, heads, dim], got {q_nope.shape=} and {q_pe.shape=}"
            )
        num_tokens = q_nope.shape[0]
        if topk_length.shape != (num_tokens,):
            raise ValueError(
                "HYV4 FP8 sparse MLA requires one topk_length per flat token "
                f"(s_q=1); got {tuple(topk_length.shape)} for {num_tokens} tokens"
            )

        q_packed = q_nope.new_empty(
            (num_tokens, self._fp8_q_heads, _HYV4_DSA_FP8_ROW_BYTES),
            dtype=torch.uint8,
        )
        q_scale = torch.empty(
            (num_tokens, self._fp8_q_heads),
            dtype=torch.float32,
            device=q_nope.device,
        )
        ixf_ops.glm_dsa_fp8_q_pack(q_nope, q_pe, q_packed, q_scale)
        out = ixf_ops.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
            q_packed.view(num_tokens, 1, self._fp8_q_heads, _HYV4_DSA_FP8_ROW_BYTES),
            kv_cache.unsqueeze(2),
            topk_indices.view(num_tokens, 1, -1),
            topk_length,
            q_scale,
            k_scale,
            self.softmax_scale,
            attn_sink=self.attn_sink,
        ).squeeze(1)
        return out[:, : self.num_heads, :].contiguous()

    # [ILU override] IluFlashMLASparseImpl._bf16_flash_mla_kernel — use the
    # paged DSV4 facade, which pads HY4 TP8's H=8 to H=16 and forwards sinks.
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
        output = ixf_ops.dsa_flash_mla_sparse_fwd_diff_kv(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            d_v=512,
            topk_length=topk_length,
            attn_sink=self.attn_sink,
        )
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
        layer,
        kv_cache_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_fp8_attn:
            return super().forward_mqa(
                q, kv_c_and_k_pe_cache, attn_metadata, layer, kv_cache_scale
            ), None
        if not isinstance(q, tuple):
            raise ValueError("HYV4 FP8 sparse MLA requires separate NoPE and RoPE Q")
        q_nope, q_pe = q
        num_tokens = q_nope.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        global_indices, topk_length = self._global_topk(
            num_tokens, topk_indices, attn_metadata
        )
        output = self._fp8_flash_mla_kernel(
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            global_indices,
            topk_length,
            layer._k_scale,
        )
        return output, None


class HYV4FlashMLASparseBackend(IluFlashMLASparseBackend):
    """Sparse MLA backend advertising sink support for HY V4."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

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
