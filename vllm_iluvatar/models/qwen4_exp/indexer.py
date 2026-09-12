# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp weight-free QSA indexer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from ixformer.inference.functions.qwen38 import (
    qwen_qsa_indexer_mqa_logits,
    qwen_qsa_indexer_q_norm_rope,
    qwen_qsa_indexer_topk_expand,
    qwen_qsa_kpool_compress_store,
    qwen_qsa_kpool_tail_update,
)
from ixformer.inference.functions.qwen38.qsa_common import align_logits_cols
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from .config import Qwen4ExpTextConfig
from .qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
)


@dataclass
class QSAPoolCarry:
    """Per-request partial sums of the QSA compression group being filled.

    A compression group spans ``compress_ratio`` consecutive positions, so it
    routinely straddles a chunked-prefill boundary and always straddles decode
    steps. Keeping the running fp32 sum here -- one row per (QSA layer,
    request-state slot) -- is what lets the indexer drop its paged ring of raw
    index keys. ``Qwen4ExpModelState`` allocates this and clears a row when a
    request takes the slot; ``Qwen4ExpModel.forward`` refreshes
    ``state_indices`` from its own arguments so it can never go stale.
    """

    acc: torch.Tensor
    """``[num_qsa_layers, max_num_reqs, indexer_head_dim]`` float32."""

    state_indices: torch.Tensor | None = None
    """Batch row -> state slot. ``None`` on dummy runs, which pool from a zero
    carry and skip the write-back so live rows stay untouched."""


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.prefix = prefix
        # MTP step 0 selects the target-aligned rows; later steps reuse them
        # while continuing to update the QSA side cache.
        self.skip_topk = False

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        if vllm_config.model_config.uses_mrope:
            # A group's compressed key is RoPE'd at the group's first position.
            # Deriving that arithmetically only works for 1D positions; exact
            # three-axis positions would have to be carried per request too.
            raise NotImplementedError(
                "Qwen4Exp QSA requires 1D positions; MRoPE checkpoints must run "
                "with language_model_only so mrope_section is stripped"
            )
        # Assigned by Qwen4ExpModelState once the model is built: the shared
        # carry and this layer's row block within it.
        self.pool_carry: QSAPoolCarry | None = None
        self.pool_row = -1
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )
        self._ws_max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self._ws_max_cols = align_logits_cols(
            max(1, int(vllm_config.model_config.max_model_len) // self.compress_ratio)
        )
        self._logits_ws: torch.Tensor | None = None
        self._req_ids_ws: torch.Tensor | None = None
        self._vis_cap_ws: torch.Tensor | None = None

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def _rope_table(self, like: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
        """The main attention's rotary table, matched to ``like``'s dtype.

        The indexer rotates the same 64 channels as the main heads and shares
        its table, so this reuses ``rotary_emb``'s own dtype cache instead of
        casting per call.
        """

        cache = self.rotary_emb._match_cos_sin_cache_dtype(like)  # noqa: SLF001
        return (
            cache,
            self.rotary_emb.rotary_dim,
            bool(getattr(self.rotary_emb, "is_neox_style", True)),
        )

    def _metadata(self) -> QSAForwardMetadata | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        return cast(QSAForwardMetadata, metadata[self.compressed_key_cache.prefix])

    def _ensure_workspace(
        self, device: torch.device, rows: int, cols: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stride = align_logits_cols(max(cols, 1))
        need_rows = max(rows, self._ws_max_tokens)
        need_cols = max(stride, self._ws_max_cols)
        if (
            self._logits_ws is None
            or self._logits_ws.device != device
            or self._logits_ws.shape[0] < need_rows
            or self._logits_ws.shape[1] < need_cols
        ):
            self._logits_ws = torch.empty(
                (need_rows, need_cols), dtype=torch.bfloat16, device=device
            )
            self._req_ids_ws = torch.empty(need_rows, dtype=torch.int32, device=device)
            self._vis_cap_ws = torch.empty(need_rows, dtype=torch.int32, device=device)
        assert self._logits_ws is not None
        assert self._req_ids_ws is not None
        assert self._vis_cap_ws is not None
        return (
            self._logits_ws[:rows, :need_cols],
            self._req_ids_ws[:rows],
            self._vis_cap_ws[:rows],
        )

    def _run_indexer(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        md: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        rows = q.shape[0]
        width = self.output_width
        if out is None:
            out = torch.empty((rows, width), dtype=torch.int32, device=q.device)
        seq_lens = md.seq_lens
        pos = md.logical_positions
        t2r = md.token_to_req

        if md.prefill is not None:
            for chunk in md.prefill.chunks:
                sl = slice(chunk.token_start, chunk.token_end)
                t = chunk.token_end - chunk.token_start
                logits_ws, req_ws, vis_ws = self._ensure_workspace(
                    q.device, t, chunk.max_context_len
                )
                if chunk.max_context_len > 0:
                    logits = qwen_qsa_indexer_mqa_logits(
                        q[sl],
                        chunk.cu_seqlens_q,
                        chunk.cu_seq_lens,
                        cache,
                        chunk.block_table,
                        chunk.valid_lens,
                        logits=logits_ws[:t],
                        max_q_len=chunk.max_q_len,
                        max_kv_len=chunk.max_kv_len,
                        max_context_len=chunk.max_context_len,
                        req_ids=req_ws[:t],
                        vis_cap=vis_ws[:t],
                        backend="ixformer_native",
                    )
                else:
                    logits = logits_ws[:t]
                qwen_qsa_indexer_topk_expand(
                    logits,
                    chunk.valid_lens,
                    pos[sl],
                    seq_lens,
                    t2r[sl],
                    self.token_topk,
                    self.compress_ratio,
                    out[sl],
                    backend="ixformer_native",
                )

        if md.num_decodes > 0 and md.decode is not None:
            d = md.decode
            t = md.num_decode_tokens
            logits_ws, req_ws, vis_ws = self._ensure_workspace(
                q.device, t, d.max_context_len
            )
            logits = qwen_qsa_indexer_mqa_logits(
                q[:t],
                d.cu_seqlens_q,
                d.cu_seqlens_kv,
                cache,
                d.block_table,
                d.valid_lens,
                logits=logits_ws[:t],
                max_q_len=d.max_q_len,
                max_kv_len=d.max_kv_len,
                max_context_len=d.max_context_len,
                req_ids=req_ws[:t],
                vis_cap=vis_ws[:t],
                backend="ixformer_native",
            )
            qwen_qsa_indexer_topk_expand(
                logits,
                d.valid_lens,
                pos[:t],
                seq_lens,
                t2r[:t],
                self.token_topk,
                self.compress_ratio,
                out[:t],
                backend="ixformer_native",
            )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        compressed_metadata = metadata
        num_tokens = compressed_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]

        # Q/K projection
        projected_qk, _ = self.index_qk_proj(hidden_states)
        projected_q, raw_keys = projected_qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        compressed_key_cache = self.compressed_key_cache.kv_cache
        cos_sin_cache, rotary_dim, is_neox_style = self._rope_table(projected_qk)
        # One token per request means decode-shaped work: scoring degenerates
        # to a GEMV and pooling only has to slide the carry.
        is_decode = num_tokens == compressed_metadata.query_start_loc.shape[0] - 1

        q = qwen_qsa_indexer_q_norm_rope(
            projected_q.reshape(-1, self.index_n_heads, self.index_head_dim),
            self.q_layernorm.weight,
            cos_sin_cache,
            positions,
            self.q_layernorm.variance_epsilon,
            rotary_dim,
            is_neox_style,
            backend="ixformer_native",
        )

        carry = self.pool_carry
        pool = (
            qwen_qsa_kpool_tail_update if is_decode else qwen_qsa_kpool_compress_store
        )
        pool(
            raw_keys.reshape(-1, 1, self.index_head_dim),
            compressed_key_cache,
            self.k_layernorm.weight,
            cos_sin_cache,
            compressed_metadata.token_to_req,
            compressed_metadata.query_start_loc,
            compressed_metadata.seq_lens,
            compressed_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.k_layernorm.variance_epsilon,
            rotary_dim,
            self.compress_ratio,
            None if carry is None else carry.acc[self.pool_row],
            None if carry is None else carry.state_indices,
            is_neox_style,
            backend="ixformer_native",
        )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out

        return self._run_indexer(q, compressed_key_cache, compressed_metadata, out)


__all__ = ["QSAIndexer", "QSAPoolCarry"]
