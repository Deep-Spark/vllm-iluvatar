"""DeepSeek-V4.1 KV compressor."""
from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_iluvatar.models.deepseek_v41.paged_caches import (
    BLOCK_SIZE,
    V41AttentionMetadata,
    V41CompressorStateCache,
    gather_window_rows,
)


class IluDeepseekV41Compressor(nn.Module):

    def __init__(self, config, compress_ratio: int, prefix: str) -> None:
        super().__init__()
        head_dim = config.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, config.rms_norm_eps)
        # Checkpoint stores these bf16; declaring fp32 produced NaN on this platform.
        self.wkv = ReplicatedLinear(
            config.hidden_size, head_dim, bias=False, prefix=f"{prefix}.wkv"
        )
        if compress_ratio > 1:
            self.wgate = ReplicatedLinear(
                config.hidden_size, head_dim, bias=False, prefix=f"{prefix}.wgate"
            )
            self.state_cache = V41CompressorStateCache(
                state_dim=2 * head_dim,
                compress_ratio=compress_ratio,
                block_size=BLOCK_SIZE,
                prefix=f"{prefix}.state",
            )
        else:
            self.state_cache = None

    @staticmethod
    def groups_completed(start_pos: int, seq_len: int, ratio: int) -> tuple[int, int]:
        return max((start_pos + ratio) // ratio - 1, 0), seq_len // ratio

    def _group_rows(
        self,
        groups: range,
        md: V41AttentionMetadata,
        req: int,
        t0: int,
        start_pos: int,
        kv_all: torch.Tensor,
        score_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.head_dim
        ratio = self.compress_ratio
        dev = kv_all.device
        positions = torch.arange(groups.start * ratio, groups.stop * ratio, device=dev)
        local = positions - start_pos
        here = (local >= 0).unsqueeze(1)
        src = (local.clamp(min=0) + t0).long()
        kv_span = kv_all.index_select(0, src)
        score_span = score_all.index_select(0, src)

        rows = gather_window_rows(
            self.state_cache.kv_cache.view(-1, 2 * width),
            md,
            req,
            groups.start * ratio,
            positions.numel(),
        )

        kv = torch.where(here, kv_span, rows[:, :width].to(kv_span.dtype))
        score = torch.where(here, score_span, rows[:, width:].to(score_span.dtype))
        return kv.contiguous(), score.contiguous()

    def forward(
        self, x: torch.Tensor, spans: list[tuple[int, int, int, int]]
    ) -> torch.Tensor | None:
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x)[0])

        kv_all = self.wkv(x)[0]
        score_all = self.wgate(x)[0]

        md: V41AttentionMetadata = get_forward_context().attn_metadata[
            self.state_cache.prefix
        ]
        state = self.state_cache.kv_cache
        if not state.numel():
            return None

        ape = torch.zeros(ratio, self.head_dim, dtype=torch.float32, device=kv_all.device)
        ixfops.ds4_compressor_save_partial_states(
            kv_all.float().contiguous(),
            score_all.float().contiguous(),
            ape,
            md.positions,
            md.slot_mapping,
            state,
            md.block_size,
            ratio,
        )

        pooled_parts = []
        for req, t0, t1, seq_len in spans:
            start_pos = seq_len - (t1 - t0)
            first_g, last_g = self.groups_completed(start_pos, seq_len, ratio)
            if last_g <= first_g:
                continue
            kv, score = self._group_rows(
                range(first_g, last_g), md, req, t0, start_pos, kv_all, score_all
            )
            pooled_parts.append(ixfops.ds4_v41_compress_pool(kv, score, ratio))

        if not pooled_parts:
            return None
        return self.norm(torch.cat(pooled_parts, dim=0))
