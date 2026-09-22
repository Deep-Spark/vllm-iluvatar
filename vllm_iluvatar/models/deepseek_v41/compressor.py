"""DeepSeek-V4.1 KV compressor, the caches it publishes into, and the inserts.

Follows ``vllm/models/deepseek_v4/compressor.py``, which keeps the compressor's
state cache next to the module that writes it. The backend, the shared metadata
and the paged-cache base come from ``sparse_mla.py``; V4.1 has one metadata type
for all four caches, so they are not duplicated per cache group here.
"""
from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
    BLOCK_SIZE,
    V41AttentionMetadata,
    V41PagedCache,
)


def latent_rows_are_tokens(ratio: int) -> bool:
    """Whether the latent carries one row per token rather than per closed group.

    The pooling layers emit one row per token because ``ds4_v41_compress_prepare_out``
    does, which is what makes them capturable; a ``ratio == 1`` layer has no pooling
    to fuse and keeps the packed per-group insert.
    """
    return ratio > 1


_SCRATCH: dict[tuple[str, int, torch.dtype, torch.device], torch.Tensor] = {}


def token_scratch(
    name: str,
    rows: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    capacity: int,
) -> torch.Tensor:
    """A ``[max_num_batched_tokens, width]`` row buffer shared by every layer.

    Allocated once at the largest batch and sliced per call, so the address is
    fixed before capture and no forward allocates. Sharing is safe because every
    user writes the buffer, reads it within one layer's forward, and never looks
    at it again, and layers run one at a time -- while one copy per layer would
    cost 32 MiB each at head_dim=512, i.e. ~580 MiB out of the KV cache for the
    18 layers that pool.
    """
    key = (name, width, dtype, device)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(max(capacity, rows), width, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    if rows > buf.size(0):
        raise RuntimeError(
            f"compressor scratch {name!r} was sized for {buf.size(0)} tokens but "
            f"got {rows}; it follows max_num_batched_tokens ({capacity})"
        )
    return buf[:rows]


def compress_rope_qdq_insert_by_token(
    cache: torch.Tensor,
    x: torch.Tensor,
    md: "V41AttentionMetadata",
    ratio: int,
    qdq_block: int,
    qdq_fmt: str,
    cos_sin: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> None:
    """``compress_rope_qdq_insert`` for rows indexed by token, not by group.
    """
    n = x.size(0)
    ixfops.v41_rope_qdq_insert_by_token(
        cache,
        x,
        md.positions[:n],
        md.slot_mapping[:n],
        md.block_table,
        md.token_to_req_indices[:n],
        cos_sin,
        md.block_size,
        ratio,
        qdq_block,
        qdq_fmt,
        norm_weight,
        norm_eps,
    )


def compress_rope_qdq_insert(
    cache: torch.Tensor,
    x: torch.Tensor,
    md: "V41AttentionMetadata",
    ratio: int,
    qdq_block: int,
    qdq_fmt: str,
    cos_sin: torch.Tensor,
) -> None:
    """Publish this batch's compressed rows: RoPE, qdq, and the cache write.

    One kernel for what was four calls -- derive the slots, rotate, quantize,
    scatter -- of which the slot derivation alone cost a cumsum and two scatters
    over the whole batch. See ``ixformer``'s ``v41_compress_rope_qdq_insert``.

    The slots are a closed form of vLLM's own per-token ``slot_mapping``, which
    is why no block-table gather is needed. A compressed cache holds one row per
    ``ratio`` tokens, so it is addressed in the group domain, but for a token at
    ``pos`` closing group ``g = pos // ratio`` the two pick the *same* physical
    block -- ``g // (bs // ratio) == pos // bs``. So the block number is
    ``slot_mapping[t] // bs`` and the offset within it ``(slot_mapping[t] % bs)
    // ratio``, and a group is closed by its last token, ``(pos + 1) % ratio == 0``.

    ``x`` is left untouched, where the chain rotated and quantized it in place.
    No caller reads it afterwards.
    """
    n = md.num_actual_tokens
    ixfops.v41_compress_rope_qdq_insert(
        cache, x, md.slot_mapping[:n], md.positions[:n], cos_sin,
        md.block_size, ratio, qdq_block, qdq_fmt,
    )


class V41CompressKVCache(V41PagedCache):

    def __init__(self, *, head_size: int, dtype: torch.dtype, compress_ratio: int,
                 block_size: int, prefix: str) -> None:
        super().__init__(head_size=head_size, dtype=dtype, block_size=block_size, prefix=prefix)
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        if block_size % compress_ratio:
            raise ValueError(
                f"block_size {block_size} must be a multiple of compress_ratio "
                f"{compress_ratio}, or a page would hold a fractional row (layer {prefix})"
            )
        self.compress_ratio = compress_ratio

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )


class V41CompressorStateCache(V41PagedCache):

    def __init__(self, *, state_dim: int, compress_ratio: int, block_size: int,
                 prefix: str) -> None:
        super().__init__(head_size=state_dim, dtype=torch.float32, block_size=block_size,
                         prefix=prefix)
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        self.state_dim = state_dim
        self.compress_ratio = compress_ratio
        self.sliding_window = compress_ratio

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=512,
            compress_ratio=1,
        )


class IluDeepseekV41Compressor(nn.Module):

    def __init__(self, config, compress_ratio: int, prefix: str) -> None:
        super().__init__()
        head_dim = config.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rms_norm_eps = config.rms_norm_eps
        # Read here, not in the forward: the vLLM config is only in scope while
        # the model is being built.
        self.max_num_tokens = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
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

    def _forward_pooled(self, x: torch.Tensor) -> torch.Tensor | None:
        """One pre-RoPE latent row per token, in one launch.

        Pooling cannot be driven from the host: which requests close a ratio group
        in a step depends on ``(position + 1) % ratio``, so a per-request loop gives
        a row count that moves from step to step where a graph needs it fixed, and
        18 of the 40 layers pay a dozen launches per request for it.
        ``ds4_v41_compress_prepare_out`` does the state write, the pooling and the
        norm together, which is also why the rows come out per token rather than per
        closed group -- see ``latent_rows_are_tokens``.
        """
        state = self.state_cache.kv_cache
        if not state.numel():
            # Before vLLM allocated the paged caches; the callers skip the cache
            # writes when this is None.
            return None

        latent = token_scratch(
            "latent",
            x.size(0),
            self.head_dim,
            x.dtype,
            x.device,
            self.max_num_tokens,
        )
        md: V41AttentionMetadata = get_forward_context().attn_metadata[
            self.state_cache.prefix
        ]
        ixfops.ds4_v41_compress_prepare_out(
            self.wkv(x)[0].float(),
            self.wgate(x)[0].float(),
            state,
            md.token_to_req_indices,
            md.positions,
            md.slot_mapping,
            md.block_table,
            self.norm.weight,
            latent,
            self.compress_ratio,
            self.rms_norm_eps,
        )
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.compress_ratio == 1:
            # Each token is its own group, so pooling is the identity: no gate,
            # no state cache, and the rows are already one per token.
            return self.norm(self.wkv(x)[0])
        return self._forward_pooled(x)
