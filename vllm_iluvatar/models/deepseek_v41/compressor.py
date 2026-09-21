"""DeepSeek-V4.1 KV compressor."""
from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_iluvatar.models.deepseek_v41.paged_caches import (
    BLOCK_SIZE,
    V41AttentionMetadata,
    V41CompressorStateCache,
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
