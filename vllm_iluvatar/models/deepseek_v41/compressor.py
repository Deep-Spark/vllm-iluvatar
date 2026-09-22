"""DeepSeek-V4.1 KV compressor, the caches it publishes into, and the inserts.

Follows ``vllm/models/deepseek_v4/compressor.py``, which keeps the compressor's
state cache next to the module that writes it. The backend, the shared metadata
and the paged-cache base come from ``sparse_mla.py``; V4.1 has one metadata type
for all four caches, so they are not duplicated per cache group here.
"""
from __future__ import annotations

import warnings

import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm_iluvatar.models.deepseek_v41.common.ops import fused_save_compress_norm
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


# The wkv/wgate projection is bf16 in, fp32 out, which cublasLt has no kernel
# for on this platform. ATen retries on cublas and gets the right answer -- and
# faster than any of our fp32-out ops -- but it re-reports the miss on every
# call, and the shapes in the text defeat Python's once-per-message dedup.
# Filtered at import so the hot path carries no catch_warnings block, which
# would split the compiled region in two.
warnings.filterwarnings(
    "ignore", message=".*bgemm_internal_cublaslt error.*", category=UserWarning
)


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

    def __init__(self, config, compress_ratio: int, prefix: str,
                 k_cache_prefix: str) -> None:
        super().__init__()
        head_dim = config.head_dim
        # A ratio-1 layer has no state ring of its own, so the token count for
        # its rows comes from the cache it publishes into.
        self.k_cache_prefix = k_cache_prefix
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rms_norm_eps = config.rms_norm_eps
        # Read here, not in the forward: the vLLM config is only in scope while
        # the model is being built.
        self.max_num_tokens = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
        self.norm = RMSNorm(head_dim, config.rms_norm_eps)
        # wkv and wgate read the same hidden state, so they are one GEMM. Its
        # output is fp32 because ``ds4_v41_compress_prepare_out`` takes fp32
        # states; going through bf16 first only threw away mantissa.
        # Checkpoint stores these bf16; declaring fp32 produced NaN on this platform.
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            config.hidden_size,
            [head_dim, head_dim] if compress_ratio > 1 else [head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        if compress_ratio > 1:
            self.state_cache = V41CompressorStateCache(
                state_dim=2 * head_dim,
                compress_ratio=compress_ratio,
                block_size=BLOCK_SIZE,
                prefix=f"{prefix}.state",
            )
        else:
            self.state_cache = None

    def _kv_score(self, x: torch.Tensor) -> torch.Tensor:
        """The merged wkv/wgate projection, fp32.

        cublasLt has no bf16 x bf16 -> fp32 kernel here, so ATen recovers on
        cublas; that path is still faster than any fp32-out op we have, and the
        warning it repeats is filtered where the layer is defined.
        """
        return torch.mm(
            x, self.fused_wkv_wgate.weight.T, out_dtype=torch.float32
        )

    def _forward_pooled(self, x: torch.Tensor) -> torch.Tensor | None:
        """One pre-RoPE latent row per token, in one launch.

        Pooling cannot be driven from the host: which requests close a ratio group
        in a step depends on ``(position + 1) % ratio``, so a per-request loop gives
        a row count that moves from step to step where a graph needs it fixed, and
        18 of the 40 layers pay a dozen launches per request for it.
        ``ds4_v41_compress_prepare_out`` does the state write, the pooling and the
        norm together, which is also why the rows come out per token rather than per
        closed group.
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
        fused_save_compress_norm(
            self._kv_score(x),
            md.positions,
            state,
            md.slot_mapping,
            md.query_start_loc,
            md.token_to_req_indices,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
            block_table=md.block_table,
        )
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.compress_ratio == 1:
            # Each token is its own group, so pooling is the identity: no gate
            # and no state ring, and the rows are already one per token.
            md: V41AttentionMetadata = get_forward_context().attn_metadata[
                self.k_cache_prefix
            ]
            latent = token_scratch(
                "latent", x.size(0), self.head_dim, x.dtype, x.device,
                self.max_num_tokens,
            )
            fused_save_compress_norm(
                self._kv_score(x),
                md.positions,
                None,
                md.slot_mapping,
                md.query_start_loc,
                None,
                self.norm.weight,
                self.rms_norm_eps,
                self.compress_ratio,
                latent,
            )
            return latent
        return self._forward_pooled(x)
