# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 mxfp8 attention layers (dense + block-sparse).

Indexer / sparse backends live alongside this file. W4A8 token-major
adapters are in ``minimax_m3.compressed_tensors``.
"""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.models.minimax_m3.mxfp8.indexer import (
    SUPPORTED_INDEXER_KV_DTYPES,
    MiniMaxM3IndexerIxformerImpl,
)
from vllm_iluvatar.models.minimax_m3.mxfp8.sparse_attention import (
    SUPPORTED_KV_CACHE_DTYPES,
    SUPPORTED_KV_CACHE_TORCH_DTYPES,
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseIxformerImpl,
)


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    """Per-head Gemma QK-norm + partial RoPE, optional K/V-split NHD insert.

    Community's same-named op writes a packed KV layout; ixformer here assumes
    this plugin's split NHD cache. Not registered on ``vllm._custom_ops``.
    """
    if kv_cache_dtype not in ("auto", "bfloat16", "fp16", "float16"):
        raise NotImplementedError(
            f"ixformer M3 fused insert does not support {kv_cache_dtype} KV cache"
        )
    ixf_ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


class MiniMAXGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm implemented by ixformer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        result = ixf_ops.gemma_rms_norm(
            input=x,
            weight=self.weight,
            eps=self.variance_epsilon,
            residual=residual,
        )
        if residual is None:
            return result

        output, residual_output = result
        x.copy_(output)
        residual.copy_(residual_output)
        return x, residual


def _require_unquantized_m3_caches(
    kv_cache_dtype: str,
    kv_cache_torch_dtype: torch.dtype,
    indexer_kv_dtype: str,
) -> None:
    """ixformer M3 sparse + indexer kernels are BF16-only.

    Checks the *resolved* torch dtype rather than only the ``--kv-cache-dtype``
    string: ``"auto"`` follows ``--dtype``, so a float16 model would otherwise
    pass a string-only guard and fail much later inside the kernel. The
    accepted sets come from the backend / indexer modules so this guard cannot
    disagree with what those kernels actually run.
    """
    if kv_cache_dtype not in ("auto", *SUPPORTED_KV_CACHE_DTYPES):
        raise NotImplementedError(
            "Iluvatar MiniMax-M3 sparse attention requires kv_cache_dtype in "
            f"{('auto', *SUPPORTED_KV_CACHE_DTYPES)}, got {kv_cache_dtype!r}."
        )
    if kv_cache_torch_dtype not in SUPPORTED_KV_CACHE_TORCH_DTYPES:
        raise NotImplementedError(
            "Iluvatar MiniMax-M3 sparse attention requires a KV cache in "
            f"{sorted(str(d) for d in SUPPORTED_KV_CACHE_TORCH_DTYPES)}; "
            f"kv_cache_dtype={kv_cache_dtype!r} resolved to "
            f"{kv_cache_torch_dtype}."
        )
    if indexer_kv_dtype not in SUPPORTED_INDEXER_KV_DTYPES:
        raise NotImplementedError(
            "Iluvatar MiniMax-M3 indexer requires indexer_kv_dtype in "
            f"{SUPPORTED_INDEXER_KV_DTYPES}, got {indexer_kv_dtype!r}."
        )


class MiniMaxM3Attention(nn.Module):
    """Dense attention with per-head QK norm and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # Fused per-head Gemma QK-norm + partial NeoX RoPE on q/k, in place.

        fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer branch.

    This is a merged attention layer: it owns the projections (qkv + index
    q/k), per-head QK norms and RoPE, *and* the attention-backend wiring that a
    generic ``Attention`` layer would normally provide — it binds the
    ``MiniMaxM3SparseBackend`` + main impl, registers the main paged K/V cache,
    and owns the lightning indexer (``MiniMaxM3IndexerIxformerImpl``), which
    holds the index-key side cache.

    The index branch (index_{q,k}_proj + index_{q,k}_norm) feeds the sparse
    top-k block selection. M3 always disables the index value/output
    projections (``sparse_disable_index_value`` set for every sparse layer), so
    ``index_{v,o}_proj`` are never created.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Sparse "index" branch dims. index_q has the same head count as the KV
        # heads (sparse_num_index_heads == num_key_value_heads), so it shards
        # identically -- including replication when tp_size > num_key_value_heads.
        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim

        # Single fused projection: q, k, v, index_q, index_k in one GEMM.
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )

        # Attention-backend wiring.
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # MiniMax-M3 sparse attention owns its KV-cache insert/read path instead
        # of wrapping the generic Attention module, so these are set here rather
        # than by Attention.__init__. ixformer only reads BF16 KV and never
        # applies a descale, but the ``AttentionLayer`` protocol that
        # ``MiniMaxM3SparseImpl.forward`` is typed against declares
        # ``_q_scale_float`` / ``_k_scale_float``, and registering them as
        # buffers keeps them in the state dict and moving with the module, the
        # same reasoning community gives in ``layers/attention/attention.py``.
        # Four scalars per layer. Note this does not make an FP8-KV checkpoint
        # loadable: ``load_weights`` matches against ``named_parameters()``, so
        # a ``*.k_scale`` tensor still finds no home and is skipped.
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)
        # Indexer side-cache dtype, mirroring --kv-cache-dtype for the main
        # cache (--attention-config '{"indexer_kv_dtype": ...}').
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype
        _require_unquantized_m3_caches(
            self.kv_cache_dtype,
            self.kv_cache_torch_dtype,
            self.indexer_kv_dtype,
        )

        # Shared top-k buffer: the indexer writes the selected blocks into it and
        # the attend impl reads them back (so nothing crosses the eager break as a
        # Python value, which would freeze at capture).
        self.topk_indices_buffer = topk_indices_buffer
        # Indexer (top-k selection) and main attention are separate impls; both
        # are ixformer, so there is nothing to select. impl is AttentionImplBase
        # (broader than the AttentionImpl that AttentionLayerBase annotates).
        self.attn_backend = MiniMaxM3SparseBackend
        self.impl = MiniMaxM3SparseIxformerImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        # Self-contained nn.Module: owns its side cache. Unlike community, there
        # is nothing to select here -- ixformer is the only indexer kernel.
        self.indexer = MiniMaxM3IndexerIxformerImpl(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            cache_config=cache_config,
            indexer_kv_dtype=self.indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

    def get_attn_backend(self) -> type[MiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Main GQA K/V cache. Block size may change after load, refresh it.
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Single fused projection emitting [q | k | v | index_q | index_k].
        qkv, _ = self.qkv_proj(hidden_states)

        # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on the
        # main (q/k) and index (index_q/index_k) branches, all read straight out
        # of the single fused ``qkv`` tensor (the "5 results").  Once the paged
        # caches are bound the kernel also inserts k/v and the index key into
        # them; the initial memory-profiling run (caches unbound, no slot_mapping)
        # short-circuits to zeros below. k/v and index_k are rewritten in place
        # inside qkv (and scatter-inserted into the caches); q and index_q are
        # de-interleaved
        # straight into the dedicated contiguous ``q``/``index_q`` buffers below.

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon
        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            # Reserve the maximum TCU densify scratch before vLLM locks its
            # workspace manager; all M3 layers reuse this simultaneous slot.
            self.impl.reserve_prefill_workspace(num_tokens)
            return qkv.new_zeros((num_tokens, self.hidden_size))

        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]
        q = qkv.new_empty((num_tokens, self.q_size))
        # index_q matches the index-K cache dtype; ixformer's indexer is BF16-only.
        index_q = qkv.new_empty(
            (num_tokens, self.index_q_size),
            dtype=self.indexer.index_cache.dtype,
        )
        fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            rotary_dim,
            eps,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            main_slot_mapping,
            index_slot_mapping,
            self.kv_cache,
            self.indexer.index_cache.kv_cache,
            self.kv_cache.size(2),  # paged-cache block size
            q,
            index_q,
            self.kv_cache_dtype,
        )

        output = torch.empty_like(q)
        attn_output = self._run_attention(q, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

    def _run_attention(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Captured with the rest of the decode forward under FULL_DECODE_ONLY.
        # Prefill stays eager (runtime mode NONE). The indexer writes top-k into
        # the shared ``topk_indices_buffer``; the attend reads it back. Split-K
        # scratch is the C++ grow-only buffer so capture does not allocate and
        # free graph-pool memory inside this call.
        self.indexer(index_query)
        return self.impl.forward(self, query, self.kv_cache, output)


