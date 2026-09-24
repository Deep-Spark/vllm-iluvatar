"""DeepSeek-V4.1-Flash attention: projections, caches, sparse MLA."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, cast

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.models.deepseek_v41.common.ops import (
    indexer_k_norm_rope_store,
    move_rows_to_paged_cache,
)
from vllm_iluvatar.models.deepseek_v41.compressor import (
    IluDeepseekCompressor,
)
from vllm_iluvatar.models.deepseek_v41.rope import apply_rope
from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
    IluDeepseekSparseSWAFlashMLABackend,
    IluDeepseekV4FlashMLABackend,
    IluDeepseekV4FlashMLAMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV41IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import IluDeepseekV4SWACache

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


warnings.filterwarnings(
    "ignore", message=".*bgemm_internal_cublaslt error.*", category=UserWarning
)

_INDEX_TILE = 256
_PACKED_MLA_HEADS = (1, 8, 16, 32, 64, 128)


def _tiled_index_rows(idx: torch.Tensor) -> torch.Tensor:
    """Index rows whose stride(0) is a positive multiple of 256.

    ``ds4_sparse_mla_fp8_decode`` reads index tiles of that width. A window of
    128 is narrower, so the row is copied into a wider allocation and the
    returned view keeps the padded stride.
    """
    if idx.dim() == 3:
        idx = idx.reshape(idx.shape[0], -1)
    width = idx.shape[-1]
    if (
        width % _INDEX_TILE == 0
        and idx.stride(1) == 1
        and idx.stride(0) > 0
        and idx.stride(0) % _INDEX_TILE == 0
    ):
        return idx
    if width % _INDEX_TILE == 0:
        return idx.contiguous()
    padded = width + (-width % _INDEX_TILE)
    wide = idx.new_empty((idx.shape[0], padded))
    wide[:, :width].copy_(idx)
    return wide[:, :width]


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
    packed_kv_cache_dtype: str = "fp8_ds_mla",
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    packed formats are ``uint8``-backed: ``fp8_ds_mla`` is UE8M0 block-scaled
    fp8 throughout, ``nvfp4_ds_mla`` keeps that sliding-window record and
    stores the compressed cache as NVFP4. An unspecific ``--kv-cache-dtype``
    (``auto`` / ``fp8``) resolves to ``packed_kv_cache_dtype``, the record this
    layer's kernel prefers, and the canonical string is written back onto
    ``cache_config`` so the page-size specs pick the right per-token slot.
    Plain-row backends store each token's KV row in its element dtype: bf16 or
    per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        if kv_cache_dtype in ("auto", "fp8"):
            kv_cache_dtype = packed_kv_cache_dtype
        elif not kv_cache_dtype.endswith("_ds_mla"):
            raise ValueError(
                "DeepseekV4 packed KV layouts only support fp8 kv-cache, got "
                f"{kv_cache_dtype}. Please set `--kv-cache-dtype fp8` or "
                "select a backend that supports bfloat16 KV cache."
            )
        if cache_config is not None and cache_config.cache_dtype != kv_cache_dtype:
            cache_config.cache_dtype = kv_cache_dtype
        logger.info_once("Using DeepSeek's %s KV cache format.", kv_cache_dtype)
        return kv_cache_dtype, torch.uint8
    if kv_cache_dtype.endswith("_ds_mla"):
        raise ValueError(
            f"{kv_cache_dtype} is a packed FlashMLA DeepSeek V4.1 KV cache "
            "format; the selected backend stores plain KV rows."
        )

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


def _replace_layer_index(prefix: str, layer_id: int) -> str:
    """Swap the layer index inside a ``...layers.<idx>...`` prefix."""
    new_prefix, n = re.subn(r"\.layers\.\d+\.", f".layers.{layer_id}.", prefix)
    assert n == 1, f"Cannot locate layer index in prefix {prefix}"
    return new_prefix


# Record the indexer K cache holds, and with it the query's.
#
# Deliberately a constant rather than ``--kv-cache-dtype``: the two are
# independent choices, and reading the cache config here used to be a hazard.
# ``_resolve_dsv4_kv_cache_dtype`` *rewrites* ``cache_config.cache_dtype`` as a
# side effect, and it runs later in ``__init__`` than the indexer cache is
# built, so whichever layer a rank happened to construct first decided the
# indexer layout for that rank. On a cut where a rank starts on a kv source
# (pp=4 rank 2 starts at layer 20, pp=5 rank 1 at layer 8) that rank ended up
# with a bf16 cache while its peers had fp8, and the pipeline replica then
# failed on a dtype mismatch.
#
# Moving to "fp8" is not this constant alone: the scoring call in
# ``sparse_attn_indexer.py`` has to become
# ``dsa_indexer_mqa_logits_with_blocks_fp8`` (ivcore30-only, and it wants a
# page whose row count is a multiple of 64) and the query goes through
# ``v41_indexer_q_rope_fp8``, whose folded weights come back fp32. The kernels
# for that path exist and are tested; only the wiring is missing.
_INDEXER_KV_DTYPE = "bf16"


def _indexer_k_cache_row(index_head_dim: int, kv_dtype: str) -> tuple[int, torch.dtype]:
    """Width and element type of one paged indexer K cache row.

    FP8 is the V3.2 layout the paged scoring kernel reads: a page of
    ``index_head_dim`` E4M3 bytes per row followed by a page of one FP32 scale
    per row, so the row is ``index_head_dim + 4`` bytes. BF16 rows are the
    values alone.
    """
    if kv_dtype == "fp8":
        return index_head_dim + 4, torch.uint8
    if kv_dtype == "bf16":
        return index_head_dim, torch.bfloat16
    raise ValueError(
        f"unsupported indexer cache dtype {kv_dtype!r}; expected 'bf16' or 'fp8'"
    )


class IluDeepseekV41IndexerCache(torch.nn.Module, AttentionLayerBase):
    """The paged index-key cache an ``owns_k`` layer publishes into."""

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        self.block_size = cache_config.block_size
        if self.block_size % compress_ratio:
            raise ValueError(
                f"block_size {self.block_size} must be a multiple of compress_ratio "
                f"{compress_ratio}, or a page would hold a fractional row "
                f"(layer {prefix})"
            )
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            # The indexer page shares a physical block with the main KV cache,
            # so it carries the same padding.
            alignment=512,
            compress_ratio=self.compress_ratio,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV41IndexerBackend


class IluDeepseekV41Indexer(nn.Module):
    """Index-query projections, the K cache for key owners, and the selection.

    Key ownership is ``layer_id in kv_source_layer_ids``: index keys come from the
    compressor's pre-RoPE latent, so only those layers own ``wk`` / ``k_norm`` and a
    cache to write them to. Everyone else scores against the cache a source
    published.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int,
        prefix: str,
        *,
        owns_k: bool,
        k_cache: IluDeepseekV41IndexerCache,
        main_head_dim: int,
        candidate_block_buffer: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.k_row_buffer: torch.Tensor | None = None
        self.store_fp8_kv = _INDEXER_KV_DTYPE == "fp8"
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "FP8" if self.store_fp8_kv else "BF16",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        if owns_k:
            # wk maps the main compressor's (pre-RoPE) latent to index keys.
            # The checkpoint stores it in bf16 with no quantization scales.
            self.wk = ReplicatedLinear(
                main_head_dim,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_cache = k_cache

        # Scoring and selection live behind the platform's subclass of official's
        # SparseAttnIndexer, same as v3.2 and glm_moe_dsa.
        self.indexer_op = IluSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=False,
            compress_ratio=compress_ratio,
            candidate_blocks=candidate_block_buffer,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )

    def _produce_k(
        self, latent: torch.Tensor | None, positions: torch.Tensor, rotary_emb
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or latent is None:
            # Profile run: the indexer K cache is not bound yet and the
            # compressor skipped its latent output.
            return
        assert self.owns_k
        indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        k_cache = self.k_cache.kv_cache
        # The slot mapping is compressed-domain, so the cache's second
        # dimension has to be the compressed row count the builder sized those
        # slots against, or the kernel's slot split lands on the wrong row.
        assert k_cache.shape[1] * self.compress_ratio == self.k_cache.block_size, (
            f"indexer cache holds {k_cache.shape[1]} rows per page, but the "
            f"slots were built for {self.k_cache.block_size} tokens at "
            f"compress_ratio {self.compress_ratio}"
        )
        # ReplicatedLinear returns (output, bias); bias is None. Rows at
        # non-boundary tokens hold garbage latent and are skipped by the
        # store kernel.
        k_pre, _ = self.wk(latent)
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            rotary_emb.cos_sin_cache,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            k_cache,
            indexer_metadata.slot_mapping,
            self.compress_ratio,
            self.store_fp8_kv,
            row_out=self.k_row_buffer,
        )

    def forward(
        self,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
        qr_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Publish this layer's index keys, then prepare the index query.

        Returns the prepared query, its scale (none in this layout) and the
        per-head weights, or three Nones when the short-context bypass filled
        the top-k outright and there is nothing to score.
        """
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if (
                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
                and not torch.cuda.is_current_stream_capturing()
            ):
                if self.owns_k:
                    self._produce_k(latent, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                topk_lens = getattr(self, "topk_lens_buffer", None)
                assert topk_lens is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    ixfops.v41_fill_short_context_topk(
                        self.topk_indices_buffer[:num_tokens],
                        topk_lens[:num_tokens],
                        positions,
                        self.compress_ratio,
                    )
                return None, None, None

        if self.owns_k:
            # K write must land before indexer_op reads the cache
            # (skip_k_cache_insert=True).
            self._produce_k(latent, positions, rotary_emb)

        # ReplicatedLinear returns (output, bias).
        q = self.wq_b(qr)[0].view(-1, self.n_head, self.head_dim)
        if self.store_fp8_kv:
            # The per-(token, head) q_scale comes back folded into the weights,
            # which the FP8 scoring kernel wants as fp32.
            q_fp8, folded_weights = torch.ops._C.v41_indexer_q_rope_fp8(
                q,
                positions,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale * self.n_head**-0.5,
            )
            return q_fp8, None, folded_weights
        # In-place: q is rotated, weights are rescaled.
        torch.ops._C.v41_indexer_q_rope(
            q,
            positions,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            indexer_weights,
            self.softmax_scale * self.n_head**-0.5,
        )
        return q, None, indexer_weights


class IluDeepseekV41Attention(nn.Module, AttentionLayerBase):
    """Q/KV/O projections, vLLM-owned caches, and the sparse MLA forward.

    ``compress_ratio > 0`` does not mean this layer compresses its own KV: only
    ``kv_source_layer_ids`` do; the rest read that cache.
    """

    backend_cls: ClassVar[type[AttentionBackend]] = IluDeepseekV4FlashMLABackend
    swa_backend_cls: ClassVar[type[AttentionBackend]] = (
        IluDeepseekSparseSWAFlashMLABackend
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        is_kv_forwarder: bool = False,
        is_kv_sender: bool = False,
        kv_source_override: int | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)
        self.layer_id = layer_id

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # ---- v4.1 sparse-attention topology ----
        # compress_ratios has one entry per layer (MTP layers included):
        # 0 = pure sliding window, 1 = full-length compressed cache,
        # 2 = ratio-2 compressed. Compressors and compressed-KV caches live
        # only on ``kv_source_layer_ids``; indexers only on
        # ``index_source_layer_ids``. Consumers reuse the most recently
        # published source below them.
        compress_ratios = getattr(config, "compress_ratios", None)
        if compress_ratios is not None and layer_id < len(compress_ratios):
            self.compress_ratio = int(compress_ratios[layer_id])
        else:
            # MTP layers past the configured list are pure sliding-window.
            self.compress_ratio = 0
        if self.compress_ratio not in (0, 1, 2):
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_id} has compress_ratio="
                f"{self.compress_ratio}; only 0 (sliding window), 1 and 2 are "
                "supported."
            )
        self.kv_source_layers = tuple(
            getattr(config, "kv_source_layer_ids", None) or ()
        )
        self.index_source_layers = tuple(
            getattr(config, "index_source_layer_ids", None) or ()
        )
        self.candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        self.candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        self.candidate_block_size = getattr(config, "candidate_block_size", 0)

        is_backbone = layer_id < config.num_hidden_layers
        self.is_kv_source = is_backbone and layer_id in self.kv_source_layers
        self.is_index_source = is_backbone and layer_id in self.index_source_layers
        # Pipeline roles, assigned by the model from the rank's cut plan. A
        # sender's group reaches consumers on a later rank, so it mirrors its
        # records into dense buffers the model ships across. A forwarder is the
        # first stranded layer on that later rank: it owns a replica cache and
        # replays what arrives into it. Neither role builds a compressor or an
        # indexer it would not otherwise have.
        self.is_kv_forwarder = is_kv_forwarder
        self.is_kv_sender = is_kv_sender
        # Bound by the model once the layer list exists.
        self.main_kv_rows: torch.Tensor | None = None
        if self.compress_ratio > 0:
            if not self.kv_source_layers or not self.index_source_layers:
                raise ValueError(
                    "DeepSeek V4.1 requires kv_source_layer_ids / "
                    "index_source_layer_ids in the config for compressed "
                    f"layers (layer {layer_id} has "
                    f"compress_ratio={self.compress_ratio})."
                )
            # ``kv_source_override`` points at this rank's forwarder when the
            # real source sits on an earlier one; everything downstream
            # addresses the cache owner by this id alone.
            self.kv_source_layer_id = (
                kv_source_override
                if kv_source_override is not None
                else max(s for s in self.kv_source_layers if s <= layer_id)
            )
            self.index_source_layer_id = max(
                s for s in self.index_source_layers if s <= layer_id
            )
        else:
            self.kv_source_layer_id = None
            self.index_source_layer_id = None
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self._o_proj_block_size = (
            32 if getattr(self.wo_a, "weight_block_size", None) == [1, 32] else 128
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer

        # Register with compilation context for metadata lookup. Done before
        # indexer/compressor creation so consumers can resolve their source
        # layers through it.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])
        self._static_forward_context = compilation_config.static_forward_context

        self.indexer = None
        if self.is_index_source:
            index_k_cache: IluDeepseekV41IndexerCache | None
            # Index K cache: owned only by kv-source layers (their indexer has
            # wk/k_norm); non-owning index sources share the K cache of the
            # latest kv source below them (its indexer produces the keys).
            if self.is_kv_source:
                row_bytes, row_dtype = _indexer_k_cache_row(
                    config.index_head_dim, _INDEXER_KV_DTYPE
                )
                index_k_cache = IluDeepseekV41IndexerCache(
                    head_dim=row_bytes,
                    dtype=row_dtype,
                    prefix=f"{prefix}.indexer.k_cache",
                    cache_config=cache_config,
                    compress_ratio=self.compress_ratio,
                )
            else:
                assert self.kv_source_layer_id is not None
                k_cache_prefix = (
                    f"{_replace_layer_index(prefix, self.kv_source_layer_id)}"
                    ".indexer.k_cache"
                )
                index_k_cache = self._static_forward_context.get(k_cache_prefix)
                if index_k_cache is None:
                    raise RuntimeError(
                        f"Indexer K cache source {k_cache_prefix} not found on "
                        f"this rank for layer {layer_id}. A pipeline cut inside "
                        "a kv-sharing group is meant to be covered by a "
                        "forwarder replica; the cut plan and the layer's "
                        "kv_source_override disagree."
                    )
            is_candidate_source = layer_id == self.candidate_source_layer
            uses_candidates = 0 <= self.candidate_source_layer < layer_id
            self.indexer = IluDeepseekV41Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=index_k_cache,
                main_head_dim=self.head_dim,
                candidate_block_buffer=(
                    candidate_block_buffer
                    if (is_candidate_source or uses_candidates)
                    else None
                ),
                candidate_block_size=self.candidate_block_size,
                candidate_write=is_candidate_source,
            )

        # A forwarder replicates a group that started on an earlier rank. It
        # runs no compressor and no indexer, but it owns both caches the group
        # would have published locally: the compressed KV one through
        # ``get_kv_cache_spec`` below, and this index-key one, which the
        # consumers above it resolve through ``kv_source_layer_id``.
        self.forwarder_indexer_cache: IluDeepseekV41IndexerCache | None = None
        if self.is_kv_forwarder:
            assert cache_config is not None
            assert not self.is_kv_source, (
                f"layer {layer_id} is its own kv source and cannot also forward one"
            )
            row_bytes, row_dtype = _indexer_k_cache_row(
                config.index_head_dim, _INDEXER_KV_DTYPE
            )
            self.forwarder_indexer_cache = IluDeepseekV41IndexerCache(
                head_dim=row_bytes,
                dtype=row_dtype,
                prefix=f"{prefix}.indexer.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below. ``auto`` means the
        # packed record here: it is what the backend advertises and what the
        # decode kernel reads, so it must not fall through to the plain-row
        # branch the way a plain ``startswith("fp8")`` test would let it.
        packed_layout = cache_config.cache_dtype in (
            "auto",
            "fp8",
        ) or cache_config.cache_dtype.endswith("_ds_mla")
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            packed_layout,
            cache_config.cache_dtype,
            cache_config,
            "fp8_ds_mla",
        )
        if self.kv_cache_torch_dtype is not torch.uint8:
            # The plain-row store kernel exists and is tested, but nothing
            # downstream reads such a cache: the sliding-window write goes
            # through ``v41_qkv_rope_fp8_insert`` and the attention through
            # ``ds4_sparse_mla_fp8_decode``, both of which take the packed
            # 584-byte record only. Refusing here beats producing garbage.
            raise NotImplementedError(
                f"DeepSeek V4.1 on this platform stores the main KV as the "
                f"packed fp8_ds_mla record; --kv-cache-dtype "
                f"{cache_config.cache_dtype} would need a bf16 sliding-window "
                "insert and a bf16 sparse-MLA decode, neither of which exists "
                "yet. Use --kv-cache-dtype fp8 (or auto)."
            )
        self.swa_bytes_per_token = 584
        # Width and dtype of one record on the wire, for the dense buffers the
        # model allocates when a pipeline cut strands this layer's consumers.
        # These are the record's own shape, not the cache's page geometry.
        self.main_transfer_width = self.swa_bytes_per_token
        self.main_transfer_dtype = self.kv_cache_torch_dtype
        (
            self.indexer_transfer_width,
            self.indexer_transfer_dtype,
        ) = _indexer_k_cache_row(config.index_head_dim, _INDEXER_KV_DTYPE)
        self.kv_page_alignment = 576

        swa_bounded_replay = getattr(cache_config, "swa_bounded_replay", False)
        if swa_bounded_replay and not vllm_config.use_v2_model_runner:
            logger.warning_once(
                "SWA bounded replay needs model runner V2 (only it skips the "
                "paged-KV writes of replayed tokens); the sliding-window cache "
                "takes part in prefix caching instead."
            )
            swa_bounded_replay = False
        if (
            swa_bounded_replay
            and vllm_config.parallel_config.prefill_context_parallel_size > 1
        ):
            logger.warning_once(
                "SWA bounded replay is off under prefill context parallelism "
                "(the replayed tokens' slot padding knows the rank-local batch "
                "only); the sliding-window cache takes part in prefix caching "
                "instead."
            )
            swa_bounded_replay = False
        self.swa_cache_layer = IluDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            backend_cls=self.swa_backend_cls,
            block_size=cache_config.block_size,
            packed_bytes_per_token=self.swa_bytes_per_token,
            packed_page_alignment=self.kv_page_alignment,
            bounded_replay=swa_bounded_replay,
        )

        # The attention layer itself was already registered with the
        # compilation context above (before indexer/compressor creation).

        # Compressors live only on kv-source layers; consumers read the
        # source's compressed cache through the forward context.
        self.compressor = None
        if self.is_kv_source:
            self.compressor = IluDeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )
        # Prefix of the attention layer owning this layer's compressed KV
        # cache (self for kv sources).
        if self.compress_ratio > 0:
            assert self.kv_source_layer_id is not None
            self.compressed_cache_prefix: str | None = _replace_layer_index(
                prefix, self.kv_source_layer_id
            )
            if (
                not self.is_kv_source
                and self.compressed_cache_prefix not in self._static_forward_context
            ):
                raise NotImplementedError(
                    f"Compressed-KV source {self.compressed_cache_prefix} not "
                    "found on this rank; PP splits inside a v4.1 kv-sharing "
                    "group are not supported."
                )
        else:
            self.compressed_cache_prefix = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Only kv-source layers own a compressed-KV cache: consumers read the
        # source's through the forward context and cr==0 layers are pure SWA.
        # The window cache is registered separately as IluDeepseekV4SWACache.
        # A forwarder owns one too -- it is the local replica of a source that
        # a pipeline cut left on an earlier rank.
        if not self.is_kv_source and not self.is_kv_forwarder:
            return None
        # Tied to the resolved record rather than restated, so the spec and the
        # tensor the store kernel writes cannot drift apart.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            model_version="deepseek_v4",
            alignment=64,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def _compressed_kv_cache(self) -> torch.Tensor:
        """The compressed-KV cache this layer reads.

        Its own for a kv source, and equally for a forwarder: the forwarder's
        cache *is* the local copy of the source it stands in for.
        """
        if self.is_kv_source or self.is_kv_forwarder:
            return self.kv_cache
        assert self.compressed_cache_prefix is not None
        source = self._static_forward_context[self.compressed_cache_prefix]
        return source.kv_cache

    def accept_remote_kv(
        self,
        main_rows: torch.Tensor,
        indexer_rows: torch.Tensor,
        topk_rows: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        """Replay a previous rank's records into this forwarder's caches.

        The producing rank already wrote these records into its own paged
        caches and kept a dense copy in token order. Here they go into the
        local pages under this rank's slot mapping, so the replica grows into
        a full copy of the source's cache one step at a time.

        Returns without writing during a profile run, when no metadata is
        bound and the caches are still empty.
        """
        assert self.is_kv_forwarder
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or not self.kv_cache.numel():
            return

        main_metadata = cast(Any, attn_metadata[self.prefix])
        move_rows_to_paged_cache(
            main_rows,
            self.kv_cache,
            main_metadata.slot_mapping,
            positions,
            self.compress_ratio,
        )

        indexer_cache = self.forwarder_indexer_cache
        assert indexer_cache is not None
        if indexer_cache.kv_cache.numel():
            indexer_metadata = cast(Any, attn_metadata[indexer_cache.prefix])
            move_rows_to_paged_cache(
                indexer_rows,
                indexer_cache.kv_cache,
                indexer_metadata.slot_mapping,
                positions,
                self.compress_ratio,
            )

        # The selection is a plain per-rank buffer, so taking it over is a
        # copy rather than a paged write.
        if topk_rows is not None and self.topk_indices_buffer is not None:
            num_tokens = topk_rows.shape[0]
            self.topk_indices_buffer[:num_tokens].copy_(topk_rows)

    @property
    def cos_sin(self) -> torch.Tensor:
        return self.rotary_emb.cos_sin_cache

    def _load_attn_sink(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rank = get_tensor_model_parallel_rank()
        shard = loaded.narrow(0, rank * self.n_local_heads, self.n_local_heads)
        param.data.copy_(shard.to(param.dtype))

    def bind_index_buffers(
        self,
        topk_lens: torch.Tensor,
        candidate_idxs: torch.Tensor | None,
        candidate_lens: torch.Tensor | None,
    ) -> None:
        """Take the selection buffers allocated after the layer list exists."""
        self.topk_lens_buffer = topk_lens
        self.candidate_idxs_buffer = candidate_idxs
        self.candidate_lens_buffer = candidate_lens
        indexer = self.indexer
        uses_candidates = self.layer_id == self.candidate_source_layer or (
            0 <= self.candidate_source_layer < self.layer_id
        )
        if indexer is not None:
            indexer.topk_lens_buffer = topk_lens
            if uses_candidates and candidate_idxs is not None:
                indexer.indexer_op.candidate_blocks = candidate_idxs

    def _o_proj(self, o: torch.Tensor) -> torch.Tensor:
        """Grouped output projection. One group per rank uses the fp8 linear."""
        if self.n_local_groups != 1:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            z = torch.einsum("bsgd,grd->bsgr", o.to(wo_a.dtype), wo_a)
            return self.wo_b(z.flatten(2))
        return self.wo_b(self.wo_a(o.flatten(0, -2)))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(
            hidden_states
        )
        qr, kv = self._split_qkv_and_norm(qr_kv)
        attn_out = torch.empty(
            (hidden_states.shape[0], self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._prepare_and_attn(
            hidden_states, qr, kv, kv_score, indexer_weights, positions, attn_out
        )
        num_tokens = hidden_states.shape[0]
        apply_rope(attn_out.unsqueeze(0), positions, self.cos_sin, inverse=True)
        return self._o_proj(attn_out.view(1, num_tokens, self.n_local_groups, -1)).view(
            num_tokens, -1
        )

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The layer's input GEMMs over the same hidden state.

        The v4.1 indexer derives K from the kv-source compressor's latent, so
        unlike v4.0 there is no indexer K GEMM over hidden_states here.

        Official fans these out over aux streams, putting the heaviest
        (``fused_wqa_wkv``) on the default one. They run in sequence here: this
        platform has no aux-stream path wired, and the prefill trace already
        sits at 88-91% GPU occupancy, so there is little to overlap into.
        """
        kv_score = None
        indexer_weights = None

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type.
            compressor = self.compressor
            kv_score = torch.mm(
                hidden_states,
                compressor.fused_wkv_wgate.weight.T,
                out_dtype=torch.float32,
            )

        if self.indexer is not None:
            # ReplicatedLinear returns (output, bias); bias is None.
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)

        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        return qr_kv, kv_score, indexer_weights

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the fused q-lora / kv projection and RMSNorm both halves."""
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return ixfops.rms_norm_qk(
            qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
        )

    def _wq_b_proj(self, qr: torch.Tensor) -> torch.Tensor:
        return self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        """RoPE q in place and publish kv as a packed fp8 window record."""
        if not isinstance(attn_metadata, dict):
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        ixfops.v41_qkv_rope_fp8_insert(
            q,
            self.swa_cache_layer.kv_cache,
            kv,
            swa_metadata.slot_mapping,
            positions,
            self.cos_sin,
        )
        return q

    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Attention input preparation followed by the sparse indexer and MLA.

        Q preparation and the KV insert precede compression; once the latent is
        ready the main-cache insert and the indexer K insert both run off it,
        and both finish before sparse attention reads either cache.
        """
        attn_metadata = get_forward_context().attn_metadata
        indexer = self.indexer
        compressor = self.compressor

        index_q: torch.Tensor | None = None
        index_weights_out: torch.Tensor | None = None
        latent: torch.Tensor | None = None

        if compressor is not None:
            latent = compressor(kv_score, positions)
            compressor.insert_cache(
                latent,
                positions,
                self.rotary_emb,
                row_out=self.main_kv_rows if self.is_kv_sender else None,
            )

        q = self._wq_b_proj(qr)
        self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        index_q, index_weights_out = None, None

        if indexer is not None:
            index_q, _, index_weights_out = indexer(
                qr,
                latent,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
            )

        self._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_weights_out,
            q,
            kv,
            positions,
            out,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            IluDeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            prefill_q = q[num_decode_tokens:]
            prefill_out = output[num_decode_tokens:]

            self._forward_prefill(
                q=prefill_q,
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=prefill_out,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            decode_out = output[:num_decode_tokens]
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                positions=positions[:num_decode_tokens],
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_only=swa_only,
                output=decode_out,
            )

    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            self.indexer.indexer_op(
                hidden_states,
                index_q,
                None,
                index_weights,
            )

        self.forward_mqa(q, kv, positions, out)

    def _window_indices(
        self,
        swa_metadata: DeepseekSparseSWAMetadata,
        begin: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SWA slots for batch tokens ``[begin, end)``.

        The builder only fills decode rows. Prefill tokens sit later in the
        same batch, so this recomputes ``[0, end)`` and returns the slice.
        """
        width = self.window_size
        buf = getattr(self, "_prefill_swa_idx", None)
        lens = getattr(self, "_prefill_swa_lens", None)
        if buf is None or buf.shape[0] < end:
            n = max(end, self.max_num_batched_tokens)
            device = swa_metadata.block_table.device
            buf = torch.empty(n, 1, width, dtype=torch.int32, device=device)
            lens = torch.empty(n, dtype=torch.int32, device=device)
            self._prefill_swa_idx = buf
            self._prefill_swa_lens = lens
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.is_valid_token is not None
        ixfops.ds4_compute_swa_decode_indices_and_lens(
            swa_indices=buf,
            swa_lens=lens,
            window_size=width,
            query_start_loc=swa_metadata.query_start_loc,
            seq_lens=swa_metadata.seq_lens,
            token_to_req_indices=swa_metadata.token_to_req_indices,
            is_valid_token=swa_metadata.is_valid_token,
            block_table=swa_metadata.block_table,
            block_size=swa_metadata.block_size,
            num_decode_tokens=end,
        )
        return buf[begin:end], lens[begin:end]

    def _sparse_fp8_mla(
        self,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        heads = q.shape[1]
        if heads not in _PACKED_MLA_HEADS:
            raise ValueError(
                f"ds4_sparse_mla_fp8_decode needs h_q in {_PACKED_MLA_HEADS}, got {heads}"
            )
        q_packed, q_scales = ixfops.ds4_dynamic_quant_mla_q_packed_fp8(q)
        ixfops.ds4_sparse_mla_fp8_decode(
            q_packed,
            q_scales,
            self.swa_cache_layer.kv_cache,
            _tiled_index_rows(swa_indices),
            self.scale,
            swa_lens,
            self.attn_sink,
            extra_cache,
            None if extra_indices is None else _tiled_index_rows(extra_indices),
            extra_lens,
            output,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        positions: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        extra_cache = None
        extra_indices = None
        extra_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            token_to_req = swa_metadata.token_to_req_indices
            assert token_to_req is not None
            extra_indices, extra_lens = ixfops.ds4_compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                token_to_req[:num_decode_tokens],
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
                positions,
                self.compress_ratio,
            )
            extra_cache = kv_cache
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        self._sparse_fp8_mla(
            q,
            swa_metadata.decode_swa_indices,
            swa_metadata.decode_swa_lens,
            extra_cache,
            extra_indices,
            extra_lens,
            output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        del swa_k_cache
        swa_only = self.compress_ratio == 0
        if not swa_only and self.compress_ratio not in (1, 2):
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; expected 0, 1, or 2."
            )

        n0 = swa_metadata.num_decode_tokens
        n1 = n0 + swa_metadata.num_prefill_tokens
        swa_indices, swa_lens = self._window_indices(swa_metadata, n0, n1)
        extra_cache = None
        extra_indices = None
        extra_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            assert swa_metadata.is_valid_token is not None
            assert swa_metadata.token_to_req_indices is not None
            assert self.topk_indices_buffer is not None
            extra_indices, extra_lens = ixfops.ds4_compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[n0:n1],
                swa_metadata.token_to_req_indices[n0:n1],
                attn_metadata.block_table,
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[n0:n1],
                positions,
                self.compress_ratio,
            )
            extra_cache = compressed_k_cache
        self._sparse_fp8_mla(
            q,
            swa_indices,
            swa_lens,
            extra_cache,
            extra_indices,
            extra_lens,
            output,
        )
