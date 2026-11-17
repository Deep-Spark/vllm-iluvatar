"""DeepSeek-V4.1-Flash attention: projections, caches, sparse MLA."""

from __future__ import annotations

import os
import re
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import ixformer.inference.functions as ixfshared
import torch
from ixformer.inference.functions import (
    flash_mla_sparse_fwd,
    flash_mla_sparse_with_kvcache,
)
from ixformer.inference.functions.models import deepseek_v41 as ixfops
from torch import nn
from vllm import envs
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
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
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
    add_skip_layers,
)
from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.models.deepseek_v41.common.ops import (
    indexer_k_norm_rope_store,
    move_rows_to_paged_cache,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    main_kv_format,
    move_quantized_records,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
    CompressedFamilySelection,
    TensorLayoutKey,
    TensorViewKey,
    compact_owner_capacity,
    consumer_selection,
    family_key,
    prefill_validity_family,
    union_validity,
)
from vllm_iluvatar.models.deepseek_v41.compressor import (
    IluDeepseekCompressor,
)
from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
    SPLIT_INDEXER_FORMAT,
    indexer_kv_layout,
)
from vllm_iluvatar.models.deepseek_v41.query_quant import (
    QueryActivation,
    can_fuse_query_quant,
    fused_query_norm,
    query_inputs_are_compatible,
)
from vllm_iluvatar.models.deepseek_v41.rope import apply_rope
from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
    IluDeepseekSparseSWAFlashMLABackend,
    IluDeepseekV4FlashMLABackend,
    IluDeepseekV4FlashMLAMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    get_max_prefill_buffer_size,
    indexer_backend_cls,
    indexer_kv_dtype,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import IluDeepseekV4SWACache

if TYPE_CHECKING:
    from ixformer.inference.functions.models.deepseek_v41.mla_sparse.float.quantized_sparse_mla import (
        GroupedInt8AttentionOutput,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(f"vllm.{__name__}")


def _resolve_swa_bounded_replay(
    cache_config: CacheConfig, vllm_config: VllmConfig
) -> bool:
    """Match the community runner and context-parallel replay constraints."""
    if not cache_config.swa_bounded_replay:
        return False
    if not vllm_config.use_v2_model_runner:
        logger.warning_once(
            "SWA bounded replay needs model runner V2; the sliding-window "
            "cache takes part in prefix caching instead."
        )
        return False
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        logger.warning_once(
            "SWA bounded replay is off under prefill context parallelism; "
            "the sliding-window cache takes part in prefix caching instead."
        )
        return False
    return True


warnings.filterwarnings(
    "ignore", message=".*bgemm_internal_cublaslt error.*", category=UserWarning
)

_INDEX_TILE = 256
_PACKED_MLA_HEADS = (1, 8, 16, 32, 64, 128)
_PT_MLA_HEADS = (1, 4, 8, 16, 32, 64, 128)
_V41_PREFILL_FMHADSA_ENV = os.environ.get("IXFORMER_V41_FMHADSA", "1") == "1"


def _v41_prefill_fmhadsa() -> bool:
    """Whether prefill should call ixformer ``fmhadsa``.

    Native FMHADSA is an ixmma kernel registered for ivcore30 only. Forcing
    ``backend=ixformer_native`` on ivcore11 would either error or fall through
    to the torch reference; take ``sparse_mla_fp8_pt`` instead.
    """
    if not _V41_PREFILL_FMHADSA_ENV:
        return False
    if not is_ivcore30():
        logger.warning_once(
            "V4.1 prefill FMHADSA is off: the ixmma kernel is ivcore30-only; "
            "falling back to sparse_mla_fp8_pt."
        )
        return False
    if not hasattr(ixfops, "fmhadsa"):
        logger.warning_once(
            "V4.1 prefill FMHADSA is off: this ixformer has no fmhadsa; "
            "falling back to sparse_mla_fp8_pt."
        )
        return False
    logger.info_once(
        "V4.1 prefill FMHADSA=on (decode stays on ixformer sparse_mla_fp8_pt)"
    )
    return True


# Requests per workspace-gathered prefill chunk. Each one costs a row of
# ``M * head_dim`` bf16, so this bounds the workspace rather than the batch.
PREFILL_CHUNK_SIZE = 4


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
    wide = idx.new_full((idx.shape[0], padded), -1)
    wide[:, :width].copy_(idx)
    return wide[:, :width]


def _batched_index_rows(idx: torch.Tensor) -> torch.Tensor:
    """Index rows as ``[tokens, s_q, width]``.

    ``ds4_flash_mla_with_kvcache`` batches over tokens, so it wants the query
    axis spelled out. The SWA builder already emits that shape; the global
    top-k op returns one row per token.
    """
    return idx if idx.dim() == 3 else idx.unsqueeze(1)


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


def _resolve_kv_cache_dtype(cache_config: CacheConfig) -> tuple[str, torch.dtype]:
    """Resolve V4.1 defaults while retaining all three formal KV layouts."""
    if cache_config.cache_dtype in ("auto", "fp8"):
        cache_config.cache_dtype = "fp8_e4m3"
    return _resolve_dsv4_kv_cache_dtype(
        cache_config.cache_dtype.endswith("_ds_mla"),
        cache_config.cache_dtype,
        cache_config,
        "fp8_ds_mla",
    )


def _replace_layer_index(prefix: str, layer_id: int) -> str:
    """Swap the layer index inside a ``...layers.<idx>...`` prefix."""
    new_prefix, n = re.subn(r"\.layers\.\d+\.", f".layers.{layer_id}.", prefix)
    assert n == 1, f"Cannot locate layer index in prefix {prefix}"
    return new_prefix


# Record the indexer K cache holds, and with it the query's.
#
# Deliberately a startup setting separate from ``--kv-cache-dtype``: the two are
# independent choices, and reading the cache config here used to be a hazard.
# ``_resolve_dsv4_kv_cache_dtype`` *rewrites* ``cache_config.cache_dtype`` as a
# side effect, and it runs later in ``__init__`` than the indexer cache is
# built, so whichever layer a rank happened to construct first decided the
# indexer layout for that rank. On a cut where a rank starts on a kv source
# (pp=4 rank 2 starts at layer 20, pp=5 rank 1 at layer 8) that rank ended up
# with a bf16 cache while its peers had fp8, and the pipeline replica then
# failed on a dtype mismatch.
#
# Resolve before constructing any indexer cache, independently of main MLA.
# ``VLLM_ILU_V41_INDEXER_KV_DTYPE``; unset stays fp8.
_INDEXER_KV_DTYPE = indexer_kv_dtype()

# Legacy packed pages reserve a power-of-two row width to satisfy common-page
# unification. Split INT8 pages instead allocate 128-byte values here and budget
# FP32 scales through indexer_kv, preserving the 132-byte PP record on the wire.
# ``bind_kv_cache`` still narrows packed views; FP8 stays on the 132-byte spec.
_INDEX_PAGE_ROW_BYTES = 256

_INDEXER_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_INDEXER_LOGITS_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_ATTN_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}

_DBG_TOPK = os.environ.get("VLLM_V41_DEBUG_TOPK", "0") == "1"

_dbg_topk_seen: dict[str, int] = {}


def _dbg_topk(
    prefix: str,
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    topk_buffer: torch.Tensor,
) -> None:
    """Signature of one index source's inputs and its selection.

    Comparable across TP / PP layouts: keyed by layer prefix, and the last row
    is the token whose answer we look at.
    """
    seen = _dbg_topk_seen.get(prefix, 0)
    if seen >= 2:
        return
    _dbg_topk_seen[prefix] = seen + 1
    n = index_q.shape[0]
    sel = topk_buffer[:n].to(torch.int64)
    valid = sel >= 0
    last = sel[-1][valid[-1]]
    logger.info(
        "[DBG-TOPK] %s #%s n=%s q=%.5f w=%.5f sel_sum=%s valid=%s last_n=%s last16=%s",
        prefix,
        seen,
        n,
        float(index_q.float().abs().sum()),
        float(index_weights.float().abs().sum()),
        int(sel[valid].sum()),
        int(valid.sum()),
        int(last.numel()),
        sorted(last[:16].tolist()),
    )


def _indexer_k_cache_row(index_head_dim: int, kv_dtype: str) -> tuple[int, torch.dtype]:
    """Width and element type of one paged indexer K cache row.

    FP8 is the V3.2 layout the paged scoring kernel reads: a page of
    ``index_head_dim`` E4M3 bytes per row followed by a page of one FP32 scale
    per row, so the row is ``index_head_dim + 4`` bytes. INT8 is the same
    geometry with the values reinterpreted. BF16 rows are the values alone.
    """
    if kv_dtype in ("fp8", "int8"):
        return index_head_dim + 4, torch.uint8
    if kv_dtype == "bf16":
        return index_head_dim, torch.bfloat16
    raise ValueError(
        f"unsupported indexer cache dtype {kv_dtype!r}; expected 'bf16', "
        "'fp8' or 'int8'"
    )


class IluDeepseekV41IndexerCache(torch.nn.Module, AttentionLayerBase):
    """The paged index-key cache an ``owns_k`` layer publishes into."""

    split_indexer = False
    prefix = ""

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
        self.record = _INDEXER_KV_DTYPE
        self.compress_ratio = compress_ratio
        self.split_indexer = indexer_kv_layout() == "split"
        self.kv_cache_scales = None
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def block_size(self) -> int:
        # Read live: ``update_block_size_for_backend`` picks the backend's
        # preferred page only once the layers exist, so it lands after this
        # module is built. A value captured in ``__init__`` would stay at the
        # 16-token default while the main KV spec reports the negotiated 64,
        # and the two MLA specs would no longer group together.
        return self.cache_config.block_size

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        cache = kv_cache.squeeze(1)
        cache[0].zero_()
        if self.split_indexer:
            if cache.shape[-1] != 128:
                raise ValueError("Split indexer payload must contain 128 INT8 values")
            owner = getattr(self, "_indexer_scale_owner", None)
            if owner is None:
                raise RuntimeError(
                    "Split indexer scales must be bound before its payload"
                )
            owner.bind(self.prefix, cache)
            self.kv_cache = cache
            return
        if cache.size(-1) != self.head_dim:
            # INT8 pages are allocated at ``_INDEX_PAGE_ROW_BYTES`` and narrowed
            # here. Page pitch stays in stride(0); the row is the 132-byte record.
            cache = cache.as_strided(
                (cache.size(0), cache.size(1), self.head_dim),
                (cache.stride(0), self.head_dim, 1),
            )
        self.kv_cache = cache

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        # INT8 mqa-logits pages 64 or 256 states. A kernel block is counted in
        # tokens, so ratio-1 stays at 64 and ratio-2 needs 128. Platform
        # negotiation leaves ``cache_config.block_size`` at the INT8 preferred
        # 64, and v0.31 then packs same-``block_size`` MLA specs into one group
        # and asks ``select_common_block_size`` for one kernel size. Forcing
        # both pages here keeps the two ratios in different groups and makes
        # the manager size match the ratio-2 backend's [128].
        record = getattr(
            self, "record", "int8" if self.dtype is torch.uint8 else "bf16"
        )
        block_size = self.block_size
        if record == "int8":
            if self.compress_ratio == 1:
                block_size = 64
            elif self.compress_ratio == 2:
                block_size = 128
        if block_size % self.compress_ratio:
            raise ValueError(
                f"block_size {block_size} must be a multiple of "
                f"compress_ratio {self.compress_ratio}, or a page would hold a "
                f"fractional row (layer {self.prefix})"
            )
        rows_per_page = block_size // self.compress_ratio
        if record == "fp8" and self.prefix and rows_per_page % 64:
            raise ValueError(
                f"FP8 indexer requires rows_per_page % 64 == 0, got "
                f"{rows_per_page} at {self.prefix}; use --block-size 128"
            )
        # INT8 is allocated wider than the record so the page byte count is a
        # power of two. FP8 keeps the contiguous 132-byte row.
        head_size = (
            _INDEX_PAGE_ROW_BYTES if record in ("int8", "fp8") else self.head_dim
        )
        logger.info_once(
            "Indexer cache %s: record=%s dtype=%s row_width=%d rows_per_page=%d",
            self.prefix,
            record,
            self.dtype,
            head_size,
            rows_per_page,
        )
        if record == "int8":
            return MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=128 if self.split_indexer else head_size,
                dtype=self.dtype,
                tokens_per_state=self.compress_ratio,
                cache_dtype_str=SPLIT_INDEXER_FORMAT if self.split_indexer else None,
            )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=self.dtype,
            # The indexer page shares a physical block with the main KV cache,
            # so it carries the same padding.
            alignment=512,
            tokens_per_state=self.compress_ratio,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return indexer_backend_cls(self.compress_ratio)


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
        self.record = _INDEXER_KV_DTYPE
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            self.record.upper(),
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
        if self.record in ("fp8", "int8"):
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            dev_idx = torch.cuda.current_device()
            idx_key = (dev_idx, max_tokens, self.n_head, self.head_dim, self.record)
            idx_scratch = _INDEXER_SCRATCH_CACHE.get(idx_key)
            if idx_scratch is None:
                idx_scratch = {
                    "weights_fp32": torch.empty(
                        (max_tokens, self.n_head),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                }
                if self.record == "fp8":
                    idx_scratch["q_quant"] = torch.empty(
                        (max_tokens, self.n_head, self.head_dim),
                        dtype=torch.float8_e4m3fn,
                        device="cuda",
                    )
                else:
                    idx_scratch["q_quant"] = torch.empty(
                        (max_tokens, self.n_head, self.head_dim),
                        dtype=torch.int8,
                        device="cuda",
                    )
                _INDEXER_SCRATCH_CACHE[idx_key] = idx_scratch
            self.weights_fp32_buffer = idx_scratch["weights_fp32"]
            if self.record == "fp8":
                self.q_fp8_buffer = idx_scratch["q_quant"]
            else:
                self.q_int8_buffer = idx_scratch["q_quant"]
            if not hasattr(k_cache, "logits_buffer"):
                decode_rows = max(
                    vllm_config.compilation_config.max_cudagraph_capture_size or 0,
                    vllm_config.scheduler_config.max_num_seqs
                    * (
                        1
                        + (
                            vllm_config.speculative_config.num_speculative_tokens
                            if vllm_config.speculative_config
                            else 0
                        )
                    ),
                )
                stride = (self.max_model_len + 31) // 32 * 32
                full_stride = max(
                    stride,
                    (vllm_config.model_config.max_model_len + 31) // 32 * 32,
                )
                logits_key = (dev_idx, decode_rows)
                shared_logits = _INDEXER_LOGITS_CACHE.get(logits_key)
                if (
                    shared_logits is None
                    or shared_logits.numel() < decode_rows * full_stride
                ):
                    shared_logits = torch.empty(
                        (decode_rows * full_stride,),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    _INDEXER_LOGITS_CACHE[logits_key] = shared_logits
                k_cache.logits_buffer = shared_logits[: decode_rows * stride].view(
                    decode_rows, stride
                )

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
            owned_decode_state=(
                self.record == "int8"
                and self.n_head == 32
                and self.head_dim == 128
                and topk_indices_buffer is not None
                and topk_indices_buffer.is_cuda
                and 1 <= vllm_config.scheduler_config.max_num_seqs <= 5
                and vllm_config.speculative_config is not None
                and vllm_config.speculative_config.num_speculative_tokens == 5
            ),
            max_num_decode_seqs=vllm_config.scheduler_config.max_num_seqs,
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
        # Compare against the builder's own row count rather than deriving it
        # from cache_config.block_size: the KV cache coordinator scales a
        # group's manager block size up to equalize page sizes across groups,
        # so the page can be wider than this layer's configured block.
        assert k_cache.shape[1] == indexer_metadata.num_states, (
            f"indexer cache holds {k_cache.shape[1]} rows per page, but the "
            f"slots were built for {indexer_metadata.num_states} "
            f"(cache shape {tuple(k_cache.shape)}, "
            f"compress_ratio {self.compress_ratio}, "
            f"prefix {self.k_cache.prefix})"
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
            self.record,
            row_out=self.k_row_buffer,
            scales=self.k_cache.kv_cache_scales,
        )

    def forward(
        self,
        qr: QueryActivation,
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
                    ixfops.fill_short_context_topk(
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
        if self.record == "fp8":
            # The per-(token, head) q_scale comes back folded into the weights,
            # which the FP8 scoring kernel wants as fp32.
            q_fp8 = self.q_fp8_buffer[: q.shape[0]]
            folded_weights = self.weights_fp32_buffer[: q.shape[0]]
            ixfops.indexer_q_rope_fp8_out(
                q,
                positions,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale * self.n_head**-0.5,
                q_fp8,
                folded_weights,
            )
            return q_fp8, None, folded_weights
        if self.record == "int8":
            # Same contract as FP8: the INT8 scoring kernel reads folded fp32
            # weights and takes no companion scale tensor.
            q_int8 = self.q_int8_buffer[: q.shape[0]]
            folded_weights = self.weights_fp32_buffer[: q.shape[0]]
            ixfshared.ds4_fused_indexer_q_rope_int8_quant(
                q,
                positions,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                self.rope_dim,
                q_int8_out=q_int8,
                weights_out=folded_weights,
            )
            return q_int8, None, folded_weights
        # In-place: q is rotated, weights are rescaled.
        ixfops.indexer_q_rope(
            q,
            positions,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            indexer_weights,
            self.softmax_scale * self.n_head**-0.5,
        )
        return q, None, indexer_weights


class IluDeepseekV41SWACache(IluDeepseekV4SWACache):
    def bind_kv_cache(self, kv_cache):
        super().bind_kv_cache(kv_cache)
        # Sparse FP8 tiles read null-page payload for masked lanes. A zero
        # probability cannot remove a NaN value from the PV accumulation.
        self.kv_cache[0].zero_()
        state = getattr(self, "_engram_hash_state", None)
        if state is not None:
            state.ensure_cache()


class IluDeepseekV41Attention(nn.Module, AttentionLayerBase):
    """Q/KV/O projections, vLLM-owned caches, and the sparse MLA forward.

    ``compress_ratio > 0`` does not mean this layer compresses its own KV: only
    ``kv_source_layer_ids`` do; the rest read that cache.
    """

    compressed_kv_format = None
    compressed_decode_restore = False
    kv_cache_scales = None
    compressed_kv_workspace = None
    main_kv_scale_rows = None
    main_scale_transfer_width = 0

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
        defer_o_proj_reduce: bool = False,
    ) -> None:
        super().__init__()
        self._defer_o_proj_reduce = defer_o_proj_reduce
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
        self.compressed_kv_format = main_kv_format()
        self.compressed_decode_restore = envs.VLLM_ILU_DS41_DECODE_KV_RESTORE
        self.kv_cache_scales = None
        self.compressed_kv_workspace = None
        self.main_kv_scale_rows = None
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
        # ``wo_a`` is contracted per output group, not as one flat linear, and
        # the checkpoint stores it flat as
        # ``[n_local_groups * o_lora_rank, heads_per_group * head_dim]``. These
        # four numbers are the only part of that split a quant method cannot
        # recover on its own (``output_partition_sizes`` carries the product
        # alone), so hand them over for ``process_weights_after_loading``.
        # Always set: a method that does not regroup simply ignores it.
        self.wo_a.bmm_geom = SimpleNamespace(
            n_local_groups=self.n_local_groups,
            o_lora_rank=self.o_lora_rank,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            head_dim=self.head_dim,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not defer_o_proj_reduce,
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
        self._configure_grouped_int8_output(vllm_config.model_config.dtype)
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        device = current_platform.device_type
        dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
        window_cols = max(256, ((self.window_size + 255) // 256) * 256)
        scratch_key = (
            dev_idx,
            max_tokens,
            self.n_local_heads,
            self.head_dim,
            config.index_topk,
            self.window_size,
        )
        scratch = _ATTN_SCRATCH_CACHE.get(scratch_key)
        if scratch is None:
            scratch = {
                "_attn_out": torch.zeros(
                    max_tokens,
                    self.n_local_heads,
                    self.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "_global_topk": torch.full(
                    (max_tokens, config.index_topk),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                "_global_topk_lens": torch.zeros(
                    max_tokens, dtype=torch.int32, device=device
                ),
                # A 256-wide row satisfies the FP8 reader's tile-stride contract.
                "_window_idx": torch.full(
                    (max_tokens, window_cols),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                "_window_lens": torch.zeros(
                    max_tokens, dtype=torch.int32, device=device
                ),
                "_insert_slots": torch.full(
                    (max_tokens,), -1, dtype=torch.int64, device=device
                ),
                "_window_offsets": torch.arange(self.window_size, device=device),
                "_draft_window_offsets": torch.arange(window_cols, device=device),
            }
            _ATTN_SCRATCH_CACHE[scratch_key] = scratch
        self._attn_out = scratch["_attn_out"]
        self._global_topk = scratch["_global_topk"]
        self._global_topk_lens = scratch["_global_topk_lens"]
        self._window_idx = scratch["_window_idx"]
        self._window_lens = scratch["_window_lens"]
        self._insert_slots = scratch["_insert_slots"]
        self._window_offsets = scratch["_window_offsets"]
        self._draft_window_offsets = scratch["_draft_window_offsets"]

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

        # A forwarder replicates a group that started on an earlier rank. It
        # runs no compressor, but it owns both caches the group would have
        # published locally: the compressed KV one through
        # ``get_kv_cache_spec`` below, and this index-key one, which the
        # consumers above it resolve through ``kv_source_layer_id``.
        #
        # Built before the indexer because the forwarder layer can be an index
        # source itself -- a cut landing on 24 / 28 / 32 / 36, which is what an
        # even pp=5 or pp=10 split does. Such a layer resolves its own prefix
        # below, so the cache has to exist by then.
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
            # The replica only relays inside this PP instance. PD transfers
            # the source rank's caches, not the forwarder's copy.
            kv_transfer_config = vllm_config.kv_transfer_config
            if (
                kv_transfer_config is not None
                and kv_transfer_config.kv_role == "kv_consumer"
            ):
                raise NotImplementedError(
                    f"layer {layer_id} is a kv forwarder on a PD decode "
                    "instance; its replica would miss the transferred prefix. "
                    "Pick a decode PP split that does not cut a kv-sharing group."
                )
            add_skip_layers(prefix, self.forwarder_indexer_cache.prefix)

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

        query_linears = [self.wq_b]
        if self.indexer is not None:
            query_linears.append(self.indexer.wq_b)
        self._fuse_query_quant = can_fuse_query_quant(
            query_linears, self.q_lora_rank, self.head_dim
        )

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_kv_cache_dtype(
            cache_config
        )
        # Width and dtype of one record on the wire, for the dense buffers the
        # model allocates when a pipeline cut strands this layer's consumers.
        # These are the record's own shape, not the cache's page geometry.
        if self.kv_cache_torch_dtype is torch.uint8:
            # Packed fp8_ds_mla: 448 E4M3 NoPE bytes, 64 BF16 RoPE values and
            # 8 bytes of UE8M0 scales.
            self.swa_bytes_per_token = 584
            self.main_transfer_width = 584
        elif self.kv_cache_torch_dtype is torch.bfloat16:
            # Plain [448 NoPE | 64 RoPE] bf16 row, which is also the record
            # width because the two coincide when nothing is packed.
            self.swa_bytes_per_token = self.head_dim * 2
            self.main_transfer_width = self.head_dim
        elif self.kv_cache_torch_dtype is torch.float8_e4m3fn:
            self.swa_bytes_per_token = self.head_dim
            self.main_transfer_width = self.head_dim
        else:
            raise NotImplementedError(
                f"Unsupported DeepSeek V4.1 KV dtype: {cache_config.cache_dtype}. "
                "Use fp8_ds_mla, bfloat16 or fp8_e4m3."
            )
        self.main_transfer_dtype = self.kv_cache_torch_dtype
        self.main_scale_transfer_width = 0
        if self.compressed_kv_format is not None:
            self.main_transfer_width = self.compressed_kv_format.payload_bytes(
                self.head_dim
            )
            self.main_transfer_dtype = torch.uint8
            self.main_scale_transfer_width = self.compressed_kv_format.scale_groups(
                self.head_dim
            )
        (
            self.indexer_transfer_width,
            self.indexer_transfer_dtype,
        ) = _indexer_k_cache_row(config.index_head_dim, _INDEXER_KV_DTYPE)
        self.kv_page_alignment = 512 if self.kv_cache_dtype == "fp8_e4m3" else 576
        self.kv_scale = 1.0
        if self.kv_cache_dtype == "fp8_e4m3":
            self._q_fp8_pt = torch.empty_like(self._attn_out, dtype=torch.float8_e4m3fn)
            self._q_scale_pt = torch.empty(
                self._attn_out.shape[0], dtype=torch.float32, device=device
            )
            self._q_quant_workspace_pt = torch.empty(
                256, dtype=torch.float32, device=device
            )

        swa_bounded_replay = _resolve_swa_bounded_replay(cache_config, vllm_config)
        self.swa_cache_layer = IluDeepseekV41SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            backend_cls=self.swa_backend_cls,
            block_size=None,
            packed_bytes_per_token=self.swa_bytes_per_token,
            packed_page_alignment=self.kv_page_alignment,
            # Match the community spec: rebuild private window KV after a hit;
            # the model state protects cached compressed KV from replay writes.
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
        # Bounded replay rebuilds the window and the compressor ring on D,
        # so those caches are not part of the PD transfer.
        if swa_bounded_replay:
            add_skip_layers(self.swa_cache_layer.prefix)
            if self.compressor is not None and self.compressor.state_cache is not None:
                add_skip_layers(self.compressor.state_cache.prefix)
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
        if self.compressed_kv_format is not None:
            fmt = self.compressed_kv_format
            block_size = fmt.rows_per_page(self.head_dim) * self.compress_ratio
            return MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=torch.uint8,
                state_content_bytes=fmt.payload_bytes(self.head_dim),
                tokens_per_state=self.compress_ratio,
                cache_dtype_str=fmt.name,
                storage_block_size=block_size,
            )
        # Tied to the resolved record rather than restated, so the spec and the
        # tensor the store kernel writes cannot drift apart.
        if self.kv_cache_dtype != "fp8_ds_mla":
            # A plain row needs none of the packed record's bookkeeping: the
            # page size follows from head_size and the element type, and there
            # is no 64B-aligned block base to respect.
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                tokens_per_state=self.compress_ratio,
                cache_dtype_str=self.kv_cache_dtype,
                model_version=None,
            )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            model_version="deepseek_v4",
            # The packed record is wider than ``head_size`` says (448 E4M3 NoPE
            # bytes + 64 BF16 RoPE + 8 UE8M0 scale bytes vs a 512-element NoPE
            # content), so the page size is stated, not derived. See the same
            # note on the SWA spec in ``ilu_sparse_swa.py``.
            state_content_bytes=self.main_transfer_width,
            alignment=64,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)
        if self.compressed_kv_format is not None:
            fmt = self.compressed_kv_format
            if self.kv_cache.dtype is not torch.uint8 or self.kv_cache.shape[1:] != (
                fmt.rows_per_page(self.head_dim),
                fmt.payload_bytes(self.head_dim),
            ):
                raise ValueError(
                    "Main compressed KV view does not match its immutable format"
                )
            if (
                self.kv_cache_scales is None
                or self.kv_cache_scales.shape[:2] != self.kv_cache.shape[:2]
            ):
                raise ValueError(
                    "Main compressed KV scale view is not paired with its payload"
                )
        else:
            self.kv_cache[0].zero_()

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

    def _compressed_kv_scales(self) -> torch.Tensor:
        source = (
            self
            if self.is_kv_source or self.is_kv_forwarder
            else self._static_forward_context[self.compressed_cache_prefix]
        )
        if (
            source.compressed_kv_format != self.compressed_kv_format
            or source.kv_cache_scales is None
        ):
            raise ValueError(
                "Compressed KV consumer has no matching source scale plane"
            )
        return source.kv_cache_scales

    def accept_remote_kv(
        self,
        main_rows: torch.Tensor,
        indexer_rows: torch.Tensor,
        topk_rows: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        main_scale_rows: torch.Tensor | None = None,
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
        if self.compressed_kv_format is not None:
            if main_scale_rows is None or self.kv_cache_scales is None:
                raise ValueError(
                    "Compressed KV PP forwarding requires both payload and scales"
                )
            move_quantized_records(
                main_rows,
                main_scale_rows,
                self.kv_cache,
                self.kv_cache_scales,
                main_metadata.slot_mapping,
                positions,
                self.compress_ratio,
                format=self.compressed_kv_format,
            )
            if self.compressed_kv_workspace is not None:
                self.compressed_kv_workspace.main_cache_written()
        else:
            move_rows_to_paged_cache(
                main_rows,
                self.kv_cache,
                main_metadata.slot_mapping,
                positions,
                self.compress_ratio,
            )

        indexer_cache = self.forwarder_indexer_cache
        assert indexer_cache is not None
        if (
            indexer_rows.ndim != 2
            or indexer_rows.shape[1] != self.indexer_transfer_width
            or indexer_rows.dtype != self.indexer_transfer_dtype
        ):
            raise ValueError(
                "remote indexer row width/dtype does not match local cache"
            )
        if indexer_cache.kv_cache.numel():
            indexer_metadata = cast(Any, attn_metadata[indexer_cache.prefix])
            if indexer_cache.split_indexer:
                ixfops.indexer_int8_insert_split(
                    indexer_rows,
                    indexer_cache.kv_cache,
                    indexer_cache.kv_cache_scales,
                    indexer_metadata.slot_mapping,
                    positions,
                    self.compress_ratio,
                )
            else:
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
    def cos_sin(self) -> torch.Tensor | None:
        rotary_emb = getattr(self, "rotary_emb", None)
        return None if rotary_emb is None else rotary_emb.cos_sin_cache

    def _configure_grouped_int8_output(self, dtype) -> None:
        from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
            IluV4W8A8BMMLinearMethod,
            IluV4W8A16BMMLinearMethod,
        )

        self._grouped_int8_o_proj = False
        self._woa_input_mode = envs.VLLM_ILU_DS41_WOA_INPUT_MODE
        if self._woa_input_mode not in ("legacy", "reader_int8"):
            raise ValueError("WO-A input mode must be legacy or reader_int8")
        method = self.wo_a.quant_method
        explicit = isinstance(method, IluV4W8A8BMMLinearMethod)
        if not explicit and not envs.VLLM_ILU_DS41_GROUPED_INT8_O_PROJ:
            return
        supported = IluV4W8A8BMMLinearMethod.supports_geometry(
            self.wo_a.bmm_geom, dtype, self.rope_head_dim
        )
        if explicit:
            if not supported:
                raise ValueError(
                    "Grouped INT8 output projection geometry is unsupported"
                )
            self._grouped_int8_o_proj = True
        elif supported and type(method) is IluV4W8A16BMMLinearMethod:
            self.wo_a.quant_method = IluV4W8A8BMMLinearMethod(
                method.parent_config, {"bmm_w8a8_gemm_format": "NN"}
            )
            self._grouped_int8_o_proj = True

    def _uses_fused_inverse_rope(self) -> bool:
        """Quantized main readers rotate their result before BF16 rounding.

        A grouped INT8 projection owns its inverse instead, for either main
        or window-only attention. Other window-only layers retain the inverse.
        The decision is layer-static and shared by the reader and its caller,
        so mixed decode/prefill slices cannot rotate twice or omit a rotation.
        """
        return (
            self.compressed_kv_format is not None
            and self.compress_ratio > 0
            and self.cos_sin is not None
            and not getattr(self, "_grouped_int8_o_proj", False)
        )

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
        """Grouped output projection over ``[tokens, n_local_heads, head_dim]``.

        The heads of one output group are contracted together, so what reaches
        ``wo_a`` is ``[tokens, n_local_groups, heads_per_group * head_dim]``.

        Which contraction runs depends on what the quant method left behind.
        The Iluvatar-quantized checkpoint routes this layer to
        ``IluV4W8A16BMMLinearMethod``, which repacks the INT8 weight into a
        Marlin layout and *deletes* ``weight`` / ``weight_scale``; the einsum
        below would have nothing to read, and ``marlin_w8a16`` applies the
        per-(group, channel) scale itself against a bf16 activation. An
        unquantized ``wo_a`` keeps its weight and takes the einsum.
        """
        if hasattr(self.wo_a, "W_repacked"):
            z = ixfshared.marlin_w8a16(
                o.view(o.shape[0], self.n_local_groups, -1),
                self.wo_a.W_repacked,
                self.wo_a.W_scale_repacked,
                group_size=-1,
                format=self.wo_a.W_format,
                batch_first=False,
            )
            return self.wo_b(z.flatten(1))
        if self.n_local_groups == 1:
            # One group is an ordinary linear; let it run its own apply. The
            # whole group's heads are contracted together, so ``wo_a`` sees
            # ``[tokens, n_local_heads * head_dim]`` -- keep tokens as the rows
            # instead of folding heads into them.
            return self.wo_b(self.wo_a(o.reshape(o.shape[0], -1)))
        w_view = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.einsum(
            "bgd,grd->bgr", o.view(o.shape[0], self.n_local_groups, -1), w_view
        )
        return self.wo_b(z.flatten(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        *,
        cache_only: bool = False,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(
            hidden_states
        )
        qr, kv = self._split_qkv_and_norm(qr_kv)
        attn_out = None
        if not cache_only:
            attn_out = self._attn_out[: hidden_states.shape[0]]
            covered = self._covered_tokens(attn_out.shape[0])
            if covered < attn_out.shape[0]:
                attn_out[covered:].zero_()
        projection_input = self._prepare_and_attn(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_weights,
            positions,
            attn_out,
            cache_only=cache_only,
        )
        if cache_only:
            # No consumer uses this hidden state: the producer ends here and
            # PD discards its sampled token. All transferred caches are ready.
            return hidden_states
        assert attn_out is not None
        if getattr(self, "_grouped_int8_o_proj", False):
            if projection_input is not None:
                from ixformer.inference.functions.models.deepseek_v41.mla_sparse.float.quantized_sparse_mla import (
                    GroupedInt8AttentionOutput,
                )

                if not isinstance(projection_input, GroupedInt8AttentionOutput):
                    raise TypeError("Unsupported grouped attention reader output")
                if not projection_input.inverse_rope_applied:
                    raise ValueError("Grouped WO-A reader did not apply inverse RoPE")
                grouped = self.wo_a.quant_method.apply_quantized(
                    self.wo_a,
                    projection_input.values,
                    projection_input.scales,
                    output_dtype=hidden_states.dtype,
                )
            else:
                grouped = self.wo_a.quant_method.apply_with_rope(
                    self.wo_a, attn_out, positions, self.cos_sin
                )
            return self.wo_b(grouped.flatten(1))
        if not self._uses_fused_inverse_rope():
            apply_rope(attn_out.unsqueeze(0), positions, self.cos_sin, inverse=True)
        result = self._o_proj(attn_out)
        return result

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
            kv_score = ixfshared.gemm_bf16_bf16_fp32(
                hidden_states,
                compressor.fused_wkv_wgate.weight,
                format="TN",
            )

        if self.indexer is not None:
            # ReplicatedLinear returns (output, bias); bias is None.
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)

        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        return qr_kv, kv_score, indexer_weights

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[QueryActivation, torch.Tensor]:
        """Split the fused q-lora / kv projection and RMSNorm both halves."""
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if getattr(self, "_fuse_query_quant", False) and query_inputs_are_compatible(
            qr, kv
        ):
            return fused_query_norm(
                qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
            )
        return ixfshared.rms_norm_qk(
            qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
        )

    def _wq_b_proj(self, qr: QueryActivation) -> torch.Tensor:
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
        """RoPE q in place and publish kv as this layer's window record."""
        if not isinstance(attn_metadata, dict):
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        if self.kv_cache_dtype == "fp8_e4m3":
            ixfops.qkv_rope_fp8_pt_insert(
                q,
                self.swa_cache_layer.kv_cache,
                kv,
                self._padded_insert_slots(swa_metadata.slot_mapping, q.shape[0]),
                positions,
                self.cos_sin,
                kv_scale=self.kv_scale,
                backend="ixformer_native",
            )
            return q
        insert = (
            ixfops.qkv_rope_packed_fp8_insert
            if self.kv_cache_dtype == "fp8_ds_mla"
            else ixfops.qkv_rope_insert
        )
        insert(
            q,
            self.swa_cache_layer.kv_cache,
            kv,
            self._padded_insert_slots(swa_metadata.slot_mapping, q.shape[0]),
            positions,
            self.cos_sin,
        )
        return q

    def _padded_insert_slots(
        self, slots: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        if slots.shape[0] >= num_tokens:
            return slots[:num_tokens]
        # Eager DP padding can exceed the local metadata's token count.
        out = self._insert_slots[:num_tokens]
        out.fill_(-1)
        n = min(num_tokens, slots.shape[0])
        out[:n].copy_(slots[:n])
        return out

    def _covered_tokens(self, num_tokens: int) -> int:
        """Rows the sparse-MLA kernels are guaranteed to write this step."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return 0
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        if swa_metadata is None:
            return 0
        return min(
            num_tokens,
            swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens,
        )

    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: QueryActivation,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor | None,
        *,
        cache_only: bool = False,
    ) -> GroupedInt8AttentionOutput | None:
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
            latent = compressor(
                kv_score,
                positions,
                rotary_emb=self.rotary_emb,
                row_out=self.main_kv_rows if self.is_kv_sender else None,
                scale_out=self.main_kv_scale_rows if self.is_kv_sender else None,
            )
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

        # Caches for this layer are written. The producer counts one call per
        # layer; the rank send starts only after that count hits zero.
        if isinstance(attn_metadata, dict) and has_kv_transfer_group():
            get_kv_transfer_group().save_kv_layer(
                self.prefix, self.kv_cache, attn_metadata
            )

        if cache_only:
            return None
        assert out is not None
        return self._sparse_indexer_and_attn(
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
    ) -> GroupedInt8AttentionOutput | None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            output.zero_()
            return None

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

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        quantize_output = (
            getattr(self, "_grouped_int8_o_proj", False)
            and getattr(self, "_woa_input_mode", "legacy") == "reader_int8"
            and num_prefills == 0
            and num_decodes > 0
            and q.shape[0] == num_decode_tokens
            and not swa_only
        )
        selection = None
        if (
            self.compressed_kv_format is not None
            and not swa_only
            and self.compressed_kv_workspace is not None
            and (
                num_prefills > 0
                or (
                    num_decodes > 0
                    and (
                        self.compressed_kv_workspace.capturing
                        or getattr(self, "compressed_decode_restore", False)
                    )
                )
            )
        ):
            if flashmla_metadata is None or self_kv_cache is None:
                raise RuntimeError(
                    "Compressed KV attention requires FlashMLA metadata and a main cache"
                )
            projection_input = None
            with self.compressed_kv_workspace.read_scope() as workspace:
                selection = self._prepare_compressed_family(
                    q if num_prefills else q[:num_decode_tokens],
                    positions if num_prefills else positions[:num_decode_tokens],
                    output if num_prefills else output[:num_decode_tokens],
                    self_kv_cache,
                    flashmla_metadata,
                    swa_metadata,
                    workspace,
                )
                if num_prefills > 0:
                    prefill_q = q[num_decode_tokens:]
                    prefill_out = output[num_decode_tokens:]
                    self._forward_prefill_as_decode(
                        q=prefill_q,
                        positions=positions[num_decode_tokens:],
                        compressed_k_cache=self_kv_cache,
                        output=prefill_out,
                        attn_metadata=flashmla_metadata if not swa_only else None,
                        swa_metadata=swa_metadata,
                        selection=selection,
                    )
                if num_decodes > 0:
                    decode_out = output[:num_decode_tokens]
                    projection_input = self._forward_decode(
                        q=q[:num_decode_tokens],
                        kv_cache=self_kv_cache,
                        positions=positions[:num_decode_tokens],
                        swa_metadata=swa_metadata,
                        attn_metadata=flashmla_metadata if not swa_only else None,
                        swa_only=swa_only,
                        output=decode_out,
                        selection=selection,
                        quantize_output=quantize_output,
                    )
            return projection_input

        if num_prefills > 0:
            prefill_q = q[num_decode_tokens:]
            prefill_out = output[num_decode_tokens:]

            # Three prefill paths, same math, very different shapes. The
            # default gathers the compressed pool and the window into one bf16
            # workspace and runs a single sparse-MLA kernel over a combined
            # index list. ``_opt`` skips the workspace and merges two paged
            # attentions by LSE. ``_as_decode`` reuses the per-token decode
            # kernel, which treats every query token as its own batch element
            # -- fine for correctness, but the query side is large in prefill.
            self._forward_prefill_as_decode(
                q=prefill_q,
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                output=prefill_out,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_metadata=swa_metadata,
                selection=selection,
            )
        if num_decodes > 0:
            decode_out = output[:num_decode_tokens]
            return self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                positions=positions[:num_decode_tokens],
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_only=swa_only,
                output=decode_out,
                selection=selection,
                quantize_output=quantize_output,
            )
        return None

    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> GroupedInt8AttentionOutput | None:
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            self.indexer.indexer_op(
                hidden_states,
                index_q,
                None,
                index_weights,
            )
            if _DBG_TOPK and self.topk_indices_buffer is not None:
                _dbg_topk(self.prefix, index_q, index_weights, self.topk_indices_buffer)
            if self.compressed_kv_workspace is not None:
                self.compressed_kv_workspace.restore.invalidate()

        return self.forward_mqa(q, kv, positions, out)

    def _window_indices(
        self,
        swa_metadata: DeepseekSparseSWAMetadata,
        begin: int,
        end: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SWA slots for batch tokens ``[begin, end)``.

        The builder only fills decode rows. Prefill tokens sit later in the
        same batch, so compute their slots directly from device positions.
        """
        # Positions and all indexing stay on device, including chunked prefill.
        req = swa_metadata.token_to_req_indices[begin:end].long()
        valid = swa_metadata.is_valid_token[begin:end]
        req = req.clamp(0, swa_metadata.block_table.shape[0] - 1)
        lower = (positions + 1 - self.window_size).clamp_min(0)
        replay = getattr(swa_metadata, "replay_start", None)
        width = self.window_size
        upper = positions
        offsets = self._window_offsets
        if not getattr(swa_metadata, "causal", True):
            query_len = (
                swa_metadata.query_start_loc[req + 1]
                - swa_metadata.query_start_loc[req]
            )
            upper = swa_metadata.seq_lens[req] - 1
            lower = (
                swa_metadata.seq_lens[req] - query_len - self.window_size
            ).clamp_min(0)
            offsets = self._draft_window_offsets
            width = 256
        if replay is not None:
            lower = torch.maximum(lower, replay[req])
        logical = lower[:, None] + offsets[None, :]
        columns = logical // swa_metadata.block_size
        table = swa_metadata.block_table
        pages = table[req[:, None], columns.clamp(0, table.shape[1] - 1)].long()
        keep = (
            valid[:, None]
            & (logical <= upper[:, None])
            & (columns < table.shape[1])
            & (pages > 0)
        )
        slots = pages * swa_metadata.block_size + logical % swa_metadata.block_size
        out = self._window_idx[begin:end]
        out.fill_(-1)
        out[:, :width].copy_(torch.where(keep, slots, -1))
        lens = self._window_lens[begin:end]
        lens.copy_(keep.sum(-1))
        return out, lens

    def _global_indices(self, local, req, table, block_size, valid, positions):
        n = local.shape[0]
        if hasattr(self, "_global_topk") and hasattr(self, "_global_topk_lens"):
            indices = self._global_topk[:n]
            lens = self._global_topk_lens[:n]
            ixfops.compute_global_topk_indices_and_lens(
                local,
                req,
                table,
                block_size,
                valid,
                positions,
                self.compress_ratio,
                output=indices,
                output_lens=lens,
            )
            return indices, lens
        return ixfops.global_topk_indices_and_lens(
            local,
            req,
            table,
            block_size,
            valid,
            positions,
            self.compress_ratio,
            0,
            False,
        )

    def _prepare_compressed_family(
        self, q, positions, output, cache, attn_metadata, swa_metadata, workspace
    ) -> CompressedFamilySelection:
        """Map once and prepare one main-KV family in the yielded worker span.

        The old and new query slices share the same map and retained rows.
        New steps/captures always record preparation; only later consumers
        in the current forward may reuse the active family. Pure decode uses
        mapping reuse during capture; its optional selected-row path restores
        the full decode selection inside the same bounded worker lease.
        Producer-only partial prefill may replace the linear plan with a
        bounded compact plan; mixed batches retain suffix-only restoration.
        """
        assert self.topk_indices_buffer is not None
        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.token_to_req_indices is not None
        n0 = swa_metadata.num_decode_tokens
        n1 = n0 + swa_metadata.num_prefill_tokens
        restore_decode = swa_metadata.num_prefills == 0 and getattr(
            self, "compressed_decode_restore", False
        )
        restore_start = 0 if restore_decode else n0
        restore_tokens = n1 - restore_start
        topk = self.topk_indices_buffer[:n1]
        reqs = swa_metadata.token_to_req_indices[:n1]
        valid = swa_metadata.is_valid_token[:n1]
        scales = self._compressed_kv_scales()
        swa_cache = self.swa_cache_layer.kv_cache
        reader_ranges = []
        if swa_metadata.num_prefills > 0:
            reader_ranges.append((slice(n0, n1), swa_metadata.prefill_swa_indices))
        if swa_metadata.num_decodes > 0:
            reader_ranges.append((slice(0, n0), swa_metadata.decode_swa_indices))
        family_masks = prefill_validity_family(
            self, get_forward_context().attn_metadata, swa_metadata, n1
        )
        key = family_key(
            owner=self.compressed_cache_prefix,
            index_source_layer_id=self.index_source_layer_id,
            payload=cache,
            scales=scales,
            workspace=workspace,
            query=q,
            output=output,
            indices=topk,
            format=self.compressed_kv_format,
            mapping_layout=(
                TensorViewKey.of(topk),
                TensorViewKey.of(reqs),
                # Physical selection is independent of a layer's query mask.
                # Each decode consumer still masks its own main lengths below.
                "per-consumer-decode-validity"
                if restore_decode
                else (
                    (
                        "prefill-mask-union",
                        tuple(TensorViewKey.of(m) for m in family_masks),
                    )
                    if family_masks
                    else TensorViewKey.of(valid)
                ),
                TensorViewKey.of(positions),
                TensorViewKey.of(attn_metadata.block_table),
                attn_metadata.block_size,
                self.compress_ratio,
            ),
            reader_layouts=(
                TensorLayoutKey.of(swa_cache),
                tuple(TensorLayoutKey.of(indices) for _, indices in reader_ranges),
            ),
            restore_start=restore_start,
            restore_tokens=restore_tokens,
        )
        tracker = self.compressed_kv_workspace.restore
        selection = tracker.lookup(key)
        if selection is not None:
            return consumer_selection(selection, valid) if family_masks else selection
        ticket = tracker.begin_prepare(key)
        consumer_valid = valid
        if family_masks:
            valid = union_validity(family_masks)
        if restore_decode:
            # A later consumer can have a valid query where the first does
            # not. Prepare its physical rows too, while guarding every request
            # index before the native mapper reads the block table.
            valid = (reqs >= 0) & (reqs < attn_metadata.block_table.shape[0])
        indices, lengths = ixfops.global_topk_indices_and_lens(
            topk,
            reqs,
            attn_metadata.block_table,
            attn_metadata.block_size // self.compress_ratio,
            valid,
            positions,
            self.compress_ratio,
            0,
            False,
        )
        if swa_metadata.num_prefills == 0 and not restore_decode:
            selection = CompressedFamilySelection(indices, lengths, None, 0)
            tracker.publish(ticket, selection)
            return selection

        from ixformer.inference.functions.models.deepseek_v41.mla_sparse.common.quantized_sparse_mla import (
            QuantizedSparseMlaRestoreCapacityError,
            make_quantized_sparse_mla_restore_plan,
            prepare_quantized_sparse_mla_restore_plan,
            quantized_sparse_mla_scratch_bytes,
        )

        scratch_bytes = max(
            quantized_sparse_mla_scratch_bytes(
                q[token_slice].unsqueeze(1),
                swa_cache.unsqueeze(-2),
                _batched_index_rows(swa_indices),
                cache,
                scales,
                _batched_index_rows(indices[token_slice]),
                format=self.compressed_kv_format,
                output=output[token_slice].unsqueeze(1),
            )
            for token_slice, swa_indices in reader_ranges
        )
        try:
            plan = make_quantized_sparse_mla_restore_plan(
                workspace,
                _batched_index_rows(indices),
                q.shape[-1],
                scratch_bytes=scratch_bytes,
                restore_start=restore_start,
                restore_tokens=restore_tokens,
                owner=self.compressed_cache_prefix,
                index_source_layer_id=self.index_source_layer_id,
                compressed_capacity=cache.shape[0] * cache.shape[1],
            )
        except QuantizedSparseMlaRestoreCapacityError:
            # A complete restore map may be unaffordable for a valid large
            # chunk. The packed reader still consumes every selected row.
            plan = None
        compact_plan = None
        if (
            getattr(self.compressed_kv_workspace, "prefill_producer", False)
            and swa_metadata.num_decodes == 0
            and n0 == 0
            and (
                plan is None
                or (
                    plan.mode == 0
                    and plan.selected_flags is None
                    and plan.rows.shape[0]
                    < plan.restore_tokens * plan.row_map.shape[-1]
                )
            )
        ):
            compact_plan = self._prepare_compact_family(
                q,
                output,
                cache,
                scales,
                indices,
                lengths,
                swa_metadata.prefill_swa_indices,
                workspace,
                scratch_bytes,
            )
        if compact_plan is not None:
            plan = None
            scratch_bytes = compact_plan.scratch.numel()
        elif plan is not None:
            prepare_quantized_sparse_mla_restore_plan(
                cache,
                scales,
                _batched_index_rows(indices),
                lengths,
                plan,
                format=self.compressed_kv_format,
            )
        selection = CompressedFamilySelection(
            indices, lengths, plan, scratch_bytes, compact_plan=compact_plan
        )
        tracker.publish(ticket, selection)
        return (
            consumer_selection(selection, consumer_valid) if family_masks else selection
        )

    def _prepare_compact_family(
        self,
        q,
        output,
        cache,
        scales,
        indices,
        lengths,
        swa_indices,
        workspace,
        scratch_bytes,
    ):
        """Prepare one P-only compact plan inside the existing family lease."""
        capacity = cache.shape[0] * cache.shape[1]
        owners = compact_owner_capacity(indices.numel(), capacity)
        if owners is None:
            return None
        from ixformer.inference.functions.models.deepseek_v41.mla_sparse.common.quantized_sparse_mla_compact import (
            QuantizedSparseMlaCompactCapacityError,
            make_quantized_sparse_mla_compact_plan,
            prepare_quantized_sparse_mla_compact_plan,
            quantized_sparse_mla_compact_prepare_scratch_bytes,
            quantized_sparse_mla_compact_scratch_bytes,
        )

        indices = _batched_index_rows(indices)
        prepare_bytes = quantized_sparse_mla_compact_prepare_scratch_bytes(
            cache,
            scales,
            indices,
            lengths,
            q.shape[-1],
            format=self.compressed_kv_format,
        )
        arguments = dict(
            compressed_capacity=capacity,
            owner_capacity=owners,
            owner_kind="hash",
            prepare_scratch_bytes=prepare_bytes,
        )
        try:
            plan = make_quantized_sparse_mla_compact_plan(
                workspace,
                indices,
                q.shape[-1],
                scratch_bytes=scratch_bytes,
                **arguments,
            )
            needed = quantized_sparse_mla_compact_scratch_bytes(
                q.unsqueeze(1),
                self.swa_cache_layer.kv_cache.unsqueeze(-2),
                _batched_index_rows(swa_indices),
                cache,
                scales,
                indices,
                format=self.compressed_kv_format,
                workspace=plan.view(),
                output=output.unsqueeze(1),
            )
            if needed > plan.scratch.numel():
                plan = make_quantized_sparse_mla_compact_plan(
                    workspace,
                    indices,
                    q.shape[-1],
                    scratch_bytes=max(scratch_bytes, needed),
                    **arguments,
                )
        except QuantizedSparseMlaCompactCapacityError:
            return None
        prepare_quantized_sparse_mla_compact_plan(
            cache, scales, indices, lengths, plan, format=self.compressed_kv_format
        )
        return plan

    def _sparse_mla(
        self,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        output: torch.Tensor,
        use_fmhadsa: bool = False,
        *,
        positions: torch.Tensor | None = None,
        prefill: bool = False,
        restore_plan=None,
        quantize_output: bool = False,
    ) -> GroupedInt8AttentionOutput | None:
        """Sparse MLA over the window and, for compressed layers, the top-k.

        The kernels below are per-token -- every query token is its own batch
        element with ``s_q = 1`` -- so prefill goes through them too, with the
        window indices :meth:`_window_indices` recomputes for its token range.
        They differ in the record they read: the packed fp8_ds_mla row wants a
        packed and per-(token, head) quantized query, while the plain bf16 row
        takes the query as it is and the caches with an explicit single-KV-head
        axis.
        """
        if (
            self.compressed_kv_format is not None
            and self.compress_ratio > 0
            and extra_cache is None
        ):
            raise ValueError(
                "Compressed attention requires its main payload and scale planes"
            )
        if self.compressed_kv_format is not None and extra_cache is not None:
            if extra_indices is None or extra_lens is None:
                raise ValueError(
                    "Compressed KV attention requires physical row indices and lengths"
                )
            if prefill:
                from ixformer.inference.functions.models.deepseek_v41.mla_sparse.common.quantized_sparse_mla_prefill import (
                    quantized_sparse_mla_prefill as reader,
                )
            else:
                from ixformer.inference.functions.models.deepseek_v41.mla_sparse.common.quantized_sparse_mla import (
                    quantized_sparse_mla as reader,
                )
            if self.compressed_kv_workspace is None:
                raise RuntimeError(
                    "Compressed KV attention workspace is not initialized"
                )
            fused_inverse = self._uses_fused_inverse_rope() and positions is not None
            with self.compressed_kv_workspace.acquire(nested=True) as workspace:
                if quantize_output and not prefill:
                    from ixformer.inference.functions.models.deepseek_v41.mla_sparse.float.quantized_sparse_mla import (
                        QuantizedSparseMlaAffineView,
                        quantized_sparse_mla_int8,
                    )

                    retained = workspace if restore_plan is None else restore_plan
                    if isinstance(
                        retained, (torch.Tensor, QuantizedSparseMlaAffineView)
                    ):
                        return quantized_sparse_mla_int8(
                            q.unsqueeze(1),
                            self.swa_cache_layer.kv_cache.unsqueeze(-2),
                            _batched_index_rows(swa_indices),
                            swa_lens,
                            extra_cache,
                            self._compressed_kv_scales(),
                            _batched_index_rows(extra_indices),
                            extra_lens,
                            self.scale,
                            format=self.compressed_kv_format,
                            groups=self.wo_a.bmm_geom.n_local_groups,
                            attn_sink=self.attn_sink,
                            positions=positions,
                            cos_sin_cache=self.cos_sin,
                            workspace=retained,
                            bf16_scratch=output.unsqueeze(1),
                        )
                reader(
                    q.unsqueeze(1),
                    self.swa_cache_layer.kv_cache.unsqueeze(-2),
                    _batched_index_rows(swa_indices),
                    swa_lens,
                    extra_cache,
                    self._compressed_kv_scales(),
                    _batched_index_rows(extra_indices),
                    extra_lens,
                    self.scale,
                    format=self.compressed_kv_format,
                    attn_sink=self.attn_sink,
                    output=output.unsqueeze(1),
                    positions=positions if fused_inverse else None,
                    cos_sin_cache=self.cos_sin if fused_inverse else None,
                    workspace=workspace if restore_plan is None else restore_plan,
                )
            return None
        if self.kv_cache_dtype == "fp8_ds_mla":
            self._sparse_fp8_mla(
                q,
                swa_indices,
                swa_lens,
                extra_cache,
                extra_indices,
                extra_lens,
                output,
            )
            return
        if self.kv_cache_dtype == "fp8_e4m3":
            self._sparse_fp8_pt_mla(
                q,
                swa_indices,
                swa_lens,
                extra_cache,
                extra_indices,
                extra_lens,
                output,
                use_fmhadsa=use_fmhadsa,
            )
            return
        # ixformer 3817cf75: mla_with_kvcache 与 ds4 的 decode 稀疏 MLA
        # 统一为 ops 层的 flash_mla_sparse_with_kvcache（签名逐参一致）。
        # 原 ixformer_native 后端已并入 ixformer_perf（同为手写 CUDA 路径）。
        flash_mla_sparse_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=self.swa_cache_layer.kv_cache.unsqueeze(-2),
            indices=_batched_index_rows(swa_indices),
            topk_length=swa_lens,
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=None if extra_cache is None else extra_cache.unsqueeze(-2),
            extra_indices=(
                None if extra_indices is None else _batched_index_rows(extra_indices)
            ),
            extra_topk_length=extra_lens,
            output=output.unsqueeze(1),
        )

    def _tile_window(self, indices):
        rows = indices.reshape(indices.shape[0], -1)
        if rows.stride(0) % 256 == 0:
            return rows
        out = self._window_idx[: rows.shape[0]]
        out.fill_(-1)
        out[:, : rows.shape[1]].copy_(rows)
        return out[:, : rows.shape[1]]

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
        q_packed, q_scales = ixfops.quant_mla_q_packed_fp8(q)
        ixfops.sparse_mla_packed_fp8(
            q_packed,
            q_scales,
            self.swa_cache_layer.kv_cache,
            self._tile_window(swa_indices),
            self.scale,
            swa_lens,
            self.attn_sink,
            extra_cache,
            None if extra_indices is None else _tiled_index_rows(extra_indices),
            extra_lens,
            output,
        )

    def _sparse_fp8_pt_mla(
        self,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        output: torch.Tensor,
        use_fmhadsa: bool = False,
    ) -> None:
        if q.shape[1] not in _PT_MLA_HEADS:
            raise ValueError(f"Unsupported per-tensor FP8 MLA head count: {q.shape[1]}")
        q_fp8 = self._q_fp8_pt[: q.shape[0]]
        q_scale = self._q_scale_pt[: q.shape[0]]
        ixfops.quant_mla_q_fp8_pt(
            q,
            q_fp8,
            q_scale,
            self._q_quant_workspace_pt,
            backend="ixformer_native",
        )
        sparse_op = (
            ixfops.fmhadsa
            if use_fmhadsa and _v41_prefill_fmhadsa()
            else ixfops.sparse_mla_fp8_pt
        )
        sparse_op(
            q_fp8,
            q_scale,
            self.swa_cache_layer.kv_cache,
            self._tile_window(swa_indices),
            output,
            self.scale,
            self.kv_scale,
            swa_lens,
            self.attn_sink,
            extra_cache,
            None if extra_indices is None else _tiled_index_rows(extra_indices),
            extra_lens,
            backend="ixformer_native",
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        positions: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None,
        swa_only: bool = False,
        output: torch.Tensor | None = None,
        selection: CompressedFamilySelection | None = None,
        quantize_output: bool = False,
    ) -> GroupedInt8AttentionOutput | None:
        if output is None and isinstance(swa_metadata, torch.Tensor):
            output = swa_metadata
            swa_metadata = attn_metadata
            attn_metadata = None
            swa_only = self.compress_ratio == 0
        if selection is not None:
            if selection.compact_plan is not None:
                raise ValueError("Compact plans require a pure-prefill family")
            indices, lengths, plan = selection.for_queries(
                slice(0, swa_metadata.num_decode_tokens)
            )
            if (
                getattr(self, "compressed_decode_restore", False)
                and swa_metadata.num_prefills == 0
            ):
                assert swa_metadata.is_valid_token is not None
                lengths = torch.where(
                    swa_metadata.is_valid_token[: swa_metadata.num_decode_tokens],
                    lengths,
                    0,
                )
            if (
                plan is not None
                and selection.plan.restore_start >= swa_metadata.num_decode_tokens
            ):
                plan = plan.scratch
            return self._sparse_mla(
                q,
                swa_metadata.decode_swa_indices,
                swa_metadata.decode_swa_lens,
                kv_cache,
                indices,
                lengths,
                output,
                positions=positions,
                restore_plan=plan,
                quantize_output=quantize_output,
            )
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
            extra_indices, extra_lens = self._global_indices(
                self.topk_indices_buffer[:num_decode_tokens],
                token_to_req[:num_decode_tokens],
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
                positions,
            )
            extra_cache = kv_cache
        if (
            not getattr(swa_metadata, "causal", True)
            and getattr(swa_metadata, "decode_swa_indices", None) is not None
            and swa_metadata.decode_swa_indices.shape[-1] >= 256
        ):
            swa_indices = swa_metadata.decode_swa_indices
            swa_lens = swa_metadata.decode_swa_lens
        elif not getattr(swa_metadata, "causal", True):
            swa_indices, swa_lens = self._window_indices(
                swa_metadata, 0, num_decode_tokens, positions
            )
        else:
            swa_indices = swa_metadata.decode_swa_indices
            swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        kwargs = (
            dict(
                use_fmhadsa=False,
                positions=positions,
                quantize_output=quantize_output,
            )
            if getattr(self._sparse_mla, "__func__", None)
            is IluDeepseekV41Attention._sparse_mla
            else {}
        )
        return self._sparse_mla(
            q,
            swa_indices,
            swa_lens,
            extra_cache,
            extra_indices,
            extra_lens,
            output,
            **kwargs,
        )

    def _forward_prefill_as_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
        selection: CompressedFamilySelection | None = None,
    ) -> None:
        """Prefill through the per-token decode kernel.

        Well-defined because the window and the compressed top-k are sourced
        purely by position, so a prefill token asks for exactly what it would
        ask for as a decode token. The query side is the catch: every token
        becomes its own batch element with ``s_q = 1``, which throws away the
        shared-K reuse a real prefill kernel gets across a request's tokens.

        Causal windows come from ``prefill_swa_indices``, filled once per step
        by ``compute_swa_decode_indices_and_lens``. The non-causal draft
        window still goes through :meth:`_window_indices` because that path
        also applies the replay lower bound.
        """
        swa_only = self.compress_ratio == 0
        if swa_metadata is None:
            swa_metadata = attn_metadata
            attn_metadata = None
        n0 = swa_metadata.num_decode_tokens
        n1 = n0 + swa_metadata.num_prefill_tokens
        if selection is not None:
            indices, lengths, plan = selection.for_queries(slice(n0, n1))
            self._sparse_mla(
                q,
                swa_metadata.prefill_swa_indices,
                swa_metadata.prefill_swa_lens,
                compressed_k_cache,
                indices,
                lengths,
                output,
                positions=positions,
                prefill=True,
                restore_plan=plan,
            )
            return
        if not swa_only and self.compress_ratio not in (1, 2):
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; expected 0, 1, or 2."
            )

        n0 = swa_metadata.num_decode_tokens
        n1 = n0 + swa_metadata.num_prefill_tokens
        if getattr(swa_metadata, "causal", True):
            swa_indices = swa_metadata.prefill_swa_indices
            swa_lens = swa_metadata.prefill_swa_lens
            if swa_indices is None or swa_lens is None:
                raise RuntimeError(
                    "V4.1 causal prefill requires prefill_swa_indices from "
                    "compute_swa_decode_indices_and_lens."
                )
        else:
            swa_indices, swa_lens = self._window_indices(
                swa_metadata, n0, n1, positions
            )
        extra_cache = None
        extra_indices = None
        extra_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            assert swa_metadata.is_valid_token is not None
            assert swa_metadata.token_to_req_indices is not None
            assert self.topk_indices_buffer is not None
            extra_indices, extra_lens = self._global_indices(
                self.topk_indices_buffer[n0:n1],
                swa_metadata.token_to_req_indices[n0:n1],
                attn_metadata.block_table,
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[n0:n1],
                positions,
            )
            extra_cache = compressed_k_cache
        kwargs = (
            dict(
                use_fmhadsa=_v41_prefill_fmhadsa(),
                positions=positions,
                prefill=True,
            )
            if getattr(self._sparse_mla, "__func__", None)
            is IluDeepseekV41Attention._sparse_mla
            else dict(prefill=True)
        )
        self._sparse_mla(
            q,
            swa_indices,
            swa_lens,
            extra_cache,
            extra_indices,
            extra_lens,
            output,
            **kwargs,
        )

    def _gather_k_cache(
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ) -> None:
        """Gather a paged K cache into the bf16 workspace.

        The packed fp8_ds_mla record has to be dequantized on the way out; a
        plain bf16 row is copied as it is.
        """
        if self.compressed_kv_format is not None and k_cache.dtype is torch.uint8:
            raise ValueError(
                "Compressed main KV must use its tile reader instead of a full-pool gather"
            )
        if self.kv_cache_dtype == "fp8_e4m3":
            ixfops.gather_fp8_pt_k_cache(
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                offset,
                kv_scale=self.kv_scale,
                backend="ixformer_native",
            )
            return
        gather = (
            ixfops.gather_packed_fp8_k_cache
            if self.kv_cache_dtype == "fp8_ds_mla"
            else ixfops.gather_k_cache
        )
        gather(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        """Prefill via a gathered workspace and one sparse-MLA kernel.

        Port of V4's ``_forward_prefill`` to the V4.1 ratios (0 / 1 / 2). Per
        chunk of ``PREFILL_CHUNK_SIZE`` requests, a row of the workspace holds
        the compressed pool in ``[0, N)`` and the sliding window at ``[N, ...)``;
        :func:`ds4_combine_topk_swa_indices` then rewrites the indexer's
        per-request compressed row ids and the window into that row's
        coordinates and concatenates them, so one
        :func:`ds4_flash_mla_sparse_fwd` covers both.

        ``N`` is the whole compressed pool, not ``top_k``: the indices the
        indexer produces range over the request's compressed rows, and the
        workspace has to be addressable at all of them. At ratio 1 that makes
        the row ``max_model_len`` wide, which is the price this path pays for
        not touching the paged caches during attention --
        :meth:`_forward_prefill_opt` is the variant that skips the workspace.

        ``attn_metadata`` is ``None`` on a sliding-window-only layer; the loop
        then gathers the window alone and hands the combiner
        ``compress_ratio=1, top_k=0, N=0`` so the layout collapses to pure SWA.
        """
        swa_only = attn_metadata is None
        if not swa_only and self.compress_ratio not in (1, 2):
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; expected 0, 1, or 2."
            )

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        if swa_only:
            # Unused when ``top_k == 0``, but the combiner still shape-checks a
            # 2D int32 buffer. Slice on dim 0 only so it stays contiguous.
            top_k = 0
            N = 0
        else:
            topk_indices = topk_indices[:num_prefill_tokens]
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio

        # Workspace row stride: compressed pool, then the window, then room for
        # the in-batch query rows the sparse kernel reads.
        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        (kv,) = current_workspace_manager().get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start

            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                self._gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            self._gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = ixfops.combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio if not swa_only else 1,
                top_k,
                M,
                N,
                replay_start=swa_metadata.replay_start[
                    num_decodes + chunk_start : num_decodes + chunk_end
                ],
            )

            # ixformer 3817cf75: sparse_mla_prefill → ops 层
            # flash_mla_sparse_fwd（kwargs 全兼容；return_lse 移为纯 kwarg，
            # 本处未用）。
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                output=output[query_start:query_end],
            )

    def _forward_prefill_opt(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        compressed_k_cache: torch.Tensor | None = None,
        attn_metadata: IluDeepseekV4FlashMLAMetadata | None = None,
    ) -> None:
        """Prefill straight off the paged caches, no workspace.

        A sliding-window-only layer is one windowed attention. A compressed
        layer (ratio 1 or 2) is two attentions over disjoint key sets --
        window and compressed top-k -- merged by log-sum-exp, which is exactly
        a single softmax over their union.
        """
        if self.compress_ratio == 0:
            self._prefill_swa_only(q, swa_k_cache, output, swa_metadata)
            return
        if self.compress_ratio not in (1, 2):
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio} in "
                "_forward_prefill_opt; expected 0, 1, or 2."
            )
        assert compressed_k_cache is not None
        assert attn_metadata is not None
        self._prefill_compressed(
            q,
            swa_k_cache,
            output,
            swa_metadata,
            compressed_k_cache,
            attn_metadata,
        )

    def _prefill_swa_only(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        """Sliding-window attention with the sink, written into ``output``."""
        ixfops.swa_attn(
            q=q,
            kv_cache=swa_k_cache,
            block_table=swa_metadata.prefill_swa_block_table,
            cu_seqlens_q=swa_metadata.prefill_cu_seqlens_q,
            cu_seqlens_k=swa_metadata.prefill_cu_seqlens_k,
            max_seqlen_q=swa_metadata.prefill_max_seqlen_q,
            max_seqlen_k=swa_metadata.prefill_max_seqlen_k,
            window_size=self.window_size,
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            output=output,
        )

    def _prefill_compressed(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        compressed_k_cache: torch.Tensor,
        attn_metadata: IluDeepseekV4FlashMLAMetadata,
    ) -> None:
        """Paged sparse-compressed + window attention, merged by LSE."""
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        cu_seqlens_q = swa_metadata.prefill_cu_seqlens_q
        cu_seqlens_k = swa_metadata.prefill_cu_seqlens_k
        prefill_seq_lens = swa_metadata.prefill_seq_lens
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert prefill_seq_lens is not None

        # The sink belongs to the union, so it enters once, at the merge; this
        # half must not fold it into its own denominator.
        o_swa, lse_swa = ixfops.swa_attn(
            q=q,
            kv_cache=swa_k_cache,
            block_table=swa_metadata.prefill_swa_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=swa_metadata.prefill_max_seqlen_q,
            max_seqlen_k=swa_metadata.prefill_max_seqlen_k,
            window_size=self.window_size,
            sm_scale=self.scale,
            attn_sink=None,
        )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + num_prefill_tokens
        ]

        # The kernel resolves the indexer's per-request compressed row ids
        # against the block table itself and derives each token's causal
        # top-k length from cu_seqlens_q / seq_lens / compress_ratio, so no
        # global-index pass is needed here.
        o_cmp, lse_cmp = ixfops.paged_sparse_attn(
            q=q,
            kv_cache=compressed_k_cache,
            block_table=attn_metadata.block_table[num_decodes:],
            topk_indices=topk_indices,
            cu_seqlens_q=cu_seqlens_q,
            seq_lens=prefill_seq_lens,
            compress_ratio=self.compress_ratio,
            block_size=attn_metadata.block_size // self.compress_ratio,
            sm_scale=self.scale,
            return_lse=True,
        )

        ixfops.merge_attn_outputs(
            output,
            o_cmp,
            lse_cmp,
            o_swa,
            lse_swa,
            attn_sink=self.attn_sink,
        )
