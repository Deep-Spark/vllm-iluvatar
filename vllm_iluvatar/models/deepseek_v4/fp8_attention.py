# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4-Flash attention for the *raw FP8* checkpoint.

Module 3 (Stage 2) wires the sparse-MLA forward: the input/output projections
run on the module-2 block-fp8 LinearMethod, the KV path is ``fp8_ds_mla``, and
the heavy ops (fused q-norm/rope/KV-insert, decode/prefill sparse MLA,
compressor, inverse-RoPE, lightning indexer) reuse ixformer ``ds4_*`` /
``dsa_*`` kernels. The computation follows the NV public version; where
ixformer lacks a matching op we use a torch equivalent -- currently only
``wo_a`` (fp8 grouped einsum), dequantized to bf16 (see ``_o_proj``).

To stay on the proven fp8_ds_mla path (identical to the sibling W4A8 fork) the
cache/compressor/indexer holders are the ilu variants
(``IluDeepseekV4SWACache`` / ``IluDeepseekCompressor`` /
``IluDeepseekV4Indexer`` with a bf16 indexer K cache); community
``DeepseekCompressor`` cannot run on Iluvatar (its Triton kernels pass
``launch_pdl``) and the community SWA/indexer backends build NV-shaped
metadata, not the fields the ``ds4_*`` kernels read.

``backend_cls`` is bound directly to ``IluDeepseekV4FlashMLASparseBackend``
(the platform selector would otherwise hand back V3.2's ``FLASHMLA_SPARSE``).
Unlike W4A8 this module does **not** register a torch custom op: ``forward``
calls ``attention_impl`` directly (TODO(ilu_fp8_align_custom_op)).

Deviations from the NV public version are marked inline with
``TODO(ilu_fp8_align_*)`` (indexer bf16 cache, wo_a torch einsum, no head
padding, single-stream, no custom op, prefill bf16 workspace); see
``模块3-attn.md`` §8/§10.

Community counterparts: ``vllm/models/deepseek_v4/attention.py`` (base),
``vllm/models/deepseek_v4/nvidia/flashmla.py`` (FlashMLA),
``vllm/models/deepseek_v4/nvidia/ops/o_proj.py`` (o_proj).
"""

from __future__ import annotations

from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    _e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.models.deepseek_v4.attention import (
    PREFILL_CHUNK_SIZE,
    IluDeepseekCompressor,
    IluDeepseekV4Indexer,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluDeepseekV4FlashMLASparseBackend,
    IluFlashMLASparseMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekV4SWACache,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module 3 (Stage 2): the Stage-1 raise-only ``IluDeepseekV4Fp8IndexerShell``
# is retired. Its whole reason to exist was the DeepGEMM gate in community
# ``DeepseekV4Indexer`` (risk R1); the sibling W4A8 ``IluDeepseekV4Indexer`` is
# already a DeepGEMM-free, bf16-indexer-cache implementation with a fully wired
# ``forward`` (ixformer ``ds4_fused_indexer_q_rope`` + bf16
# ``dsa_indexer_mqa_logits_with_blocks`` + ``dsa_update_topk_indices``), so we
# reuse it verbatim. Its sub-modules (``wq_b`` / ``weights_proj`` /
# ``compressor`` / ``k_cache``) carry the same design §5.2 names/shapes, so the
# checkpoint-loading rules and the ``_EXPECTED_INDEXER`` table are unchanged
# (only the indexer K cache goes fp8-132B -> bf16, plan §1.2). ``wq_b`` still
# takes ``quant_config`` so the raw-FP8 ``indexer.wq_b`` weight loads normally.
#
# See ``模块3-attn.md`` §1.1/§1.2 and TODO(ilu_fp8_align_indexer_fp8): the bf16
# indexer cache is an ixformer-capability downgrade from NV's fp8/fp4 indexer,
# not a functional loss for "跑通 + 单句正确".


def _ds4_sparse_mla_fwd_torch(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Pure-torch equivalent of ixformer ``ds4_flash_mla_sparse_fwd``.

    Gather-based sparse MLA prefill in the absorbed latent space (key == value
    == the gathered kv row, since ``d_qk == d_v == head_dim``). Semantics match
    the FlashMLA sparse kernel (vllm ``flash_mla_sparse_prefill``) plus the two
    ilu-added arguments (per-head attention sink, per-token ``topk_length``):

    * ``q``           : ``(s_q, h_q, d)`` bf16.
    * ``kv``          : ``(s_kv, 1, d)`` bf16 (single MQA kv head).
    * ``indices``     : ``(s_q, 1, topk)`` int; entries ``< 0`` / ``>= s_kv`` and
                        everything at/after ``topk_length`` are invalid (kernel
                        convention: "invalid indices set to -1 or >= s_kv").
    * ``sm_scale``    : softmax scale applied to ``q @ kᵀ``.
    * ``attn_sink``   : ``(h_q,)`` per-head sink logit -- an extra softmax column
                        whose value vector is 0, so it only drains probability
                        mass (matches the kernel's sink slot). Default ``-inf``
                        makes the sink a no-op.
    * ``topk_length`` : ``(s_q,)`` number of valid indices per query token.
    * ``output``      : ``(s_q, h_q, d)`` written in place.

    Correctness-first (plan acceptance = "跑通 + 单句正确"); this deliberately
    ignores the kernel's memory/latency profile (materializes the gathered keys
    and the full ``(s_q, h_q, topk)`` logits). TODO(ilu_fp8_align_sparse_fwd):
    restore ``ixfops.ds4_flash_mla_sparse_fwd`` once validated on corex.
    """
    s_q, h_q, d = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    idx = indices[:, 0, :].long()  # (s_q, topk)
    positions = torch.arange(topk, device=q.device)
    valid = positions[None, :] < topk_length.to(torch.long)[:, None]
    valid = valid & (idx >= 0) & (idx < s_kv)

    idx_safe = idx.clamp_(0, s_kv - 1)
    k = kv[:, 0, :].index_select(0, idx_safe.reshape(-1))
    k = k.reshape(s_q, topk, d).float()  # (s_q, topk, d)

    # scores: (s_q, h_q, topk) = q (s_q,h_q,d) . k (s_q,topk,d)
    scores = torch.einsum("qhd,qkd->qhk", q.float(), k) * float(sm_scale)
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))

    # Attention sink: one extra column per head with a value vector of zeros.
    sink = attn_sink.to(scores.dtype).view(1, h_q, 1).expand(s_q, h_q, 1)
    logits = torch.cat([scores, sink], dim=-1)  # (s_q, h_q, topk + 1)
    weights = torch.softmax(logits, dim=-1)
    # Rows with no valid key and an -inf sink softmax to NaN -> emit zeros.
    weights = torch.nan_to_num(weights, nan=0.0)

    attn = torch.einsum("qhk,qkd->qhd", weights[..., :topk], k)  # sink dropped
    output.copy_(attn.to(output.dtype))


class IluDeepseekV4Fp8Attention(nn.Module, AttentionLayerBase):
    """DeepSeek-V4-Flash MLA attention, raw-FP8 checkpoint (Module 3 wired).

    Exposed contract (pinned by ``tests/model_executor/models/
    test_ilu_dsv4_fp8_attention.py`` and ``..._fp8_kv_spec.py``):

    * ``IluDeepseekV4Fp8Attention(vllm_config, prefix)`` -- positional.
    * ``backend_cls`` is the bound backend class, not a selector call.
    * ``get_padded_num_q_heads`` is a classmethod: ``64 if n <= 64 else 128``
      (kept as a helper; Module 3 does **not** pad -- ``attn_sink`` and ``o``
      carry exactly ``n_local_heads``).
    * ``get_kv_cache_spec`` only reads ``compress_ratio`` / ``head_dim`` /
      ``kv_cache_dtype`` / ``kv_cache_torch_dtype``.
    * ``compress_ratio`` (``1`` for dense layers), ``compressor`` / ``indexer``
      (``None`` when the layer has none), ``n_local_heads``, ``attn_sink``
      (shape ``(n_local_heads,)``).
    """

    # Bound directly -- IluPlatform.get_attn_backend_cls returns V3.2's
    # FLASHMLA_SPARSE for (use_mla and use_sparse), not FLASHMLA_SPARSE_DSV4.
    # Community does the same: attention.py:603 `return self.backend_cls`.
    backend_cls: ClassVar[type[AttentionBackend]] = IluDeepseekV4FlashMLASparseBackend
    # UE8M0 block-scaled fp8 packed as uint8 (`fp8_ds_mla`).
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Community: attention.py:123 (bounds the bf16 kv-gather workspace). Kept so
    # Stage 2 does not have to rediscover the constant.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q / output buffers are allocated at.

        Community: ``nvidia/flashmla.py:58-66`` -- the FP8 decode kernel only
        supports ``h_q in {64, 128}``.
        """
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # alias the compressor keys its k-cache off
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
        # compress_ratios has num_hidden_layers + 1 entries (the trailing one is
        # the MTP layer) and dense layers store 0; both degrade to 1.
        # Community: attention.py:187-190.
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Module 3 / plan §1.3 -- no head padding. ixformer's sparse MLA
        # kernels (``ds4_flash_mla_with_fp8_kvcache`` /
        # ``ds4_flash_mla_sparse_fwd``) accept an arbitrary ``h_q``, so the
        # NV ``padded_heads`` (64/128) slots are dropped: ``attn_sink`` carries
        # exactly ``n_local_heads`` values and ``o`` stays
        # ``(T, n_local_heads, head_dim)``. TODO(ilu_fp8_align_head_pad): if a
        # kernel that requires padding is ever adopted, restore ``padded_heads``
        # and the leading-``n_local_heads``-valid ``attn_sink`` semantics.
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        # ---- quantised linears (community Fp8LinearMethod, zero changes) ----
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
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        # Lazily-built bf16 dequant of the block-fp8 ``wo_a`` weight, reshaped to
        # the grouped ``(n_local_groups, o_lora_rank, heads_per_group*head_dim)``
        # layout the ``_o_proj`` einsum consumes (see ``_wo_a_bf16_view``).
        self._wo_a_bf16: torch.Tensor | None = None
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # cos_sin_cache only -- no kernel launch at build time.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        # Only C4A layers run sparse attention and therefore own an indexer.
        # Community: attention.py:258-273.
        self.indexer: IluDeepseekV4Indexer | None = None
        if self.compress_ratio == 4:
            self.indexer = self._build_indexer(
                vllm_config,
                config=config,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.indexer",
            )

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # `--kv-cache-dtype fp8_ds_mla` is a hard precondition: the community
        # resolver asserts `kv_cache_dtype.startswith("fp8")` and normalises the
        # string back onto cache_config so the page specs pick the 576B slot.
        # Community: attention.py:68-101 (reused verbatim, not re-implemented).
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        # Module 3: use the ilu SWA cache (its backend builds the
        # ``DeepseekSparseSWAMetadata`` fields -- ``decode_swa_indices`` /
        # ``prefill_gather_lens`` / ``slot_mapping`` etc. -- that the ixformer
        # ``ds4_*`` kernels read). It handles the ``fp8_ds_mla`` uint8 layout,
        # matching the proven W4A8 fp8 KV path.
        self.swa_cache_layer = IluDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with the compilation context for metadata lookup.
        # Community: attention.py:301-307.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Built after the attention setup so the compressor sees self.prefix.
        # Community: attention.py:308-320.
        #
        # Module 3 / plan §2: use ``IluDeepseekCompressor`` (bf16 params
        # identical to community, so checkpoint loading is unchanged) rather
        # than community ``DeepseekCompressor``. The community forward launches
        # upstream Triton kernels with ``launch_pdl=False``, which Iluvatar
        # Triton rejects; the ilu fork routes the two steps through
        # ``ixfops.ds4_compressor_*`` and, because ``kv_cache_dtype ==
        # "fp8_ds_mla"`` here, writes the packed 584B main-MLA cache via
        # ``ds4_compressor_pool_norm_rope_fp8_kv_insert``. It reads
        # ``kv_cache_dtype`` off the layer registered at ``k_cache_prefix``
        # (this attention: ``self.prefix``), which carries ``self.kv_cache_dtype``.
        self.compressor: IluDeepseekCompressor | None = None
        if self.compress_ratio > 1:
            self.compressor = IluDeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )
            self.compressor.kv_cache_dtype = self.kv_cache_dtype

    # ------------------------------------------------------------------ #
    # sub-module construction
    # ------------------------------------------------------------------ #
    # Module 3: build the sibling W4A8 ``IluDeepseekV4Indexer`` (plan §3). It is
    # DeepGEMM-free (the R1 gate that forced the Stage-1 shell lives only in
    # community ``DeepseekV4Indexer`` -> ``SparseAttnIndexer``), stores a bf16
    # indexer K cache, and has a fully wired ``forward``. Its ``wq_b`` takes
    # ``quant_config`` so the raw-FP8 ``indexer.wq_b`` weight still loads; every
    # other sub-module name/shape matches design §5.2, so checkpoint loading and
    # the ``_EXPECTED_INDEXER`` table are unchanged.
    def _build_indexer(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        quant_config,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> IluDeepseekV4Indexer:
        """Build the lightning indexer for a ``compress_ratio == 4`` layer."""
        return IluDeepseekV4Indexer(
            vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=self.topk_indices_buffer,
            compress_ratio=self.compress_ratio,
            prefix=prefix,
        )

    # ------------------------------------------------------------------ #
    # backend / KV cache spec
    # ------------------------------------------------------------------ #
    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Whether this instance stores fp8 KV in the ``fp8_ds_mla`` layout."""
        return self.use_fp8_ds_mla_layout

    def get_attn_backend(self) -> type[AttentionBackend]:
        # Community: attention.py:602-603 -- deliberately bypasses the platform
        # backend selector.
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Main MLA spec. Community: ``attention.py:605-625``.

        Reads only ``compress_ratio`` / ``head_dim`` / ``kv_cache_dtype`` /
        ``kv_cache_torch_dtype`` so it stays callable against a duck-typed
        ``self`` (see ``test_ilu_dsv4_fp8_kv_spec.py``).
        """
        if self.compress_ratio <= 1:
            # SWA-only layer: its cache is allocated by DeepseekV4SWACache.
            return None
        # Module 3: the fp8_ds_mla main MLA cache is read/written by the same
        # ixformer ``ds4_*`` kernels as the sibling W4A8 fork, which require
        # 64B-aligned block bases (W4A8 ``attention.py`` ``get_kv_cache_spec``;
        # C128 pages 2*584=1168 pad to 1216). We match that value instead of
        # NV's 576 so the page algebra is identical to the proven fp8 path.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=64 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    # ------------------------------------------------------------------ #
    # forward path -- Stage 2 (design §7 group A)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse MLA attention forward (raw-FP8 checkpoint, fp8_ds_mla KV).

        NV-aligned pipeline (community
        ``vllm/models/deepseek_v4/attention.py:324-371`` + its
        ``nvidia`` FlashMLA/o_proj ops), implemented with ixformer ``ds4_*``
        kernels on the fp8 KV path (plan §2). The block-fp8 input projections
        (``fused_wqa_wkv`` / ``wq_b`` / ``wo_b``) run through the module-2
        FP8 block-scaled LinearMethod transparently; only ``wo_a`` needs a
        torch-equivalent (``_o_proj``).

        Unlike the sibling W4A8 wrapper this does **not** register / dispatch
        through ``torch.ops.vllm.ilu_deepseek_v4_attention`` (plan §7,
        TODO(ilu_fp8_align_custom_op)); ``attention_impl`` is called directly
        so this module never re-registers an op name owned by W4A8.
        """
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        num_tokens = hidden_states.shape[0]
        # No head-count padding (plan §1.3): ``o`` carries exactly
        # ``n_local_heads`` heads; ixformer sparse MLA kernels accept any h_q.
        o = torch.empty(
            (num_tokens, self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self.attention_impl(hidden_states, qr, kv, positions, o)

        return self._o_proj(o, positions)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Sequential sparse-MLA body (NV logic, ixformer kernels).

        Mirrors the W4A8 ``attention_impl`` but on the fp8_ds_mla KV path and
        with no aux-stream overlap (plan §1.4 / TODO(ilu_fp8_align_single_stream)
        -- indexer / compressor / kv-insert run in order on the default stream).
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # 1) Fused q/kv RMSNorm (learnable weights) -> ixformer fused kernel.
        qr, kv = ixfops.rms_norm_qk(
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.eps,
        )

        # 2) Q projection (block-fp8 Linear): (T, q_lora) -> (T, H, head_dim).
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # 3) Per-layer-type sequencing (plan §2 branches).
        if self.indexer is not None:
            # C4A: indexer (own bf16 compressor + topk) + main compressor
            #      (writes fp8 main MLA cache) + SWA fp8 insert.
            self.indexer(hidden_states, qr, positions, self.indexer_rotary_emb)
            assert self.compressor is not None
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        elif self.compressor is not None:
            # C128A: main compressor + SWA fp8 insert (no indexer).
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        else:
            # SWA-only: just the fused Q/KV norm+rope+insert.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # 4) Dummy / profiling run: reserve the prefill workspace at the real
        #    execution shape, zero the output, and return (matches W4A8).
        if not isinstance(attn_metadata, dict):
            swa_only = self.compress_ratio <= 1
            if swa_only:
                N = 0
            else:
                N = (
                    self.max_model_len + self.compress_ratio - 1
                ) // self.compress_ratio
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            out.zero_()
            return

        # 5) All KV caches are now written -- push to a KV connector if present.
        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            connector.save_kv_layer("layer_name", "kv_cache", "attn_metadata")

        # 6) Sparse MLA forward into the pre-allocated ``out`` (T, H, head_dim).
        self.forward_mqa(q, kv, positions, out)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA forward: split ``[decode | prefill]`` and dispatch.

        Community: ``nvidia/flashmla.py`` ``forward_mqa`` (decode via
        ``flash_mla_with_kvcache`` + chunked prefill via
        ``flash_mla_sparse_fwd``). ``kv`` is unused here -- the rotated K rows
        were already written into the SWA (and, for cr>1, main MLA) caches by
        the compressor / ``_fused_qnorm_rope_kv_insert`` step; it is accepted
        only to keep the NV call signature.
        """
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict), (
            "fp8 V4 MLA forward_mqa requires a dict-shaped attn_metadata; got "
            f"{type(attn_metadata).__name__}. Profiling/dummy runs are "
            "short-circuited by attention_impl before reaching here."
        )

        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "Missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; the SWA KVCacheGroup must be "
            "registered alongside the main MLA cache."
        )

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=self.swa_cache_layer.kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata,
        attn_metadata: IluFlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Decode-side sparse MLA (fp8_ds_mla). Community ``_forward_decode``.

        Three-branch topk sourcing (SWA-only / C4A local->global / C128A
        pre-computed) is identical to the bf16 W4A8 path; only the kernel is
        the fp8 variant ``ds4_flash_mla_with_fp8_kvcache``.
        """
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices: torch.Tensor | None = None
        topk_lens: torch.Tensor | None = None
        if not swa_only:
            assert attn_metadata is not None, (
                "FlashMLASparseMetadata is required when compress_ratio > 1; "
                f"prefix={self.prefix!r}"
            )
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = (
                    ixfops.ds4_compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices[:num_decode_tokens],
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            elif self.compress_ratio == 128:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Decode treats every query token as its own batch element.
        q = q.unsqueeze(1)

        swa_cache = self.swa_cache_layer.kv_cache
        ixfops.ds4_flash_mla_with_fp8_kvcache(
            q=q,
            k_cache=swa_cache,
            indices=swa_indices,
            topk_length=swa_lens,
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
            output=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata | None,
        swa_metadata,
    ) -> None:
        """Prefill-side sparse MLA (fp8_ds_mla). Community ``_forward_prefill``.

        Chunk by ``PREFILL_CHUNK_SIZE``; dequant-gather the compressed + SWA
        segments into a shared bf16 workspace (``ds4_gather_fp8_k_cache``),
        combine the indexer local topk with the SWA window
        (``ds4_combine_topk_swa_indices``), then sparse MLA prefill
        (``ds4_flash_mla_sparse_fwd``). See plan §3 and
        TODO(ilu_fp8_align_prefill_dequant): fp8 KV is dequantized to a bf16
        512-dim workspace before the sparse attention (ixformer prefill path).
        """
        swa_only = attn_metadata is None

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

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            elif self.compress_ratio == 128:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
                assert topk_indices is not None
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        (kv,) = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start

            # 1) Compressed K segment (dequant fp8 -> bf16 workspace [0:N)).
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                ixfops.ds4_gather_fp8_k_cache(
                    out=kv[:chunk_size],
                    k_cache=compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # 2) SWA K segment (dequant fp8 -> bf16 workspace [N:N+gather_len)).
            swa_block_table = swa_metadata.block_table[num_decodes:]
            ixfops.ds4_gather_fp8_k_cache(
                out=kv[:chunk_size],
                k_cache=swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # 3) Combine local topk + SWA window into workspace-row coords.
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = ixfops.ds4_combine_topk_swa_indices(
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
            )

            # 4) Sparse MLA prefill: read bf16 workspace, write output slice.
            # TODO(ilu_fp8_align_sparse_fwd): torch equivalent of
            # ``ixfops.ds4_flash_mla_sparse_fwd`` (see ``_ds4_sparse_mla_fwd_torch``).
            _ds4_sparse_mla_fwd_torch(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                output=output[query_start:query_end],
            )

    def _wo_a_bf16_view(self) -> torch.Tensor:
        """Dequantize the block-fp8 ``wo_a`` weight to the grouped bf16 layout.

        Plan §5 / TODO(ilu_fp8_align_wo_a_gemm): NV runs ``wo_a`` as a fp8
        grouped einsum (``deep_gemm.fp8_einsum`` with fp8-quantized activation).
        ixformer has no fp8 grouped einsum, so we dequantize the weight once
        (fp8 E4M3 + block ``weight_scale_inv`` -> bf16) and run a bf16
        ``einsum`` in ``_o_proj`` (activation stays bf16). Result is cached: the
        weight is static after load. Layout mirrors the W4A8 bf16 ``wo_a`` view:
        ``(n_local_groups, o_lora_rank, heads_per_group * head_dim)``.
        """
        if self._wo_a_bf16 is None:
            w = self.wo_a.weight  # fp8 e4m3, [N, K] = [n_groups*o_lora, hpg*hd]
            ws = getattr(self.wo_a, "weight_scale_inv", None)
            if ws is None:
                ws = getattr(self.wo_a, "weight_scale", None)
            assert ws is not None, (
                "wo_a has no block weight scale (weight_scale_inv/weight_scale); "
                "cannot dequantize the fp8 grouped weight."
            )
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _e8m0_to_fp32(ws)
            ws = ws.to(torch.float32)

            N, K = w.shape
            # DeepSeek-V4 block quant is (128, 128); broadcast the block scale
            # back to full [N, K] then dequant (same as module-2's kernel).
            bn, bk = 128, 128
            ws_full = ws.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
            ws_full = ws_full[:N, :K]
            w_bf16 = (w.to(torch.float32) * ws_full).to(torch.bfloat16)

            heads_per_group = self.n_local_heads // self.n_local_groups
            self._wo_a_bf16 = w_bf16.view(
                self.n_local_groups,
                self.o_lora_rank,
                heads_per_group * self.head_dim,
            )
        return self._wo_a_bf16

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + grouped ``wo_a`` + row-parallel ``wo_b``.

        Community: ``nvidia/flashmla.py`` ``deep_gemm_fp8_o_proj``. We reuse
        ixformer ``ds4_inv_rope`` (rotates the RoPE band of ``o`` in place),
        then run the ``wo_a`` grouped projection as a bf16 ``einsum`` against the
        dequantized weight (see ``_wo_a_bf16_view`` and
        TODO(ilu_fp8_align_wo_a_gemm) -- no fp8 activation quant). ``wo_b`` is a
        normal block-fp8 RowParallelLinear (module 2).
        """
        num_tokens = o.shape[0]

        ixfops.ds4_inv_rope(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            inplace=True,
        )

        heads_per_group = self.n_local_heads // self.n_local_groups
        o_view = o.view(
            num_tokens, self.n_local_groups, heads_per_group * self.head_dim
        )
        w_view = self._wo_a_bf16_view()
        z = torch.einsum("bhr,hdr->bhd", o_view, w_view)

        return self.wo_b(z.flatten(1))

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        """Fused per-head q RMSNorm + q/kv RoPE + fp8 KV paged insert (SWA).

        Community: ``vllm/models/deepseek_v4/attention.py:513-599`` (the
        ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` fp8 sibling). On
        the fp8_ds_mla path ``ds4_fused_qnorm_rope_fp8_kv_insert`` writes the
        584B packed SWA cache: q gets weightless per-head RMSNorm + GPT-J RoPE
        (last ``rope_head_dim`` dims), kv NoPE(448) -> fp8 E4M3 + UE8M0 scale and
        RoPE(64) -> bf16. Tokens with ``slot_mapping[t] < 0`` are skipped on the
        KV side inside the kernel. ``q`` and the SWA cache are mutated in place.
        """
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "fp8 V4 attention_impl: missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; check that the SWA KVCacheGroup "
            "is registered alongside the main MLA cache."
        )

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # 3D uint8 cache passed directly: the fp8 insert kernel uses stride(0)
        # and must not flatten padded (non-contiguous) blocks.
        ixfops.ds4_fused_qnorm_rope_fp8_kv_insert(
            q,
            kv,
            swa_kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
