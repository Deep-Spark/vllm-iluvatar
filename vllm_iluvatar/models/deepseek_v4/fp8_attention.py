# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4-Flash attention skeleton for the *raw FP8* checkpoint.

Stage 1 scope (spec ``deepseek-v4-fp8-weight-loading``, task 1.3): build every
weight the checkpoint carries, expose the KV-cache specs, and stop at the first
forward entry. No kernel is wired, the vendor kernel package is not imported,
and no torch custom op is registered -- the sibling W4A8 ``attention.py`` owns
``vllm::ilu_deepseek_v4_attention``, and re-registering any op name from this
module would break worker start-up (both modules are imported eagerly by
``ilu_register._iter_package_modules``).

Design references (``.kiro/specs/deepseek-v4-fp8-weight-loading/design.md``):

* §4.1 -- self-written class instead of subclassing community
  ``DeepseekV4Attention``: the four abstract seams do not reach the fused
  kv-insert ops, the multi-stream GEMM orchestration or the deep_gemm
  ``_o_proj``.
* §4.2 -- the three cache holders (``DeepseekV4SWACache`` /
  ``DeepseekV4IndexerCache`` / ``CompressorStateCache``) and
  ``DeepseekCompressor`` are reused verbatim; the lightning **indexer** is a
  self-written weights-only shell (``IluDeepseekV4Fp8IndexerShell``) because
  community ``DeepseekV4Indexer`` hard-constructs ``SparseAttnIndexer``, whose
  ``__init__`` refuses to run without DeepGEMM (risk R1, confirmed on device).
  ``backend_cls`` is bound directly to ``IluDeepseekV4FlashMLASparseBackend``
  because the platform selector would hand back V3.2's ``FLASHMLA_SPARSE``
  instead.
* §5.2 -- parameter names / shapes.
* §6.1 -- component checklist. ``attn_gemm_parallel_execute`` /
  ``attention_impl`` / aux streams / ``ln_events`` are deliberately absent.
* §7 -- TODO numbering ``A1``..``A4`` plus ``B2`` (indexer forward).

Community counterpart: ``vllm/models/deepseek_v4/attention.py`` (base) and
``vllm/models/deepseek_v4/nvidia/flashmla.py`` (FlashMLA subclass).
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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
from vllm.models.deepseek_v4.attention import (
    DeepseekV4IndexerCache,
    _resolve_dsv4_kv_cache_dtype,
)
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluDeepseekV4FlashMLASparseBackend,
)

logger = init_logger(__name__)


class IluDeepseekV4Fp8IndexerShell(nn.Module):
    """Lightning-indexer *weights only*, for ``compress_ratio == 4`` layers.

    Why this exists instead of community ``DeepseekV4Indexer``
    ---------------------------------------------------------
    Community ``DeepseekV4Indexer.__init__``
    (``vllm/models/deepseek_v4/attention.py:668-771``) hard-constructs
    ``SparseAttnIndexer`` at ``attention.py:753``, and
    ``SparseAttnIndexer.__init__`` ends with a hard gate
    (``vllm/model_executor/layers/sparse_attn_indexer.py:750-754``)::

        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment.")

    ``IluPlatform`` derives from ``CudaPlatform`` so ``is_cuda()`` is True,
    while DeepGEMM is an NVIDIA-only package that the Iluvatar container will
    never carry. That makes the gate structural, not a missing dependency --
    see the ``FALLBACK ENABLED`` note on
    ``IluDeepseekV4Fp8Attention._build_indexer`` for the observed failure.

    What it does build (design §5.2, ``indexer.*`` rows -- names and shapes are
    identical to community, because both the checkpoint loading rules and the
    expectation table in ``test_ilu_dsv4_fp8_attention.py::_EXPECTED_INDEXER``
    are written against them):

    * ``wq_b``        -- ``ReplicatedLinear(q_lora_rank, n_head * head_dim)``,
      quantised (``weight`` + ``weight_scale_inv``).
    * ``weights_proj``-- ``ReplicatedLinear(hidden_size, n_head)``, bf16.
    * ``k_cache``     -- community ``DeepseekV4IndexerCache``, ``head_dim`` 132.
    * ``compressor``  -- community ``DeepseekCompressor`` at ``head_dim`` 128.

    What it does *not* build: ``SparseAttnIndexer`` (the blocked point) and the
    ``ln_events`` CUDA events (design §6.1 keeps Stage 1 free of aux streams).
    Every scalar community keeps is kept here under the same name and meaning so
    Stage 2 can wire the op without renaming anything.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        # --- scalars, community attention.py:683-727 (same names, same meaning)
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk  # 512
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = vllm_config.attention_config.use_fp4_indexer_cache
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )
        self.prefix = prefix
        # Kept for signature parity only; Stage 1 owns no aux stream, and the
        # two `torch.cuda.Event()` objects community allocates here are skipped.
        self.aux_stream = aux_stream

        # --- weights, community attention.py:699-713 (no tensor parallel) ---
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

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # FP8 indexer cache reuses the V3.2 layout: 128 fp8 + 4B fp32 scale.
        # Community: attention.py:730-741.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        # head_dim=128 is one of the two values DeepseekCompressor accepts
        # (compressor.py:284/:291 branches, :300-303 raises for anything else).
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        # Community: vllm/models/deepseek_v4/attention.py:773-807
        # (DeepseekV4Indexer.forward -- wq_b + fused_indexer_q_rope_quant, the
        # compressor branch, then `self.indexer_op(...)`). The final
        # `SparseAttnIndexer` call is what this shell skips: its __init__ gate at
        # vllm/model_executor/layers/sparse_attn_indexer.py:750-754 demands
        # DeepGEMM, so Stage 2 has to supply an Iluvatar-side equivalent of
        # `sparse_attn_indexer` (top-k logits over the packed indexer k-cache).
        raise NotImplementedError(
            "TODO(ilu_fp8_B2): lightning indexer forward is not wired in "
            "Stage 1 (weight loading only); the community path needs DeepGEMM."
        )


class IluDeepseekV4Fp8Attention(nn.Module, AttentionLayerBase):
    """DeepSeek-V4-Flash MLA attention, raw-FP8 checkpoint, weights only.

    Exposed contract (pinned by ``tests/model_executor/models/
    test_ilu_dsv4_fp8_attention.py`` and ``..._fp8_kv_spec.py``):

    * ``IluDeepseekV4Fp8Attention(vllm_config, prefix)`` -- positional.
    * ``backend_cls`` is the bound backend class, not a selector call.
    * ``get_padded_num_q_heads`` is a classmethod: ``64 if n <= 64 else 128``.
    * ``get_kv_cache_spec`` only reads ``compress_ratio`` / ``head_dim`` /
      ``kv_cache_dtype`` / ``kv_cache_torch_dtype``.
    * ``compress_ratio`` (``1`` for dense layers), ``compressor`` / ``indexer``
      (``None`` when the layer has none), ``n_local_heads``, ``padded_heads``,
      ``attn_sink``.
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

        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # -inf == no sink effect. load_weights fills the first n_local_heads
        # slots and leaves the rest at -inf. Community: attention.py:196-203.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
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
        self.indexer: IluDeepseekV4Fp8IndexerShell | None = None
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

        self.swa_cache_layer = DeepseekV4SWACache(
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
        self.compressor: DeepseekCompressor | None = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    # ------------------------------------------------------------------ #
    # sub-module construction
    # ------------------------------------------------------------------ #
    # FALLBACK ENABLED (design risk R1 / requirement 9.4) -- 2026-xx, task 1.4.
    #
    # Requirement 9.1 asks Stage 1 to reuse community `DeepseekV4Indexer` first
    # and to fall back only once a construction-time failure is observed. It was
    # observed. Running
    #   pytest tests/model_executor/models/test_ilu_dsv4_fp8_attention.py -v
    # in the Iluvatar container gave 8 passed / 8 failed: every dense and
    # cr==128 assertion passed, and all eight cr==4 cases died on the same
    # stack (line numbers below are from the pre-fallback revision of this
    # file) --
    #   fp8_attention.py:209 __init__
    #     -> fp8_attention.py:293 _build_indexer
    #       -> DeepseekV4Indexer(...)
    #         -> vllm/models/deepseek_v4/attention.py:753 SparseAttnIndexer(...)
    #           -> vllm/model_executor/layers/sparse_attn_indexer.py:751
    #   RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM support
    #                 in the current vLLM environment.
    # (the gate is `if current_platform.is_cuda() and not has_deep_gemm()`).
    # `IluPlatform` derives from `CudaPlatform`, so `is_cuda()` is True, while
    # DeepGEMM is NVIDIA-only and will not exist in an Iluvatar container --
    # a structural block, not a missing package, so installing something cannot
    # clear it.
    #
    # Fallback taken: build `IluDeepseekV4Fp8IndexerShell` instead. It creates
    # the same four sub-modules (`wq_b` / `weights_proj` / `k_cache` /
    # `compressor`) with the design §5.2 names and shapes, keeps every community
    # scalar, skips `SparseAttnIndexer`, and raises the B2 TODO on forward
    # (design §7 group B). This method stays the single construction entry
    # point: if the DeepGEMM gate is ever removed upstream, or Iluvatar ships an
    # equivalent implementation, swapping the shell back for community
    # `DeepseekV4Indexer` is a one-line change here and nothing outside this
    # method moves.
    def _build_indexer(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        quant_config,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> IluDeepseekV4Fp8IndexerShell:
        """Build the lightning indexer for a ``compress_ratio == 4`` layer.

        Community counterpart: ``attention.py:261-272`` (the
        ``DeepseekV4Indexer`` construction). ``aux_stream`` stays ``None``:
        Stage 1 owns no aux streams (design §6.1 excludes them).
        """
        return IluDeepseekV4Fp8IndexerShell(
            vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=self.topk_indices_buffer,
            compress_ratio=self.compress_ratio,
            prefix=prefix,
            aux_stream=None,
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
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
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
        # Community: vllm/models/deepseek_v4/attention.py:324-371
        # (input GEMMs + fused q/kv RMSNorm + attention_impl + _o_proj).
        raise NotImplementedError(
            "TODO(ilu_fp8_A1): IluDeepseekV4Fp8Attention.forward is not wired "
            "in Stage 1 (weight loading only)."
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Community: vllm/models/deepseek_v4/nvidia/flashmla.py:68-143
        # (sparse MLA decode via flash_mla_with_kvcache + chunked prefill via
        # flash_mla_sparse_fwd).
        raise NotImplementedError(
            "TODO(ilu_fp8_A2): sparse MLA decode/prefill is not wired in "
            "Stage 1 (weight loading only)."
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Community: vllm/models/deepseek_v4/nvidia/flashmla.py:42-56
        # (deep_gemm_fp8_o_proj: inverse-RoPE + fp8 quant + wo_a bmm + wo_b).
        raise NotImplementedError(
            "TODO(ilu_fp8_A3): inverse-RoPE + wo_a/wo_b output projection is "
            "not wired in Stage 1 (weight loading only)."
        )

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        # Community: vllm/models/deepseek_v4/attention.py:513-599
        # (torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert and
        # its bf16 / per-tensor-fp8 siblings).
        raise NotImplementedError(
            "TODO(ilu_fp8_A4): fused q-RMSNorm + RoPE + KV quant + paged "
            "insert is not wired in Stage 1 (weight loading only)."
        )
