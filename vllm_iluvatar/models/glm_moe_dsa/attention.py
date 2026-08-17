# SPDX-License-Identifier: Apache-2.0
"""Self-contained GLM-5.2 (GlmMoeDsa = DeepSeek-V3.2) MLA + DSA attention.

Aligned with ``vllm_iluvatar.models.deepseek_v4.attention``: the
whole attention *algorithm* chain lives in the plugin, so GLM never dispatches
through the community ``MultiHeadLatentAttentionWrapper`` / ``MLAAttention`` /
community ``SparseMLAAttentionImpl``. Only *infrastructure* is reused
(``*Linear`` / ``RMSNorm`` / ``get_rope`` / the fork ``IluFlashMLASparseBackend``
metadata, ``IluDeepseekV32IndexerBackend`` cache and the ixformer / FlashMLA
sparse kernels). The DSA top-k step uses the plugin ``SparseAttnIndexer``
registered as ``vllm::ilu_sparse_attn_indexer``.

Three layers, mirroring V4:

- :class:`IluGlmMoeDsaMLAAttention` -- outer wrapper. Owns the projections
  (``fused_qkv_a_proj`` / ``q_b_proj`` / ``kv_b_proj`` / ``o_proj``), the norms,
  the DSA indexer and ``o_proj``; runs the ``forward_opt``-equivalent pre/post
  processing. Analogue of ``IluDeepseekV4MultiHeadLatentAttentionWrapper``.
- :class:`IluGlmMoeDsaSparseMLA` -- inner sparse MLA (``AttentionLayerBase``).
  Owns ``get_attn_backend`` / ``get_kv_cache_spec`` and drives rope + absorbed
  k-up + KV insert + sparse MLA + v-up. Default KV is bf16 latent (576) via
  ``concat_and_cache_mla`` + ``dsa_flash_mla_sparse_fwd``. With
  ``--kv-cache-dtype fp8`` it uses the GLM DSA 640-byte packed path
  (``glm_dsa_fp8_kv_insert`` + ``dsa_flash_mla_with_fp8_kvcache`` /
  ``sparse_flash_attn_fp8``). The absorbed up-proj is grouped W8A16 marlin:
  k-up / v-up read ``kv_b_proj.W_UK`` / ``W_UV`` (repacked at load time by
  ``_IluGlmKVBProjW8A16Method``) and call ``ixf_ops.marlin_w8a16``. Slimmed
  fork of ``IluFlashMLASparseImpl`` (GLM/V3.2 only: no compress_ratio / SWA /
  attn_sink / C128A / DCP / quantized-output paths). Analogue of
  ``IluDeepseekV4MLAAttention``.
- :class:`IluGlmMoeDsaIndexer` / :class:`IluGlmMoeDsaIndexerCache` -- the DSA
  lightning indexer, driving the shared plugin ``vllm::ilu_sparse_attn_indexer``
  op.

The ``q_a_proj`` + ``kv_a_proj_with_mqa`` fusion is done at *load time* via a
``MergedColumnParallelLinear`` named ``fused_qkv_a_proj`` (see ``model.py``
``load_weights``); there is no runtime ``_inject_merge_qkv_a`` monkeypatch.
"""

import json
import math
from functools import lru_cache
from types import SimpleNamespace

import ixformer.inference.functions as ixf_ops
import torch
import torch.nn as nn
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
    IluFlashMLASparseMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerBackend,
)

# GLM DSA fp8 packed latent: 512 e4m3 NoPE + 64 bf16 RoPE = 640 bytes.
_GLM_DSA_FP8_ROW_BYTES = 640
_GLM_DSA_FP8_Q_HEADS = 64
_GLM_DSA_FP8_E4M3_MAX = 448.0

logger = init_logger(__name__)


@lru_cache(maxsize=4)
def _load_glm_dsa_fp8_kv_scale_table(path: str) -> dict[str, float]:
    """Load per-layer GLM DSA fp8 KV scales: ``{layer.self_attn.attn: {kv: float}}``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH={path!r}: expected a JSON "
            f"object of layer -> {{'kv': float}}, got {type(raw).__name__}"
        )
    out: dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "kv" in val:
            out[key] = float(val["kv"])
        elif isinstance(val, (int, float)):
            out[key] = float(val)
        else:
            raise ValueError(
                f"VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH={path!r}: entry "
                f"{key!r} must be {{'kv': float}} or a float, got {val!r}"
            )
    logger.info("Loaded %d GLM DSA fp8 KV scales from %s", len(out), path)
    return out


def _resolve_glm_dsa_fp8_kv_scale(prefix: str) -> tuple[float | None, str]:
    """Resolve GLM DSA fp8 KV dequant scale for one attention layer.

    Priority:
      1. ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE`` — a single float shared by **all**
         layers (not per-layer). Prefer this only for quick bring-up.
      2. ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH`` — JSON table keyed by layer
         prefix (``{prefix: {"kv": float}}`` or ``{prefix: float}``).
      3. Otherwise ``None`` → online ``amax/448`` on the first eager KV insert
         (skipped during CUDA-graph capture; prefer 1/2 for production).
    """
    scale_env = envs.VLLM_ILU_GLM_DSA_FP8_KV_SCALE
    if scale_env.strip():
        return float(scale_env), "VLLM_ILU_GLM_DSA_FP8_KV_SCALE"

    path = envs.VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH
    if path.strip():
        table = _load_glm_dsa_fp8_kv_scale_table(path.strip())
        candidates = [
            prefix,
            prefix.removeprefix("model."),
            f"model.{prefix}" if not prefix.startswith("model.") else prefix,
        ]
        for cand in candidates:
            if cand in table:
                return table[cand], f"{path}:{cand}"
        logger.warning(
            "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH=%r has no entry for %r "
            "(tried %s); falling back to online amax/448 calibration",
            path,
            prefix,
            candidates,
        )
        return None, "online_amax_missing_json_key"

    return None, "online_amax"


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class IluGlmMoeDsaIndexerCache(torch.nn.Module, AttentionLayerBase):
    """BF16 lightning-indexer K cache for GLM-5.2 (ilu W4A8).

    Mirrors the community ``DeepseekV32IndexerCache``: a single-vector paged
    MLA cache whose backend is :class:`IluDeepseekV32IndexerBackend`.
    """

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV32IndexerBackend


class IluGlmMoeDsaIndexer(nn.Module):
    """DSA lightning indexer for GLM-5.2 (ilu W4A8).

    Only built on ``full`` indexer layers. ``wq_b`` is W8A8 INT8; ``wk`` +
    ``weights_proj`` are fused into a single BF16 ``wk_weights_proj`` (the
    GLM checkpoint ships both as BF16, so no FP8 upcast path is needed). Topk
    is computed by the shared plugin ``SparseAttnIndexer`` (which writes into
    the ``topk_indices_buffer`` consumed by the sparse MLA layer).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # q projection for the indexer (W8A8 INT8, matches dense_linears
        # target ``indexer.wq_b``).
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: one BF16 GEMM producing
        # [head_dim + n_head]. Both source tensors are BF16 in the GLM ckpt.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self.n_head_scale = self.n_head**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.k_cache = IluGlmMoeDsaIndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = IluSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        # Fused wk + weights_proj: one GEMM, then split.
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce leading dims during compilation; reshape
        # back to token-flattened shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)

        weights = weights.unsqueeze(-1) * self.softmax_scale * self.n_head_scale
        weights = weights.squeeze(-1)

        return self.indexer_op(
            hidden_states,
            q,
            k,
            weights,
        )


class IluGlmMoeDsaSparseMLA(nn.Module, AttentionLayerBase):
    """Sparse MLA inner attention for GLM-5.2 (ilu W4A8).

    Self-contained slim fork of ``IluFlashMLASparseImpl`` -- keeps only the
    GLM/V3.2 path: rope + absorbed k-up + KV insert + sparse MLA (single MQA
    path for prefill & decode) + v-up. Drops compress_ratio / SWA / attn_sink /
    C128A / DCP / quantized output. Registered as an ``AttentionLayerBase`` so
    the runner allocates its paged KV cache and builds
    :class:`IluFlashMLASparseMetadata` keyed on ``prefix``.

    KV cache modes:
      * default / ``bfloat16`` — 576-wide bf16 latent via
        ``concat_and_cache_mla`` + ``dsa_flash_mla_sparse_fwd``.
      * ``fp8`` / ``fp8_e4m3`` / ``fp8_glm_dsa`` — 640-byte packed rows
        (512 e4m3 NoPE + 64 bf16 RoPE) via ``glm_dsa_fp8_kv_insert`` and
        either ``dsa_flash_mla_with_fp8_kvcache`` (local q_heads <= 32) or
        ``sparse_flash_attn_fp8`` (64-head tile). Not DeepSeek-V4
        ``fp8_ds_mla`` (584B).

    ``kv_b_proj`` is grouped W8A16 marlin: :class:`IluGlmKVBProjW8A16Method `
    (in ``compressed-tensors`` / ``IluQuantConfig``) splits it into ``W_UK`` / ``W_UV`` and repacks
    each into Marlin layout at load time (deterministic, loader-driven). The
    k-up / v-up here just read ``self.kv_b_proj.W_UK`` / ``W_UV`` and call
    ``ixf_ops.marlin_w8a16`` -- mirroring R1's ``_k_up_proj_w8a8`` /
    ``_v_up_proj_w8a8`` (``vllm/v1/attention/backends/mla/ilu_mla.py``). There
    is no BF16 absorb fallback (the converted checkpoint is always INT8).
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        rotary_emb: nn.Module,
        topk_indices_buffer: torch.Tensor | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads  # local (TP-sharded) head count
        self.num_kv_heads = 1
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # Semantic MLA latent width (kv_lora + rope). FP8 packed rows use
        # ``_GLM_DSA_FP8_ROW_BYTES`` as the physical cache head size instead.
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.scale = float(scale)
        self.softmax_scale = float(scale)
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        self.prefix = prefix

        kv_cache_dtype = (
            cache_config.cache_dtype
            if cache_config is not None and cache_config.cache_dtype not in (None, "")
            else "auto"
        )
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "bfloat16"
        if kv_cache_dtype == "fp8_ds_mla":
            raise ValueError(
                "IluGlmMoeDsaSparseMLA does not use DeepSeek-V4 fp8_ds_mla "
                "(584B) layout. Pass --kv-cache-dtype fp8 for the GLM DSA "
                "640-byte packed path, or bfloat16 for the bf16 latent."
            )
        self.use_fp8_attn = kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_glm_dsa")
        init_scale = 1.0
        self._fp8_kv_scale_frozen = False
        # q_heads > 32: fmha (fixed 64-head tile). Else dsv4 (h_q % 16 == 0).
        self._fp8_use_dsv4 = self.use_fp8_attn and self.num_heads <= 32
        self._fp8_q_heads = (
            max(16, (self.num_heads + 15) // 16 * 16)
            if self._fp8_use_dsv4
            else _GLM_DSA_FP8_Q_HEADS
        )
        if self.use_fp8_attn:
            kv_cache_dtype = "fp8"
            if self.num_heads > _GLM_DSA_FP8_Q_HEADS:
                raise ValueError(
                    "FP8 sparse MLA requires <= "
                    f"{_GLM_DSA_FP8_Q_HEADS} local Q heads; got {self.num_heads}"
                )
            resolved, source = _resolve_glm_dsa_fp8_kv_scale(prefix)
            if resolved is not None:
                init_scale = resolved
                self._fp8_kv_scale_frozen = True
                logger.info(
                    "GLM DSA fp8 KV scale for %s = %.8g (%s)",
                    prefix,
                    init_scale,
                    source,
                )
            elif prefix.endswith(".layers.0.self_attn.attn") or prefix.endswith(
                "layers.0.self_attn.attn"
            ):
                logger.warning(
                    "GLM DSA fp8 KV scale for %s uses online amax/448 "
                    "(%s). Prefer VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH "
                    "(per-layer) or VLLM_ILU_GLM_DSA_FP8_KV_SCALE "
                    "(single float shared by ALL layers). Online scale "
                    "freezes on the first eager insert and is skipped "
                    "during CUDA-graph capture.",
                    prefix,
                    source,
                )
            if prefix.endswith(".layers.0.self_attn.attn") or prefix.endswith(
                "layers.0.self_attn.attn"
            ):
                logger.info(
                    "GLM DSA fp8 attn = %s (local q_heads=%d)",
                    "dsv4" if self._fp8_use_dsv4 else "fmha",
                    self.num_heads,
                )
        self.kv_cache_dtype = kv_cache_dtype

        self.register_buffer(
            "_k_scale",
            torch.full((1,), init_scale, dtype=torch.float32),
            persistent=False,
        )

        self.kv_cache = torch.tensor([])
        self._vllm_config = get_current_vllm_config()
        compilation_config = self._vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluFlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.use_fp8_attn:
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=_GLM_DSA_FP8_ROW_BYTES,
                dtype=torch.uint8,
                cache_dtype_str="fp8",
            )
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        # Absorbed k-up as a grouped W8A16 marlin GEMM (R1 _k_up_proj_w8a8).
        # q_nope: (B, N, P); kv_b_proj.W_UK: per-head (K=P, N=kv_lora) marlin.
        # k-up's scale sits on the contraction axis -> k16n16_grouped_n.
        return ixf_ops.marlin_w8a16(
            q_nope,
            self.kv_b_proj.W_UK,
            self.kv_b_proj.W_UK_S,
            group_size=-1,
            format="k16n16_grouped_n",
            batch_first=False,
        )  # (B, N, L)

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        # Absorbed v-up as a grouped W8A16 marlin GEMM (R1 _v_up_proj_w8a8).
        # x: (B, N, L); kv_b_proj.W_UV: per-head (K=kv_lora, N=v) marlin.
        return ixf_ops.marlin_w8a16(
            x,
            self.kv_b_proj.W_UV,
            self.kv_b_proj.W_UV_S,
            group_size=-1,
            format="k16n16",
            batch_first=False,
        )  # (B, N, V)

    @staticmethod
    def flash_mla_sparse_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse attention prefill kernel

        Args:
        - q: [s_q, h_q, d_qk], bfloat16
        - kv: [s_kv, h_kv, d_qk], bfloat16
        - indices: [s_q, h_kv, topk], int32.
            Invalid indices should be set to -1 or numbers >= s_kv
        - sm_scale: float
        - d_v: The dimension of value vectors. Can only be 512

        Returns:
        - (output, max_logits, lse)
            About the definition of output,
            max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
        """
        results = ops.sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
        return results

    def _maybe_init_fp8_kv_scale(self, kv_c: torch.Tensor) -> None:
        """Freeze kv_scale from the first eager insert; skip CUDA-graph capture.

        Online calibration is a bring-up fallback. Prefer
        ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH`` / ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE``
        so scale is fixed before graph capture. If the first inserts happen
        only under capture, scale stays at the init value (usually 1.0).
        """
        if self._fp8_kv_scale_frozen or kv_c.numel() == 0:
            return
        if torch.cuda.is_current_stream_capturing():
            if not getattr(self, "_fp8_kv_scale_capture_warned", False):
                logger.warning(
                    "GLM DSA fp8 KV scale for %s is still unfrozen during "
                    "CUDA-graph capture; keeping scale=%s. Warm up with an "
                    "eager insert or set VLLM_ILU_GLM_DSA_FP8_KV_SCALE(_PATH).",
                    self.prefix,
                    float(self._k_scale.item()),
                )
                self._fp8_kv_scale_capture_warned = True
            return
        amax = kv_c.detach().float().abs().amax().clamp_min(1e-12)
        scale = (amax / _GLM_DSA_FP8_E4M3_MAX).reshape(1)
        self._k_scale.copy_(scale)
        self._fp8_kv_scale_frozen = True
        logger.info(
            "GLM DSA fp8 KV scale for %s frozen online = %.8g (amax/448)",
            self.prefix,
            float(scale.item()),
        )

    def _global_topk(
        self,
        num_tokens: int,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_indices = self.topk_indices_buffer[:num_tokens]
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        return topk_indices.view(num_tokens, 1, -1), topk_length

    def _sparse_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> torch.Tensor:
        """DSA sparse MLA via ixformer ``dsa_flash_mla_sparse_fwd`` (qk=576, v=512)."""
        num_tokens = q.shape[0]
        topk_indices, topk_length = self._global_topk(num_tokens, attn_metadata)
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        return ixf_ops.dsa_flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            d_v=self.kv_lora_rank,
            topk_length=topk_length,
        )

    def _sparse_mla_kernel_fp8(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> torch.Tensor:
        """Packed 640B sparse MLA. ``q_heads > 32``: fmha; else dsv4."""
        num_tokens = q_nope.shape[0]
        q_heads = self._fp8_q_heads
        topk_indices, topk_length = self._global_topk(num_tokens, attn_metadata)
        q_packed = q_nope.new_empty(
            (num_tokens, q_heads, _GLM_DSA_FP8_ROW_BYTES), dtype=torch.uint8
        )
        q_scale = torch.empty(
            (num_tokens, q_heads), dtype=torch.float32, device=q_nope.device
        )
        ixf_ops.glm_dsa_fp8_q_pack(q_nope, q_pe, q_packed, q_scale)
        if self._fp8_use_dsv4:
            out = ixf_ops.dsa_flash_mla_with_fp8_kvcache(
                q_packed.view(num_tokens, 1, q_heads, _GLM_DSA_FP8_ROW_BYTES),
                kv_cache.unsqueeze(2),
                topk_indices,
                topk_length,
                q_scale,
                self._k_scale,
                self.softmax_scale,
            ).squeeze(1)
        else:
            out = ixf_ops.sparse_flash_attn_fp8(
                q_packed,
                kv_cache.view(-1, 1, kv_cache.shape[-1]),
                topk_indices,
                q_scale,
                self._k_scale,
                self.softmax_scale,
                topk_length=topk_length,
            )
        if self.num_heads == q_heads:
            return out
        return out[:, : self.num_heads, :].contiguous()

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse MLA forward. Returns ``(T, num_heads * v_head_dim)``.

        ``q``: ``(T, num_heads, qk_head_dim)``; ``kv_c_normed``: ``(T, kv_lora)``;
        ``k_pe``: ``(T, qk_rope_head_dim)``. Rope is applied here (not in the
        wrapper), matching the fork ``forward_full`` contract.
        """
        num_tokens = q.shape[0]
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.prefix)

        if attn_metadata is None:
            # Profiling / dummy run: reserve a comparable workspace so peak
            # memory is measured, then return zeros (matches fork forward_full).
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadataBuilder,
            )

            ws = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                self._vllm_config
            )
            _ = torch.empty(
                (ws, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            return q.new_zeros((num_tokens, self.num_heads * self.v_head_dim))

        num_actual_toks = attn_metadata.num_actual_tokens
        kv_cache = self.kv_cache

        k_pe = k_pe.unsqueeze(1)
        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # Absorbed k-up: (B, N, P) -> (B, N, kv_lora_rank).
        q_nope = self._k_up_proj(q_nope)
        q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping.flatten()[:num_actual_toks]
            if self.use_fp8_attn:
                self._maybe_init_fp8_kv_scale(kv_c_normed)
                ixf_ops.glm_dsa_fp8_kv_insert(
                    kv_c_normed,
                    k_pe,
                    kv_cache,
                    slot_mapping,
                    self._k_scale,
                    attn_metadata.block_size,
                )
            else:
                ops.concat_and_cache_mla(
                    kv_c_normed,
                    k_pe,
                    kv_cache,
                    slot_mapping,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

        if self.use_fp8_attn:
            attn_out = self._sparse_mla_kernel_fp8(
                q_nope, q_pe, kv_cache, attn_metadata
            )
        else:
            q = torch.cat([q_nope, q_pe], dim=-1)
            attn_out = self._sparse_mla_kernel(q, kv_cache, attn_metadata)

        output = q_nope.new_empty((num_tokens, self.num_heads, self.v_head_dim))
        output[:num_actual_toks] = self._v_up_proj(attn_out)
        return output.view(num_tokens, self.num_heads * self.v_head_dim)


class IluGlmMoeDsaMLAAttention(nn.Module):
    """MLA + DSA attention wrapper for a GLM-5.2 decoder layer (ilu W4A8).

    Self-contained analogue of the community
    ``MultiHeadLatentAttentionWrapper.forward_opt``: a single fused
    ``fused_qkv_a_proj`` GEMM produces ``[q_lora | kv_lora | rope]``, then q/kv
    norms, the DSA indexer (:class:`IluGlmMoeDsaIndexer` driving the shared
    plugin ``SparseAttnIndexer``), the inner :class:`IluGlmMoeDsaSparseMLA`
    and ``o_proj``. No community wrapper / ``MLAAttention`` in the chain.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        assert self.q_lora_rank is not None, (
            "IluGlmMoeDsaMLAAttention requires q_lora_rank (GLM-5.2 uses "
            "q_lora_rank=2048); the no-lora path is not supported here."
        )
        # Fused q_a_proj + kv_a_proj_with_mqa: merged at load time via
        # stacked_params_mapping (no runtime _inject_merge_qkv_a). disable_tp
        # keeps it replicated (both source projections are un-sharded).
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # kv_b_proj is grouped W8A16 marlin (absorbed MLA up-proj): the ilu
        # ``IluGlmKVBProjW8A16Method `` splits it into W_UK/W_UV and repacks
        # each at load time. Attach ``bmm_geom`` here (always -- the method
        # reads it in process_weights_after_loading) so the flat 2D weight can
        # be re-grouped per head. All dims are TP-local.
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.kv_b_proj.bmm_geom = SimpleNamespace(
            num_heads=self.num_local_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = _yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # DSA indexer is only built on ``full`` layers; ``skip_topk`` (``shared``)
        # layers reuse the previous full layer's topk buffer, so no indexer is
        # built there. GLM-5.2 ships an explicit per-layer ``indexer_types``
        # list ("full"/"shared"), which is the source of truth. Fall back to the
        # ``index_topk_pattern`` ("S") / frequency formula only if it is absent
        # (the two are equivalent for the current config, but the explicit list
        # is more robust to config variants).
        layer_id = extract_layer_index(prefix)
        _indexer_types = getattr(config, "indexer_types", None)
        if _indexer_types is not None and 0 <= layer_id < len(_indexer_types):
            self.skip_topk = _indexer_types[layer_id] != "full"
        else:
            _index_topk_freq = getattr(config, "index_topk_freq", 1)
            _index_topk_pattern = getattr(config, "index_topk_pattern", None)
            _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
            if _index_topk_pattern is None:
                self.skip_topk = (
                    max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq
                    != 0
                )
            elif 0 <= layer_id < len(_index_topk_pattern):
                self.skip_topk = _index_topk_pattern[layer_id] == "S"
            else:
                self.skip_topk = False

        # MTP draft layers (layer_id >= num_hidden_layers) must always build a
        # ``full`` indexer: they compute their own top-k at draft step 0 and the
        # proposer toggles skip_topk at runtime for steps 1+
        # (index_share_for_mtp_iteration). Starting with skip_topk=True here
        # would leave no indexer built and the draft reading a never-written
        # top-k buffer. Mirrors community deepseek_v2.py's is_mtp_layer handling.
        _num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if _num_hidden_layers is not None and layer_id >= _num_hidden_layers:
            self.skip_topk = False

        if not self.skip_topk:
            self.indexer = IluGlmMoeDsaIndexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer = None

        self.mla_attn = IluGlmMoeDsaSparseMLA(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack fused norm-quant tuple: W8A8 gemm wants (q,s,dtype);
        # indexer wk is BF16 and needs the normed activation.
        hidden_bf16: torch.Tensor
        if isinstance(hidden_states, tuple):
            x_q, x_s, third = hidden_states
            if isinstance(third, torch.Tensor):
                hidden_bf16 = third
                gemm_in: torch.Tensor | tuple = (x_q, x_s, third.dtype)
            else:
                gemm_in = (x_q, x_s, third)
                # No BF16 payload: indexer (if any) must not run on int8.
                hidden_bf16 = x_q  # placeholder; has_indexer path needs bf16
        else:
            hidden_bf16 = hidden_states
            gemm_in = hidden_states

        q_latent_kpe = self.fused_qkv_a_proj(gemm_in)[0]
        output_padding_size = getattr(self.fused_qkv_a_proj, "output_padding_size", 0)
        q_c, kv_a, k_pe, _ = q_latent_kpe.split(
            [
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                output_padding_size,
            ],
            dim=1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c_normed = self.kv_a_layernorm(kv_a)

        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_bf16, q_c, positions, self.rotary_emb)

        attn_out = self.mla_attn(q, kv_c_normed, k_pe, positions)
        return self.o_proj(attn_out)[0]
