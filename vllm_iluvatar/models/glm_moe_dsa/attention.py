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
  (``glm_dsa_fp8_kv_insert`` + ``dsa_flash_mla_with_fp8_kvcache``). The
  absorbed up-proj is grouped W8A16 marlin:
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
from vllm.logger import init_logger
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
from vllm.v1.attention.ops.pcp import (
    maybe_gather_indexer_k,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)
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

logger = init_logger(__name__)

# GLM DSA fp8 packed latent: 512 e4m3 NoPE + 64 bf16 RoPE = 640 bytes.
_GLM_DSA_FP8_ROW_BYTES = 640
_GLM_DSA_FP8_Q_HEADS = 64
_GLM_DSA_FP8_E4M3_MAX = 448.0


DynamicInt8Input = tuple[torch.Tensor, torch.Tensor, torch.dtype]
_RMS_ROPE_FP8_KV_ENGAGE_LOGGED = False
_RMS_ROPE_FP8_KV_SKIP_LOGGED = False
_Q_A_NORM_QUANT_FAILED = False


def _gather_and_cache_packed_fp8_kv(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor,
    pcp_world_size: int,
    num_decode_tokens: int,
) -> None:
    """Pack locally with a shared, frozen scale; gather 640 bytes per token.

    The runner supplies rank-major padded slots. Pack only actual local rows;
    padded rows still participate in the collective but never enter the cache.
    Replicated decode rows are retained once by the upstream PCP helper.
    """
    local_tokens = slot_mapping.numel() // pcp_world_size
    packed = torch.empty(
        (local_tokens, 1, _GLM_DSA_FP8_ROW_BYTES),
        dtype=torch.uint8,
        device=kv_c.device,
    )
    if kv_c.shape[0] < local_tokens:
        packed[kv_c.shape[0] :].zero_()
    if kv_c.shape[0]:
        ixf_ops.glm_dsa_fp8_kv_insert(
            kv_c,
            k_pe,
            packed,
            torch.arange(kv_c.shape[0], device=kv_c.device),
            scale,
            1,
        )
    # This public helper handles any single tensor, including packed byte rows.
    packed, slots = maybe_gather_indexer_k(
        packed[:, 0], slot_mapping, num_decode_tokens, True
    )
    # The uint8 specialization is a vectorized byte copy, with negative-slot
    # masking. No dequantization or second FP8 conversion takes place here.
    ixf_ops.concat_and_cache_mla(
        packed[:, :512],
        packed[:, 512:].unsqueeze(1),
        kv_cache,
        slots,
        "auto",
        scale,
    )


def _can_fuse_q_a_norm_quant(
    attn: "IluGlmMoeDsaMLAAttention",
    q_c: torch.Tensor,
) -> bool:
    if (
        _Q_A_NORM_QUANT_FAILED
        or not envs.VLLM_ILU_FUSE_NORM_QUANT
        or q_c.dim() != 2
        or q_c.stride(-1) != 1
        or not isinstance(
            getattr(attn.q_b_proj, "scheme", None),
            IluCompressedTensorsW8A8Int8,
        )
    ):
        return False
    return attn.indexer is None or isinstance(
        getattr(attn.indexer.wq_b, "scheme", None),
        IluCompressedTensorsW8A8Int8,
    )


def _fuse_q_a_norm_quant(
    attn: "IluGlmMoeDsaMLAAttention",
    q_c: torch.Tensor,
) -> DynamicInt8Input | None:
    if not _can_fuse_q_a_norm_quant(attn, q_c):
        return None
    global _Q_A_NORM_QUANT_FAILED
    try:
        q_int8, _, _, scales = ixf_ops.residual_rmsnorm_with_quant(
            q_c,
            attn.q_a_layernorm.weight,
            eps=attn.q_a_layernorm.variance_epsilon,
        )
        return q_int8, scales, q_c.dtype
    except Exception as exc:  # noqa: BLE001 - fail closed to the BF16 path
        logger.warning(
            "GLM-5.2 q_a RMSNorm+INT8 fusion unavailable (%s); "
            "falling back to split RMSNorm",
            exc,
        )
        _Q_A_NORM_QUANT_FAILED = True
        return None


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


class IluGlmMoeDsaPCPBackend(IluFlashMLASparseBackend):
    """PCP support for the GLM cache path, independent of the V4 backend."""

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


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

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]: the indexer kernels index a 3-D
        # block-major cache (see ixf_ops.indexer_k_cache dim==3 assert).
        self.kv_cache = kv_cache.squeeze(1)

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
        # The fused K op inserts only local rows; PCP must gather normalized,
        # rotated K before inserting the replicated cache.
        self.use_prepare_fusion = (
            envs.VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
        )
        if self.use_prepare_fusion:
            if self.head_dim != 128 or self.rope_dim != 64:
                raise ValueError(
                    "GLM indexer preparation fusion requires head_dim=128, rope_dim=64"
                )
            for name in (
                "glm_dsa_indexer_q_rope_weights",
                "glm_dsa_indexer_k_norm_rope_cache",
            ):
                if not callable(getattr(ixf_ops, name, None)) or not hasattr(
                    torch.ops._ixformer_C, name
                ):
                    raise RuntimeError(
                        f"VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION=1 requires "
                        f"ixformer with {name} (Python and native op)"
                    )

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
        qr: torch.Tensor | DynamicInt8Input,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # Fused wk + weights_proj: one GEMM, then split.
        # Small decode batches are faster through the BF16 torch GEMM than
        # through the quantized linear dispatch. Same cutoff as the 0.27
        # GLM path (one DP rank, bs <= 128).
        if (
            hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[0] <= 128
            and self.wk_weights_proj.weight.dtype == torch.bfloat16
        ):
            kw = torch.nn.functional.linear(hidden_states, self.wk_weights_proj.weight)
            logger.info_once("GLM decode indexer BF16 torch GEMM path")
        else:
            kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        if self.use_prepare_fusion:
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                metadata = attn_metadata.get(self.k_cache.prefix)
                if metadata is not None and self.k_cache.kv_cache.numel() > 0:
                    if rotary_emb.is_neox_style:
                        raise ValueError(
                            "GLM indexer preparation fusion requires interleaved RoPE"
                        )
                    # Match the original indexer cache insert's padded token
                    # extent. Negative slots are skipped by the fused K op.
                    slots = metadata.slot_mapping
                    num_tokens = slots.shape[0]
                    if num_tokens > k.shape[0]:
                        raise ValueError("Indexer slot mapping exceeds K token count")
                    cos_sin = rotary_emb._match_cos_sin_cache_dtype(q)
                    weights = ixf_ops.glm_dsa_indexer_q_rope_weights(
                        q,
                        weights,
                        positions,
                        cos_sin,
                        softmax_scale=self.softmax_scale,
                        head_scale=self.n_head_scale,
                    )
                    ixf_ops.glm_dsa_indexer_k_norm_rope_cache(
                        k[:num_tokens],
                        self.k_norm.weight,
                        self.k_norm.bias,
                        positions[:num_tokens],
                        cos_sin,
                        self.k_cache.kv_cache,
                        slots,
                        eps=self.k_norm.eps,
                    )
                    # K remains raw: only the fused op's cache output is used
                    # downstream. Never insert raw K over that prepared cache.
                    return self.indexer_op(
                        hidden_states, q, k, weights, skip_k_cache_insert=True
                    )

        # Also used by metadata-free dummy/profiling runs.
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
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


class GlmDsaTopkConvertCache:
    """Reuse one request-local -> global top-k conversion across shared layers.

    Invalidate after a ``full`` layer rewrites ``topk_indices_buffer``, or after
    a PP stage copies a buffer from upstream (different KV cache).
    """

    def __init__(self) -> None:
        self._generation = 0
        self._cached_generation = -1
        self._num_tokens = -1
        self._value: tuple[torch.Tensor, torch.Tensor] | None = None

    def invalidate(self) -> None:
        self._generation += 1

    def get(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            self._value is None
            or self._cached_generation != self._generation
            or self._num_tokens != num_tokens
        ):
            return None
        return self._value

    def put(self, num_tokens: int, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._cached_generation = self._generation
        self._num_tokens = num_tokens
        self._value = value


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
        (512 e4m3 NoPE + 64 bf16 RoPE) via ``glm_dsa_fp8_kv_insert`` +
        ``dsa_flash_mla_with_fp8_kvcache``; ixinfer picks DSV4 vs FMHA from the
        local ``q_heads`` (16-aligned for <= 32, else 64). Not DeepSeek-V4
        ``fp8_ds_mla`` (584B).

    ``kv_b_proj`` is grouped W8A16 marlin: :class:`IluGlmKVBProjW8A16Method `
    (in ``compressed-tensors`` / ``IluQuantConfig``) splits it into ``W_UK`` / ``W_UV`` and repacks
    each into Marlin layout at load time (deterministic, loader-driven). The
    k-up / v-up here just read ``self.kv_b_proj.W_UK`` / ``W_UV`` and call
    ``ixf_ops.marlin_w8a16`` -- mirroring R1's ``_k_up_proj_w8a8`` /
    ``_v_up_proj_w8a8`` (``vllm/v1/attention/backends/mla/ilu_mla.py``). There
    is no BF16 absorb fallback (the converted checkpoint is always INT8).
    """

    _fp8_kv_scale_calibration_enabled: bool = True

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
        topk_convert_cache: GlmDsaTopkConvertCache | None = None,
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
        self.topk_convert_cache = topk_convert_cache
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
        self._fp8_q_heads = self.num_heads
        if self.use_fp8_attn:
            kv_cache_dtype = "fp8"
            if self.num_heads > _GLM_DSA_FP8_Q_HEADS:
                raise ValueError(
                    "FP8 sparse MLA requires <= "
                    f"{_GLM_DSA_FP8_Q_HEADS} local Q heads; got {self.num_heads}"
                )
            # ixinfer: <=32 DSV4 (h_q % 16 == 0); >32 FMHA (pad to 64).
            self._fp8_q_heads = (
                _GLM_DSA_FP8_Q_HEADS
                if self.num_heads > 32
                else max(16, (self.num_heads + 15) // 16 * 16)
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
                    "freezes on the first eager insert with amax>0; "
                    "capture and all-zero dummy tensors are skipped.",
                    prefix,
                    source,
                )
            if prefix.endswith(".layers.0.self_attn.attn") or prefix.endswith(
                "layers.0.self_attn.attn"
            ):
                logger.info(
                    "GLM DSA fp8 attn: local q_heads=%d -> padded q_heads=%d",
                    self.num_heads,
                    self._fp8_q_heads,
                )
        self.kv_cache_dtype = kv_cache_dtype
        from vllm import envs

        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        self._decode_prepare_enabled = (
            envs.VLLM_ILU_GLM_DSA_PREPARE_FUSION
            and is_ivcore30()
            and self.use_fp8_attn
            and self.num_heads in (16, 32, 64)
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
        )

        self.register_buffer(
            "_k_scale",
            torch.full((1,), init_scale, dtype=torch.float32),
            persistent=False,
        )
        self._vllm_config = get_current_vllm_config()
        use_pcp = self._vllm_config.parallel_config.prefill_context_parallel_size > 1
        fusion_requested = envs.VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV
        # Prefer the native Torch schema over the Python facade: the facade can
        # exist from a Python-only ixformer update while `_ixformer_C` still
        # lacks the ivcore30 kernel. Kernel contract is H in {8,16,32,64}.
        fusion_op_available = False
        try:
            fusion_op_available = hasattr(
                torch.ops._ixformer_C, "fused_glm_dsa_rmsnorm_rope_fp8_kv_insert"
            )
        except Exception:
            fusion_op_available = False
        fusion_heads_supported = self.num_heads in (8, 16, 32, 64)
        self._rms_rope_fp8_kv_fusion_available = (
            fusion_requested
            and self.use_fp8_attn
            and fusion_op_available
            and fusion_heads_supported
            and not use_pcp
        )
        self.use_rms_rope_fp8_kv_fusion = (
            self._rms_rope_fp8_kv_fusion_available and self._fp8_kv_scale_frozen
        )
        if fusion_requested and not self.use_rms_rope_fp8_kv_fusion:
            global _RMS_ROPE_FP8_KV_SKIP_LOGGED
            if not _RMS_ROPE_FP8_KV_SKIP_LOGGED:
                reasons = []
                if use_pcp:
                    reasons.append("PCP requires gathering normalized/rotated KV")
                if not self.use_fp8_attn:
                    reasons.append("KV cache is not fp8")
                if not self._fp8_kv_scale_frozen:
                    reasons.append("KV scale is not frozen")
                if not fusion_op_available:
                    reasons.append("ixformer native op is unavailable")
                if not fusion_heads_supported:
                    reasons.append(
                        f"local Q heads={self.num_heads} not in {{8,16,32,64}}"
                    )
                logger.warning(
                    "GLM-5.2 RMSNorm+RoPE+FP8 KV fusion requested but not "
                    "armed (%s). Falling back to split path. Further layers "
                    "will not repeat this warning.",
                    ", ".join(reasons) or "unknown",
                )
                _RMS_ROPE_FP8_KV_SKIP_LOGGED = True
        elif self.use_rms_rope_fp8_kv_fusion:
            global _RMS_ROPE_FP8_KV_ENGAGE_LOGGED
            if not _RMS_ROPE_FP8_KV_ENGAGE_LOGGED:
                logger.info(
                    "GLM DSA RMSNorm+RoPE+FP8 KV fusion armed "
                    "(VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV=1)"
                )
                _RMS_ROPE_FP8_KV_ENGAGE_LOGGED = True

        self.kv_cache = torch.tensor([])
        if (
            self._vllm_config.parallel_config.prefill_context_parallel_size > 1
            and self._vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError("GLM DSA PCP currently requires DCP=1")
        compilation_config = self._vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]: the sparse MLA ixformer kernels
        # (fused_glm_dsa_*_kv_insert / dsa_flash_mla_*) index a 3-D cache.
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluGlmMoeDsaPCPBackend

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

    def _v_up_proj(self, x: torch.Tensor, outputs=None) -> torch.Tensor:
        # Absorbed v-up as a grouped W8A16 marlin GEMM (R1 _v_up_proj_w8a8).
        # x: (B, N, L); kv_b_proj.W_UV: per-head (K=kv_lora, N=v) marlin.
        return ixf_ops.marlin_w8a16(
            x,
            self.kv_b_proj.W_UV,
            self.kv_b_proj.W_UV_S,
            group_size=-1,
            format="k16n16",
            batch_first=False,
            outputs=outputs,
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

    def _maybe_init_fp8_kv_scale(
        self, kv_c: torch.Tensor, slot_mapping: torch.Tensor | None = None
    ) -> None:
        """Freeze kv_scale from the first eager insert that has a real amax.

        Online calibration is a bring-up fallback. Prefer
        ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH`` / ``VLLM_ILU_GLM_DSA_FP8_KV_SCALE``.

        Skip CUDA-graph capture (no reduction / D2H). Skip non-finite or
        non-positive amax: PP stages > 0 see an all-zero recv buffer during
        the profiling dummy run; freezing ``clamp(0)/448`` saturates later
        real tokens. Prefill stays eager under ``FULL_DECODE_ONLY``, so the
        first real insert still calibrates. Ignore padding and V2 warmup rows
        with slot_mapping == -1: their projections can be nonzero even though
        no KV is inserted.
        """
        if (
            not self._fp8_kv_scale_calibration_enabled
            or self._fp8_kv_scale_frozen
            or kv_c.numel() == 0
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        abs_kv = kv_c.detach().float().abs()
        if slot_mapping is not None:
            abs_kv = abs_kv.masked_fill(slot_mapping.reshape(-1, 1) < 0, 0.0)
        amax = float(abs_kv.amax())
        if not math.isfinite(amax) or amax <= 0.0:
            return
        scale = amax / _GLM_DSA_FP8_E4M3_MAX
        self._k_scale.fill_(scale)
        self._fp8_kv_scale_frozen = True
        # The first real eager insert calibrates on normalized KV. Subsequent
        # calls can consume raw KV in the fused op, using that same frozen scale.
        self.use_rms_rope_fp8_kv_fusion = getattr(
            self, "_rms_rope_fp8_kv_fusion_available", False
        )
        logger.info(
            "GLM DSA fp8 KV scale for %s frozen online = %.8g (amax/448)",
            self.prefix,
            scale,
        )

    def _global_topk(
        self,
        num_tokens: int,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.topk_convert_cache
        if cache is not None:
            hit = cache.get(num_tokens)
            if hit is not None:
                return hit

        topk_indices = self.topk_indices_buffer[:num_tokens]
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        result = (topk_indices.view(num_tokens, 1, -1), topk_length)
        if cache is not None:
            cache.put(num_tokens, result)
        return result

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
        # Sparse MLA returns latent V (kv_lora_rank=512); v_up_proj maps to
        # v_head_dim afterwards. Do not pass self.v_head_dim (256 on GLM-5.2).
        return ixf_ops.dsa_flash_mla_sparse_fwd_diff_kv(
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
        prepared_q: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Packed 640B sparse MLA via ``dsa_flash_mla_with_fp8_kvcache``."""
        num_tokens = q_nope.shape[0]
        q_heads = self._fp8_q_heads
        topk_indices, topk_length = self._global_topk(num_tokens, attn_metadata)

        if prepared_q is not None:
            q_packed, q_scale = prepared_q
        else:
            q_packed = q_nope.new_empty(
                (num_tokens, q_heads, _GLM_DSA_FP8_ROW_BYTES),
                dtype=torch.uint8,
            )
            q_scale = torch.empty(
                (num_tokens, q_heads),
                dtype=torch.float32,
                device=q_nope.device,
            )
            ixf_ops.glm_dsa_fp8_q_pack(q_nope, q_pe, q_packed, q_scale)

        out = ixf_ops.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
            q_packed.view(num_tokens, 1, q_heads, _GLM_DSA_FP8_ROW_BYTES),
            kv_cache.unsqueeze(2),
            topk_indices,
            topk_length,
            q_scale,
            self._k_scale,
            self.softmax_scale,
        ).squeeze(1)
        if self.num_heads == q_heads:
            return out
        return out[:, : self.num_heads, :].contiguous()

    def forward(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        kv_rms_weight: torch.Tensor | None = None,
        kv_rms_eps: float | None = None,
    ) -> torch.Tensor:
        """Sparse MLA forward. Returns ``(T, num_heads * v_head_dim)``.

        ``kv_c`` is already RMS-normalized unless ``kv_rms_weight`` is provided,
        in which case the FP8 fusion consumes the raw latent. RoPE is applied
        here (or inside the fused kernel when armed).
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
        if num_actual_toks == 0 and attn_metadata.pcp_world_size <= 1:
            return q.new_zeros((num_tokens, self.num_heads * self.v_head_dim))
        kv_cache = self.kv_cache
        use_pcp = attn_metadata.pcp_world_size > 1

        k_pe = k_pe.unsqueeze(1)
        q = q[:num_actual_toks]
        kv_c = kv_c[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]

        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        use_decode_prepare = (
            self._decode_prepare_enabled
            and is_ivcore30()
            and not use_pcp
            and kv_rms_weight is not None
            and q.dtype == torch.bfloat16
            and 0 < num_actual_toks <= 65535
            and attn_metadata.block_size == 64
            and kv_cache.numel() > 0
            and (self._fp8_kv_scale_frozen or torch.cuda.is_current_stream_capturing())
        )
        prepared_q = None
        use_rms_rope_fp8_kv_fusion = (
            (self.use_rms_rope_fp8_kv_fusion or use_decode_prepare)
            and not use_pcp
            and kv_rms_weight is not None
            and kv_cache.numel() > 0
        )
        if kv_rms_weight is not None and kv_rms_eps is None:
            raise ValueError("kv_rms_eps is required with kv_rms_weight")
        if (
            kv_rms_weight is not None
            and not use_rms_rope_fp8_kv_fusion
            and num_actual_toks > 0
        ):
            kv_c_normed = ixf_ops.rms_norm(kv_c, kv_rms_weight, kv_rms_eps)
        else:
            kv_c_normed = kv_c

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if not use_rms_rope_fp8_kv_fusion and num_actual_toks > 0:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # Absorbed k-up: (B, N, P) -> (B, N, kv_lora_rank).
        if num_actual_toks > 0:
            q_nope = self._k_up_proj(q_nope)
            q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping.flatten()[:num_actual_toks]
            cache_kv_c, cache_k_pe = kv_c_normed, k_pe
            use_packed_pcp = (
                use_pcp
                and self.use_fp8_attn
                and self._fp8_kv_scale_frozen
                and attn_metadata.num_decode_tokens
                < attn_metadata.slot_mapping.numel() // attn_metadata.pcp_world_size
            )
            if use_packed_pcp:
                _gather_and_cache_packed_fp8_kv(
                    kv_c_normed,
                    k_pe,
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    self._k_scale,
                    attn_metadata.pcp_world_size,
                    attn_metadata.num_decode_tokens,
                )
            elif use_pcp:
                # Collective sizes and rank-major slot strides use the runner's
                # padded length, including ranks with no local prefill tokens.
                slot_mapping = attn_metadata.slot_mapping.flatten()
                local_tokens = slot_mapping.numel() // attn_metadata.pcp_world_size
                padding = local_tokens - num_actual_toks
                if padding:
                    cache_kv_c = torch.nn.functional.pad(cache_kv_c, (0, 0, 0, padding))
                    cache_k_pe = torch.nn.functional.pad(
                        cache_k_pe, (0, 0, 0, 0, 0, padding)
                    )
                cache_kv_c, cache_k_pe, slot_mapping = (
                    maybe_gather_mla_latent_cache_inputs(
                        cache_kv_c,
                        cache_k_pe,
                        slot_mapping,
                        attn_metadata.num_decode_tokens,
                        True,
                    )
                )
            if self.use_fp8_attn and not use_packed_pcp:
                if use_decode_prepare:
                    # BF16 absorbed einsum can return head-major storage.
                    # Keep compatible token-major (including padded) Q as-is.
                    if q_nope.stride(0) < self.num_heads * q_nope.stride(1):
                        q_nope = q_nope.contiguous()
                    logger.info_once("GLM decode glm_dsa_prepare path")
                    prepared_q = ixf_ops.glm_dsa_prepare(
                        q_nope,
                        q_pe,
                        k_pe.squeeze(1),
                        kv_c,
                        kv_rms_weight,
                        kv_cache,
                        slot_mapping,
                        positions,
                        self.rotary_emb._match_cos_sin_cache_dtype(q_pe),
                        self._k_scale,
                        eps=kv_rms_eps,
                    )
                elif use_rms_rope_fp8_kv_fusion:
                    ixf_ops.fused_glm_dsa_rmsnorm_rope_fp8_kv_insert(
                        q_pe,
                        k_pe,
                        kv_c,
                        kv_rms_weight,
                        kv_rms_eps,
                        kv_cache,
                        slot_mapping,
                        positions,
                        self.rotary_emb._match_cos_sin_cache_dtype(q_pe),
                        self._k_scale,
                        attn_metadata.block_size,
                    )
                else:
                    self._maybe_init_fp8_kv_scale(cache_kv_c, slot_mapping)
                    ixf_ops.glm_dsa_fp8_kv_insert(
                        cache_kv_c,
                        cache_k_pe,
                        kv_cache,
                        slot_mapping,
                        self._k_scale,
                        attn_metadata.block_size,
                    )
            elif not self.use_fp8_attn:
                ops.concat_and_cache_mla(
                    cache_kv_c,
                    cache_k_pe,
                    kv_cache,
                    slot_mapping,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

        if num_actual_toks == 0:
            return q.new_zeros((num_tokens, self.num_heads * self.v_head_dim))

        if self.use_fp8_attn:
            attn_out = self._sparse_mla_kernel_fp8(
                q_nope, q_pe, kv_cache, attn_metadata, prepared_q
            )
        else:
            q = torch.cat([q_nope, q_pe], dim=-1)
            attn_out = self._sparse_mla_kernel(q, kv_cache, attn_metadata)

        if num_actual_toks == num_tokens:
            return self._v_up_proj(attn_out).reshape(
                num_tokens, self.num_heads * self.v_head_dim
            )
        output = q_nope.new_empty((num_tokens, self.num_heads, self.v_head_dim))
        self._v_up_proj(attn_out, outputs=output[:num_actual_toks])
        if use_pcp and num_actual_toks < num_tokens:
            output[num_actual_toks:].zero_()
        return output.view(num_tokens, self.num_heads * self.v_head_dim)


def glm_dsa_layer_skip_topk(config, layer_id: int) -> bool:
    """Whether ``layer_id`` reuses a previous ``full`` layer's top-k indices.

    ``full`` layers build a DSA indexer and write ``topk_indices_buffer``;
    ``shared`` layers only read it. The per-layer ``indexer_types`` list is the
    source of truth; the ``index_topk_pattern`` ("S") / frequency formula is a
    fallback for config variants lacking it (both agree on GLM-5.2).

    MTP draft layers (``layer_id >= num_hidden_layers``) always return False:
    they compute their own top-k at draft step 0 and the proposer toggles
    ``skip_topk`` for steps 1+ (``index_share_for_mtp_iteration``). True here
    would leave no indexer built and the draft reading a never-written buffer.
    Mirrors community deepseek_v2.py's is_mtp_layer.

    ``IluGlmMoeDsaModel`` also uses this to pick which PP stage boundaries
    carry the buffer, so both callers must agree.
    """
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False

    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None and 0 <= layer_id < len(indexer_types):
        return indexer_types[layer_id] != "full"

    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is None:
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
    if 0 <= layer_id < len(index_topk_pattern):
        return index_topk_pattern[layer_id] == "S"
    return False


class IluGlmMoeDsaMLAAttention(nn.Module):
    """MLA + DSA attention wrapper for a GLM-5.2 decoder layer (ilu W4A8).

    Self-contained analogue of the community
    ``MultiHeadLatentAttentionWrapper.forward_opt``: a single fused
    ``fused_qkv_a_proj`` GEMM produces ``[q_lora | kv_lora | rope]``, then q/kv
    norms, the DSA indexer (:class:`IluGlmMoeDsaIndexer` driving the shared
    plugin ``SparseAttnIndexer``), the inner :class:`IluGlmMoeDsaSparseMLA`
    and ``o_proj``. No community wrapper / ``MLAAttention`` in the chain.

    ``sparse_mla_cls`` lets the FP8 variant swap in its own inner attention
    (see ``fp8_attention.py``) without duplicating this ``__init__``.
    """

    sparse_mla_cls: type[IluGlmMoeDsaSparseMLA] = IluGlmMoeDsaSparseMLA

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
        topk_convert_cache: GlmDsaTopkConvertCache | None = None,
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

        # DSA indexer is only built on ``full`` layers; ``skip_topk``
        # (``shared``) layers reuse the previous full layer's topk buffer.
        layer_id = extract_layer_index(prefix)
        self.skip_topk = glm_dsa_layer_skip_topk(config, layer_id)
        self.topk_convert_cache = topk_convert_cache

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

        self.mla_attn = self.sparse_mla_cls(
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
            topk_convert_cache=topk_convert_cache,
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
        q_projection_input = _fuse_q_a_norm_quant(self, q_c)
        if q_projection_input is None:
            q_projection_input = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_projection_input)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )

        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_bf16, q_projection_input, positions, self.rotary_emb)
            if self.topk_convert_cache is not None:
                self.topk_convert_cache.invalidate()

        if (
            self.mla_attn.use_rms_rope_fp8_kv_fusion
            or self.mla_attn._decode_prepare_enabled
        ):
            attn_out = self.mla_attn(
                q,
                kv_a,
                k_pe,
                positions,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
            )
        else:
            kv_c_normed = self.kv_a_layernorm(kv_a)
            attn_out = self.mla_attn(q, kv_c_normed, k_pe, positions)
        return self.o_proj(attn_out)[0]
