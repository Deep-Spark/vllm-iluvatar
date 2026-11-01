# SPDX-License-Identifier: Apache-2.0
"""MLA attention and lightning indexer for HY V4 on Iluvatar.

Corresponds to: vllm/models/hy_v4/nvidia/attention.py (upstream #54160)

Reuses community `MLAAttention` rather than reimplementing the algorithm chain:
``IluPlatform`` already redirects ``AttentionBackendEnum.FLASHMLA_SPARSE`` to
``IluFlashMLASparseBackend``, so a community ``MLAAttention(use_sparse=True)``
lands on ixformer kernels with no model-side work. The HY V4 deltas are the
per-head learnable sink (bound through `.flashmla_sparse`) and the gated-MLA
output gate.

The indexer follows ``--kv-cache-dtype``, the same switch as the main MLA
cache. ``fp8`` / ``fp8_e4m3`` selects the grouped fp8 K cache and the ds4
kernels (see ``fp8_indexer.py``). Any other dtype keeps the BF16 indexer
cache used by V3.2 and GLM DSA. The community fp8 indexer op is not used:
``per_token_group_quant_fp8`` needs ``torch.ops._C.per_token_group_fp8_quant``,
which Iluvatar does not build, and the ixformer lane quantizes on its own.

Delta vs upstream: the HPC fused output gate (``hpc_gated_mla_gemm``) is
dropped. It needs the hpc ``.so`` plus sm100/sm103 and is off by default
(``VLLM_ENABLE_HPC_OPS=0``), so upstream's own default is the eager gate kept
below.
"""

from __future__ import annotations

import copy
import json
import math
import os
from functools import lru_cache
from typing import cast

import regex as re
import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig
from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.models.deepseek_v32.attention import IluDeepseekV32IndexerCache
from vllm_iluvatar.models.hy_v4.fp8_indexer import (
    HYV4Fp8IndexerCache,
    hy4_fp8_indexer_forward,
)

from .topk_index_cache import HYV4TopkIndexCache

logger = init_logger(__name__)

_SPARSE_LAYER_TYPES = (
    "sparse_attention",
    "sparse",
    "deepseek_sparse_attention",
    "indexed_attention",
)
_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_HYV4_DSA_FP8_ROW_BYTES = 640
_HYV4_DSA_FP8_E4M3_MAX = 448.0
_HYV4_DSA_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_MLA_KV_FUSE_LOGGED = False


def apply_qk_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    mrope_emb: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate MLA q/k.

    ``q_pe`` is ``[tokens, heads, rope]`` and ``k_pe`` is ``[tokens, 1, rope]``
    or ``[tokens, rope]``. Three-row positions always use M-RoPE, including
    text with identical rows. Selecting by device values would synchronize
    during CUDA Graph capture and freeze the choice across replay. The M-RoPE
    kernel reads ``[tokens, heads * rope]``; its 1-D flash-attn path is unused.
    """
    if positions.ndim == 2:
        if positions.shape[0] != 3 or mrope_emb is None:
            raise ValueError(
                "HYV4 2-D positions require three rows and an M-RoPE module"
            )
        q_out, k_out = mrope_emb(
            positions,
            # M-RoPE rotates these buffers in place. A reshape may alias the
            # original slices (especially a one-token K), and its returned
            # view can have different singleton strides. The caller copies
            # results back into q/k; independent buffers avoid overlapping
            # copy_ and keep this contract valid for prefill and decode.
            q_pe.reshape(q_pe.shape[0], -1).clone(),
            k_pe.reshape(k_pe.shape[0], -1).clone(),
        )
        return q_out.reshape(q_pe.shape), k_out.reshape(k_pe.shape)
    return rotary_emb(positions, q_pe, k_pe)


def build_hy4_ropes(
    head_size: int,
    max_position: int,
    rope_parameters: dict | None,
    *,
    is_neox_style: bool = False,
) -> tuple[nn.Module, nn.Module | None]:
    """1-D Iluvatar RoPE, plus M-RoPE when the checkpoint asks for it.

    ``mrope_section`` stays on the HF config so the runner supplies 3-row
    positions. It is removed from the 1-D module: ``MRotaryEmbedding``'s 1-D
    path imports flash-attn, which is not available here.
    """
    params = dict(rope_parameters or {})
    section = params.pop("mrope_section", None)
    interleaved = bool(params.pop("mrope_interleaved", False))
    rotary = get_rope(
        head_size,
        max_position=max_position,
        rope_parameters=params,
        is_neox_style=is_neox_style,
    )
    mrope = None
    if section:
        mrope = get_rope(
            head_size,
            max_position=max_position,
            rope_parameters={
                **params,
                "mrope_section": section,
                "mrope_interleaved": interleaved,
            },
            is_neox_style=is_neox_style,
        )
    return rotary, mrope


@lru_cache(maxsize=4)
def _load_hyv4_fp8_kv_scale_table(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"VLLM_ILU_HYV4_DSA_FP8_KV_SCALE_PATH={path!r}: expected a JSON "
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
                f"VLLM_ILU_HYV4_DSA_FP8_KV_SCALE_PATH={path!r}: entry "
                f"{key!r} must be {{'kv': float}} or a float, got {val!r}"
            )
    logger.info("Loaded %d HYV4 DSA fp8 KV scales from %s", len(out), path)
    return out


def _resolve_hyv4_fp8_kv_scale(prefix: str) -> tuple[float | None, str]:
    scale_env = os.environ.get("VLLM_ILU_HYV4_DSA_FP8_KV_SCALE", "")
    if scale_env.strip():
        return float(scale_env), "VLLM_ILU_HYV4_DSA_FP8_KV_SCALE"

    path = os.environ.get("VLLM_ILU_HYV4_DSA_FP8_KV_SCALE_PATH", "")
    if path.strip():
        table = _load_hyv4_fp8_kv_scale_table(path.strip())
        candidates = [
            prefix,
            prefix.removeprefix("model."),
            f"model.{prefix}" if not prefix.startswith("model.") else prefix,
        ]
        for candidate in candidates:
            if candidate in table:
                return table[candidate], f"{path}:{candidate}"
        logger.warning(
            "VLLM_ILU_HYV4_DSA_FP8_KV_SCALE_PATH=%r has no entry for %r "
            "(tried %s); falling back to online amax/448 calibration",
            path,
            prefix,
            candidates,
        )
        return None, "online_amax_missing_json_key"
    return None, "online_amax"


def _hyv4_fp8_q_heads(num_heads: int) -> int:
    if num_heads > 64:
        raise ValueError(
            f"HYV4 fp8 sparse MLA requires <=64 local heads, got {num_heads}"
        )
    return 64 if num_heads > 32 else max(16, (num_heads + 15) // 16 * 16)


# Token count up to which the fused single-kernel up-projection wins. Measured
# per layer on ivcore30 (TP2, G=32): the fused kernel runs 19.4 us at T=16
# against 25.6 for quant+einsum, and loses at T=32 (32.4 vs 27.9) where the
# einsum's larger tiles take over. Production maps concurrency to T as
# ``T = seqs_per_rank * 4`` (DP8 + MTP3), so this covers 32-way concurrency.
_FUSED_MAX_TOKENS = 16
_FP8_E4M3_MAX = 448.0


def _ixf_functions():
    """Imported lazily, like the rest of this module's ixformer use."""
    import ixformer.inference.functions as ixf_ops

    return ixf_ops


def _quant_per_out_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, D, R]`` BF16 -> e4m3 bytes plus a read-safe fp32 scale ``[G, D]``.

    The scale sits on the output axis D, which is what the up-projection ops
    dequantise with; the einsum reads it in 128-wide chunks, hence read-safe.
    """
    groups, dim = w.shape[0], w.shape[1]
    wf = w.float()
    scale = (wf.abs().amax(dim=-1) / _FP8_E4M3_MAX).clamp_min(1e-12)
    q = (
        (wf / scale.unsqueeze(-1))
        .clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    padded = _ixf_functions().alloc_readsafe_scale(groups, dim, device=w.device)
    padded.copy_(scale)
    return q, padded


def indexer_fused_quant_config(quant_config, prefix):
    """Expose the BF16 assembled wk/weights_proj to the runtime resolver.

    Mixed checkpoints dequantize wk before loading shard 0, while shard 1 is
    BF16. Keep that existing assembly contract, including MTP, and let only an
    explicit runtime group quantize the fully assembled fused parameter. The
    shared target/draft quant config and their other dispatches stay unchanged.
    """
    from vllm_iluvatar.model_executor.layers.quantization.fp8 import IluFp8Config

    if not isinstance(quant_config, IluFp8Config):
        return None
    fused_config = copy.copy(quant_config)
    fused_config.ignored_layers = list(quant_config.ignored_layers) + [
        f"{prefix}.wk_weights_proj"
    ]
    return fused_config


class HYV4MLAAttentionLayer(MLAAttention):
    """Community MLA layer with HYV4's physical 640-byte fp8 cache spec."""

    def __init__(self, *args, **kwargs) -> None:
        cache_config = kwargs.get("cache_config")
        requested_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.use_hyv4_fp8_attn = requested_dtype in _HYV4_DSA_FP8_CACHE_DTYPES
        prefix = kwargs.get("prefix", "")
        init_scale = 1.0
        scale_frozen = False
        if self.use_hyv4_fp8_attn:
            resolved, source = _resolve_hyv4_fp8_kv_scale(prefix)
            if resolved is not None:
                if not math.isfinite(resolved) or resolved <= 0:
                    raise ValueError(
                        f"HYV4 fp8 KV scale must be finite and >0, got {resolved}"
                    )
                init_scale = resolved
                scale_frozen = True
                logger.info(
                    "HYV4 DSA fp8 KV scale for %s = %.8g (%s)",
                    prefix,
                    init_scale,
                    source,
                )
            kwargs["hyv4_use_fp8_attn"] = True
            kwargs["hyv4_fp8_kv_scale_frozen"] = scale_frozen
            kwargs["hyv4_fp8_prefix"] = prefix
            # MLAAttention canonicalizes every quantized FLASHMLA_SPARSE cache
            # to DeepSeek's 656-byte ``fp8_ds_mla`` format and mutates the
            # shared CacheConfig in place. HYV4 uses a different 640-byte GLM
            # layout implemented by HYV4FlashMLASparseImpl, so give the
            # community layer a private, non-quantized facade config. The
            # override below still publishes the real uint8/640-byte spec.
            # Keeping ``self.kv_cache_dtype == "auto"`` also prevents the
            # community forward path from viewing our packed uint8 cache as a
            # plain float8 tensor.
            facade_cache_config = copy.copy(cache_config)
            facade_cache_config.cache_dtype = "auto"
            kwargs["cache_config"] = facade_cache_config
        super().__init__(*args, **kwargs)
        # FP8 absorb is the default for HY4's all-sparse-MQA path. Keep the
        # community BF16 implementation for execution modes it cannot serve.
        self.fp8_k_up = (
            self.impl.is_sparse
            and self.impl.dcp_world_size == 1
            and not self.use_pcp
            and self._vllm_config.attention_config.sparse_mla_force_mqa
        )
        self.W_UK_FP8: torch.Tensor | None = None
        self.W_UK_FP8_S: torch.Tensor | None = None
        if self.use_hyv4_fp8_attn:
            self._k_scale.fill_(init_scale)
            self._non_persistent_buffers_set.add("_k_scale")
        self.fuse_v_up = os.environ.get(
            "VLLM_ILU_HYV4_FUSE_KV_UP", "1"
        ) != "0" and hasattr(_ixf_functions(), "hy4_kv_up_gate")
        # Folding the gate is separable from running v-up in FP8, which keeps
        # the two halves of this change independently switchable.
        self.fuse_gate = (
            self.fuse_v_up
            and os.environ.get("VLLM_ILU_HYV4_FUSE_KV_UP_GATE", "1") != "0"
        )
        self.W_UV_FP8: torch.Tensor | None = None
        self.W_UV_FP8_S: torch.Tensor | None = None
        # Gate logits for the current token batch, handed over by the attention
        # module so ``_v_up_proj`` can fold ``sigmoid * mul`` into its store,
        # and the number of leading rows that fold actually covered.
        self.gate_logits: torch.Tensor | None = None
        self.gated_rows = 0

    # [ILU override] MLAAttention.process_weights_after_loading — delta: keep an
    # FP8 absorbed weights, laid out with scales on the output channel axis.
    def _dequant_per_channel_kv_b(self, act_dtype: torch.dtype) -> None:
        """Undo the Iluvatar block->per-channel requant before MLA absorbs kv_b.

        ``IluFp8BlockScaledMMKernel`` replaces the 128x128 ``weight_scale_inv``
        with a fp32 ``[N]`` channel scale. Community ``MLAAttention`` then
        dequantizes with ``group_shape=(128, 128)`` and rejects that 1D scale.
        Materialize BF16 and drop the quant method so the absorb sees a plain
        weight. BF16 ``kv_b_proj`` (no channel scale) is left untouched.
        """
        layer = self.kv_b_proj
        while hasattr(layer, "base_layer") and hasattr(
            layer.base_layer, "quant_method"
        ):
            layer = layer.base_layer
        weight = getattr(layer, "weight", None)
        if weight is None or weight.dtype != torch.float8_e4m3fn:
            return
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(layer, "weight_scale", None)
        if (
            scale is None
            or scale.ndim != 1
            or scale.shape[0] != weight.shape[0]
            or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.utils import replace_parameter

        bf16 = (weight.to(act_dtype) * scale.to(act_dtype).unsqueeze(-1)).contiguous()
        replace_parameter(layer, "weight", bf16)
        layer.quant_method = UnquantizedLinearMethod()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self._dequant_per_channel_kv_b(act_dtype)
        super().process_weights_after_loading(act_dtype)
        if self.fp8_k_up:
            # Community W_UK_T is [H, nope, latent]. GLM grouped einsum
            # consumes [H, latent, nope] with scales [H, latent].
            self.W_UK_FP8, self.W_UK_FP8_S = _quant_per_out_channel(
                self.W_UK_T.transpose(1, 2)
            )
            self.W_UK_T = None
        if self.fuse_v_up:
            # Community W_UV[N, L, V] -> [G, D=V, R=L], scale per D.
            self.W_UV_FP8, self.W_UV_FP8_S = _quant_per_out_channel(
                self.W_UV.transpose(1, 2)
            )
            self.W_UV = None
        if self.fp8_k_up or self.fuse_v_up:
            logger.info(
                "HYV4 absorbed projection ready: layer=%s K=%s V=%s gate_fused=%s",
                self.layer_name,
                tuple(self.W_UK_FP8.shape) if self.fp8_k_up else "BF16",
                tuple(self.W_UV_FP8.shape) if self.fuse_v_up else "BF16",
                self.fuse_gate,
            )

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        """Absorb K into Q using GLM's existing FP8 grouped GEMM contract."""
        ops = _ixf_functions()
        a, a_s = ops.fp8_grouped_per_token_quant(q_nope, self.num_heads)
        return ops.fp8_per_channel_einsum(a, self.W_UK_FP8, a_s, self.W_UK_FP8_S)

    def forward_impl(
        self,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        quant_group_size=None,
        quant_scale_ue8m0=None,
        quant_col_major=None,
        quant_tma_aligned=None,
        q_dcp_replicated=None,
    ):
        """HY4's all-MQA FP8 absorb path; cache insertion remains in vLLM.

        Corresponds to MLAAttention.forward_impl at community 341bf7d4.
        The baseline inlines BF16 K-up BMM without an override hook. This
        model-owned path retains HY4's sink-aware backend and existing V-up
        hook, consuming only real tokens from graph-padded inputs. DCP/PCP
        and dense MHA retain the original BF16 path, selected at construction.
        Output quantization fusion is not
        supported here: HY4 applies its gate before the following Linear.
        """
        if not self.fp8_k_up:
            return super().forward_impl(
                q,
                k_c_normed,
                k_pe,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                quant_group_size,
                quant_scale_ue8m0,
                quant_col_major,
                quant_tma_aligned,
                q_dcp_replicated,
            )
        if (
            output.dtype != q.dtype
            or output_scale is not None
            or output_block_scale is not None
            or q_dcp_replicated is not None
        ):
            raise ValueError("HYV4 FP8 K-up requires unquantized, local-head output")
        if attn_metadata is None:
            # Profile/warmup contains no real KV or query. In particular DP
            # dummy ranks must not route uninitialized attention outputs.
            return output.zero_()
        n = attn_metadata.num_actual_tokens
        if n:
            q_nope, q_pe = q[:n].split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_latent = self._k_up_proj(q_nope)
            attn_out, _ = self.impl.forward_mqa(
                (q_latent, q_pe), kv_cache, attn_metadata, self
            )
            self._v_up_proj(attn_out, out=output[:n])
        output[n:].zero_()
        return output

    # [ILU override] MLAAttention._v_up_proj — delta: FP8 instead of a BF16 bmm,
    # plus HY4's elementwise output gate folded into the store so the layer no
    # longer pays a ``sigmoid_`` and a ``mul_`` over [T, heads * v_head_dim].
    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor) -> None:
        if not self.fuse_v_up:
            super()._v_up_proj(x, out)
            return
        ixf_ops = _ixf_functions()
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        out_3d = out.view(-1, self.num_heads, self.v_head_dim)
        gate = self.gate_logits
        if self.fuse_gate:
            # The module stopped applying the gate itself, so a missing or
            # too-short hand-over must not silently drop it.
            if gate is None:
                raise RuntimeError(
                    f"{self.layer_name}: gate folding is on but the attention "
                    "module handed over no gate logits before attention ran."
                )
            if gate.shape[0] < x.shape[0]:
                raise RuntimeError(
                    f"{self.layer_name}: gate logits cover {gate.shape[0]} "
                    f"tokens, attention produced {x.shape[0]}."
                )
        if gate is not None:
            # In a mixed batch this runs on the leading MQA rows only; the dense
            # MHA prefill rows bypass ``_v_up_proj`` entirely, so the module
            # gates whatever ``gated_rows`` leaves over.
            gate = gate[: x.shape[0]].view(x.shape[0], self.num_heads, self.v_head_dim)
            self.gated_rows = x.shape[0]
        if x.shape[0] <= _FUSED_MAX_TOKENS:
            ixf_ops.hy4_kv_up_gate(x, self.W_UV_FP8, self.W_UV_FP8_S, gate, out=out_3d)
            return
        a, a_s = ixf_ops.fp8_grouped_per_token_quant(x, x.shape[1])
        ixf_ops.hy4_fp8_einsum_gate(
            a, self.W_UV_FP8, a_s, self.W_UV_FP8_S, gate, out=out_3d
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.use_hyv4_fp8_attn:
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=_HYV4_DSA_FP8_ROW_BYTES,
                dtype=torch.uint8,
                cache_dtype_str="fp8",
                non_causal_multi_token_decode=self.non_causal_multi_token_decode,
            )
        return super().get_kv_cache_spec(vllm_config)


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.
    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


def hyv4_layer_skip_topk(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether ``layer_id`` reads a previous full layer's top-k buffer.

    This is the same predicate as ``HYV4MLAAttention.skip_topk``. ``HYV4Model``
    uses it to choose which pipeline-stage boundaries carry the buffer, so the
    two must stay identical: a mismatch makes the send and recv key sets differ
    and the pipeline hangs.

    MTP draft layers (``layer_id >= num_hidden_layers``) always return False.
    They build their own indexer. ``compute_skip_topk_layers`` only covers
    backbone layers, and returning True here would make a stage try to read a
    buffer the draft never receives.
    """
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False
    if not hasattr(config, "index_topk"):
        return False
    layer_types = getattr(config, "layer_types", None)
    if (
        layer_types is None
        or not (0 <= layer_id < len(layer_types))
        or layer_types[layer_id] not in _SPARSE_LAYER_TYPES
    ):
        return False
    return layer_id in compute_skip_topk_layers(config)


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.
    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # Quantized wk weights are upcast to BF16 while loading to keep the
        # fusion (see ``weight_utils.try_load_quantized_indexer_wk``). An
        # explicit runtime group may quantize after both shards are assembled.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=indexer_fused_quant_config(quant_config, prefix),
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self.n_head_scale = self.n_head**-0.5
        # Both scales are constants, so fold them once here: the product is
        # 1/sqrt(index_head_dim * index_n_heads), which is an exact power of two
        # for HY4 (1/64), so the single multiply in ``prepare_inputs`` is one
        # bf16 rounding fewer than applying the two factors in turn.
        self.weights_scale = self.softmax_scale * self.n_head_scale

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        assert cache_config is not None, "HYV4 indexer requires cache_config"
        # Same switch as the main MLA cache. The community layer later swaps in
        # a private cache_dtype="auto" facade; this indexer is built earlier,
        # from the original CacheConfig, so it still sees --kv-cache-dtype.
        kv_cache_dtype = cache_config.cache_dtype or "auto"
        self.use_fp8 = kv_cache_dtype in _HYV4_DSA_FP8_CACHE_DTYPES
        if self.use_fp8:
            # Grouped fp8 K cache + the ds4 fp8 kernel lane; see fp8_indexer.py.
            self.k_cache = HYV4Fp8IndexerCache(
                head_dim=self.head_dim,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
            )
        else:
            # BF16 indexer K cache, matching the other Iluvatar DSA models.
            self.k_cache = IluDeepseekV32IndexerCache(
                head_dim=self.head_dim,
                dtype=torch.bfloat16,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
            )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

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
        if self.use_fp8:
            return hy4_fp8_indexer_forward(
                self, hidden_states, qr, positions, rotary_emb
            )
        hidden_states, q, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        kw, _ = self.wk_weights_proj(hidden_states)
        k = self.k_norm(kw[:, : self.head_dim])

        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims, and both
        # tensors already hold no_pe first, so RoPE can rotate the tail in
        # place. ``ops.rotary_embedding`` reads its strides from the tensor
        # descriptor, so the views below are rotated inside q / k themselves and
        # the two [num_tokens, head_dim] cats that used to reassemble them are
        # gone. q and k only have to agree with each other -- the indexer logits
        # kernel dots over the whole head_dim -- so the checkpoint order is kept.
        nope_dim = self.head_dim - self.rope_dim
        q_pe = q[..., nope_dim:]
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k_pe = k[:, nope_dim:].unsqueeze(1)
        q_pe_out, k_pe_out = apply_qk_rope(
            rotary_emb, positions, q_pe, k_pe, getattr(self, "mrope_emb", None)
        )
        if q_pe_out is not q_pe:
            # The CustomOp fallback (``forward_native``) returns fresh tensors
            # instead of rotating in place, so copy them back.
            q_pe.copy_(q_pe_out.reshape(q_pe.shape))
            k_pe.copy_(k_pe_out.reshape(k_pe.shape))

        # The multiply doubles as the contiguous copy the indexer logits kernel
        # requires: ``weights`` is a [num_tokens, n_head] column slice of the
        # fused wk/weights_proj GEMM, so it is row-strided on its own.
        weights = kw[:, self.head_dim :] * self.weights_scale
        return hidden_states, q, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    `.flashmla_sparse`; if no backend on this platform can consume sinks, the
    weight is still loaded but the bias is disabled with a warning.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
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
        layer_idx: int = 0,
        topk_index_cache: HYV4TopkIndexCache | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = hyv4_layer_skip_topk(config, self.layer_id)
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        self.use_hyv4_fp8_attn = kv_cache_dtype in _HYV4_DSA_FP8_CACHE_DTYPES
        self._fp8_q_heads = (
            _hyv4_fp8_q_heads(self.num_local_heads)
            if self.use_hyv4_fp8_attn
            else self.num_local_heads
        )
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.fused_qkv_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            # ``q_a_proj`` and ``kv_a_proj_with_mqa`` read the same
            # ``hidden_states`` and are both TP-replicated, so they run as one
            # GEMM. The checkpoint keeps them separate; `load_weights` merges
            # them through ``stacked_params_mapping``.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb, self.mrope_emb = build_hy4_ropes(
            qk_rope_head_dim,
            max_position_embeddings,
            config.rope_parameters,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer_mrope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb, self.indexer_mrope_emb = build_hy4_ropes(
                qk_rope_head_dim,
                max_position_embeddings,
                config.rope_parameters,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
            self.indexer.mrope_emb = self.indexer_mrope_emb
        else:
            self.indexer_rope_emb = None
            self.indexer_mrope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
        else:
            self.linear_gate = None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            enable_sink = sink_backend is not None
            self.learnable_sink_param = nn.Parameter(
                torch.full(
                    (self._fp8_q_heads,),
                    float("-inf"),
                    # The kernels require fp32 sinks; the disabled path keeps
                    # the checkpoint dtype since the value is never consumed.
                    dtype=torch.float32 if enable_sink else torch.bfloat16,
                )
            )
            if enable_sink:
                sinks = self.learnable_sink_param
                self._force_sparse_mqa()

        extra_impl_args: dict[str, object] = {} if sinks is None else {"sinks": sinks}
        if self.is_sparse and topk_index_cache is not None:
            # A "shared" layer reads what the closest preceding full layer left
            # in ``topk_indices_buffer``, so it can also reuse everything that
            # layer derived from it (see ``topk_index_cache``).
            extra_impl_args["topk_index_cache"] = topk_index_cache
            extra_impl_args["reuses_topk_indices"] = self.skip_topk
        self.mla_attn = HYV4MLAAttentionLayer(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )
        self.assert_topk_buffer_shared()

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        """The sparse top-k index buffer this layer reads from.

        The real consumer is ``mla_attn.impl``, an ``AttentionImpl`` rather than
        an ``nn.Module``, so a plain ``named_modules()`` walk cannot reach it.
        The MTP proposer rebinds the draft layers onto the target model's buffer
        through such a walk, so expose it here: without this the draft attention
        would keep reading its own buffer while the indices it needs are written
        to the target's.
        """
        return getattr(self.mla_attn.impl, "topk_indices_buffer", None)

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor) -> None:
        # Three objects hold this buffer and all three must move together: the
        # impl *reads* the top-k rows, while the one that *writes* them is the
        # indexer's opaque op. Rebinding only the impl would leave the draft
        # reading the target's rows while its own indexer fills a buffer nobody
        # looks at -- top-k would be silently wrong (no error, accept rate near
        # zero). ``assert_topk_buffer_shared`` checks the invariant.
        self.mla_attn.impl.topk_indices_buffer = buffer  # type: ignore[attr-defined]
        if self.indexer is not None:
            self.indexer.topk_indices_buffer = buffer
            self.indexer.indexer_op.topk_indices_buffer = buffer

    def assert_topk_buffer_shared(self) -> None:
        """Fail fast unless reader and writer share one top-k buffer.

        Cheap (three ``data_ptr()`` reads, no device sync) and only meaningful
        on a layer that owns an indexer: a shared-topk layer has no writer of
        its own by design.
        """
        if self.indexer is None:
            return
        impl_buffer = self.topk_indices_buffer
        if impl_buffer is None:
            return
        holders = {
            "mla_attn.impl": impl_buffer,
            "indexer": self.indexer.topk_indices_buffer,
            "indexer.indexer_op": self.indexer.indexer_op.topk_indices_buffer,
        }
        pointers = {
            name: None if tensor is None else tensor.data_ptr()
            for name, tensor in holders.items()
        }
        if len(set(pointers.values())) != 1:
            raise RuntimeError(
                f"HYV4 layer {self.layer_id} top-k index buffer is not shared "
                f"between its reader and writer: {pointers}. The sparse "
                "attention would read indices the indexer never wrote."
            )

    def _resolve_sink_backend(
        self, kv_cache_dtype: str
    ) -> type[AttentionBackend] | None:
        """Return an MLA backend that can apply this layer's learnable sink.

        The sink is part of the architecture, so a backend that cannot apply it
        changes the model's output. Resolution order:

        1. If the backend the selector would pick already advertises
           `supports_sink`, keep it -- this also honours an explicit
           ``--attention-backend`` choice.
        2. Otherwise fall back to the sink-capable Iluvatar sparse subclass in
           `.flashmla_sparse`, whose ixformer kernel accepts ``attn_sink``,
           provided it validates against the current runtime configuration.
        3. Otherwise give up on the bias rather than failing the load.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The backend class to bind, or None when no sink-capable backend is
            available; the caller then loads the sink weight but disables the
            bias.
        """
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            # Stringify before logging: warning_once dedupes on the arguments,
            # and a fresh exception object per layer would defeat it.
            logger.warning_once(
                "HYV4 failed to select an MLA backend for the learnable sink "
                "(%s); the sink parameter is loaded but the sink bias is "
                "disabled.",
                str(exc),
            )
            return None

        if selected_cls.supports_sink():
            return selected_cls

        from .flashmla_sparse import HYV4FlashMLASparseBackend

        # Mirror how the selector derives the configuration-dependent inputs so
        # this check accepts exactly what the backend would accept at runtime.
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the device compute "
                "capability is unknown. The sink parameter is loaded but the "
                "sink bias is disabled."
            )
            return None
        cache_config = get_current_vllm_config().cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
        )
        if invalid_reasons:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the selected backend %s "
                "cannot apply sinks and the sink-capable Iluvatar sparse path "
                "is invalid here (%s). The sink parameter is loaded but the "
                "sink bias is disabled.",
                selected_cls.get_name(),
                ", ".join(invalid_reasons),
            )
            return None

        logger.info_once(
            "HYV4 learnable sink enabled: using the sink-capable Iluvatar "
            "sparse MLA impl instead of %s, which cannot apply sinks.",
            selected_cls.get_name(),
        )
        return HYV4FlashMLASparseBackend

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def _prepare_mla_qkv(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE, and on the fp8 path fuse RMSNorm + RoPE + KV insert.

        ``q`` and ``k_pe`` are updated in place. Returns the latent that
        ``mla_attn`` should see: raw ``kv_c`` when the fused insert already
        wrote the cache, otherwise the RMSNorm output.
        """
        impl = self.mla_attn.impl
        fused = self._try_fuse_mla_kv(q, kv_c, k_pe, positions)
        impl._skip_fp8_kv_insert = fused
        if fused:
            return kv_c
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q_pe = q[..., self.qk_nope_head_dim :]
        q_out, k_out = apply_qk_rope(
            self.rotary_emb, positions, q_pe, k_pe, self.mrope_emb
        )
        if q_out is not q_pe:
            q_pe.copy_(q_out.reshape(q_pe.shape))
            k_pe.copy_(k_out.reshape(k_pe.shape))
        return kv_c_normed

    def _try_fuse_mla_kv(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        global _MLA_KV_FUSE_LOGGED
        if os.environ.get("VLLM_ILU_HYV4_FUSE_MLA_KV", "1") == "0":
            return False
        impl = getattr(self.mla_attn, "impl", None)
        if impl is None or not getattr(impl, "use_fp8_attn", False):
            return False
        # Under PCP the community MLAAttention layer gathers kv_c/k_pe across
        # ranks before do_kv_cache_update. This fused insert bypasses that path,
        # so it would cache only the local shard. Let the gather path run.
        if getattr(impl, "pcp_world_size", 1) > 1:
            return False
        if not getattr(impl, "_fp8_kv_scale_frozen", False):
            return False
        if q.shape[-2] not in (8, 16, 32, 64):
            return False
        if self.kv_a_layernorm.weight.dtype != kv_c.dtype:
            return False
        kv_cache = self.mla_attn.kv_cache
        if kv_cache.numel() == 0:
            return False
        try:
            import ixformer.inference.functions as ixf_ops
        except Exception:
            return False
        if not hasattr(ixf_ops, "fused_glm_dsa_rmsnorm_rope_fp8_kv_insert"):
            return False
        ctx = get_forward_context()
        slot_mapping = getattr(ctx, "slot_mapping", None)
        if not isinstance(slot_mapping, dict):
            return False
        slots = slot_mapping.get(self.mla_attn.layer_name)
        if slots is None:
            return False
        n = q.shape[0]
        slots = slots.reshape(-1)
        # The fused insert consumes a 1-D table; use the M-RoPE path for all
        # three-row inputs without inspecting device values during capture.
        if positions.ndim == 2:
            return False
        pos = positions.reshape(-1)
        if slots.numel() < n or pos.numel() < n or kv_c.shape[0] < n:
            return False
        q_pe = q[..., self.qk_nope_head_dim :]
        ixf_ops.fused_glm_dsa_rmsnorm_rope_fp8_kv_insert(
            q_pe,
            k_pe,
            kv_c[:n],
            self.kv_a_layernorm.weight,
            float(self.kv_a_layernorm.variance_epsilon),
            kv_cache,
            slots[:n].contiguous(),
            pos[:n].contiguous(),
            self.rotary_emb._match_cos_sin_cache_dtype(q_pe),
            self.mla_attn._k_scale,
            int(kv_cache.shape[1]),
        )
        if not _MLA_KV_FUSE_LOGGED:
            logger.info(
                "HYV4 MLA RMSNorm+RoPE+fp8 KV insert fusion armed "
                "(VLLM_ILU_HYV4_FUSE_MLA_KV=1)"
            )
            _MLA_KV_FUSE_LOGGED = True
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.q_proj is not None
            assert self.kv_a_proj_with_mqa is not None
            q = self.q_proj(hidden_states)[0]
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        kv_c_normed = self._prepare_mla_qkv(q, kv_c, k_pe, positions)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # HY4's elementwise gate is folded into the absorbed v-up store, so the
        # gate GEMM has to run *before* attention and hand its logits over. The
        # headwise variant reshapes the output and is left on the eager path.
        fused_gate = (
            self.gated_mla
            and self.linear_gate is not None
            and self.config.gating_type != "headwise"
            and getattr(self.mla_attn, "fuse_gate", False)
        )
        gate_score = None
        if fused_gate:
            gate_score = self.linear_gate(hidden_states)[0]
            self.mla_attn.gate_logits = gate_score
            self.mla_attn.gated_rows = 0

        if is_breakable_cudagraph_enabled():
            # Breakable CUDA graphs replay the eager segment without a Python
            # return value, so it needs a caller-owned stable output buffer.
            attn_out = torch.empty(
                output_shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            self._indexer_and_attn(
                hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
            )
        else:
            # FULL_DECODE_ONLY (the HY4 path) captures the whole function and
            # does not need the breakable in-place contract. Return the
            # MLAAttention buffer directly, avoiding one [T,H*V] allocation and
            # one full copy per layer.
            if self.indexer is not None and self.is_sparse and not self.skip_topk:
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=output_shape,
            )

        if fused_gate:
            gated = self.mla_attn.gated_rows
            self.mla_attn.gate_logits = None
            if gated < attn_out.shape[0]:
                # Mixed batch: the dense-MHA prefill rows never reached the
                # gated v-up, so gate the tail here.
                assert gate_score is not None
                attn_out[gated:].mul_(gate_score[gated:].sigmoid())
        elif self.gated_mla and self.linear_gate is not None:
            gate_score = self.linear_gate(hidden_states)[0]
            if self.config.gating_type == "headwise":
                gate_score = gate_score.unsqueeze(-1)
                attn_out = attn_out.reshape(*attn_out.shape[:-1], -1, self.v_head_dim)
                attn_out = attn_out * torch.sigmoid(gate_score)
                attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
            else:
                # HY4 is ``elementwise``, one gate per output element. Both
                # operands are this layer's own freshly allocated buffers (the
                # gate GEMM output and, since O2, the MLAAttention output), so
                # gating in place drops two [num_tokens, heads * v_head_dim]
                # allocations per layer without aliasing anything o_proj reads.
                attn_out = attn_out.mul_(gate_score.sigmoid_())

        out, _ = self.o_proj(attn_out)
        return out

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary.
        """
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
            )
        )
