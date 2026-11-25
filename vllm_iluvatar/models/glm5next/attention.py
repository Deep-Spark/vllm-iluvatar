# SPDX-License-Identifier: Apache-2.0
"""GLM5Next projections and complete Iluvatar Kpool indexing."""

from types import MethodType

import ixformer.inference.functions as ixfops
import torch
import triton
import triton.language as tl
import vllm.envs as envs
from ixformer.cuda import cuda_arch
from ixformer.inference import ixattn_backend
from ixformer.inference.functions.models.glm5next import (
    fwht128_quant_int8_with_weights,
    global_pool_indices,
)
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
from vllm.models.glm5next.sparse_indexer import SparseAttnIndexerKpool
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.ilu_util.arch import is_ivcore11, is_ivcore30
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (  # noqa: E501
    IluW8A8OnlineLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import IluFp8Config


def resolve_glm5next_indexer_kv_dtype(
    quant_config: QuantizationConfig | None = None,
) -> str:
    if envs.is_set("VLLM_ILU_INDEXER_KV_DTYPE"):
        return envs.VLLM_ILU_INDEXER_KV_DTYPE
    if is_ivcore30() and isinstance(quant_config, IluFp8Config):
        return "fp8"
    return "bf16"


@triton.jit
def _scale_cast_bf16_kernel(
    in_ptr,
    out_ptr,
    scale,
    T,
    H,
    in_s0,
    in_s1,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_H)
    mask = col < H
    x = tl.load(in_ptr + row * in_s0 + col * in_s1, mask=mask, other=0.0)
    tl.store(out_ptr + row * H + col, (x * scale).to(tl.bfloat16), mask=mask)


def _scale_to_bf16(weights: torch.Tensor, scale: float) -> torch.Tensor:
    T, H = weights.shape
    out = torch.empty((T, H), dtype=torch.bfloat16, device=weights.device)
    _scale_cast_bf16_kernel[(T,)](
        weights,
        out,
        scale,
        T,
        H,
        weights.stride(0),
        weights.stride(1),
        BLOCK_H=next_power_of_2(H),
    )
    return out


def glm5next_uses_compact_pool_indices(vllm_config: VllmConfig, kpool: int) -> bool:
    if kpool == 4 and glm5next_fp8_attn_enabled_for_config(vllm_config):
        return True
    if not (
        kpool == 4
        and vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        and vllm_config.model_config.dtype == torch.bfloat16
    ):
        return False
    arch = getattr(torch, "corex_arch", None)
    if arch == "ivcore11":
        return True
    if arch != "ivcore30" or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8":
        return False

    config = vllm_config.model_config.hf_text_config
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    return (
        ixattn_backend.ixattn_available()
        and config.kv_lora_rank == 512
        and config.qk_nope_head_dim == 256
        and config.qk_rope_head_dim == 0
        and config.num_attention_heads % tp_size == 0
        and config.num_attention_heads // tp_size in (16, 32, 64)
    )


def glm5next_topk_buffer_width(
    vllm_config: VllmConfig, topk_tokens: int, kpool: int
) -> int:
    if glm5next_uses_compact_pool_indices(vllm_config, kpool):
        return topk_tokens // kpool
    width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
    return (width + 127) // 128 * 128


logger = init_logger(__name__)

# One set of query staging tensors per (device, capacity, heads, kv_lora_rank),
# shared across every NoPE MLA layer in the process for zero-alloc CUDA graphs.
_FP8_MLA_SCRATCH: dict[tuple, dict[str, torch.Tensor]] = {}

# The architectures this module's layers back, for config-phase checks that run
# before the model class is resolved.
GLM5NEXT_ARCHS = (
    "Glm5NextForConditionalGeneration",
    "Glm5NextForCausalLM",
)


def _glm5next_fp8_mla_unmet_reason(vllm_config: VllmConfig) -> str | None:
    if not is_ivcore30() or getattr(torch, "corex_arch", "ivcore30") != "ivcore30":
        return "requires ivcore30"
    if not ixattn_backend.ixattn_available():
        return "ixformer built without CQ ixattn"
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "dtype", None) != torch.bfloat16:
        return f"requires dtype=bfloat16, got {getattr(model_config, 'dtype', None)}"
    hf_text = getattr(model_config, "hf_text_config", None)
    if (
        getattr(hf_text, "qk_rope_head_dim", None) != 0
        or getattr(hf_text, "kv_lora_rank", None) != 512
        or getattr(hf_text, "qk_nope_head_dim", None) != 256
        or getattr(hf_text, "index_kpool", None) != 4
    ):
        return (
            "requires NoPE MLA (qk_rope_head_dim=0, kv_lora_rank=512, "
            "qk_nope_head_dim=256, index_kpool=4)"
        )
    num_heads = getattr(hf_text, "num_attention_heads", None)
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    tp = (
        getattr(parallel_cfg, "tensor_parallel_size", 1)
        if parallel_cfg is not None
        else 1
    )
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    draft_tp = (
        getattr(spec_cfg, "draft_tensor_parallel_size", None)
        if spec_cfg is not None
        else None
    )
    for shard_tp in (tp,) if draft_tp is None else (tp, draft_tp):
        if (
            not isinstance(num_heads, int)
            or shard_tp <= 0
            or num_heads % shard_tp != 0
            or (num_heads // shard_tp) not in (16, 32, 64)
        ):
            return f"local heads ({num_heads} // tp={shard_tp}) must be 16, 32, or 64"
    return None


def glm5next_fp8_attn_enabled_for_config(vllm_config: VllmConfig | None) -> bool:
    """Single-source-of-truth config predicate for CQ GLM5Next FP8 NoPE MLA."""
    if vllm_config is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    archs = getattr(model_config, "architectures", None) or []
    if not any(arch in archs for arch in GLM5NEXT_ARCHS):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    requested = (
        envs.VLLM_ILU_GLM5_FP8_ATTN
        if envs.is_set("VLLM_ILU_GLM5_FP8_ATTN")
        else bool(
            getattr(cache_config, "_ilu_glm5next_fp8_kv", False)
            or envs.VLLM_ILU_GLM5_FP8_ATTN
        )
    )
    if not requested:
        return False
    reason = _glm5next_fp8_mla_unmet_reason(vllm_config)
    if reason is not None:
        logger.warning_once(
            "Disabling GLM5Next FP8 MLA (%s); falling back to BF16 MLA.",
            reason,
        )
        return False
    return True


def glm5next_fp8_attn_enabled(
    qk_rope_head_dim: int,
    vllm_config: VllmConfig | None = None,
) -> bool:
    return qk_rope_head_dim == 0 and glm5next_fp8_attn_enabled_for_config(vllm_config)


def _fp8_mla_scratch(
    device: torch.device,
    capacity: int,
    num_heads: int,
    kv_lora_rank: int,
) -> dict[str, torch.Tensor]:
    key = (device, capacity, num_heads, kv_lora_rank)
    scratch = _FP8_MLA_SCRATCH.get(key)
    if scratch is None:
        scratch = {
            "q_fp8": torch.empty(
                (capacity, num_heads, kv_lora_rank),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            "q_scale": torch.empty(
                (capacity, num_heads), dtype=torch.float32, device=device
            ),
            "empty_pe": torch.empty(
                (capacity, 1, 0), dtype=torch.bfloat16, device=device
            ),
            "unit_kv_scale": torch.ones((1,), dtype=torch.float32, device=device),
        }
        _FP8_MLA_SCRATCH[key] = scratch
    return scratch


def _resolve_fp8_kv_scale(layer, scratch: dict[str, torch.Tensor]) -> torch.Tensor:
    k_scale = getattr(layer, "_k_scale", None)
    if (
        isinstance(k_scale, torch.Tensor)
        and k_scale.is_cuda
        and k_scale.dtype == torch.float32
        and k_scale.numel() == 1
    ):
        return k_scale.reshape(1)
    return scratch["unit_kv_scale"]


def _fp8_mla_get_kv_cache_spec(self, vllm_config: VllmConfig) -> MLAAttentionSpec:
    """512 pure-E4M3 bytes per token (no RoPE tail, per-tensor static scale)."""
    return MLAAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=self.head_size,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str="fp8_e4m3",
        kv_quant_mode=get_kv_quant_mode("fp8_e4m3"),
        is_index_group_leader=self.indexer is not None,
        non_causal_multi_token_decode=self.non_causal_multi_token_decode,
        block_stride_alignment=self.head_size,
    )


def _fp8_mla_forward_mqa(
    self,
    q: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata,
    layer,
    kv_cache_scale: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA over E4M3 query and latent cache via ixattn kpool=4."""
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return q
    scratch = self._fp8_mla_scratch
    assert num_tokens <= scratch["q_fp8"].shape[0], (
        f"FP8 NoPE MLA staging holds {scratch['q_fp8'].shape[0]} tokens, "
        f"got {num_tokens}"
    )
    q_fp8 = scratch["q_fp8"][:num_tokens]
    q_scale = scratch["q_scale"][:num_tokens]
    ixfops.dynamic_scaled_fp8_quant(q_fp8, q, q_scale)

    assert self.topk_indices_buffer is not None
    selected_pools = attn_metadata.topk_tokens // 4
    pool_ids = self.topk_indices_buffer[:num_tokens, :selected_pools]
    if getattr(self, "use_ixattn_compact_pools", False) and positions is not None:
        phys_pools, topk_lens = self._ixattn_physical_pool_rows(
            num_tokens, pool_ids, attn_metadata, positions
        )
    else:
        phys_pools, topk_lens = global_pool_indices(
            pool_ids,
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            attn_metadata.block_size,
            expand=False,
        )

    kv_scale = _resolve_fp8_kv_scale(layer, scratch)
    ixfops.flash_mla_sparse_fwd(
        q=q_fp8,
        kv=kv_c_and_k_pe_cache.view(-1, 1, self.kv_lora_rank),
        indices=phys_pools.view(num_tokens, 1, -1),
        sm_scale=self.softmax_scale,
        topk_length=topk_lens,
        output=q,
        kpool=4,
        q_scale=q_scale,
        kv_scale=kv_scale,
        backend=backend if backend is not None else "ixattn",
    )
    return q


def _fp8_mla_forward_full(
    self,
    layer,
    q: torch.Tensor,
    k_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
    positions: torch.Tensor | None = None,
    kv_cache_scale: list[torch.Tensor] | None = None,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    """NoPE sparse MLA forward_full writing E4M3 KV cache and running ixattn."""
    if attn_metadata is None:
        num_toks = q.shape[0]
        _ = torch.empty(
            (num_toks, self.num_heads, self.kv_lora_rank),
            device=k_c_normed.device,
            dtype=k_c_normed.dtype,
        )
        _ = torch.empty((num_toks, 576), device=k_c_normed.device, dtype=torch.int32)
        return torch.empty(
            output.shape[0],
            self.v_head_dim * self.num_heads,
            device=q.device,
            dtype=q.dtype,
        )
    if self.dcp_world_size == -1:
        self.dcp_world_size = get_dcp_group().world_size

    num_actual_toks = attn_metadata.num_actual_tokens
    q = q[:num_actual_toks]
    k_c_normed = k_c_normed[:num_actual_toks]
    if num_actual_toks < output.shape[0]:
        output[num_actual_toks:].zero_()
    if num_actual_toks > 0:
        scratch = self._fp8_mla_scratch
        ql_nope = q.new_empty((num_actual_toks, self.num_heads, self.kv_lora_rank))
        self._k_up_proj(q, out=ql_nope)
        if kv_cache.numel() > 0:
            ixfops.concat_and_cache_mla(
                k_c_normed,
                scratch["empty_pe"][:num_actual_toks],
                kv_cache,
                attn_metadata.slot_mapping.flatten()[:num_actual_toks],
                kv_cache_dtype="fp8_e4m3",
                scale=_resolve_fp8_kv_scale(layer, scratch),
            )
        attn_out = self.forward_mqa(
            ql_nope,
            kv_cache,
            attn_metadata,
            layer,
            backend=backend,
            positions=positions,
        )
        output_view = output.view(-1, self.num_heads, self.v_head_dim)
        self._v_up_proj(attn_out, out=output_view[:num_actual_toks])
    return output.view(output.shape[0], self.v_head_dim * self.num_heads)


def install_glm5next_fp8_mla(layer, vllm_config: VllmConfig) -> None:
    """Route one validated NoPE MLA layer onto ixattn's kpool=4 FP8 forward."""
    impl = layer.impl
    assert (
        layer.head_size == layer.kv_lora_rank == 512
        and impl.num_heads in (16, 32, 64)
        and impl.softmax_scale == 1 / 16
        and (
            impl._indexer is None
            or getattr(impl._indexer, "compact_pool_indices", False)
        )
    )
    layer.kv_cache_dtype = "fp8_e4m3"
    impl.kv_cache_dtype = "fp8_e4m3"

    device = torch.device(
        vllm_config.device_config.device_type, torch.cuda.current_device()
    )
    impl._fp8_mla_scratch = _fp8_mla_scratch(
        device,
        vllm_config.scheduler_config.max_num_batched_tokens,
        impl.num_heads,
        impl.kv_lora_rank,
    )

    layer.get_kv_cache_spec = MethodType(_fp8_mla_get_kv_cache_spec, layer)
    impl.forward_full = MethodType(_fp8_mla_forward_full, impl)
    impl.forward_mqa = MethodType(_fp8_mla_forward_mqa, impl)


class IluGlm5NextIndexerCache(DeepseekV32IndexerCache):
    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
        scale_fmt: str,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "IluGlm5NextIndexerCache expects index_kpool > 1"
        assert cache_config.block_size % index_kpool == 0, (
            "IluGlm5NextIndexerCache: cache_config.block_size "
            f"({cache_config.block_size}) must be a multiple of index_kpool "
            f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
        )
        self._index_kpool = index_kpool
        self._scale_fmt = scale_fmt

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, tokens_per_state=self._index_kpool)

        storage_block_size = spec.block_size // self._index_kpool
        assert (
            spec.block_size % self._index_kpool == 0 and storage_block_size % 32 == 0
        ), (
            "IluGlm5NextIndexerCache: kpool indexer requires cache block_size to "
            f"be a multiple of index_kpool * 32 ({self._index_kpool * 32}) so "
            "that DeepGEMM paged-MQA pool pages (32 or 64 entries) tile the "
            f"storage block, got block_size={spec.block_size} -> "
            f"storage_block_size={storage_block_size}."
        )
        max_page_size = max(PAGED_MQA_PAGE_SIZES)
        min_page_size = min(PAGED_MQA_PAGE_SIZES)
        if storage_block_size <= max_page_size:
            page_size = storage_block_size
        elif storage_block_size % max_page_size == 0:
            page_size = max_page_size
        else:
            page_size = min_page_size
        assert self._scale_fmt == "None" or page_size in (64, 256), (
            "IluGlm5NextIndexerCache: the quantized index cache needs a pool page of "
            f"64 or 256, got {page_size} from block_size={spec.block_size} and "
            f"index_kpool={self._index_kpool}. Launch with --block-size "
            f"{64 * self._index_kpool} (or any multiple of it)."
        )
        return replace(
            spec,
            storage_block_size=page_size * self._index_kpool,
        )

    def get_attn_backend(self):
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            IluDeepseekV32IndexerBackend,
        )

        return IluDeepseekV32IndexerBackend


class IluGlm5NextTailCache(DeepseekV32IndexerCache):
    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "IluGlm5NextTailCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        span = self._index_kpool + vllm_config.num_speculative_tokens
        ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))
        assert self.cache_config.block_size % ring == 0, (
            f"IluGlm5NextTailCache: cache_config.block_size "
            f"({self.cache_config.block_size}) must be a multiple of the "
            f"tail ring ({ring})"
        )
        return KpoolTailSpec(
            block_size=ring,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=ring,
        )

    def get_attn_backend(self):
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            IluKpoolTailBackend,
        )

        return IluKpoolTailBackend


class IluIndexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
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
        self.supports_query_stream_overlap = is_ivcore11()
        self.quant_config = quant_config
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
        )

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.register_buffer("_kw_gate_weight", None, persistent=False)
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.quant_block_size = 128
        indexer_kv_dtype = resolve_glm5next_indexer_kv_dtype(quant_config)
        if indexer_kv_dtype == "bf16":
            self.k_cache_dtype = torch.bfloat16
            self.k_cache_head_dim = self.head_dim
            self.scale_fmt = "None"
        elif indexer_kv_dtype == "int8":
            self.k_cache_dtype = torch.uint8
            self.k_cache_head_dim = self.head_dim + 4
            self.scale_fmt = "float32"
        else:
            self.k_cache_dtype = torch.uint8
            self.k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
            self.scale_fmt = "ue8m0"
        self.topk_indices_buffer = topk_indices_buffer
        self._wp_fp32: torch.Tensor | None = None
        self._k_norm_cast: tuple[torch.Tensor, torch.Tensor] | None = None

        self.k_cache = IluGlm5NextIndexerCache(
            head_dim=self.k_cache_head_dim,
            dtype=self.k_cache_dtype,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
            scale_fmt=self.scale_fmt,
        )
        self.tail_cache = IluGlm5NextTailCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.tail_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
        )
        self.max_pool_len = vllm_config.model_config.max_model_len // self.index_kpool
        self.prefix = prefix
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.compact_pool_indices = glm5next_uses_compact_pool_indices(
            vllm_config, self.index_kpool
        )
        self.indexer_op = SparseAttnIndexerKpool(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            tail_cache=self.tail_cache,
        )
        self.indexer_op.compact_pool_indices = self.compact_pool_indices

    def prepare_projection_weights(self) -> None:
        previous = self._kw_gate_weight
        self._kw_gate_weight = None
        wk = self.wk_weights_proj
        gate = self.index_kpool_compress_gate
        if (
            cuda_arch != "ivcore30"
            or not wk.weight.is_cuda
            or not isinstance(wk.quant_method, UnquantizedLinearMethod)
            or wk.weight.dtype not in (torch.bfloat16, torch.float16)
            or wk.weight.dtype != gate.dtype
            or wk.weight.device != gate.device
        ):
            return
        split = self.head_dim + self.n_head
        shape = (split + self.head_dim, wk.weight.shape[1])
        if (
            previous is not None
            and previous.shape == shape
            and previous.dtype == wk.weight.dtype
            and previous.device == wk.weight.device
        ):
            combined = previous
            if wk.weight.data_ptr() != combined.data_ptr():
                combined[:split].copy_(wk.weight.detach())
            if gate.data_ptr() != combined[split:].data_ptr():
                combined[split:].copy_(gate.detach())
        else:
            combined = torch.cat((wk.weight.detach(), gate.detach()))
        # Keep checkpoint names and loader attributes on the original Parameters.
        wk.weight.data = combined[:split]
        gate.data = combined[split:]
        self._kw_gate_weight = combined

    def _project_k_weights_gate(self, hidden_states: torch.Tensor):
        # Larger batches keep the separate GEMMs: the merged-N dispatch is not
        # uniformly faster across prefill shapes.
        if self._kw_gate_weight is not None and hidden_states.shape[0] <= 128:
            output = ixfops.linear(hidden_states, self._kw_gate_weight)
            split = self.head_dim + self.n_head
            return (
                output[:, : self.head_dim],
                output[:, self.head_dim : split],
                output[:, split:],
            )
        kw = ixfops.linear(hidden_states, self.wk_weights_proj.weight)
        return kw[:, : self.head_dim], kw[:, self.head_dim :], None

    def _fused_indexer_k_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dim: int,
        eps: float,
    ) -> torch.Tensor:
        if self._k_norm_cast is None or self._k_norm_cast[0].dtype != x.dtype:
            self._k_norm_cast = (weight.to(x.dtype), bias.to(x.dtype))
        cast_weight, cast_bias = self._k_norm_cast
        return ixfops.layer_norm(x, [dim], cast_weight, cast_bias, eps)

    def _fused_indexer_weight_scale(
        self, weights: torch.Tensor, q_scale: torch.Tensor, scale: float
    ) -> torch.Tensor:
        return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        k, weights, gate_score = self._project_k_weights_gate(hidden_states)

        k = self._fused_indexer_k_norm(
            k, self.k_norm.weight, self.k_norm.bias, self.head_dim, self.k_norm.eps
        )
        if self.rope_dim > 0:
            q, k = rotary_emb(positions, q, k.unsqueeze(1))

        assert self.head_dim == 128 and self.quant_block_size == 128
        weight_scale = self.softmax_scale * self.n_head**-0.5
        if self.scale_fmt == "ue8m0":
            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = fwht128_quant_fp8(q)
            q = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)
            weights = self._fused_indexer_weight_scale(weights, q_scale, weight_scale)
        elif self.scale_fmt == "float32":
            q, weights = fwht128_quant_int8_with_weights(q, weights, weight_scale)
        else:
            weights = _scale_to_bf16(weights, weight_scale)

        if gate_score is None:
            gate_score = ixfops.linear(hidden_states, self.index_kpool_compress_gate)

        return self.indexer_op(
            hidden_states,
            q,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )


def prepare_indexer_projection_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, IluIndexer):
            module.prepare_projection_weights()


class IluGlm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
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
        input_size: int | None = None,
        skip_rope: bool | None = False,
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

        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

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
                proj_input_size,
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

        if envs.VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8 and (
            quant_config is None or quant_config.get_name() == "ilu_f4f8"
        ):
            projections = (
                (self.fused_qkv_a_proj, self.q_b_proj, self.o_proj)
                if self.q_lora_rank is not None
                else (self.kv_a_proj_with_mqa, self.q_proj, self.o_proj)
            )
            for projection in projections:
                if not isinstance(projection.quant_method, UnquantizedLinearMethod):
                    continue
                projection.quant_method = IluW8A8OnlineLinearMethod(
                    projection.quant_method
                )

        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_llama_scaling"
                    if config.rope_parameters.get("attention_factor") == 1.0
                    else "deepseek_yarn"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
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
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        self.is_v32 = config.index_topk is not None

        if self.is_v32:
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not config.indexer_rope_interleave,
            )
            assert q_lora_rank is not None
            self.indexer: IluIndexer | None = IluIndexer(
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
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            None,
            prefix,
            skip_topk=False,
            fuse_qkv_rmsnorm=True,
        )

        self.fp8_attn = self.is_v32 and glm5next_fp8_attn_enabled(
            self.qk_rope_head_dim, vllm_config
        )
        if self.fp8_attn:
            assert topk_indices_buffer is not None
            install_glm5next_fp8_mla(self.mla_attn.mla_attn, vllm_config)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)
