# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 native FP8 attention integration on the 0.31 plugin API.

Cache allocation and scheduling remain in vLLM. This module owns only metadata
adaptation, the independent ixformer call sequence and preallocated buffers.
The initial service route is eager; decode Graph is enabled only after its
framework metadata/capture path has been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import ixformer.inference.functions as ixf
import torch
from ixformer.inference.backend_registry import registry
from ixformer.inference.functions.models.minimax_m3.data_quant.fp8.fp8_attention_output_quant import (
    fp8_attention_output_quant as _ixf_fp8_attention_output_quant,
)
from ixformer.inference.functions.models.minimax_m3.fused_kv.fp8.qknorm_rope_fp8_kv_insert import (
    qknorm_rope_fp8_kv_insert,
)
from ixformer.inference.functions.models.minimax_m3.indexer.float.index_topk import (
    index_topk,
)
from ixformer.inference.functions.models.minimax_m3.indexer.fp8.fp8_index_score import (
    fp8_index_score,
)
from ixformer.inference.functions.models.minimax_m3.sparse.fp8.fp8_dsa import fp8_dsa
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend as CommunityIndexerBackend,
)
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerCache
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend as CommunitySparseBackend,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.models.minimax_m3.mxfp8.attention import MiniMAXGemmaRMSNorm

logger = init_logger(__name__)


def _resolve_fp8_indexer_dtype(attention_config):
    dtype = attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype != "fp8":
        raise NotImplementedError("MiniMax FP8 requires E4M3 index cache")
    return dtype


@dataclass(frozen=True)
class DenseQuerySegment:
    request_start: int
    request_end: int
    token_start: int
    token_end: int
    decode_query_len: int
    cu: torch.Tensor | None = None


@dataclass
class FP8Metadata(AttentionMetadata):
    num_actual_tokens: int
    cu: torch.Tensor
    seq: torch.Tensor
    prefix: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # Host counts also drive the MoE graph-safe decode dispatch.
    num_prefills: int = 0
    num_decodes: int = 0
    dense_segments: tuple[DenseQuerySegment, ...] | None = None
    dense_valid_tokens: int | None = None


class FP8MetadataBuilder(AttentionMetadataBuilder[FP8Metadata]):
    # Prefill/mixed batches keep the eager metadata partitioning path. Decode
    # kernels consume stable buffers whose values change between replays.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    reorder_batch_threshold = None

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        spec = vllm_config.speculative_config
        if spec is not None and spec.use_dspark():
            # Uniform verification reuses the causal ragged DSA path. cu/seq,
            # block tables and slots remain runner-owned stable buffers; the
            # per-builder prefix buffer is refreshed before every replay.
            # Scratch includes all bonus+draft query tokens (allocated below).
            return AttentionCGSupport.UNIFORM_BATCH
        return cls._cudagraph_support

    def build_for_cudagraph_capture(self, common_attn_metadata):
        c = common_attn_metadata
        lengths = (
            c.query_start_loc_cpu[1 : c.num_reqs + 1]
            - c.query_start_loc_cpu[: c.num_reqs]
        )
        if lengths.numel() and not bool((lengths == self.capture_query_width).all()):
            raise ValueError(
                "MiniMax FP8 Graph requires uniform configured verification width"
            )
        return self.build(0, c)

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        spec = vllm_config.speculative_config
        self.capture_query_width = 1 + (
            spec.num_speculative_tokens if spec is not None else 0
        )
        self.prefix_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        c = common_attn_metadata
        n = c.num_reqs
        # Framework metadata prep, outside attention forward/capture. This is
        # one GPU-to-GPU batch copy, never a device-scalar Python decision.
        prefix = self.prefix_buffer[:n]
        prefix.copy_(c.compute_num_computed_tokens()[:n], non_blocking=True)
        # Classify using framework CPU metadata, preserving the existing
        # request order shared by the main cache and DSA indexer. Consecutive
        # single-query requests form decode segments; other requests use ragged
        # prefill metadata. No GPU scalar read or scheduler reorder is added.
        cpu_cu = getattr(c, "query_start_loc_cpu", None)
        if cpu_cu is None and c.query_start_loc.device.type == "cpu":
            cpu_cu = c.query_start_loc
        if cpu_cu is None or cpu_cu.device.type != "cpu":
            raise ValueError("FP8 dense partition requires framework CPU query starts")
        starts = cpu_cu[: n + 1].tolist()
        if (
            len(starts) != n + 1
            or starts[0] != 0
            or any(b < a for a, b in zip(starts, starts[1:]))
        ):
            raise ValueError("Invalid framework CPU query starts")
        if starts[-1] > c.num_actual_tokens:
            raise ValueError("Query end exceeds attention token capacity")
        segments = []
        first = 0
        while first < n:
            decode = starts[first + 1] - starts[first] == 1
            last = first + 1
            while last < n and (starts[last + 1] - starts[last] == 1) == decode:
                last += 1
            begin, end = starts[first], starts[last]
            # DSA also needs ragged cu for decode; prepare it outside forward.
            cu = c.query_start_loc[first : last + 1]
            if begin:
                cu = cu - begin
            segments.append(DenseQuerySegment(first, last, begin, end, int(decode), cu))
            first = last
        return FP8Metadata(
            num_actual_tokens=c.num_actual_tokens,
            cu=c.query_start_loc[: n + 1],
            seq=c.seq_lens[:n],
            prefix=prefix,
            block_table=c.block_table_tensor[:n],
            max_query_len=c.max_query_len,
            max_seq_len=c.max_seq_len,
            num_prefills=sum(b - a > 1 for a, b in zip(starts, starts[1:])),
            num_decodes=sum(b - a == 1 for a, b in zip(starts, starts[1:])),
            dense_segments=tuple(segments),
            dense_valid_tokens=starts[-1],
        )


class MiniMaxM3FP8Backend(CommunitySparseBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name():
        return "MINIMAX_M3_ILU_FP8"

    @staticmethod
    def get_builder_cls():
        return FP8MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [128]

    @classmethod
    def supported_kv_cache_layouts(cls):
        # Native kernels require contiguous H/N/C inside each physical page.
        # Both layouts permit official allocator page padding/interleaving.
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


class MiniMaxM3FP8IndexerBackend(CommunityIndexerBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name():
        return "MINIMAX_M3_ILU_FP8_INDEXER"

    @staticmethod
    def get_builder_cls():
        return FP8MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [128]

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


class MiniMaxM3FP8IndexerCache(MiniMaxM3IndexerCache):
    def bind_kv_cache(self, cache):
        if (
            cache.dtype != torch.float8_e4m3fn
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != (1, 128, 128)
            or cache.stride(3) != 1
            or cache.stride(2) != 128
            or cache.stride(0) < 128 * 128
        ):
            raise ValueError(
                "M3 FP8 index allocator must bind E4M3[P,1,128,128] with valid page stride"
            )
        super().bind_kv_cache(cache)
        logger.info_once(
            "M3 FP8 index cache bound: shape=%s stride=%s dtype=%s",
            tuple(self.kv_cache.shape),
            self.kv_cache.stride(),
            self.kv_cache.dtype,
        )


class _State:
    def __init__(self, config):
        self.device = config.device_config.device
        self.buffers = {}

    def buffer(self, name, shape, dtype):
        if name not in self.buffers:
            self.buffers[name] = torch.zeros(shape, dtype=dtype, device=self.device)
        value = self.buffers[name]
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise RuntimeError(f"FP8 shared buffer contract changed for {name}")
        return value


_STATES: dict[int, _State] = {}


def get_state(config, *, create=False):
    # with_hf_config() creates a distinct VL text config but preserves the
    # compilation/static layer context used by the outer worker. Own the
    # scratch lifetime with that shared context, not a temporary config wrapper.
    key = id(config.compilation_config.static_forward_context)
    if create and key not in _STATES:
        _STATES[key] = _State(config)
    return _STATES.get(key)


class MiniMaxM3FP8Attention(nn.Module, AttentionLayerBase):
    """Independent Full/DSA primitive chain over framework-owned packed caches."""

    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        cache_config=None,
        *,
        sparse=False,
    ):
        super().__init__()
        vc = get_current_vllm_config()
        if not is_ivcore30() or vc.model_config.dtype != torch.bfloat16:
            raise NotImplementedError(
                "MiniMax FP8 requires ivcore30/BF16 model activations"
            )
        if cache_config is None or cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
            raise NotImplementedError("MiniMax FP8 requires explicit E4M3 main cache")
        indexer_kv_dtype = _resolve_fp8_indexer_dtype(vc.attention_config)
        if cache_config.block_size != 128:
            raise NotImplementedError(
                "Initial M3 FP8 framework route is B128; other kernel sizes remain separately tested"
            )
        if (
            vc.parallel_config.prefill_context_parallel_size != 1
            or vc.parallel_config.decode_context_parallel_size != 1
        ):
            raise NotImplementedError("Initial M3 FP8 integration requires PCP1/DCP1")
        if getattr(vc.parallel_config, "enable_dbo", False):
            raise NotImplementedError(
                "MiniMax FP8 does not support concurrent DBO scratch use"
            )
        if vc.speculative_config is not None and (
            not vc.speculative_config.use_dspark()
            or vc.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "MiniMax FP8 speculative path requires DSpark with PP1"
            )
        if config.hidden_size != 6144 or config.head_dim != 128:
            raise NotImplementedError(
                "MiniMax FP8 native producer/projection requires C6144/D128"
            )
        if sparse and "ixformer_native" not in registry.available_backends("fp8_dsa"):
            raise RuntimeError("MiniMax FP8 DSA requires its matching native build")
        if not sparse:
            for name in ("paged_attention_prefill_fp8", "paged_attention_decode_fp8"):
                if "ixformer_native" not in registry.available_backends(name):
                    raise RuntimeError(
                        "MiniMax FP8 Full attention requires the generic native paged-attention build"
                    )
        tp = get_tensor_model_parallel_world_size()
        if tp not in (1, 2, 4, 8):
            raise NotImplementedError("MiniMax FP8 TP whitelist is 1/2/4/8")
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim, self.hidden_size = 128, 6144
        self.layer_id, self.sparse, self.layer_name = layer_id, sparse, prefix + ".attn"
        self.kv_cache_dtype = cache_config.cache_dtype
        self.state = get_state(vc, create=True)
        # DSpark verification schedules the bonus plus speculative tokens.
        # Target scratch is distinct from the dense draft and remains live for
        # one sequential target forward; auxiliary states are copied by model.py.
        self.max_tokens = (
            vc.scheduler_config.max_num_scheduled_tokens
            or vc.scheduler_config.max_num_batched_tokens
        )
        query_width = 1 + (
            vc.speculative_config.num_speculative_tokens
            if vc.speculative_config is not None
            else 0
        )
        self.max_decode_tokens = min(
            self.max_tokens, vc.scheduler_config.max_num_seqs * query_width
        )
        self.score_capacity = (vc.model_config.max_model_len + 127) // 128
        self.B = 128
        self.cache_config = cache_config
        self.quant_config = quant_config
        linear_cls = (
            MinimaxM3QKVParallelLinearWithIndexer if sparse else QKVParallelLinear
        )
        linear_args = [
            config.hidden_size,
            128,
            config.num_attention_heads,
            config.num_key_value_heads,
        ]
        if sparse:
            linear_args += [
                config.sparse_attention_config["sparse_num_index_heads"],
                128,
            ]
        self.qkv_proj = linear_cls(
            *linear_args,
            bias=False,
            quant_config=quant_config,
            prefix=prefix + ".qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * 128,
            6144,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )
        self.q_norm = MiniMAXGemmaRMSNorm(128, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(128, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            128,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_v_scale", torch.ones(1, dtype=torch.float32), persistent=False
        )
        self.calculate_kv_scales = False
        self.kv_cache = torch.tensor([])
        self.index_cache = None
        if sparse:
            cfg = config.sparse_attention_config
            actual = (
                cfg["sparse_block_size"],
                cfg["sparse_topk_blocks"],
                cfg.get("sparse_init_block", 0),
                cfg.get("sparse_local_block", 0),
                cfg.get("sparse_score_type", "max"),
            )
            if actual != (128, 16, 0, 1, "max"):
                raise NotImplementedError(
                    f"MiniMax FP8 requires Bs128/top16/init0/local1/max, got {actual}"
                )
            self.index_q_norm = MiniMAXGemmaRMSNorm(128, eps=config.rms_norm_eps)
            self.index_k_norm = MiniMAXGemmaRMSNorm(128, eps=config.rms_norm_eps)
            self.index_cache = MiniMaxM3FP8IndexerCache(
                128,
                self.layer_name + ".index_cache",
                cache_config=cache_config,
                indexer_kv_dtype=indexer_kv_dtype,
                backend_cls=MiniMaxM3FP8IndexerBackend,
            )
        context = vc.compilation_config.static_forward_context
        if self.layer_name in context:
            raise ValueError(f"Duplicate FP8 attention layer {self.layer_name}")
        context[self.layer_name] = self
        self._allocate_buffers()

    def bind_kv_cache(self, cache):
        if (
            cache.dtype != torch.float8_e4m3fn
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != (self.num_kv_heads, 128, 256)
            or cache.stride(3) != 1
            or cache.stride(2) != 256
            or cache.stride(1) != 128 * 256
            or cache.stride(0) < self.num_kv_heads * cache.stride(1)
        ):
            raise ValueError(
                "M3 FP8 main allocator must bind packed E4M3[P,Hkv,128,256] with valid HND/page strides"
            )
        self.kv_cache = cache
        logger.info_once(
            "M3 FP8 main cache bound: shape=%s stride=%s dtype=%s",
            tuple(cache.shape),
            cache.stride(),
            cache.dtype,
        )

    def get_attn_backend(self):
        return MiniMaxM3FP8Backend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=128,
            num_kv_heads=self.num_kv_heads,
            head_size=128,
            head_size_v=128,
            dtype=torch.float8_e4m3fn,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _allocate_buffers(self):
        T, H, HK = self.max_tokens, self.num_heads, self.num_kv_heads
        F = (H + 2 * HK + (HK + 1 if self.sparse else 0)) * 128
        kind = "dsa" if self.sparse else "full"
        make = lambda name, shape, dtype: self.state.buffer(name, shape, dtype)
        self.x8 = make("x8", (T, 6144), torch.float8_e4m3fn)
        self.xs = make("xs", (T,), torch.float32)
        self.packed = make(kind + "_packed", (T, F), torch.bfloat16)
        self.q8 = make("q8", (T, H, 128), torch.float8_e4m3fn)
        self.qs = make("qs", (T, H), torch.float32)
        self.attn_out = make("attn_out", (T, H, 128), torch.bfloat16)
        # Flat scratch is shared by sequential layers in this PP worker. Kernels
        # pack current [S,Tsegment,H,130] into its prefix, never retain Tmax stride.
        if self.sparse:
            # Retain the standalone 64-split scratch bound. Prefill's 32/16-split
            # plans are restricted to T*HK<=16/32. The padded-layout S4
            # fallback has floor(Tsegment/Nrequests)<=128, hence at most
            # 129*Nrequests-1 tokens; Nrequests is scheduler-bounded.
            prefill_tokens = min(
                T, 129 * self.max_decode_tokens - 1 if HK >= 2 else 128 // HK
            )
            workspace_units = max(
                64 * self.max_decode_tokens,
                4 * prefill_tokens,
                32 * min(T, 16 // HK),
                16 * min(T, 32 // HK),
            )
        else:
            workspace_units = 64 * max(self.max_decode_tokens, min(T, 1023 // HK))
        self.attn_workspace = make(
            kind + "_attn_workspace",
            (workspace_units * H * 130,),
            torch.float32,
        )
        self.y8 = make("y8", (T, H * 128), torch.float8_e4m3fn)
        self.ys = make("ys", (T,), torch.float32)
        self.result = make("result", (T, 6144), torch.bfloat16)
        if self.sparse:
            M = self.score_capacity
            self.iq = make("iq", (T, HK, 128), torch.float8_e4m3fn)
            self.scores = make("scores", (T, HK, M), torch.bfloat16)
            self.top = make("top", (T, HK, 16), torch.int32)

    def _project(self, layer, A, As, out):
        if layer.weight_scale.dim() != 1 or layer.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "M3 attention requires completed checkpoint MXFP8 per-channel requant"
            )
        # Reuse the standard ixmm projection with per-token/per-channel scales.
        return ixf.fp8_per_channel_mm(
            A=A,
            B=layer.weight,
            As=As,
            Bs=layer.weight_scale,
            output=out,
            backend="ixmm",
        )

    def forward(self, positions, hidden_states):
        T = hidden_states.shape[0]
        if self.max_tokens < T:
            raise ValueError(
                "M3 FP8 token count exceeds preallocated scheduler capacity"
            )
        md = get_forward_context().attn_metadata
        if (
            not isinstance(md, dict)
            or self.layer_name not in md
            or self.kv_cache.numel() == 0
        ):
            # Allocations were included in model memory profiling at init.
            return self.result[:T]
        main = md[self.layer_name]
        if not isinstance(main, FP8Metadata):
            raise TypeError("M3 FP8 requires its canonical metadata builder")
        n = main.num_actual_tokens
        if not 0 <= n <= T:
            raise ValueError(
                "M3 metadata query capacity exceeds padded projection capacity"
            )
        if n == 0:
            self.result[:T].zero_()
            return self.result[:T]
        # In eager prefill, DP/SP may pad the model input while attention
        # metadata and slots retain their own smaller capacity. FULL decode
        # instead pads the attention metadata too. Keep these two capacities
        # separate; cu[-1] still identifies valid queries within n on device.
        if positions.ndim != 1 or positions.shape[0] < n:
            raise ValueError("M3 positions must cover attention query capacity")
        context = get_forward_context().slot_mapping
        if not isinstance(context, dict):
            raise TypeError("M3 FP8 requires per-layer framework slot mappings")
        ixf.fp8_per_token_dynamic_scaled(
            x=hidden_states[:n],
            fp8_output=self.x8[:n],
            output_scales=self.xs[:n],
            backend="ixformer_native",
        )
        self._project(self.qkv_proj, self.x8[:n], self.xs[:n], self.packed[:n])
        producer = dict(
            packed=self.packed[:n],
            q_norm_weight=self.q_norm.weight,
            k_norm_weight=self.k_norm.weight,
            rope_cache=self.rotary_emb.cos_sin_cache,
            positions=positions[:n],
            slot_mapping=context[self.layer_name],
            kv_cache=self.kv_cache,
            q_out=self.q8[:n],
            q_scale_out=self.qs[:n],
            epsilon=self.q_norm.variance_epsilon,
            mode="dsa" if self.sparse else "full",
            k_scale=self._k_scale,
            v_scale=self._v_scale,
        )
        if self.sparse:
            producer.update(
                index_norm_q_weight=self.index_q_norm.weight,
                index_norm_k_weight=self.index_k_norm.weight,
                index_slot_mapping=context[self.index_cache.prefix],
                index_cache=self.index_cache.kv_cache,
                index_q_out=self.iq[:n],
            )
        qknorm_rope_fp8_kv_insert(**producer, backend="ixformer_native")
        if self.sparse:
            index = md[self.index_cache.prefix]
            if not isinstance(index, FP8Metadata) or index.num_actual_tokens != n:
                raise ValueError("Main/index metadata query capacities disagree")
            fp8_index_score(
                self.iq[:n],
                self.index_cache.kv_cache,
                index.block_table,
                index.cu,
                index.seq,
                self.scores[:n],
                backend="ixformer_native",
            )
            index_topk(
                scores=self.scores[:n],
                cu_seqlens_q=index.cu,
                seq_lens=index.seq,
                prefix_lens=index.prefix,
                out=self.top[:n],
                backend="ixformer_native",
            )
        segments = main.dense_segments
        if segments is None:
            # Compatibility metadata uses one ragged prefill call.
            segments = (DenseQuerySegment(0, main.seq.numel(), 0, n, 0, main.cu),)
        k_cache, v_cache = self.kv_cache[..., :128], self.kv_cache[..., 128:]
        for segment in segments:
            begin, end = segment.token_start, segment.token_end
            if begin == end:
                continue
            first, last = segment.request_start, segment.request_end
            if self.sparse:
                fp8_dsa(
                    q=self.q8[begin:end],
                    kv_cache=self.kv_cache,
                    topk_idx=self.top[begin:end],
                    block_table=main.block_table[first:last],
                    cu_seqlens_q=segment.cu,
                    seq_lens=main.seq[first:last],
                    prefix_lens=main.prefix[first:last],
                    q_scale=self.qs[begin:end],
                    k_scale=self._k_scale,
                    v_scale=self._v_scale,
                    out=self.attn_out[begin:end],
                    backend="ixformer_native",
                    phase="decode" if segment.decode_query_len else "prefill",
                    workspace=self.attn_workspace,
                )
            else:
                dense_args = dict(
                    q=self.q8[begin:end],
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=main.block_table[first:last],
                    seq_lens=main.seq[first:last],
                    q_scale=self.qs[begin:end],
                    k_scale=self._k_scale,
                    v_scale=self._v_scale,
                    out=self.attn_out[begin:end],
                    backend="ixformer_native",
                    workspace=self.attn_workspace
                    if segment.decode_query_len == 1
                    or (end - begin) * self.num_kv_heads < 1024
                    else None,
                )
                if segment.decode_query_len:
                    ixf.paged_attention_decode_fp8(
                        **dense_args, decode_query_len=segment.decode_query_len
                    )
                else:
                    ixf.paged_attention_prefill_fp8(
                        **dense_args, cu_seqlens_q=segment.cu
                    )
        if main.dense_valid_tokens is not None and main.dense_valid_tokens < n:
            self.attn_out[main.dense_valid_tokens : n].zero_()
        _ixf_fp8_attention_output_quant(
            input=self.attn_out[:n],
            out_fp8=self.y8[:n],
            out_scale=self.ys[:n],
            backend="ixformer_native",
        )
        self._project(self.o_proj, self.y8[:n], self.ys[:n], self.result[:n])
        self.result[n:T].zero_()
        return self.result[:T]
