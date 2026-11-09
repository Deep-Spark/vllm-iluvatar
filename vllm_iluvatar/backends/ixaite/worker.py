# SPDX-License-Identifier: Apache-2.0
# Source: v1/worker/gpu_worker.py at 47fb804 and 1334ec4; private profiling and warmup.
"""Worker lifecycle and warmup for the independently selected ixaite backend."""

from __future__ import annotations

import itertools
import sys
from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

logger = init_logger(__name__)

_QWEN35_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)
_QWEN35_VISION_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
# Dense Qwen3 (e.g. Qwen3-8B): linear + paged-FA warmup only; no GDN / ViT.
_QWEN3_DENSE_MODEL_TYPES = frozenset({"qwen3"})
# MiniMax-M3.1-Flash ships as VL wrapper ``minimax_m3_vl`` (text-only serve
# with image/video limits 0). Without this, prefill linear/silu shapes
# (e.g. M=256) JIT on the first real request and dominate e2e wall.
_MINIMAX_M3_MODEL_TYPES = frozenset({"minimax_m3_vl", "minimax_m3"})
_IXAITE_TEXT_WARMUP_MODEL_TYPES = (
    _QWEN35_MODEL_TYPES | _QWEN3_DENSE_MODEL_TYPES | _MINIMAX_M3_MODEL_TYPES
)
_IXAITE_ATTN_HEAD_DIMS = (128, 256)
# M=1 is the single-token decode step. Body small_m M=2..16 is warmed by
# ixaite_body_small_m_warmup_sizes, not by this dummy_run list.
_IXAITE_LINEAR_DECODE_WARMUP_SIZES = (1,)
_GDN_CHUNK_SIZE = 64
# Crop launchers quantise M to a multiple of this; M <= 16 goes to small_m and
# 17..128 to the single-tile buckets below, so 129 is the smallest logical size
# on the crop path.
_HGEMM_CROP_BM = 256
_FIRST_CROP_WARMUP_SIZE = 129
_SMALL_M_MAX = 16
_CAUSAL_CONV_WIDTH = 4
# Qwen3.5 e2e attention shapes. These run after the production KV cache has
# been bound because ixaite specializes on its padded strides and block-table
# width; a synthetic pre-KV cache would populate different JIT entries.
_IXAITE_RESHAPE_WARMUP_SIZES = (1, 65)
_IXAITE_ATTN_PAGE_SIZE = 16
# ixaite prefill uses two warp schedules (<256 and >=256 query tokens). Q tile
# count and batch are runtime grid dimensions for Qwen3.5, so these two
# representatives cover the full scheduler range.
_IXAITE_PREFILL_WARMUP_CASES = (
    (128, 1),
    (256, 1),
)
# ixaite splits hgemm at this physical M; leftover rows pick a new tile.
_HGEMM_MAX_SINGLE = 2048
# Qwen3.5 ViT: patch 16, spatial merge 2. Serving profiles use 448 and 896.
_QWEN35_VIT_PATCH = 16
_QWEN35_VIT_MERGE = 2
_QWEN35_VIT_IMAGE_SIZES = (448, 896)
_QWEN35_VIT_TEXT_TOKENS = 128
# Scheduler leftovers seen on concurrent VL prefill (not a clean merge+text).
_QWEN35_VL_OBSERVED_ROWS = (304, 926)


def ixaite_spec_graph_warmup_batches(
    query_len: int,
    max_batch: int,
    capture_sizes: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Request batches whose spec CUDA graphs will compile distinct FA keys.

    Capture sizes are token counts. Uniform spec rounds them to a multiple of
    ``query_len``, so each surviving size maps to ``tokens // query_len``
    requests. MMA launchers specialize on that packed batch, so warmup follows
    the capture list instead of only the endpoints.
    """
    query_len = max(1, int(query_len))
    max_batch = int(max_batch)
    batches: list[int] = []
    seen: set[int] = set()
    for tokens in capture_sizes or ():
        token_count = int(tokens)
        if token_count < query_len or token_count % query_len != 0:
            continue
        batch = token_count // query_len
        if batch < 1 or batch > max_batch or batch in seen:
            continue
        seen.add(batch)
        batches.append(batch)
    if not batches and max_batch >= 1:
        batches.append(1)
        if max_batch > 1:
            batches.append(max_batch)
    return tuple(batches)


def ixaite_prefill_warmup_sizes(max_num_tokens: int) -> list[int]:
    """Logical token counts that select every live linear crop launcher.

    The crop launcher cache keys on the padded M, which is
    ``ceil(M / 256) * 256`` for M > 128. One logical size just above each
    bucket's lower boundary therefore compiles every launcher a scheduled
    batch can select, up to ``max_num_tokens`` tokens in one step.
    """
    if max_num_tokens < _FIRST_CROP_WARMUP_SIZE:
        return []
    num_buckets = -(-max_num_tokens // _HGEMM_CROP_BM)
    sizes = [_FIRST_CROP_WARMUP_SIZE]
    sizes.extend((j - 1) * _HGEMM_CROP_BM + 1 for j in range(2, num_buckets + 1))
    return [size for size in sizes if size <= max_num_tokens]


def ixaite_gdn_chunk_warmup_sizes(num_k_heads: int, max_num_tokens: int) -> list[int]:
    """Token counts that compile the live WY GDN cubin forks.

    Pack and KKT fork on ``chunks * Hk > 64`` (merge); ``fwd_o`` and the
    staging launch fork on whether every sequence length is a multiple of 64
    (dense). The two flags key separate caches rather than one cubin, so two
    launches suffice for all four combinations: a dense sub-threshold T fills
    the no-merge and dense keys, a varlen above-threshold T fills the merge
    and varlen keys, and any larger T reuses them.
    """
    if num_k_heads <= 0 or max_num_tokens <= 0:
        return []
    small_chunks = max(1, 64 // num_k_heads)
    dense_small = small_chunks * _GDN_CHUNK_SIZE
    varlen_merge = (small_chunks + 1) * _GDN_CHUNK_SIZE - (_GDN_CHUNK_SIZE - 1)
    sizes: list[int] = []
    if dense_small <= max_num_tokens:
        sizes.append(dense_small)
    if varlen_merge <= max_num_tokens:
        sizes.append(varlen_merge)
    return sizes


def _hgemm_single_buckets() -> tuple[int, ...]:
    from ixaite.ops.linear_qwen35_shapes import HGEMM_SINGLE_BUCKETS

    return HGEMM_SINGLE_BUCKETS


def ixaite_body_small_m_warmup_sizes(max_num_tokens: int) -> list[int]:
    """Exact M keys for body small_m (QKV / O / gate_up).

    dummy_run of 1 compiles M=1. Graph capture only hits 2/4/8/16, so M=3/5/6/7
    and 9..15 still JIT on a short eager prompt unless 2..16 are warmed here.
    DOWN_PROJ M<=16 is routed to pad-32 hgemm and is covered by the single-tile
    dummy instead.
    """
    return [m for m in range(2, _SMALL_M_MAX + 1) if m <= max_num_tokens]


def ixaite_hgemm_single_warmup_sizes(max_num_tokens: int) -> list[int]:
    """Exact M values that select the small-tile (non-crop) hgemm launchers.

    Crop dummy starts at 129. pad_m == M for the single-tile buckets is a
    different cubin, so dummy_run of 1 or 129 does not compile them.
    """
    return [m for m in _hgemm_single_buckets() if m <= max_num_tokens]


def ixaite_hgemm_exact_warmup_sizes(max_num_tokens: int) -> list[int]:
    """Physical M values that select the exact (non-crop) hgemm launcher."""
    if max_num_tokens < _HGEMM_CROP_BM:
        return []
    return list(range(_HGEMM_CROP_BM, max_num_tokens + 1, _HGEMM_CROP_BM))


def ixaite_qwen35_vision_row_counts() -> list[int]:
    """ViT / merger / remainder rows for the Qwen3.5 square-image sizes."""
    rows: set[int] = set()
    merge_area = _QWEN35_VIT_MERGE * _QWEN35_VIT_MERGE
    for image_size in _QWEN35_VIT_IMAGE_SIZES:
        grid = image_size // _QWEN35_VIT_PATCH
        vit = grid * grid
        merged = vit // merge_area
        rows.update((vit, merged, merged + _QWEN35_VIT_TEXT_TOKENS))
        leftover = vit % _HGEMM_MAX_SINGLE
        if leftover:
            rows.add(leftover)
        leftover_m = merged % _HGEMM_MAX_SINGLE
        if leftover_m:
            rows.add(leftover_m)
    rows.update(_QWEN35_VL_OBSERVED_ROWS)
    return sorted(row for row in rows if row > _SMALL_M_MAX)


def ixaite_qwen35_vision_linear_warmup_sizes(max_num_tokens: int) -> list[int]:
    """Exact M for ViT remainder / merger. Encoder M can exceed LLM max tokens."""
    del max_num_tokens
    return ixaite_qwen35_vision_row_counts()


def ixaite_qwen35_vl_prefill_dummy_sizes(max_num_tokens: int) -> list[int]:
    """LLM dummy_run lengths for one-image prefill (GDN / silu / rms).

    Skip the full 896-grid ViT row count (3136): that forward is encoder-only.
    784 stays — it is also the 896 merger / 448 ViT length used on the LLM.
    """
    return [row for row in (784, 912, 926, 1088) if row <= max_num_tokens]


def ixaite_qwen35_vit_qkv_warmup_sizes() -> list[int]:
    """Packed-QKV seq lengths on the 448 / 896 ViT path (including remainder)."""
    return [784, 1088, 3136]


def ixaite_lm_head_warmup_sizes(max_num_seqs: int) -> list[int]:
    """Logical M values that compile lm_head small_m and hgemm buckets.

    Logits run over scheduled sequences, not the prefill token count, so
    dummy_run of T does not compile this family. small_m keys on exact M;
    hgemm forks crop vs exact at each single-tile bucket.
    """
    if max_num_seqs < 1:
        return []
    sizes = [m for m in range(1, _SMALL_M_MAX + 1) if m <= max_num_seqs]
    prev = _SMALL_M_MAX
    for bucket in _hgemm_single_buckets():
        crop = prev + 1
        if crop <= max_num_seqs:
            sizes.append(crop)
        if bucket <= max_num_seqs:
            sizes.append(bucket)
        prev = bucket
    sizes.extend(
        size
        for size in ixaite_hgemm_exact_warmup_sizes(max_num_seqs)
        if size not in sizes
    )
    sizes.extend(
        size for size in ixaite_prefill_warmup_sizes(max_num_seqs) if size not in sizes
    )
    return sizes


def _pcp_profile_local_tokens(num_tokens: int, pcp_size: int, max_num_reqs: int) -> int:
    # DualChunkSwap gives each prefill rank at most n / PCP + 2 tokens
    # per request (two rounded chunks). Single-token decodes are replicated.
    # This upper bound covers mixed batches and the maximum rank's padding.
    return min(num_tokens, (num_tokens + pcp_size - 1) // pcp_size + 2 * max_num_reqs)


# Keep provider-specific profiling and warmup beside the ixaite lifecycle.
class IxaiteWorker(IluWorker):
    @contextmanager
    def _pcp_memory_profile(self):
        runner = self.model_runner
        if (
            not getattr(self, "use_v2_model_runner", False)
            or self.parallel_config.prefill_context_parallel_size <= 1
            or self.vllm_config.speculative_config is not None
        ):
            yield
            return

        # MRV2's memory-profile dummy batch bypasses prepare_inputs and hence
        # PCP partitioning. Feeding the global budget on every rank makes the
        # MoE gather expand it by PCP again. Profile a conservative local bound
        # instead; real requests and subsequent warmup retain their usual path.
        original = runner._dummy_run
        had_override = "_dummy_run" in vars(runner)

        def profile_dummy_run(num_tokens, *args, **kwargs):
            if kwargs.get("is_profile", False):
                local_tokens = _pcp_profile_local_tokens(
                    num_tokens,
                    self.parallel_config.prefill_context_parallel_size,
                    self.scheduler_config.max_num_seqs,
                )
                logger.info(
                    "PCP memory profile: global token budget %d, local bound %d",
                    num_tokens,
                    local_tokens,
                )
                num_tokens = local_tokens
            return original(num_tokens, *args, **kwargs)

        runner._dummy_run = profile_dummy_run
        try:
            yield
        finally:
            if had_override:
                runner._dummy_run = original
            else:
                del runner._dummy_run

    # FULL decode must fit the Mamba cache reserved for this worker.
    def _fit_max_num_seqs_to_mamba_blocks(self, kv_cache_config: KVCacheConfig) -> None:
        """Lower seq caps that exceed the Mamba blocks reserved for FULL decode.

        Each FULL decode sequence needs one Mamba cache block. vLLM refuses
        to start when the cap is above the reserved count. Prefill capture
        sizes stay as configured; decode graphs already drop sizes above
        ``max_num_seqs``. A cap already within the block count is left as is.
        """
        if not getattr(kv_cache_config, "has_mamba_layers", False):
            return
        num_blocks = int(getattr(kv_cache_config, "num_blocks", 0))
        if num_blocks < 1:
            return
        mode = getattr(self.compilation_config, "cudagraph_mode", None)
        has_full = getattr(mode, "has_full_cudagraphs", None)
        if has_full is None or not has_full():
            return
        requested = int(self.scheduler_config.max_num_seqs)
        runner_limit = getattr(self.model_runner, "max_num_reqs", None)
        runner_reqs = None if runner_limit is None else int(runner_limit)
        scheduler_over = requested > num_blocks
        runner_over = runner_reqs is not None and runner_reqs > num_blocks
        if not scheduler_over and not runner_over:
            return
        reported = requested if runner_reqs is None else max(requested, runner_reqs)
        logger.warning(
            "Mamba cache has %d blocks, below max_num_seqs %d. "
            "Lowering max_num_seqs to %d so FULL decode graphs match "
            "the reserved blocks.",
            num_blocks,
            reported,
            num_blocks,
        )
        if scheduler_over:
            self.scheduler_config.max_num_seqs = num_blocks
        if runner_over:
            self.model_runner.max_num_reqs = num_blocks

    def compile_or_warm_up_model(self):
        from vllm.model_executor.warmup import kernel_warmup, v1_block_table_warmup

        # The ixaite slot mapper replaces the unavailable Triton kernel.
        original_block = v1_block_table_warmup.warm_v1_block_table_kernels
        original_kernel = kernel_warmup.warm_v1_block_table_kernels
        v1_block_table_warmup.warm_v1_block_table_kernels = lambda runner: None
        kernel_warmup.warm_v1_block_table_kernels = lambda runner: None
        try:
            if self.parallel_config.prefill_context_parallel_size <= 1:
                return super().compile_or_warm_up_model()

            from vllm_iluvatar.models.glm_moe_dsa.attention import IluGlmMoeDsaSparseMLA

            # Synthetic V2 warmup must not choose the scale for the first request.
            calibration_states = [
                (layer, layer._fp8_kv_scale_calibration_enabled)
                for layer in self.model_runner.model.modules()
                if isinstance(layer, IluGlmMoeDsaSparseMLA)
            ]
            for layer, _ in calibration_states:
                layer._fp8_kv_scale_calibration_enabled = False
            try:
                return super().compile_or_warm_up_model()
            finally:
                for layer, enabled in calibration_states:
                    layer._fp8_kv_scale_calibration_enabled = enabled
        finally:
            v1_block_table_warmup.warm_v1_block_table_kernels = original_block
            kernel_warmup.warm_v1_block_table_kernels = original_kernel

    # Retain the existing lifecycle and warmup contract.
    def determine_available_memory(self) -> int:
        orig_cls = self._try_replace_model_runner_class()
        try:
            if self._ixaite_warmup_compile_cap() is not None:
                # Under VLLM_COMPILE the first compiled call fixes the traced
                # dynamic shapes: a backed symint traced at num_tokens == 1
                # gets 0/1-specialized, collapsing every token dim to static
                # 1 and leaving only the mrope positions stride
                # (max_num_tokens + 1) as a SymInt, which then fails the
                # piecewise range check. Let the community profile run
                # (max_num_tokens >= 2) trace first, then warm the buckets
                # against the already-dynamic compiled graph.
                # Compiled probing still needs upstream PCP memory accounting.
                with self._pcp_memory_profile():
                    available_memory = Worker.determine_available_memory(self)
                if self._warm_up_ixaite_before_kv():
                    self._release_ixaite_warmup_scratch()
                return available_memory
            if self._warm_up_ixaite_before_kv():
                self._release_ixaite_warmup_scratch()
            with self._pcp_memory_profile():
                return Worker.determine_available_memory(self)
        finally:
            self.model_runner.__class__ = orig_cls

    # Retain the existing lifecycle and warmup contract.
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        orig_cls = self._try_replace_model_runner_class()
        try:
            self._fit_max_num_seqs_to_mamba_blocks(kv_cache_config)
            Worker.initialize_from_config(self, kv_cache_config)
            from vllm_iluvatar.backends.ixaite.attention.kv_cache import (
                bind_int8_kv_caches,
            )

            bind_int8_kv_caches(self.model_runner)
            self._warm_up_ixaite_attention_after_kv()
        finally:
            self.model_runner.__class__ = orig_cls

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_attention_after_kv(self) -> bool:
        """Compile Qwen3 / Qwen3.5 attention shapes against the bound production KV cache."""
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False

        from vllm_iluvatar.backends.ixaite.attention.flash_attn import (
            IxaiteFlashAttentionImpl,
        )

        layer = next(
            (
                candidate
                for candidate in self.compilation_config.static_forward_context.values()
                if isinstance(
                    getattr(candidate, "impl", None), IxaiteFlashAttentionImpl
                )
            ),
            None,
        )
        if layer is None:
            return False

        kv_cache = getattr(layer, "kv_cache", None)
        if (
            not hasattr(kv_cache, "shape")
            or kv_cache.ndim != 5
            or kv_cache.shape[0] != 2
            or kv_cache.shape[-2] != _IXAITE_ATTN_PAGE_SIZE
            or int(kv_cache.shape[-1]) not in _IXAITE_ATTN_HEAD_DIMS
        ):
            logger.warning(
                "Skipping ixaite attention warmup: unsupported bound KV shape %s",
                getattr(kv_cache, "shape", None),
            )
            return False

        block_tables = self.model_runner.input_batch.block_table
        decode_query_len = max(
            1,
            int(getattr(self.model_runner, "uniform_decode_query_len", 1)),
        )
        max_batch = min(
            self.scheduler_config.max_num_seqs,
            max(1, self.model_runner.max_num_tokens),
        )
        table_owners = getattr(block_tables, "block_tables", None)
        if table_owners is None:
            table_owners = [block_tables[0]]
        block_table_template = max(
            (owner.get_device_tensor(max_batch) for owner in table_owners),
            key=lambda table: table.shape[1],
        )
        if block_table_template.ndim != 2 or block_table_template.shape[1] <= 0:
            logger.warning(
                "Skipping ixaite attention warmup: invalid block-table shape %s",
                tuple(block_table_template.shape),
            )
            return False

        cudagraph_mode = getattr(self.compilation_config, "cudagraph_mode", None)
        decode_mode = (
            cudagraph_mode.decode_mode()
            if callable(getattr(cudagraph_mode, "decode_mode", None))
            else None
        )
        has_full_decode_graph = getattr(decode_mode, "name", None) == "FULL"
        capture_sizes: tuple[int, ...] = ()
        graph_max_batch = 0
        if has_full_decode_graph:
            capture_sizes = tuple(
                getattr(self.compilation_config, "cudagraph_capture_sizes", None) or ()
            )
            max_cudagraph_tokens = getattr(
                self.compilation_config, "max_cudagraph_capture_size", None
            )
            graph_max_batch = max_batch
            if max_cudagraph_tokens:
                graph_max_batch = min(
                    max_batch,
                    max(1, int(max_cudagraph_tokens) // decode_query_len),
                )
        self._run_ixaite_attention_warmup(
            layer.impl,
            kv_cache,
            block_table_template,
            int(self.model_runner.max_num_tokens),
            int(self.model_config.max_model_len),
            decode_query_len,
            capture_sizes,
            graph_max_batch,
        )
        return True

    # Retain the existing lifecycle and warmup contract.
    @staticmethod
    def _run_ixaite_attention_warmup(
        implementation,
        kv_cache,
        block_table_template,
        max_num_tokens: int,
        max_model_len: int,
        decode_query_len: int = 1,
        cudagraph_capture_sizes: tuple[int, ...] | list[int] | None = None,
        graph_max_batch: int | None = None,
    ) -> None:
        import torch
        from ixaite.ops.attention.planner import (
            iter_qwen_mma_decode_schedule_representatives,
            iter_qwen_simt_decode_schedule_representatives,
        )

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        reshape, reason = load_ixaite_python_op(
            "reshape_and_cache_flash", "reshape_and_cache_flash"
        )
        if reshape is None:
            raise RuntimeError(
                f"ixaite attention selected but cache-writer warmup is unavailable: {reason}"
            )

        prefill = implementation._flash_attn_op(0)
        decode = implementation._flash_attn_op(1)
        key_cache, value_cache = kv_cache.unbind(0)
        device = key_cache.device
        dtype = key_cache.dtype
        num_heads = implementation.num_heads
        num_kv_heads = implementation.num_kv_heads
        head_dim = int(kv_cache.shape[-1])
        max_batch = int(block_table_template.shape[0])
        cases = [
            (context, batch)
            for context, batch in _IXAITE_PREFILL_WARMUP_CASES
            if context * batch <= max_num_tokens and batch <= max_batch
        ]
        decode_query_len = max(1, int(decode_query_len))
        query_lens = tuple(range(1, decode_query_len + 1))
        decode_groups = tuple(
            (
                query_len,
                iter_qwen_simt_decode_schedule_representatives(
                    dtype=dtype,
                    max_batch_size=min(max_batch, max_num_tokens // query_len),
                    max_context_len=max_model_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    seqlen_q=query_len,
                ),
                iter_qwen_mma_decode_schedule_representatives(
                    max_batch_size=min(max_batch, max_num_tokens // query_len),
                    max_context_len=max_model_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    query_len=query_len,
                ),
            )
            for query_len in query_lens
        )
        graph_capacity = int(block_table_template.shape[1]) * _IXAITE_ATTN_PAGE_SIZE
        graph_batches = ixaite_spec_graph_warmup_batches(
            decode_query_len,
            max_batch if graph_max_batch is None else int(graph_max_batch),
            cudagraph_capture_sizes,
        )

        logger.info(
            "Warming ixaite attention against production KV: "
            "prefill_cases=%s decode_query_len=%s decode_schedules=%s "
            "graph_batches=%s",
            cases,
            decode_query_len,
            {
                query_len: {"simt": len(simt_reps), "mma": len(mma_reps)}
                for query_len, simt_reps, mma_reps in decode_groups
            },
            graph_batches,
        )
        with torch.inference_mode():
            for num_tokens in _IXAITE_RESHAPE_WARMUP_SIZES:
                key = torch.zeros(
                    (num_tokens, num_kv_heads, head_dim),
                    dtype=dtype,
                    device=device,
                )
                value = torch.zeros_like(key)
                slots = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
                reshape(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slots,
                    "auto",
                    1.0,
                    1.0,
                )
                del key, value, slots

            for context, batch in cases:
                actual_tokens = batch * context
                cu_seqlens = torch.arange(
                    0,
                    actual_tokens + 1,
                    context,
                    dtype=torch.int32,
                    device=device,
                )
                prefill_block_table = torch.zeros(
                    (batch, block_table_template.shape[1]),
                    dtype=torch.int32,
                    device=device,
                )
                prefill_q = torch.zeros(
                    (actual_tokens, num_heads, head_dim),
                    dtype=dtype,
                    device=device,
                )
                prefill_out = torch.empty(
                    (actual_tokens, num_heads, head_dim),
                    dtype=dtype,
                    device=device,
                )
                prefill(
                    q=prefill_q,
                    k=key_cache,
                    v=value_cache,
                    out=prefill_out,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=context,
                    max_seqlen_k=context,
                    softmax_scale=implementation.scale,
                    causal=True,
                    block_table=prefill_block_table,
                )
                del cu_seqlens, prefill_q, prefill_out, prefill_block_table

            def _warm_decode(
                batch: int,
                context: int,
                query_len: int,
                *,
                use_cuda_graph: bool,
                planning_k: int,
            ) -> None:
                if batch < 1 or batch > max_batch or batch * query_len > max_num_tokens:
                    return
                decode_block_table = torch.zeros(
                    (batch, block_table_template.shape[1]),
                    dtype=torch.int32,
                    device=device,
                )
                decode_q = torch.zeros(
                    (batch * query_len, 1, num_heads, head_dim),
                    dtype=dtype,
                    device=device,
                )
                decode_out = torch.empty_like(decode_q)
                decode(
                    q=decode_q,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=decode_block_table,
                    cache_seqlens=torch.full(
                        (batch,), context, dtype=torch.int32, device=device
                    ),
                    max_query_len=query_len,
                    cu_query_lens=torch.arange(
                        0,
                        batch * query_len + 1,
                        query_len,
                        dtype=torch.int32,
                        device=device,
                    ),
                    softmax_scale=implementation.scale,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    softcap=0.0,
                    sinks=None,
                    out=decode_out,
                    use_cuda_graph=use_cuda_graph,
                    max_context_len=planning_k,
                )
                del decode_block_table, decode_q, decode_out

            for query_len, simt_representatives, mma_representatives in decode_groups:
                for _schedule, batch, context in (
                    *simt_representatives,
                    *mma_representatives,
                ):
                    _warm_decode(
                        batch,
                        context,
                        query_len,
                        use_cuda_graph=False,
                        planning_k=context,
                    )
                if query_len == decode_query_len and query_len > 1:
                    for batch in graph_batches:
                        _warm_decode(
                            batch,
                            min(graph_capacity, max_model_len),
                            query_len,
                            use_cuda_graph=True,
                            planning_k=graph_capacity,
                        )

            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    # Retain the existing lifecycle and warmup contract.
    def shutdown(self) -> None:
        orig_cls = self._try_replace_model_runner_class()
        runner_cls = self.model_runner.__class__

        class _GraphPinReleasingModelRunner(runner_cls):
            def shutdown(runner_self) -> None:
                super().shutdown()
                # GPUModelRunner.shutdown has now destroyed its graph owners,
                # while Worker.shutdown has not yet released allocator pools.
                # Do not import the optional runtime during shutdown: if it was
                # never loaded, it cannot own graph pins in this process.
                ixaite = sys.modules.get("ixaite")
                release_graph_pins = getattr(ixaite, "release_cuda_graph_pins", None)
                if release_graph_pins is not None:
                    release_graph_pins()

        self.model_runner.__class__ = _GraphPinReleasingModelRunner
        try:
            Worker.shutdown(self)
        finally:
            self.model_runner.__class__ = orig_cls

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_before_kv(self) -> bool:
        ran_gelu = self._warm_up_ixaite_gelu()
        ran_video_evs = self._warm_up_ixaite_video_evs()
        ran_vit_qkv_rope = self._warm_up_ixaite_vit_qkv_rope()
        ran_prefill = self._warm_up_ixaite_prefill_buckets()
        ran_small_m = self._warm_up_ixaite_body_small_m()
        ran_exact = self._warm_up_ixaite_linear_exact_buckets()
        ran_lm_head = self._warm_up_ixaite_lm_head()
        ran_decode_net = self._warm_up_ixaite_decode_network()
        ran_gdn = self._warm_up_ixaite_gdn_chunk_buckets()
        ran_decode = self._warm_up_ixaite_gdn_packed_decode()
        ran_conv = self._warm_up_ixaite_gdn_conv()
        ran_prep = self._warm_up_ixaite_gdn_post_conv_prep()
        return (
            ran_gelu
            or ran_video_evs
            or ran_vit_qkv_rope
            or ran_prefill
            or ran_small_m
            or ran_exact
            or ran_lm_head
            or ran_decode_net
            or ran_gdn
            or ran_decode
            or ran_conv
            or ran_prep
        )

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_image_limit(self) -> int:
        multimodal_config = getattr(self.model_config, "multimodal_config", None)
        getter = getattr(multimodal_config, "get_limit_per_prompt", None)
        if getter is None:
            return 0
        limit = getter("image")
        if limit is None:
            return 0
        return int(limit)

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_hf_sources(self):
        cfg = getattr(self, "model_config", None)
        if cfg is None:
            return ()
        hf = getattr(cfg, "hf_config", None)
        return (
            getattr(cfg, "hf_text_config", None),
            getattr(hf, "text_config", None) if hf is not None else None,
            hf,
        )

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_model_type(self) -> str:
        for src in self._ixaite_hf_sources():
            if src is None:
                continue
            value = getattr(src, "model_type", None)
            if value:
                return str(value)
        return ""

    # Model-specific warmup eligibility belongs to the callers.
    def _ixaite_has_vision_encoder(self) -> bool:
        hf = getattr(self.model_config, "hf_config", None)
        return (
            getattr(hf, "vision_config", None) is not None
            and self._ixaite_image_limit() > 0
        )

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_hidden_size(self) -> int:
        for src in self._ixaite_hf_sources():
            if src is None:
                continue
            value = getattr(src, "hidden_size", None)
            if value:
                return int(value)
        return 0

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_quant_config_is_w8a8(self) -> bool:
        cfg = getattr(self, "model_config", None)
        text = str(getattr(cfg, "quantization", None) or "").lower()
        if "w4a8" in text:
            return False
        if "w8a8" in text:
            return True
        qc = getattr(cfg, "quantization_config", None) if cfg is not None else None
        if qc is None:
            hf = getattr(cfg, "hf_config", None) if cfg is not None else None
            qc = getattr(hf, "quantization_config", None) if hf is not None else None
        if not isinstance(qc, dict):
            return False
        blob = str(qc).lower()
        if "w4a8" in blob:
            return False
        if "w8a8" in blob:
            return True
        groups = qc.get("config_groups") or {}
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights") or {}
            acts = group.get("input_activations") or {}
            try:
                w_bits = int(weights.get("num_bits", 0) or 0)
                a_bits = int(acts.get("num_bits", 0) or 0)
            except (TypeError, ValueError):
                continue
            if w_bits == 8 and a_bits == 8:
                return True
        return False

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_has_w8a8_body_linears(self) -> bool:
        """True when MLP / full-attn qkv actually use a W8A8 linear scheme.

        compressed-tensors also covers W4A8, so quant_method alone is not
        enough. Prefer the loaded module scheme; unit tests fall back to
        quantization_config num_bits.
        """
        model = getattr(getattr(self, "model_runner", None), "model", None)
        named = getattr(model, "named_modules", None)
        if callable(named):
            from vllm_iluvatar.backends.ixaite.quantization import (
                is_w8a8_linear_scheme,
            )

            saw_scheme = False
            for name, module in named():
                scheme = getattr(module, "scheme", None)
                if scheme is None:
                    continue
                saw_scheme = True
                if is_w8a8_linear_scheme(scheme) and any(
                    key in name for key in ("qkv_proj", "gate_up_proj", "down_proj")
                ):
                    return True
            if saw_scheme:
                return False
        return self._ixaite_quant_config_is_w8a8()

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_linear_warmup_candidate_geometries(self):
        from ixaite.ops.linear_qwen35_shapes import (
            QWEN36_27B_K,
            QWEN36_35B_K,
            LinearFamily,
            geometry_for,
            qwen3_8b_geometries,
            qwen36_27b_geometries,
            qwen36_35b_a3b_geometries,
        )

        tp_size = self._ixaite_tp_size()
        hidden = self._ixaite_hidden_size()
        model_type = self._ixaite_model_type()
        if model_type in _QWEN3_DENSE_MODEL_TYPES:
            candidates = [geom for geom in qwen3_8b_geometries() if geom.tp == tp_size]
            for family in (
                LinearFamily.O_PROJ,
                LinearFamily.GATE_UP,
                LinearFamily.DOWN_PROJ,
            ):
                try:
                    candidates.append(geometry_for(family, tp_size))
                except ValueError:
                    continue
        elif model_type in _QWEN35_MODEL_TYPES and hidden == QWEN36_27B_K:
            candidates = [
                geom for geom in qwen36_27b_geometries() if geom.tp == tp_size
            ]
        elif model_type in _QWEN35_MODEL_TYPES and hidden == QWEN36_35B_K:
            candidates = [
                geom for geom in qwen36_35b_a3b_geometries() if geom.tp == tp_size
            ]
        else:
            candidates = []
            for family in LinearFamily:
                try:
                    candidates.append(geometry_for(family, tp_size))
                except ValueError:
                    continue

        if model_type in _QWEN35_MODEL_TYPES and self._ixaite_has_vision_encoder():
            have = {geom.family for geom in candidates}
            for family in (
                LinearFamily.VIS_QKV,
                LinearFamily.VIS_O_PROJ,
                LinearFamily.VIS_FC1,
                LinearFamily.VIS_FC2,
                LinearFamily.MERGER_FC1,
                LinearFamily.MERGER_FC2,
            ):
                if family in have:
                    continue
                try:
                    candidates.append(geometry_for(family, tp_size))
                except ValueError:
                    continue

        seen: set[tuple[int, int]] = set()
        unique = []
        for geom in candidates:
            if geom.nk in seen:
                continue
            seen.add(geom.nk)
            unique.append(geom)
        return unique

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_linear_warmup_geometries(self, *, include_lm_head: bool):
        from ixaite.ops.linear_qwen35_shapes import LinearFamily, ixformer_gated_nk

        gated = ixformer_gated_nk()
        has_vision = (
            self._ixaite_model_type() in _QWEN35_MODEL_TYPES
            and self._ixaite_has_vision_encoder()
        )
        skip_w8a8_body = self._ixaite_has_w8a8_body_linears()
        vision_families = (
            LinearFamily.VIS_QKV,
            LinearFamily.VIS_O_PROJ,
            LinearFamily.VIS_FC1,
            LinearFamily.VIS_FC2,
            LinearFamily.MERGER_FC1,
            LinearFamily.MERGER_FC2,
        )
        w8a8_body_families = (
            LinearFamily.QKV,
            LinearFamily.GATE_UP,
            LinearFamily.DOWN_PROJ,
        )
        selected = []
        for geom in self._ixaite_linear_warmup_candidate_geometries():
            if include_lm_head != (geom.family is LinearFamily.LM_HEAD):
                continue
            if not has_vision and geom.family in vision_families:
                continue
            if skip_w8a8_body and geom.family in w8a8_body_families:
                continue
            if geom.nk in gated:
                continue
            selected.append(geom)
        return selected

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_vision_qkv_geometry(self) -> tuple[int, int]:
        model = getattr(getattr(self, "model_runner", None), "model", None)
        if model is not None and callable(getattr(model, "modules", None)):
            from vllm_iluvatar.backends.ixaite.attention.mm_encoder_attention import (
                IluMMEncoderAttention,
            )

            # Read the actual shard geometry, including vision data parallelism.
            attention = next(
                (m for m in model.modules() if isinstance(m, IluMMEncoderAttention)), None
            )
            if attention is not None:
                return attention.num_heads, attention.head_size
        vision = getattr(self.model_config.hf_config, "vision_config", None)
        num_heads = int(getattr(vision, "num_heads", 16) or 16)
        hidden_size = int(getattr(vision, "hidden_size", 1152) or 1152)
        head_dim = hidden_size // num_heads if num_heads else 72
        tp_size = self._ixaite_tp_size()
        if num_heads % tp_size == 0:
            num_heads = num_heads // tp_size
        return num_heads, head_dim

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_load_vision_op(self, op_name: str):
        # These models share packed NeoX preparation; other encoders need their own inputs.
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _QWEN35_VISION_MODEL_TYPES | {"qwen2_5_vl"}:
            return None
        if not self._ixaite_has_vision_encoder():
            return None

        from vllm_iluvatar.backends.ixaite.runtime import (
            is_ixaite_op_enabled,
            load_ixaite_torch_op,
        )

        if not is_ixaite_op_enabled(op_name):
            return None
        implementation, _ = load_ixaite_torch_op(op_name)
        return implementation

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_iter_vit_qkv_warmup(self):
        import torch

        num_heads, head_dim = self._ixaite_vision_qkv_geometry()
        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        packed_width = 3 * num_heads * head_dim
        for seq_len in ixaite_qwen35_vit_qkv_warmup_sizes():
            packed = torch.zeros(seq_len, 1, packed_width, dtype=dtype, device=device)
            yield seq_len, packed, num_heads, head_dim

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_gelu(self) -> bool:
        """Compile every Qwen3.5 Vision GELU launcher before inference."""
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type != "qwen3_5":
            return False

        from vllm_iluvatar.backends.ixaite.runtime import (
            is_ixaite_op_enabled,
            load_ixaite_torch_op,
        )

        if not is_ixaite_op_enabled("gelu"):
            return False
        implementation, _ = load_ixaite_torch_op("gelu")
        if implementation is None:
            return False

        import torch

        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        logger.info("Warming ixaite Qwen3.5 Vision GELU cubins")
        # ixaite caches by (dtype, mode, guard_tail): one pack4 vector selects
        # the guarded launcher; one full 256-thread block selects the unguarded.
        for approximate in ("none", "tanh"):
            for numel in (4, 4 * 256):
                input = torch.zeros(numel, dtype=dtype, device=device)
                output = implementation(input, approximate)
                del input, output
        return True

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_video_evs(self) -> bool:
        """Compile the Qwen3.5 short-video EVS launcher before KV allocation."""
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        multimodal_config = getattr(self.model_config, "multimodal_config", None)
        if (
            model_type != "qwen3_5"
            or multimodal_config is None
            or getattr(multimodal_config, "video_pruning_method", None) != "evs"
            or not getattr(multimodal_config, "video_pruning_rate", None)
        ):
            return False

        from vllm_iluvatar.backends.ixaite.runtime import (
            is_ixaite_python_op_enabled,
            load_ixaite_python_op,
        )

        if not is_ixaite_python_op_enabled("video_evs", "compute_retention_mask"):
            return False
        implementation, _ = load_ixaite_python_op("video_evs", "compute_retention_mask")
        if implementation is None:
            return False

        import torch

        hidden_size = int(self.model_config.hf_config.vision_config.out_hidden_size)
        video = torch.zeros(
            (392, hidden_size),
            dtype=self._ixaite_torch_dtype(),
            device=self._ixaite_warmup_device(),
        )
        logger.info("Warming ixaite Qwen3.5 short-video EVS cubins")
        mask = implementation(
            video,
            (2, 28, 28),
            2,
            float(multimodal_config.video_pruning_rate),
        )
        del video, mask
        return True

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_vit_qkv_rope(self) -> bool:
        """Compile fused ViT QKV split + NeoX RoPE cubins before KV reserve."""
        implementation = self._ixaite_load_vision_op("vit_qkv_rope")
        if implementation is None:
            return False

        import torch
        import vllm.envs as envs

        _, head_dim = self._ixaite_vision_qkv_geometry()
        fused = (
            self._ixaite_torch_dtype() == torch.bfloat16
            and head_dim < 128
            and envs.VLLM_ILU_VIT_ATTN_OPT_LEVEL == 0
        )

        logger.info(
            "Warming ixaite ViT QKV RoPE cubins with seq lengths %s",
            ixaite_qwen35_vit_qkv_warmup_sizes()[:1] if fused else ixaite_qwen35_vit_qkv_warmup_sizes(),
        )
        for seq_len, packed, num_heads, head_dim in self._ixaite_iter_vit_qkv_warmup():
            device = packed.device
            # Two spatial axes concatenate the half-width frequency tables into full-head RoPE.
            cache_dtypes = (torch.bfloat16, torch.float32) if fused else (torch.float32,)
            for cache_dtype in cache_dtypes:
                cos = torch.zeros(
                    seq_len, head_dim // 2 if fused else head_dim // 4,
                    dtype=cache_dtype, device=device,
                )
                padding = {"padded_head_dim": ((head_dim + 127) // 128) * 128} if fused else {}
                # Cache values do not affect compilation, so one tensor covers cos and sin.
                q, k, v = implementation(packed, cos, cos, num_heads, head_dim, **padding)
                del cos, q, k, v
            if fused:
                layout = self._ixaite_load_vision_op("vit_qkv_layout")
                if layout is not None:
                    layout(packed, num_heads, head_dim, **padding)
                # Padded preparation specializes on dtype/cache dtype, batch, heads,
                # native/padded width and rotary width; sequence lengths stay dynamic.
                break

        max_context = getattr(self.model_config, "max_model_len", None)
        if fused and isinstance(max_context, int) and max_context > 0:
            from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

            attention, _ = load_ixaite_python_op("flash_attn", "flash_attn_varlen_func")
            if attention is not None:
                from ixaite.ops.attention.flash_attn_prefill import prefill_warmup_calls

                vision = self.model_config.hf_config.vision_config
                query_limit = max_context * int(getattr(vision, "spatial_merge_size", 2)) ** 2
                storage_dim = ((head_dim + 127) // 128) * 128
                cases = prefill_warmup_calls(
                    num_heads, num_heads, storage_dim, query_limit,
                    dtype=torch.bfloat16, paged=False,
                    max_seqlen_q=query_limit, max_seqlen_k=query_limit,
                )
                logger.info(
                    "Warming ixaite vision attention H=%d D=%d variants=%s",
                    num_heads, head_dim, cases,
                )
                for sequences in cases:
                    lengths = [length for length, _ in sequences]
                    total = sum(lengths)
                    length = max(lengths)
                    query = torch.zeros(
                        (total, num_heads, storage_dim), dtype=torch.bfloat16, device=device
                    )
                    offsets = torch.tensor(
                        [0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device
                    )
                    attention(
                        query, query, query, offsets, offsets, length, length,
                        softmax_scale=head_dim**-0.5, causal=False, valid_head_dim=head_dim,
                    )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_warmup_compile_cap(self) -> int | None:
        """Top compiled-range endpoint when the model is torch.compile'd.

        Returns None in eager mode (no shape restriction on dummy runs).
        """
        from vllm.config.compilation import CompilationMode

        compilation_config = self.vllm_config.compilation_config
        if compilation_config.mode != CompilationMode.VLLM_COMPILE:
            return None
        endpoints = compilation_config.compile_ranges_endpoints
        if not endpoints:
            return None
        return max(endpoints)

    def _ixaite_reachable_token_cap(self) -> int:
        """Largest token count one forward can schedule under current knobs.

        ``max_num_tokens`` alone is ``max_num_batched_tokens`` and can be far
        above what ``max_num_seqs * max_model_len`` allows (fair-bench uses
        max_num_seqs=1). Prefill dummy_run beyond that wastes minutes and
        never hits a live shape.
        """
        max_num_tokens = int(self.model_runner.max_num_tokens)
        max_model_len = int(self.model_config.max_model_len)
        max_num_seqs = int(self.vllm_config.scheduler_config.max_num_seqs)
        return max(1, min(max_num_tokens, max_num_seqs * max_model_len))

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_prefill_buckets(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import is_ixaite_op_enabled

        if not is_ixaite_op_enabled("linear"):
            return False

        max_num_tokens = self._ixaite_reachable_token_cap()
        warmup_sizes = [
            size
            for size in _IXAITE_LINEAR_DECODE_WARMUP_SIZES
            if size <= max_num_tokens
        ]
        warmup_sizes.extend(ixaite_hgemm_single_warmup_sizes(max_num_tokens))
        warmup_sizes.extend(ixaite_prefill_warmup_sizes(max_num_tokens))
        # Under VLLM_COMPILE the dummy run goes through the compiled model,
        # which only accepts token counts inside the compiled ranges; larger
        # buckets would raise "out of considered ranges". Scheduled batches
        # are capped by max_num_batched_tokens (the top endpoint), so the
        # dropped buckets are unreachable at serve time.
        compile_cap = self._ixaite_warmup_compile_cap()
        if compile_cap is not None:
            warmup_sizes = [size for size in warmup_sizes if size <= compile_cap]
        if (
            model_type in _QWEN35_VISION_MODEL_TYPES
            and self._ixaite_has_vision_encoder()
        ):
            seen = set(warmup_sizes)
            for size in ixaite_qwen35_vl_prefill_dummy_sizes(max_num_tokens):
                if size not in seen:
                    seen.add(size)
                    warmup_sizes.append(size)
        # Exact crop-boundary Ms (256, 512, ...) select a different silu / linear
        # cubin than crop warmup (129, 257, ...). Fair-bench isl=256 hits these;
        # linear-only exact warmup does not compile silu_and_mul.
        if model_type in _MINIMAX_M3_MODEL_TYPES:
            seen = set(warmup_sizes)
            for size in ixaite_hgemm_exact_warmup_sizes(max_num_tokens):
                if size not in seen:
                    seen.add(size)
                    warmup_sizes.append(size)
        if not warmup_sizes:
            return False

        logger.info(
            "Warming ixaite prefill linear buckets with token sizes %s",
            warmup_sizes,
        )
        for size in warmup_sizes:
            logger.info("ixaite prefill bucket warmup dummy_run size=%d", size)
            self.model_runner._dummy_run(
                size,
                skip_eplb=True,
                is_profile=True,
                remove_lora=False,
            )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_decode_network(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False
        if self.model_runner.max_num_tokens < 1:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import is_ixaite_op_enabled

        if is_ixaite_op_enabled("linear"):
            return False
        if not any(
            is_ixaite_op_enabled(op_name)
            for op_name in (
                "silu_and_mul",
                "rms_norm_gated",
                "embedding_gather",
            )
        ):
            return False

        logger.info("Warming ixaite decode-token network with size 1")
        self.model_runner._dummy_run(
            1,
            skip_eplb=True,
            is_profile=True,
            remove_lora=False,
        )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_linear_attn_config(self):
        for src in (
            getattr(self.model_config, "hf_text_config", None),
            getattr(self.model_config, "hf_config", None),
        ):
            if src is not None and getattr(src, "linear_num_key_heads", None):
                return src
        return None

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_gdn_chunk_buckets(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _QWEN35_MODEL_TYPES:
            return False

        cfg = self._ixaite_linear_attn_config()
        if cfg is None:
            return False
        parallel_config = getattr(self, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        num_k_heads = int(cfg.linear_num_key_heads) // tp_size
        warmup_sizes = ixaite_gdn_chunk_warmup_sizes(
            num_k_heads, self.model_runner.max_num_tokens
        )
        if not warmup_sizes:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op("gdn_chunk", "chunk_gated_delta_rule")
        if implementation is None:
            return False

        logger.info(
            "Warming ixaite GDN chunk cubins with token sizes %s",
            warmup_sizes,
        )
        for size in warmup_sizes:
            self._run_ixaite_gdn_chunk_warmup(implementation, size)
        return True

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_gdn_chunk_warmup(self, implementation, total_tokens: int) -> None:
        import torch

        cfg = self._ixaite_linear_attn_config()
        assert cfg is not None
        parallel_config = getattr(self, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        num_k_heads = int(cfg.linear_num_key_heads) // tp_size
        num_v_heads = int(cfg.linear_num_value_heads) // tp_size
        head_k_dim = int(cfg.linear_key_head_dim)
        head_v_dim = int(cfg.linear_value_head_dim)
        dtype = self.model_config.dtype
        device = getattr(self, "device", None) or self.model_runner.device

        q = torch.zeros(
            total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device
        )
        k = torch.zeros_like(q)
        v = torch.zeros(
            total_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device
        )
        g = torch.zeros(total_tokens, num_v_heads, dtype=torch.float32, device=device)
        beta = torch.zeros(total_tokens, num_v_heads, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
        implementation(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            chunk_size=_GDN_CHUNK_SIZE,
            use_qk_l2norm_in_kernel=False,
            output_final_state=True,
        )
        del q, k, v, g, beta, cu_seqlens

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_gdn_tp_geometry(self):
        cfg = self._ixaite_linear_attn_config()
        if cfg is None:
            return None
        parallel_config = getattr(self, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        return (
            int(cfg.linear_num_key_heads) // tp_size,
            int(cfg.linear_num_value_heads) // tp_size,
            int(cfg.linear_key_head_dim),
            int(cfg.linear_value_head_dim),
        )

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_gdn_packed_decode(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _QWEN35_MODEL_TYPES:
            return False
        if self._ixaite_gdn_tp_geometry() is None:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op(
            "gdn_packed_decode",
            "fused_recurrent_gated_delta_rule_packed_decode",
        )
        if implementation is None:
            return False

        logger.info("Warming ixaite GDN packed decode cubin")
        self._run_ixaite_gdn_packed_decode_warmup(implementation)
        return True

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_gdn_packed_decode_warmup(self, implementation) -> None:
        import torch

        geometry = self._ixaite_gdn_tp_geometry()
        assert geometry is not None
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
        dtype = self.model_config.dtype
        device = getattr(self, "device", None) or self.model_runner.device
        qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        mixed_qkv = torch.zeros(1, qkv_dim, dtype=dtype, device=device)
        a = torch.zeros(1, num_v_heads, dtype=dtype, device=device)
        b = torch.zeros_like(a)
        a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
        dt_bias = torch.zeros_like(a_log)
        state = torch.zeros(
            2, num_v_heads, head_v_dim, head_k_dim, dtype=dtype, device=device
        )
        out = torch.zeros(1, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)
        indices = torch.ones(1, dtype=torch.int32, device=device)
        implementation(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=head_k_dim**-0.5,
            initial_state=state,
            out=out,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        del mixed_qkv, a, b, a_log, dt_bias, state, out, indices

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_tp_size(self) -> int:
        parallel_config = getattr(self, "parallel_config", None)
        return int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_max_num_seqs(self) -> int:
        for src in (
            getattr(self, "scheduler_config", None),
            getattr(self.model_runner, "scheduler_config", None),
        ):
            value = getattr(src, "max_num_seqs", None)
            if value:
                return int(value)
        return 128

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_torch_dtype(self):
        import torch

        dtype = getattr(self.model_config, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        name = str(dtype).lower().replace("torch.", "")
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
        }.get(name, torch.bfloat16)

    # Retain the existing lifecycle and warmup contract.
    def _ixaite_warmup_device(self):
        return getattr(self, "device", None) or self.model_runner.device

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_body_small_m(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import is_ixaite_op_enabled

        if not is_ixaite_op_enabled("linear"):
            return False
        sizes = ixaite_body_small_m_warmup_sizes(self.model_runner.max_num_tokens)
        if not sizes:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op("linear", "linear")
        if implementation is None:
            return False

        logger.info("Warming ixaite body small_m linears with token sizes %s", sizes)
        self._run_ixaite_linear_family_warmup(
            implementation, sizes, include_lm_head=False
        )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_linear_exact_buckets(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import is_ixaite_op_enabled

        if not is_ixaite_op_enabled("linear"):
            return False
        max_num_tokens = self._ixaite_reachable_token_cap()
        sizes = ixaite_hgemm_exact_warmup_sizes(max_num_tokens)
        if (
            model_type in _QWEN35_VISION_MODEL_TYPES
            and self._ixaite_has_vision_encoder()
        ):
            extra = ixaite_qwen35_vision_linear_warmup_sizes(max_num_tokens)
            sizes = sorted(set(sizes) | set(extra))
        if not sizes:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op("linear", "linear")
        if implementation is None:
            return False

        logger.info(
            "Warming ixaite exact linear buckets with token sizes %s",
            sizes,
        )
        self._run_ixaite_linear_family_warmup(
            implementation, sizes, include_lm_head=False
        )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_lm_head(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _IXAITE_TEXT_WARMUP_MODEL_TYPES:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import is_ixaite_op_enabled

        if not is_ixaite_op_enabled("linear"):
            return False
        sizes = ixaite_lm_head_warmup_sizes(self._ixaite_max_num_seqs())
        if not sizes:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op("linear", "linear")
        if implementation is None:
            return False

        logger.info("Warming ixaite lm_head with token sizes %s", sizes)
        self._run_ixaite_linear_family_warmup(
            implementation, sizes, include_lm_head=True
        )
        return True

    # Retain the existing lifecycle and warmup contract.
    def _clear_ixaite_linear_staging(self) -> None:
        import sys

        linear_mod = sys.modules.get("ixaite.ops.linear")
        if linear_mod is None:
            return
        for name in (
            "_STAGING_A",
            "_STAGING_A_K",
            "_STAGING_BIAS_MR",
            "_STAGING_W_N",
            "_STAGING_W_MR",
            "_STAGING_C_N",
        ):
            cache = getattr(linear_mod, name, None)
            if cache is not None:
                cache.clear()

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_linear_family_warmup(
        self, implementation, sizes: list[int], *, include_lm_head: bool
    ) -> None:
        import torch

        # dummy_run leaves InferenceMode staging buffers; the custom op
        # then copy_s into them outside InferenceMode.
        self._clear_ixaite_linear_staging()
        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        with torch.inference_mode(False):
            for geom in self._ixaite_linear_warmup_geometries(
                include_lm_head=include_lm_head
            ):
                weight = torch.zeros(geom.n, geom.k, dtype=dtype, device=device)
                for m in sizes:
                    x = torch.zeros(m, geom.k, dtype=dtype, device=device)
                    implementation(x, weight)
                    del x
                del weight

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_gdn_conv(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _QWEN35_MODEL_TYPES:
            return False
        if self._ixaite_gdn_tp_geometry() is None:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        fn_impl, _ = load_ixaite_python_op("causal_conv1d", "causal_conv1d_fn")
        update_impl, _ = load_ixaite_python_op("causal_conv1d", "causal_conv1d_update")
        if fn_impl is None and update_impl is None:
            return False

        logger.info("Warming ixaite GDN causal conv1d cubins")
        if fn_impl is not None:
            self._run_ixaite_gdn_conv_fn_warmup(fn_impl)
        if update_impl is not None:
            self._run_ixaite_gdn_conv_update_warmup(update_impl)
        return True

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_gdn_conv_fn_warmup(self, implementation) -> None:
        import torch

        geometry = self._ixaite_gdn_tp_geometry()
        assert geometry is not None
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        width = _CAUSAL_CONV_WIDTH
        t = min(64, int(self.model_runner.max_num_tokens) or 64)
        with torch.inference_mode(False):
            x = torch.zeros(dim, t, dtype=dtype, device=device)
            weight = torch.zeros(dim, width, dtype=dtype, device=device)
            bias = torch.zeros(dim, dtype=dtype, device=device)
            conv_states = torch.zeros(2, dim, width - 1, dtype=dtype, device=device)
            query_start_loc = torch.tensor([0, t], dtype=torch.int32, device=device)
            implementation(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                activation="silu",
            )
            has_initial_state = torch.zeros(1, dtype=torch.int32, device=device)
            implementation(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                has_initial_state=has_initial_state,
                activation="silu",
            )
            input_index = torch.arange(t, dtype=torch.int32, device=device)
            implementation(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                activation="silu",
                input_index=input_index,
            )
            del x, weight, bias, conv_states, query_start_loc
            del has_initial_state, input_index

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_gdn_conv_update_warmup(self, implementation) -> None:
        import torch

        geometry = self._ixaite_gdn_tp_geometry()
        assert geometry is not None
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        width = _CAUSAL_CONV_WIDTH
        with torch.inference_mode(False):
            x = torch.zeros(1, dim, dtype=dtype, device=device)
            weight = torch.zeros(dim, width, dtype=dtype, device=device)
            bias = torch.zeros(dim, dtype=dtype, device=device)
            conv_state = torch.zeros(2, dim, width - 1, dtype=dtype, device=device)
            implementation(x, conv_state, weight, bias, activation="silu")
            input_index = torch.zeros(1, dtype=torch.int32, device=device)
            packed = torch.zeros(1, dim, dtype=dtype, device=device)
            implementation(
                packed,
                conv_state,
                weight,
                bias,
                activation="silu",
                input_index=input_index,
            )
            del x, weight, bias, conv_state, input_index, packed

    # Retain the existing lifecycle and warmup contract.
    def _warm_up_ixaite_gdn_post_conv_prep(self) -> bool:
        model_type = getattr(self.model_config.hf_config, "model_type", "")
        if model_type not in _QWEN35_MODEL_TYPES:
            return False
        if self._ixaite_gdn_tp_geometry() is None:
            return False

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        implementation, _ = load_ixaite_python_op(
            "gdn_post_conv_prep", "fused_post_conv_prep"
        )
        if implementation is None:
            return False

        logger.info("Warming ixaite GDN post-conv prep cubins")
        self._run_ixaite_gdn_post_conv_prep_warmup(implementation)
        return True

    # Retain the existing lifecycle and warmup contract.
    def _run_ixaite_gdn_post_conv_prep_warmup(self, implementation) -> None:
        import torch

        geometry = self._ixaite_gdn_tp_geometry()
        assert geometry is not None
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
        dtype = self._ixaite_torch_dtype()
        device = self._ixaite_warmup_device()
        packed_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        with torch.inference_mode(False):
            conv_out = torch.zeros(1, packed_dim, dtype=dtype, device=device)
            a = torch.zeros(1, num_v_heads, dtype=dtype, device=device)
            b = torch.zeros_like(a)
            a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
            dt_bias = torch.zeros_like(a_log)
            for apply_l2norm in (True, False):
                implementation(
                    conv_out,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    num_k_heads,
                    head_k_dim,
                    head_v_dim,
                    apply_l2norm=apply_l2norm,
                )
            output_index = torch.zeros(1, dtype=torch.int32, device=device)
            implementation(
                conv_out,
                a,
                b,
                a_log,
                dt_bias,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                apply_l2norm=False,
                output_index=output_index,
            )
            del conv_out, a, b, a_log, dt_bias, output_index

    # Retain the existing lifecycle and warmup contract.
    def _release_ixaite_warmup_scratch(self) -> None:
        import gc
        import sys

        import torch

        self._clear_ixaite_linear_staging()
        gdn_mod = sys.modules.get("ixaite.ops.gdn_chunk_wy")
        if gdn_mod is not None:
            cache = getattr(gdn_mod, "_WS_CACHE", None)
            if cache is not None:
                cache.clear()

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
