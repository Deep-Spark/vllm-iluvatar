# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker with following overrides:

1. ``determine_available_memory`` for attention quantisation support.
2. ``initialize_from_config`` for attention quantisation support.
3. ``shutdown`` for attention quantisation support.
4. ``compile_or_warm_up_model`` to exclude simulated requests from KV calibration.
5. ``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
import vllm.v1.worker.gpu.model_runner as _gpu_model_runner
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_iluvatar.ilu_util.arch import is_ivcore11

from .pp_exchange import (
    defer_pp_send_wait,
    drain_pp_send_wait,
    schema_bytes,
    schema_pp_exchange,
)

logger = init_logger(__name__)


def _execute_with_received_pp_tensors(
    runner, execute_model, scheduler_output, intermediate_tensors=None, **kwargs
):
    """Borrow received PP buffers for one eager forward, retaining their owner."""
    saved = runner.intermediate_tensors
    if intermediate_tensors is not None:
        runner.intermediate_tensors = intermediate_tensors
    try:
        return execute_model(scheduler_output, intermediate_tensors, **kwargs)
    finally:
        runner.intermediate_tensors = saved


def _persistent_runner_storage_ptrs(runner) -> set[int]:
    ptrs: set[int] = set()
    model = getattr(runner, "model", None)
    if model is not None:
        for t in model.parameters():
            ptrs.add(t.untyped_storage().data_ptr())
        for t in model.buffers():
            ptrs.add(t.untyped_storage().data_ptr())
    inter = getattr(runner, "intermediate_tensors", None)
    if inter is not None and hasattr(inter, "tensors"):
        for t in inter.tensors.values():
            if isinstance(t, torch.Tensor):
                ptrs.add(t.untyped_storage().data_ptr())
    for attr in ("input_buffers", "model_state"):
        state = getattr(runner, attr, None)
        if state is None or not hasattr(state, "__dict__"):
            continue
        for val in vars(state).values():
            if isinstance(val, torch.Tensor):
                ptrs.add(val.untyped_storage().data_ptr())
            elif isinstance(val, dict):
                for item in val.values():
                    if isinstance(item, torch.Tensor):
                        ptrs.add(item.untyped_storage().data_ptr())
    return ptrs


def _pcp_profile_local_tokens(num_tokens: int, pcp_size: int, max_num_reqs: int) -> int:
    # DualChunkSwap gives each prefill rank at most n / PCP + 2 tokens
    # per request (two rounded chunks). Single-token decodes are replicated.
    # This upper bound covers mixed batches and the maximum rank's padding.
    return min(num_tokens, (num_tokens + pcp_size - 1) // pcp_size + 2 * max_num_reqs)


class IluWorker(Worker):
    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        if (
            max_split_size_mb == 20
            and self.parallel_config.pipeline_parallel_size > 1
            and self.model_config.enforce_eager
        ):
            max_split_size_mb = 21
        return super()._scoped_allocator_max_split(max_split_size_mb)

    def _eager_pp_direct_input_supported(self) -> bool:
        parallel = getattr(self, "parallel_config", None)
        model_cfg = getattr(self, "model_config", None)
        vllm_cfg = getattr(self, "vllm_config", None)
        runner = getattr(self, "model_runner", None)
        if parallel is None or model_cfg is None or vllm_cfg is None or runner is None:
            return False
        if not getattr(self, "use_v2_model_runner", False):
            return False
        if not getattr(model_cfg, "enforce_eager", False):
            return False
        from vllm.config.compilation import CUDAGraphMode

        if vllm_cfg.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            return False
        return (
            parallel.pipeline_parallel_size > 1
            and parallel.data_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and not parallel.use_sequence_parallel_moe
            and not vllm_cfg.compilation_config.pass_config.enable_sp
            and not getattr(parallel, "enable_dbo", False)
            and getattr(runner, "ubatch_runner", None) is None
        )

    def _configure_eager_pp_exchange(self) -> None:
        self._pp_send_schema = None
        self._pp_recv_schema = None
        self._pp_defer_send_wait = False
        if not self._eager_pp_direct_input_supported() or not is_ivcore11():
            return
        if getattr(self.device, "type", None) != "cuda":
            return
        parallel = self.parallel_config
        runner = self.model_runner
        if (
            parallel.tensor_parallel_size != 1
            or getattr(runner, "is_pooling_model", False)
            or getattr(runner, "is_encoder_decoder", False)
            or getattr(runner, "use_aux_hidden_state_outputs", False)
            or getattr(
                getattr(runner, "pp_handler", None), "aux_hidden_state_relay_keys", None
            )
            or not callable(getattr(runner, "model", None))
            or not callable(getattr(runner, "_dummy_run", None))
        ):
            return

        from vllm.distributed import get_pp_group
        from vllm.sequence import IntermediateTensors

        pp = get_pp_group()
        model = runner.model
        if not hasattr(model, "register_forward_hook"):
            return

        probe_tokens = max(
            2,
            getattr(runner, "decode_query_len", 1),
            getattr(runner, "uniform_decode_query_len", 1),
        )
        persistent_ptrs = _persistent_runner_storage_ptrs(runner)
        captured: dict[str, object] = {}

        def _capture_output(_module, _args, out):
            if not pp.is_last_rank and isinstance(out, IntermediateTensors):
                valid = all(
                    t.is_cuda
                    and t.is_contiguous()
                    and t.ndim >= 1
                    and t.shape[0] == probe_tokens
                    for t in out.tensors.values()
                )
                if valid and out.tensors:
                    captured["send_schema"] = tuple(
                        (k, tuple(t.shape[1:]), t.dtype) for k, t in out.tensors.items()
                    )
                    captured["fresh_alloc"] = all(
                        t.untyped_storage().data_ptr() not in persistent_ptrs
                        for t in out.tensors.values()
                    )

        hook_handle = model.register_forward_hook(_capture_output)
        try:
            runner._dummy_run(probe_tokens, skip_attn=True, is_profile=True)
        finally:
            hook_handle.remove()
            runner.execute_model_state = None
            if not pp.is_last_rank and hasattr(runner, "step_timing"):
                runner.step_timing.forward_end()

        # Reclaim the oversized runner.intermediate_tensors allocation on
        # non-first ranks now that eager PP steps borrow received buffers.
        recv_schema = None
        if not pp.is_first_rank:
            inter = getattr(runner, "intermediate_tensors", None)
            if (
                isinstance(inter, IntermediateTensors)
                and inter.tensors
                and all(
                    t.is_cuda and t.is_contiguous() and t.ndim >= 1
                    for t in inter.tensors.values()
                )
            ):
                recv_schema = tuple(
                    (k, tuple(t.shape[1:]), t.dtype) for k, t in inter.tensors.items()
                )
                dummy_cap = max(
                    16,
                    getattr(runner, "decode_query_len", 1),
                    getattr(runner, "uniform_decode_query_len", 1),
                )
                cur_rows = next(iter(inter.tensors.values())).shape[0]
                if cur_rows > dummy_cap and hasattr(
                    model, "make_empty_intermediate_tensors"
                ):
                    runner.intermediate_tensors = model.make_empty_intermediate_tensors(
                        batch_size=dummy_cap,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )
                    torch.cuda.empty_cache()

        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        local_info = {
            "send_schema": captured.get("send_schema"),
            "recv_schema": recv_schema,
            "fresh_alloc": bool(captured.get("fresh_alloc", False)),
            "free_bytes": int(free_bytes),
        }
        gathered: list[dict] = [None] * pp.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_info, group=pp.cpu_group)

        schemas_match = all(
            gathered[h]["send_schema"] is not None
            and gathered[h]["send_schema"] == gathered[h + 1]["recv_schema"]
            for h in range(pp.world_size - 1)
        )
        if not schemas_match:
            return

        max_tokens = getattr(
            runner, "max_num_tokens", self.scheduler_config.max_num_batched_tokens
        )
        max_hop_bytes = max(
            schema_bytes(gathered[h]["send_schema"], max_tokens)
            for h in range(pp.world_size - 1)
        )
        min_free = min(info["free_bytes"] for info in gathered)
        min_headroom = max(256 << 20, 2 * max_hop_bytes)
        defer_send_wait = (
            all(gathered[h]["fresh_alloc"] for h in range(pp.world_size - 1))
            and min_free >= min_headroom
        )

        if not pp.is_last_rank:
            self._pp_send_schema = gathered[pp.rank_in_group]["send_schema"]
        if not pp.is_first_rank:
            self._pp_recv_schema = gathered[pp.rank_in_group]["recv_schema"]
        self._pp_defer_send_wait = defer_send_wait
        if pp.is_first_rank:
            logger.info(
                "ILU eager PP schema exchange active (defer_send_wait=%s, free_min_mib=%.1f)",
                defer_send_wait,
                min_free / (1024 * 1024),
            )

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from vllm_iluvatar import envs as ilu_envs

        if ilu_envs.VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8:
            from vllm_iluvatar.models.glm5next.mtp import (
                IluGlm5NextMTP,
                prepare_draft_head_w8a8,
            )

            speculator = getattr(self.model_runner, "speculator", None)
            draft = getattr(speculator, "model", None)
            if not isinstance(draft, IluGlm5NextMTP):
                raise ValueError(
                    "Draft head W8A8 requires GLM5Next MTP with model runner V2"
                )
            prepare_draft_head_w8a8(draft)

        configure = (
            getattr(self.model_runner.model_state, "configure_prefix_reuse", None)
            if self.vllm_config.use_v2_model_runner
            else None
        )
        self.prefix_state = (
            configure(self.model_runner) if configure is not None else None
        )
        import vllm.envs as envs

        if envs.VLLM_ILU_DP_VOCAB_GROUP_SIZE:
            if not self.vllm_config.use_v2_model_runner:
                raise ValueError("DP vocabulary projection requires model runner V2")
            from vllm_iluvatar.distributed.dp_vocab_projection import (
                install_dp_vocab_projection,
            )

            install_dp_vocab_projection(
                self.model_runner, envs.VLLM_ILU_DP_VOCAB_GROUP_SIZE
            )
        return result

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

    def _try_replace_model_runner_class(self) -> type[object]:
        import vllm.envs as envs

        # Preserve the actual runner class. With VLLM_USE_V2_MODEL_RUNNER=1,
        # self.model_runner is the V2 GPUModelRunner; restoring the legacy class
        # here corrupts the instance and makes V2 initialization call V1 methods.
        orig_cls = self.model_runner.__class__
        is_v2_runner = isinstance(self.model_runner, _gpu_model_runner.GPUModelRunner)

        quant_kv_cache_requested = (
            envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != ""
            or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
        )

        if quant_kv_cache_requested and not is_v2_runner:
            raise NotImplementedError(
                "Iluvatar quantized KV cache currently requires the V2 model "
                "runner. Set VLLM_USE_V2_MODEL_RUNNER=1, or disable "
                "VLLM_ILU_FLASH_ATTN_QUANT_TYPE/VLLM_ILU_MLA_ATTN_QUANT_TYPE."
            )

        # The installed ixinfer build supports the community interleaved
        # LBHNC layout in flash_attn_with_kvcache and reshape_and_cache_flash
        # (verified by kernel probe), so unquantized V2 runs keep the community
        # GPUModelRunner. Only quantized KV still swaps in
        # IluvatarGPUModelRunner for its int8 scale-cache allocation.
        if quant_kv_cache_requested:
            logger.info(
                "Enable quant kv cache, replace self.model_runner.__class__ to IluvatarGPUModelRunner temporarily"
            )
            from vllm_iluvatar.v1.worker.gpu.model_runner import (
                IluvatarGPUModelRunner,
            )

            self.model_runner.__class__ = IluvatarGPUModelRunner
        return orig_cls

    def determine_available_memory(self) -> int:
        from vllm_iluvatar import envs as ilu_envs

        if (
            not ilu_envs.VLLM_ILU_SKIP_CUDAGRAPH_MEMORY_PROFILE
            or not self.vllm_config.use_v2_model_runner
        ):
            return self._determine_available_memory()
        # Avoid the CoreX throwaway collective graph lifecycle. Activation
        # profiling and the final FULL decode graph capture still run.
        runner = self.model_runner
        previous = runner.profile_cudagraph_memory
        had_override = "profile_cudagraph_memory" in vars(runner)
        runner.profile_cudagraph_memory = lambda: 0
        logger.warning(
            "Skipping throwaway CUDA graph memory estimate; real capture remains enabled"
        )
        try:
            return self._determine_available_memory()
        finally:
            if had_override:
                runner.profile_cudagraph_memory = previous
            else:
                del runner.profile_cudagraph_memory

    def _determine_available_memory(self) -> int:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
            CacheBudget,
            main_kv_format,
            prefill_producer_for_config,
            retained_mapper_bytes_for_config,
            workspace_bytes_for_config,
        )

        fmt = main_kv_format()
        if fmt is not None:
            self._validate_compressed_kv_runner()
            workspace_bytes = workspace_bytes_for_config(self.vllm_config)
            prefill_producer = prefill_producer_for_config(self.vllm_config)
            retained_mapper_bytes = retained_mapper_bytes_for_config(self.vllm_config)
            if retained_mapper_bytes >= workspace_bytes:
                raise ValueError(
                    "Compressed mapper credit leaves no usable workspace arena"
                )
            from vllm_iluvatar.v1.worker.gpu.compressed_kv_profiling import (
                compressed_kv_profiling,
            )

            with compressed_kv_profiling(
                self.model_runner,
                fmt,
                workspace_bytes=workspace_bytes - retained_mapper_bytes,
                prefill_producer=prefill_producer,
            ):
                available = super().determine_available_memory()
            budget = CacheBudget.create(
                available,
                fmt,
                workspace_bytes=workspace_bytes,
                retained_mapper_bytes=retained_mapper_bytes,
                prefill_producer=prefill_producer,
            )
            self._compressed_kv_budget = budget
            # The scheduler receives payload bytes only. Sidecars and the
            # single fixed workspace are reserved before its block allocator.
            logger.info(
                "Compressed KV budget: format=%s total=%d payload=%d "
                "scale_page=%d workspace=%d retained_mapper=%d arena=%d",
                fmt.name,
                available,
                budget.payload_bytes,
                budget.scale_page_bytes,
                budget.workspace_bytes,
                budget.retained_mapper_bytes,
                budget.arena_bytes,
            )
            return budget.payload_bytes
        orig_cls = self._try_replace_model_runner_class()
        try:
            from contextlib import nullcontext

            from vllm_iluvatar.models.deepseek_v41.indexer_kv import indexer_kv_layout

            profiling = nullcontext()
            if indexer_kv_layout() == "split":
                if not self.vllm_config.use_v2_model_runner:
                    raise ValueError("Split indexer KV requires model runner V2")
                from vllm_iluvatar.v1.worker.gpu.compressed_kv_profiling import (
                    compressed_kv_profiling,
                )

                profiling = compressed_kv_profiling(
                    self.model_runner,
                    None,
                    workspace_bytes=0,
                    prefill_producer=prefill_producer_for_config(self.vllm_config),
                )
            with profiling, self._pcp_memory_profile():
                available = super().determine_available_memory()
            self._indexer_kv_available_memory = available
            return available
        finally:
            self.model_runner.__class__ = orig_cls

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import main_kv_format
        from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
            IndexerScaleState,
            extend_kv_zeroer,
            plan_indexer_scales,
        )

        index_layout = plan_indexer_scales(kv_cache_config)
        index_state = None
        if index_layout.placements:
            if not self.vllm_config.use_v2_model_runner:
                raise ValueError("Split indexer KV requires model runner V2")
            if getattr(self, "_indexer_kv_state", None) is not None:
                raise RuntimeError("Split indexer KV cache is already initialized")
            with self._maybe_get_memory_pool_context(tag="kv_cache"):
                index_state = IndexerScaleState(
                    index_layout,
                    self.device,
                    self.vllm_config.compilation_config.static_forward_context,
                )
            self._indexer_kv_state = index_state
            logger.info(
                "Split INT8 indexer scales: bytes=%d owners=%d blocks=%d",
                index_layout.nbytes,
                len(index_layout.placements),
                index_layout.num_blocks,
            )

        fmt = main_kv_format()
        orig_cls = self._try_replace_model_runner_class()
        try:
            if fmt is not None:
                self._initialize_compressed_kv(kv_cache_config, fmt)
            else:
                if index_state is not None:
                    payload = {t.size for t in kv_cache_config.kv_cache_tensors}
                    if (
                        len(payload) != 1
                        or payload.pop() + index_layout.nbytes
                        > self._indexer_kv_available_memory
                    ):
                        raise ValueError("Split indexer exceeds its profiled budget")
                super().initialize_from_config(kv_cache_config)
                if index_state is not None:
                    self.model_runner.kv_caches.extend(index_state.scales.values())
                    if self.model_runner.kv_block_zeroer is None:
                        self.model_runner._init_kv_zero_meta()
                    extend_kv_zeroer(
                        self.model_runner.kv_block_zeroer,
                        index_layout.num_blocks,
                        index_state.scales.values(),
                    )
            if isinstance(self.model_runner, _gpu_model_runner.GPUModelRunner):
                from vllm_iluvatar.v1.worker.gpu.model_runner import (
                    warmup_topp_split_kernels,
                )

                warmup_topp_split_kernels(self.model_runner)
            self._initialize_prefix_cache()
        except BaseException:
            if index_state is not None:
                index_state.release()
                self._indexer_kv_state = None
            raise
        finally:
            self.model_runner.__class__ = orig_cls

    def _initialize_prefix_cache(self) -> None:
        if getattr(self, "prefix_state", None) is not None:
            from vllm.distributed.kv_transfer import get_kv_transfer_group

            from vllm_iluvatar.v1.worker.prefill_state.runtime import (
                attach_prefix_state,
            )

            self.prefix_state.initialize_cache()
            attach_prefix_state(get_kv_transfer_group(), self.prefix_state)

    def _validate_compressed_kv_runner(self) -> None:
        import vllm.envs as envs

        if not self.vllm_config.use_v2_model_runner:
            raise NotImplementedError(
                "DS41 compressed main KV requires model runner V2"
            )
        model_config = getattr(self.vllm_config, "model_config", None)
        if model_config is not None:
            hf_config = getattr(
                model_config, "hf_text_config", getattr(model_config, "hf_config", None)
            )
            if getattr(hf_config, "model_type", None) != "deepseek_v41":
                raise ValueError("DS41 compressed main KV only supports DeepSeek V4.1")
        if self.cache_config.cache_dtype != "bfloat16":
            raise ValueError(
                "DS41 compressed main KV requires --kv-cache-dtype bfloat16 for SWA"
            )
        if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE:
            raise ValueError(
                "DS41 compressed main KV cannot use legacy V1 quantization settings"
            )
        if self.vllm_config.attention_config.hisparse_config is not None:
            raise NotImplementedError(
                "DS41 compressed main KV does not use the HiSparse allocator"
            )

    def _initialize_compressed_kv(self, kv_cache_config, fmt) -> None:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
            CompressedKVState,
            plan_scale_layout,
        )

        self._validate_compressed_kv_runner()
        budget = getattr(self, "_compressed_kv_budget", None)
        if budget is None:
            raise RuntimeError(
                "Compressed KV must be budgeted before cache initialization"
            )
        if getattr(self, "_compressed_kv_state", None) is not None:
            raise RuntimeError("Compressed KV cache is already initialized")
        if getattr(self, "_compressed_kv_init_attempted", False):
            raise RuntimeError(
                "Compressed KV initialization failed; restart the worker before retrying"
            )
        layout = plan_scale_layout(kv_cache_config, fmt)
        index_state = getattr(self, "_indexer_kv_state", None)
        index_bytes = index_state.layout.nbytes if index_state is not None else 0
        budget.validate(layout.payload_bytes, layout.scale_bytes + index_bytes)
        logger.info(
            "Compressed KV allocation: payload=%d main_scales=%d "
            "index_scales=%d blocks=%d",
            layout.payload_bytes,
            layout.scale_bytes,
            index_bytes,
            layout.num_blocks,
        )
        context = self.vllm_config.compilation_config.static_forward_context
        self._compressed_kv_init_attempted = True
        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            state = CompressedKVState(
                layout,
                fmt,
                self.device,
                context,
                workspace_bytes=budget.arena_bytes,
                prefill_producer=budget.prefill_producer,
            )
        self._compressed_kv_state = state
        try:
            # Bind scale views before the V2 runner registers the cache with
            # its connector. NIXL observes both planes in one layer transfer.
            super().initialize_from_config(kv_cache_config)
            # V2 uses this tensor list for physical block copies. No synthetic
            # layer names or separate block IDs enter its metadata path.
            self.model_runner.kv_caches.extend(state.scales.values())
            extra_planes = tuple(index_state.scales.values()) if index_state else ()
            self.model_runner.kv_caches.extend(extra_planes)
            if state.scales or extra_planes:
                if getattr(self.model_runner, "kv_block_zeroer", None) is None:
                    self.model_runner._init_kv_zero_meta()
                state.bind_zeroer(self.model_runner.kv_block_zeroer, extra_planes)
        except Exception:
            state.release()
            self._compressed_kv_state = None
            raise

    def _get_onchip_checker(self):
        checker = getattr(self, "_onchip_checker", None)
        if checker is None:
            from vllm_iluvatar.ilu_util.onchip_check import OnchipChecker

            checker = OnchipChecker()
            self._onchip_checker = checker
        return checker

    def _get_hbm_ecc_monitor(self):
        monitor = getattr(self, "_hbm_ecc_monitor", None)
        if monitor is None:
            from vllm_iluvatar.ilu_util.hbm_ecc_monitor import HbmEccMonitor

            monitor = HbmEccMonitor()
            self._hbm_ecc_monitor = monitor
        return monitor

    def compile_or_warm_up_model(self):
        from vllm_iluvatar.models.glm_moe_dsa.attention import IluGlmMoeDsaSparseMLA

        # v0.31 warmup can simulate requests with valid KV slots even at PCP=1.
        # Only real requests may freeze the online scale; restore on failure too.
        calibration_states = [
            (layer, layer._fp8_kv_scale_calibration_enabled)
            for layer in self.model_runner.model.modules()
            if isinstance(layer, IluGlmMoeDsaSparseMLA)
        ]
        for layer, _ in calibration_states:
            layer._fp8_kv_scale_calibration_enabled = False
        local_rank = getattr(self, "local_rank", 0)
        rank = getattr(self, "rank", 0)
        device = getattr(self, "device", None)
        physical_gpu = current_platform.device_id_to_physical_device_id(local_rank)
        checker = self._get_onchip_checker()
        monitor = self._get_hbm_ecc_monitor()
        try:
            checker.start(
                device,
                rank=rank,
                local_rank=local_rank,
                logical_device=local_rank,
                physical_gpu=physical_gpu,
            )
            monitor.start(
                rank=rank,
                local_rank=local_rank,
                logical_device=local_rank,
                physical_gpu=physical_gpu,
            )
            result = super().compile_or_warm_up_model()
            self._configure_eager_pp_exchange()
            controller = getattr(self.model_runner, "_ilu_dp_vocab_projection", None)
            if controller is not None:
                controller.enable_dynamic_capacity()
            if getattr(self, "prefix_state", None) is not None:
                self.prefix_state.start()
            return result
        except Exception:
            monitor.free()
            checker.free()
            raise
        finally:
            for layer, enabled in calibration_states:
                layer._fp8_kv_scale_calibration_enabled = enabled

    def sample_tokens(self, grammar_output):
        controller = getattr(self.model_runner, "_ilu_dp_vocab_projection", None)
        if (
            controller is None
            or not controller.dynamic_capacity
            or self.model_runner.execute_model_state is None
        ):
            return super().sample_tokens(grammar_output)
        # Read the shared CPU agreement before the runner clears its execute
        # state. Do not retain that state and its activation owners here.
        dp_sync = self.model_runner.execute_model_state.dp_sync
        with controller.pair(dp_sync):
            return super().sample_tokens(grammar_output)

    def _run_execute_model(self, scheduler_output):
        state = getattr(self, "_compressed_kv_state", None)
        if state is not None:
            # Lease the shared arena for eager work and graph replay alike.
            with state.workspace.acquire():
                return super().execute_model(scheduler_output)
        return super().execute_model(scheduler_output)

    def execute_model(self, scheduler_output):
        self._get_hbm_ecc_monitor().check()
        rows = scheduler_output.total_num_scheduled_tokens
        if rows > 0:
            self._get_onchip_checker().before_batch()
        context = getattr(self, "prefix_state", None)
        if context is not None:
            context.observe_scheduler_output(scheduler_output)

        original = None
        if rows > 0 and self._eager_pp_direct_input_supported():
            original = self.model_runner.execute_model
            self.model_runner.execute_model = partial(
                _execute_with_received_pp_tensors, self.model_runner, original
            )
        try:
            send_schema = getattr(self, "_pp_send_schema", None)
            recv_schema = getattr(self, "_pp_recv_schema", None)
            if rows > 0 and (send_schema is not None or recv_schema is not None):
                from vllm.distributed import get_pp_group

                defer = bool(getattr(self, "_pp_defer_send_wait", False))
                pending = defer_pp_send_wait(self) if defer else ()
                with schema_pp_exchange(
                    get_pp_group(),
                    rows,
                    self.device,
                    send_schema=send_schema,
                    recv_schema=recv_schema,
                    pending=pending,
                    retain_send=defer,
                ):
                    return self._run_execute_model(scheduler_output)
            if getattr(self, "_pp_deferred_sends", None):
                drain_pp_send_wait(self)
            return self._run_execute_model(scheduler_output)
        finally:
            if original is not None:
                self.model_runner.execute_model = original
            if context is not None:
                context.end_step()

    def shutdown(self) -> None:
        drain_pp_send_wait(self)
        if getattr(self, "prefix_state", None) is not None:
            self.prefix_state.shutdown()
        orig_cls = self._try_replace_model_runner_class()
        try:
            super().shutdown()
        finally:
            state = getattr(self, "_compressed_kv_state", None)
            if state is not None:
                state.release()
                self._compressed_kv_state = None
            index_state = getattr(self, "_indexer_kv_state", None)
            if index_state is not None:
                index_state.release()
                self._indexer_kv_state = None
            self.model_runner.__class__ = orig_cls

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        try:
            self.model_runner._dummy_run(
                num_tokens, uniform_decode=True, profile_seq_lens=1
            )
        finally:
            if getattr(self, "prefix_state", None) is not None:
                self.prefix_state.end_step()
