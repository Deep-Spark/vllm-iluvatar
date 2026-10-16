# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker with following overrides:

1. ``determine_available_memory`` for attention quantisation support.
2. ``initialize_from_config`` for attention quantisation support.
3. ``compile_or_warm_up_model`` / ``execute_model`` / ``shutdown`` for the
   optional on-chip storage integrity checker and HBM ECC monitor.
4. ``shutdown`` for attention quantisation support.
5. ``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.
"""

from __future__ import annotations

import vllm.v1.worker.gpu_model_runner as _gpu_model_runner
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger("vllm." + __name__)


class IluWorker(Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        configure = (
            getattr(self.model_runner.model_state, "configure_prefix_reuse", None)
            if self.vllm_config.use_v2_model_runner
            else None
        )
        self.prefix_state = (
            configure(self.model_runner) if configure is not None else None
        )
        return result

    def _try_replace_model_runner_class(self) -> type[object]:
        import vllm.envs as envs

        # Preserve the actual runner class. With VLLM_USE_V2_MODEL_RUNNER=1,
        # self.model_runner is the V2 GPUModelRunner; restoring the legacy class
        # here corrupts the instance and makes V2 initialization call V1 methods.
        orig_cls = self.model_runner.__class__
        is_v1_runner = isinstance(self.model_runner, _gpu_model_runner.GPUModelRunner)

        quant_kv_cache_requested = (
            envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != ""
            or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
        )

        if quant_kv_cache_requested and not is_v1_runner:
            raise NotImplementedError(
                "Iluvatar quantized KV cache currently requires the V1 model "
                "runner. Set VLLM_USE_V2_MODEL_RUNNER=0, or disable "
                "VLLM_ILU_FLASH_ATTN_QUANT_TYPE/VLLM_ILU_MLA_ATTN_QUANT_TYPE."
            )

        if quant_kv_cache_requested:
            logger.info(
                "Enable quant kv cache, replace self.model_runner.__class__ to IluvatarGPUModelRunner temporarily"
            )
            from vllm_iluvatar.v1.worker.gpu_model_runner import (
                IluvatarGPUModelRunner,
            )

            self.model_runner.__class__ = IluvatarGPUModelRunner
        return orig_cls

    def determine_available_memory(self) -> int:
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
            return super().determine_available_memory()
        finally:
            self.model_runner.__class__ = orig_cls

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import main_kv_format

        fmt = main_kv_format()
        if fmt is not None:
            self._initialize_compressed_kv(kv_cache_config, fmt)
            return
        orig_cls = self._try_replace_model_runner_class()
        try:
            super().initialize_from_config(kv_cache_config)
            if getattr(self, "prefix_state", None) is not None:
                from vllm.distributed.kv_transfer import get_kv_transfer_group

                self.prefix_state.initialize_cache()
                get_kv_transfer_group().attach_prefix_state(self.prefix_state)
        finally:
            self.model_runner.__class__ = orig_cls

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
        budget.validate(layout.payload_bytes, layout.scale_bytes)
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
            if state.scales:
                if getattr(self.model_runner, "kv_block_zeroer", None) is None:
                    self.model_runner._init_kv_zero_meta()
                state.bind_zeroer(self.model_runner.kv_block_zeroer)
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
        physical_gpu = current_platform.device_id_to_physical_device_id(self.local_rank)
        checker = self._get_onchip_checker()
        monitor = self._get_hbm_ecc_monitor()
        try:
            checker.start(
                self.device,
                rank=self.rank,
                local_rank=self.local_rank,
                logical_device=self.local_rank,
                physical_gpu=physical_gpu,
            )
            monitor.start(
                rank=self.rank,
                local_rank=self.local_rank,
                logical_device=self.local_rank,
                physical_gpu=physical_gpu,
            )
            result = super().compile_or_warm_up_model()
            if getattr(self, "prefix_state", None) is not None:
                self.prefix_state.start()
            return result
        except Exception:
            monitor.free()
            checker.free()
            raise

    def execute_model(self, scheduler_output):
        self._get_hbm_ecc_monitor().check()
        if scheduler_output.total_num_scheduled_tokens > 0:
            self._get_onchip_checker().before_batch()
        context = getattr(self, "prefix_state", None)
        if context is not None:
            context.observe_scheduler_output(scheduler_output)
        try:
            state = getattr(self, "_compressed_kv_state", None)
            if state is not None:
                # Lease the shared arena for eager work and graph replay alike.
                with state.workspace.acquire():
                    return super().execute_model(scheduler_output)
            return super().execute_model(scheduler_output)
        finally:
            if context is not None:
                context.end_step()

    def shutdown(self) -> None:
        if getattr(self, "prefix_state", None) is not None:
            self.prefix_state.shutdown()
        orig_cls = self._try_replace_model_runner_class()
        checker = getattr(self, "_onchip_checker", None)
        monitor = getattr(self, "_hbm_ecc_monitor", None)
        try:
            if monitor is not None:
                try:
                    monitor.check()
                except Exception:
                    logger.exception("HBM ECC monitor shutdown check failed")
            if checker is not None:
                try:
                    checker.check_shutdown()
                except Exception:
                    logger.exception("Worker health shutdown check failed")
            super().shutdown()
        finally:
            state = getattr(self, "_compressed_kv_state", None)
            if state is not None:
                state.release()
                self._compressed_kv_state = None
            if checker is not None:
                try:
                    checker.free()
                except Exception:
                    logger.exception("Failed to free worker health checker")
            if monitor is not None:
                try:
                    monitor.free()
                except Exception:
                    logger.exception("Failed to free HBM ECC monitor")
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
