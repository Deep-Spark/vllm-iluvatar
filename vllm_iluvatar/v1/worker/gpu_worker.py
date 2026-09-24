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
        orig_cls = self._try_replace_model_runner_class()
        try:
            return super().determine_available_memory()
        finally:
            self.model_runner.__class__ = orig_cls

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        orig_cls = self._try_replace_model_runner_class()
        try:
            super().initialize_from_config(kv_cache_config)
        finally:
            self.model_runner.__class__ = orig_cls

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
            return super().compile_or_warm_up_model()
        except Exception:
            monitor.free()
            checker.free()
            raise

    def execute_model(self, scheduler_output):
        self._get_hbm_ecc_monitor().check()
        if scheduler_output.total_num_scheduled_tokens > 0:
            self._get_onchip_checker().before_batch()
        return super().execute_model(scheduler_output)

    def shutdown(self) -> None:
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
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )
