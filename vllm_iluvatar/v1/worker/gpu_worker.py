# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker with following overrides:

1. ``determine_available_memory`` for attention quantisation support.
2. ``initialize_from_config`` for attention quantisation support.
3. ``shutdown`` for attention quantisation support.
4. ``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.
"""

from __future__ import annotations

import vllm.v1.worker.gpu_model_runner as _gpu_model_runner
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


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

    def shutdown(self) -> None:
        orig_cls = self._try_replace_model_runner_class()
        try:
            super().shutdown()
        finally:
            self.model_runner.__class__ = orig_cls

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )
