# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker with following overrides:

1. ``determine_available_memory`` for attention quantisation support.
2. ``initialize_from_config`` for attention quantisation support.
3. ``shutdown`` for attention quantisation support.
4. ``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.
5. ``compile_or_warm_up_model`` to exclude synthetic GLM KV calibration.
"""

from __future__ import annotations

from contextlib import contextmanager

import vllm.v1.worker.gpu.model_runner as _gpu_model_runner
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _pcp_profile_local_tokens(num_tokens: int, pcp_size: int, max_num_reqs: int) -> int:
    # DualChunkSwap gives each prefill rank at most n / PCP + 2 tokens
    # per request (two rounded chunks). Single-token decodes are replicated.
    # This upper bound covers mixed batches and the maximum rank's padding.
    return min(num_tokens, (num_tokens + pcp_size - 1) // pcp_size + 2 * max_num_reqs)


class IluWorker(Worker):
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
        orig_cls = self._try_replace_model_runner_class()
        try:
            with self._pcp_memory_profile():
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

    def compile_or_warm_up_model(self):
        if self.parallel_config.prefill_context_parallel_size <= 1:
            return super().compile_or_warm_up_model()

        from vllm_iluvatar.models.glm_moe_dsa.attention import IluGlmMoeDsaSparseMLA

        # V2 warmup includes simulated requests with valid KV slots. Keep those
        # projections out of online calibration; the first real request must
        # select the scale. Restore flags even when startup fails.
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

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )
