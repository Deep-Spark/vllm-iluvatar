# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker with two overrides:

1. ``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.
2. ``init_device`` for KVINT8 attention quantisation support.
"""

from __future__ import annotations

import vllm.v1.worker.gpu_model_runner as _gpu_model_runner
from vllm.v1.worker.gpu_worker import Worker

from vllm_iluvatar.v1.attention.static_quant_utils import is_kvint8_enabled
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)


class IluWorker(Worker):

    def init_device(self) -> None:
        import os, sys
        import vllm.envs as envs
        raw_env = os.environ.get("VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "<unset>")
        envs_val = getattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "<missing>")
        enabled = is_kvint8_enabled()
        print(
            f"[KVINT8 check] os.environ={raw_env!r}, envs.attr={envs_val!r}, "
            f"is_kvint8_enabled={enabled}",
            file=sys.stderr, flush=True,
        )
        if enabled:
            from vllm_iluvatar.v1.worker.gpu_model_runner import (
                IluvatarGPUModelRunner,
            )

            orig_cls = _gpu_model_runner.GPUModelRunner
            _gpu_model_runner.GPUModelRunner = IluvatarGPUModelRunner  # type: ignore[misc]
            print(
                f"[KVINT8] replaced GPUModelRunner -> IluvatarGPUModelRunner, "
                f"use_v2={getattr(self, 'use_v2_model_runner', '?')}",
                file=sys.stderr, flush=True,
            )
            try:
                super().init_device()
            finally:
                _gpu_model_runner.GPUModelRunner = orig_cls
            print(
                f"[KVINT8] model_runner class = {type(self.model_runner).__name__}",
                file=sys.stderr, flush=True,
            )
        else:
            super().init_device()

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )
