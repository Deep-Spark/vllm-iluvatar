# SPDX-License-Identifier: Apache-2.0
"""Honor the draft eager override in the V2 DFlash/DSpark runner."""

from functools import wraps

from vllm.config.compilation import CUDAGraphMode


def make_init_cudagraph_manager(original):
    @wraps(original)
    def init_cudagraph_manager(self, cudagraph_mode):
        if self.vllm_config.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        return original(self, cudagraph_mode)

    return init_cudagraph_manager
