# SPDX-License-Identifier: Apache-2.0
"""MRV2 PCP validation for the Iluvatar eager PP integration."""

from copy import copy
from functools import wraps

from vllm.config import CUDAGraphMode


def make_validate_config(original):
    @wraps(original)
    def validate_config(vllm_config, supports_mm_inputs):
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size <= 1
            or parallel.pipeline_parallel_size <= 1
        ):
            return original(vllm_config, supports_mm_inputs)

        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise NotImplementedError(
                "Iluvatar PCP + PP currently requires eager mode."
            )

        # Only bypass upstream's blanket PP rejection. Keep all its other PCP
        # checks (MLA, multimodal, LoRA, speculative decoding, etc.). Never
        # mutate the live topology: model construction and groups still use PP.
        validation_config = copy(vllm_config)
        validation_config.parallel_config = copy(parallel)
        validation_config.parallel_config.pipeline_parallel_size = 1
        return original(validation_config, supports_mm_inputs)

    return validate_config
