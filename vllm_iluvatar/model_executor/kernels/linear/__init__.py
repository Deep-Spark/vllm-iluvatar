# SPDX-License-Identifier: Apache-2.0
"""Iluvatar mixed-precision linear kernel selector."""

from __future__ import annotations

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear import (
    choose_mp_linear_kernel as _community_choose_mp_linear_kernel,
)
from vllm.platforms import current_platform

from vllm_iluvatar.model_executor.kernels.linear.mixed_precision.ilu_gptq_marlin import (
    IluGPTQMarlinLinearKernel,
)


def _get_linear_backend() -> str:
    ilu_backend = getattr(envs, "VLLM_ILU_LINEAR_BACKEND", "").strip()
    if ilu_backend:
        return ilu_backend

    config = get_current_vllm_config_or_none()
    if config is not None:
        return config.kernel_config.linear_backend
    return "auto"


def choose_mp_linear_kernel(
    config: MPLinearLayerConfig,
    compute_capability: int | None = None,
) -> type[MPLinearKernel]:
    """Choose a mixed-precision linear kernel with Iluvatar priority.

    Community vLLM has no plugin registration point for MPLinearKernel classes.
    The Iluvatar quant methods call this selector directly so community vLLM can
    keep its upstream kernel list. ``IluGPTQMarlinLinearKernel`` is selected only
    when ``VLLM_ILU_LINEAR_BACKEND=gptq_marlin`` is set.
    """

    linear_backend = _get_linear_backend()
    if linear_backend == "gptq_marlin":
        can_implement, _reason = IluGPTQMarlinLinearKernel.can_implement(config)
        if can_implement:
            return IluGPTQMarlinLinearKernel
        raise ValueError(
            "VLLM_ILU_LINEAR_BACKEND=gptq_marlin was requested but "
            "IluGPTQMarlinLinearKernel cannot implement this layer."
        )

    return _community_choose_mp_linear_kernel(config, compute_capability)


__all__ = [
    "IluGPTQMarlinLinearKernel",
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "choose_mp_linear_kernel",
]
