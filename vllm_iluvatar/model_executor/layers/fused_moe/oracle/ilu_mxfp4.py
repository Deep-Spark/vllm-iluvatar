# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP4 MoE backend-selection override (patch P1).

Corresponds to: vllm/model_executor/layers/fused_moe/oracle/mxfp4.py ::
``select_deepseek_v4_mxfp4_moe_backend`` (``oracle/mxfp4.py:566``)

Target   : vllm.model_executor.layers.fused_moe.oracle.mxfp4.select_deepseek_v4_mxfp4_moe_backend
Baseline : main_9c12a81ac
Why      : the community function can never return ``Mxfp4MoeBackend.NONE``
           off-ROCm — ``_get_priority_backends()`` yields TRTLLM / DEEPGEMM /
           MARLIN / BATCHED_MARLIN and, when none of them supports the config,
           the function ends in ``raise NotImplementedError("No MXFP4 MoE
           backend supports the deployment configuration.")``. Stage 1 of the
           DeepSeek-V4 FP8 weight-loading work needs ``NONE`` precisely because
           ``Mxfp4MoEMethod.process_weights_after_loading``
           (``quantization/mxfp4.py:766``) early-returns on it, which leaves the
           four routed MXFP4 params byte-identical to the checkpoint (no kernel
           format conversion, no ``del layer.w13_weight_scale``).
How      : delegation wrapper built by ``make_select_deepseek_v4_mxfp4_moe_backend``
           and installed by ``ilu_patch.general_patches.patch_dsv4_mxfp4_backend``
           — on Iluvatar return ``(Mxfp4MoeBackend.NONE, None)`` without calling
           the community function, otherwise delegate untouched. No community
           method body is copied, so no source-fingerprint guard is needed
           (patch登记表 §2, tier 2).
Scope    : DeepSeek-V4 FP8 checkpoint loading on Iluvatar (every
           ``Mxfp4MoEMethod.__init__``, i.e. every routed-MXFP4 FusedMoE layer).
Remove   : when an Iluvatar MXFP4 MoE kernel lands and a real backend can be
           selected (Stage 2+).

Platform gate: ``current_platform.device_name == "iluvatar"``. Deliberately not
``isinstance(current_platform, IluPlatform)`` (importing
``platforms/ilu_platform.py`` has import-time side effects: it stubs ``vllm._C``,
builds the platform class and touches ``torch.backends``) and not
``current_platform.is_cuda()`` (``IluPlatform`` subclasses ``CudaPlatform``, so
that predicate is always true and would hijack real CUDA runs).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

# ``IluPlatform.device_name`` (vllm_iluvatar/platforms/ilu_platform.py:115).
_ILU_DEVICE_NAME = "iluvatar"


def _is_iluvatar_platform() -> bool:
    """True when running on the Iluvatar platform.

    Kept as a module-level zero-arg predicate so the wrapper can resolve it as a
    module global at call time (monkeypatchable in tests, mirrors
    ``distributed/parallel_state.py::_is_ivcore30_fn``). ``current_platform`` is
    imported lazily: ``vllm.platforms.__getattr__`` resolves the platform on
    attribute access, and that must not happen at plugin import time.
    """
    from vllm.platforms import current_platform

    return getattr(current_platform, "device_name", None) == _ILU_DEVICE_NAME


def make_select_deepseek_v4_mxfp4_moe_backend(
    orig: Callable[[FusedMoEConfig], tuple[Any, Any]],
) -> Callable[[FusedMoEConfig], tuple[Any, Any]]:
    """Wrap the community ``select_deepseek_v4_mxfp4_moe_backend``.

    Mirrors ``vllm_iluvatar/config/speculative.py::make_hf_config_override``:
    the factory takes the community callable and returns a
    ``functools.wraps``-decorated wrapper, so name / signature / ``__wrapped__``
    still point at the community function after the rebind.
    """

    @functools.wraps(orig)
    def select_deepseek_v4_mxfp4_moe_backend(
        config: FusedMoEConfig,
    ) -> tuple[Any, Any]:
        # [ILU override] select_deepseek_v4_mxfp4_moe_backend — delta: force the
        # NONE backend on Iluvatar so MXFP4 routed weights stay byte-identical
        # to the checkpoint; every other platform keeps community behaviour.
        if _is_iluvatar_platform():
            from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
                Mxfp4MoeBackend,
            )

            return Mxfp4MoeBackend.NONE, None
        return orig(config)

    return select_deepseek_v4_mxfp4_moe_backend
