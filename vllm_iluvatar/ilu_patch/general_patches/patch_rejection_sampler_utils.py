# SPDX-License-Identifier: Apache-2.0
"""Wiring for the GPU spec-decode rejection-kernel ILU override.

Corresponds to: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

Target   : vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils._rejection_kernel
Baseline : v0.30.0 (ced6857afa)
Why      : two Iluvatar Triton behaviours the community kernel does not expect.
           ``tl_rand32`` returns a ``[1]``-shaped tensor rather than a 0-d
           scalar, which promotes the loop-carried ``verifying`` /
           ``accepted_length`` to an inconsistent type; and the tail fixup
           ``if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length <
           ...`` is a constexpr AND runtime conjunction that this build lowers
           to a runtime branch instead of folding, so it compiles the dead body
           and hits ``tl.load`` on the None ``cumulative_log_p`` pointer during
           spec-decode warmup (``rejection_sample_method='standard'``, which is
           what DSpark / MTP use).
How      : rebind ``_rejection_kernel`` to the ilu mirror
           (``vllm_iluvatar.v1.worker.gpu.spec_decode.rejection_sampler_utils``),
           a verbatim copy that reduces ``u`` with ``tl.sum`` and re-nests the
           tail guard so the constexpr ``if USE_BLOCK_VERIFICATION:`` stands
           alone and is folded. The community ``rejection_sample`` launches the
           kernel through its module globals, so the rebind reaches it.
Scope    : Iluvatar GPU spec-decode verification (standard / block methods).
Remove   : when the Iluvatar Triton build folds ``constexpr and runtime`` guards
           and ``tl.rand`` returns a 0-d scalar.
Fingerprint: the community kernel source is SHA-checked at register-time; gated
           by VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)

_KERNEL_NAME = "_rejection_kernel"
_BASELINE_SOURCE_SHA = (
    "6a49a064953cc91b844e146f12e0294a85b2aa1491e42a3f5cac2e5265ed2a1c"
)


def register() -> None:
    """Register the spec-decode rejection-kernel override."""
    p.register_once(
        __name__,
        _apply,
        detail="spec decode _rejection_kernel: scalar rand + nested constexpr guard",
    )


def _apply() -> None:
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as _m

    import vllm_iluvatar.v1.worker.gpu.spec_decode.rejection_sampler_utils as impl

    if p._enabled(p.ENV_CHECK_FINGERPRINT):
        current_sha = p.source_sha(inspect.getsource(_m._rejection_kernel.fn))
        if current_sha != _BASELINE_SOURCE_SHA:
            raise RuntimeError(
                "ILU patch: community "
                "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils."
                f"{_KERNEL_NAME} source changed vs baseline v0.30.0 (ced6857afa). "
                f"Expected {_BASELINE_SOURCE_SHA}, got {current_sha}. Re-sync "
                "vllm_iluvatar/v1/worker/gpu/spec_decode/rejection_sampler_utils.py "
                "and patch_rejection_sampler_utils.py."
            )

    p.rebind_module_func(_m, _KERNEL_NAME, impl._rejection_kernel)
