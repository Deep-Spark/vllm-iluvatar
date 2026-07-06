# SPDX-License-Identifier: Apache-2.0
"""
Patch layer: splice ILU custom ops into vllm._custom_ops.

Corresponds to: vllm/_custom_ops.py
Registered via the post-import hook (vllm_iluvatar.ilu_patch.import_hook.register)
keyed on "vllm._custom_ops".  The splat therefore runs at the EXACT moment the
community module finishes importing — early enough for module-level
(import-time) uses such as ``w8a8_utils`` calling
``cutlass_scaled_mm_supports_fp8`` inside a spawned worker subprocess, where the
general_plugins phase would run too late.  Registration happens at startup
(see patch_import_compat.py); only this lightweight module is imported there —
_custom_ops (and thus ixformer) is pulled in lazily inside the callback, so
startup stays cheap and does not force platform resolution.

Strategy: splat — community vllm._custom_ops has already executed (all community
functions available) when the callback fires, then:
  - overwrite community functions listed in _ILU_OVERRIDES, and
  - add the ILU-only functions listed in _ILU_ADDED,
both via setattr.  Never replace the entire module.  Both groups are driven by
explicit manifests in ops/custom_ops.py (no implicit module scan, so imported
helpers such as init_logger / ScalarType are never accidentally splatted).
"""

import logging

from vllm_iluvatar.ilu_patch.import_hook import register
from vllm_iluvatar.ilu_util.signature import validate_patch_signatures

logger = logging.getLogger(__name__)

# Interfaces whose ILU signature intentionally DIVERGES from community and so
# must be excluded from strict signature validation (validation compares
# against community, but in this build the callers of these functions are still
# the fork's own modules, which call with the fork convention):
#   - awq_gemm: fork omits community's `split_k_iters` and adds a no-default
#     `pack_factor`.
# TODO(iluvatar): once these functions' callers are migrated to community
# baseline, switch the ILU impls to the community signature and remove them
# from this set so they are strict-validated again.
_SKIP_SIGCHECK: set[str] = {"awq_gemm"}


def _patch_custom_ops(module=None) -> None:
    """Splat Iluvatar custom ops onto community vllm._custom_ops.

    ``module`` is the freshly loaded ``vllm._custom_ops`` passed by the
    post-import hook; falls back to importing it (already cached) when called
    without an argument.
    """
    import vllm_iluvatar._custom_ops as _ilu_co

    if module is not None:
        _co = module
    else:
        import vllm._custom_ops as _co

    # Strict-validate every override EXCEPT the known signature divergences
    # tracked in _SKIP_SIGCHECK (see note above). A community upgrade that
    # changes any other signature will then fail fast at startup.
    checked = [n for n in _ilu_co._ILU_OVERRIDES if n not in _SKIP_SIGCHECK]
    validate_patch_signatures(_co, _ilu_co, checked, strict=True)

    # 1. Overwrite community functions with ILU implementations.
    overridden = 0
    for name in _ilu_co._ILU_OVERRIDES:
        fn = getattr(_ilu_co, name, None)
        if fn is not None:
            setattr(_co, name, fn)
            overridden += 1

    # 2. Add ILU-only functions that community _custom_ops.py does not have.
    added = 0
    for name in _ilu_co._ILU_ADDED:
        fn = getattr(_ilu_co, name, None)
        if fn is not None:
            setattr(_co, name, fn)
            added += 1

    logger.debug(
        "ILU custom_ops: overrode %d functions, added %d new functions",
        overridden,
        added,
    )


# Fire the splat the moment community vllm._custom_ops finishes importing.
register("vllm._custom_ops", _patch_custom_ops)
