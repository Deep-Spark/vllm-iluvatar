# SPDX-License-Identifier: Apache-2.0
"""
signature — Shared utilities for Iluvatar patch modules.

Provides general-purpose helpers that any patch_*.py file can import.
No side effects on import; no dependency on vllm internals.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SignatureMismatchError(RuntimeError):
    """Raised when an ILU override is incompatible with the community signature."""


def validate_patch_signatures(
    community_module: object,
    ilu_module: object,
    func_names: list[str],
    strict: bool = False,
) -> None:
    """Check that ILU override functions are backward-compatible with community.

    For each name in func_names that exists in both modules, verifies:
      1. All community parameter names exist in the ILU function.
      2. Extra ILU-only parameters must have defaults (so community callers
         that omit them still work).

    Call this inside a patch function before doing setattr / sys.modules swap,
    so mismatches surface at startup rather than at inference time.

    Args:
        community_module: The community module being patched (e.g. vllm._custom_ops).
        ilu_module:        The ILU replacement module.
        func_names:        Function names to validate (typically _ILU_OVERRIDES).
        strict:            If True, raise SignatureMismatchError on mismatch.
                           If False (default), log a WARNING and continue.

    Raises:
        SignatureMismatchError: When strict=True and a mismatch is found.

    Example::

        from vllm_iluvatar.ilu_util.signature import validate_patch_signatures
        import vllm._custom_ops as _co
        import vllm_iluvatar._custom_ops as _ilu

        validate_patch_signatures(_co, _ilu, _ilu._ILU_OVERRIDES, strict=True)
    """
    errors: list[str] = []

    for name in func_names:
        comm_fn: Callable | None = getattr(community_module, name, None)
        ilu_fn: Callable | None = getattr(ilu_module, name, None)

        if comm_fn is None or ilu_fn is None:
            # ILU-only addition or community doesn't have it — nothing to check.
            continue

        try:
            comm_sig = inspect.signature(comm_fn)
            ilu_sig = inspect.signature(ilu_fn)
        except (ValueError, TypeError):
            # Built-in or C extension — skip.
            continue

        comm_params = dict(comm_sig.parameters)
        ilu_params = dict(ilu_sig.parameters)

        param_errors: list[str] = []

        # Every community param must appear in ILU.
        for param_name in comm_params:
            if param_name not in ilu_params:
                param_errors.append(
                    f"  community param '{param_name}' missing in ILU version"
                )

        # Extra ILU params must have defaults (community callers can't pass them).
        for param_name, ilu_param in ilu_params.items():
            if (
                param_name not in comm_params
                and ilu_param.default is inspect.Parameter.empty
            ):
                param_errors.append(
                    f"  ILU-only param '{param_name}' has no default — "
                    f"community callers cannot omit it"
                )

        if param_errors:
            errors.append(f"'{name}':\n" + "\n".join(param_errors))

    if errors:
        msg = (
            "ILU patch signature validation failed.\n"
            "The following ILU override functions are not backward-compatible:\n\n"
            + "\n\n".join(errors)
            + "\n\nFix: update the ILU implementation so it accepts all "
            "community parameter names, with ILU-specific extras having defaults."
        )
        if strict:
            raise SignatureMismatchError(msg)
        else:
            logger.warning(msg)
