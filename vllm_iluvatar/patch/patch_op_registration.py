"""Make ``direct_register_custom_op`` idempotent on the live community module.

Root cause: the import-time override finder evicts already-imported community
modules so the next import yields our overlay copy. For modules that register
torch custom ops at import time (e.g. ``vllm.distributed.parallel_state``
registers ``vllm::all_reduce``), the community copy already registered the op
*before* the plugin loaded, so re-executing the overlay copy raises::

    RuntimeError: Tried to register an operator (vllm::all_reduce ...) with the
    same name and overload name multiple times.

The overlay copy of ``torch_utils.py`` already guards this, but we also guard
the *live community* ``direct_register_custom_op`` here (run first, before any
redirect / re-import) as a belt-and-suspenders measure for any code path that
still holds the community reference.
"""

from __future__ import annotations

_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import torch
        import vllm.utils.torch_utils as tu
        from vllm.logger import init_logger
    except Exception:
        return

    logger = init_logger(__name__)

    orig = getattr(tu, "direct_register_custom_op", None)
    if orig is None or getattr(orig, "_ilu_idempotent", False):
        return

    def direct_register_custom_op(
        op_name,
        op_func,
        mutates_args=None,
        fake_impl=None,
        target_lib=None,
        dispatch_key=None,
        tags=(),
    ):
        # Skip if the op is already registered (community module ran first).
        try:
            my_lib = target_lib or getattr(tu, "vllm_lib", None)
            ns = getattr(my_lib, "ns", "vllm") if my_lib is not None else "vllm"
            if hasattr(getattr(torch.ops, ns, None), op_name):
                return
        except Exception:
            pass
        try:
            return orig(
                op_name,
                op_func,
                mutates_args=mutates_args,
                fake_impl=fake_impl,
                target_lib=target_lib,
                dispatch_key=dispatch_key,
                tags=tags,
            )
        except RuntimeError as e:
            if "multiple times" in str(e) or "already" in str(e).lower():
                return
            raise

    direct_register_custom_op._ilu_idempotent = True  # type: ignore[attr-defined]
    tu.direct_register_custom_op = direct_register_custom_op
    logger.info("vllm-iluvatar: made direct_register_custom_op idempotent")
