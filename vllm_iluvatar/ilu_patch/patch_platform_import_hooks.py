# SPDX-License-Identifier: Apache-2.0
"""Platform-stage post-import hooks.

ILU patch wiring registered from ``vllm_iluvatar.register_platform`` after
Iluvatar/CoreX has been confirmed, but before the target vLLM modules are first
used. The target modules are still imported by vLLM naturally; each callback runs
immediately after its module finishes importing.

Use this file for platform-scoped hooks that cannot be handled by
``general_plugins`` but must not eagerly import their target while vLLM is still
resolving ``current_platform``.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_iluvatar.ilu_patch import import_hook

_HookCallback = Callable[[object], None]

_registered = False


def _patch_distributed_utils(module: object) -> None:
    """Patch create_tcp_store once ``vllm.distributed.utils`` loads.

    Target : vllm.distributed.utils.create_tcp_store
    Why    : Elastic EP creates a TCPStore from a pre-bound listen fd; CoreX
             requires ``use_libuv=False`` for that path.
    How    : wrap the community function after module import, avoiding eager
             distributed imports while platform detection is still running.
    Remove : when upstream exposes a platform hook for TCPStore creation or no
             longer requires libuv override on CoreX.
    """
    from vllm_iluvatar.distributed.utils import _make_create_tcp_store

    module.create_tcp_store = _make_create_tcp_store(module.create_tcp_store)


def _patch_pixtral_use_xformers_ops(module: object) -> None:
    """Force Pixtral vision code to build xformers block-diagonal metadata.

    Target : vllm.model_executor.models.pixtral.USE_XFORMERS_OPS
    Why    : Iluvatar swaps Pixtral vision attention to ixformer unpadded flash
             attention, but still needs the community xformers
             BlockDiagonalMask sequence metadata (q_seqinfo/k_seqinfo).
    How    : patch the community module-level flag after import so its
             VisionTransformer.forward creates BlockDiagonalMask instead of a
             dense Tensor attention mask.
    Remove : when the Iluvatar Pixtral attention path no longer depends on
             xformers BlockDiagonalMask metadata.
    """
    module.USE_XFORMERS_OPS = True


_HOOKS: tuple[tuple[str, _HookCallback], ...] = (
    # Keep this manifest as the only edit point for platform-stage hooks.
    ("vllm.distributed.utils", _patch_distributed_utils),
    ("vllm.model_executor.models.pixtral", _patch_pixtral_use_xformers_ops),
)


def register_platform_import_hooks() -> None:
    """Register all platform-scoped post-import hooks exactly once."""
    global _registered
    if _registered:
        return
    _registered = True

    for target, callback in _HOOKS:
        import_hook.register(target, callback)
