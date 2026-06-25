# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar early compat shims.

These are corex/torch-environment compat shims (NOT vLLM logic) that must run
before any deep ``import vllm`` touches them.  They are applied via two paths:

  1. Main process: ``vllm_iluvatar.register_platform()`` calls ``apply()``
     directly (at ``current_platform`` first access, before parallel_state is
     imported) and sets ``VLLM_ILUVATAR_ENABLE_STARTUP_PATCH=1`` so children
     inherit it.
  2. Spawned worker subprocess: the installed ``vllm_iluvatar.pth`` runs at
     interpreter startup; it is inert unless the env above is set, so it only
     fires for vLLM worker processes (never for unrelated ``python`` calls).

Shims (no-ops on stock CUDA torch):
  1. torch.accelerator — corex torch has a partial/missing module; map the
     subset vLLM uses onto torch.cuda equivalents.
  2. symmetric-memory guard — corex torch ships an INCOMPLETE
     ``torch.distributed._symmetric_memory`` (importable, but missing
     ``empty``).  Community ``device_communicators/symm_mem.py`` only guards
     ``ImportError`` (not the missing attribute), so it flips
     ``symm_mem_available=True`` and later crashes at ``torch_symm_mem.empty``
     during GroupCoordinator init.  We pre-inject a stub and force-disable
     symmetric-memory allreduce via ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``.

Then installs module *stand-in* shims (``_install_import_shims`` ->
``import_shims.install_all``): whole community modules that would otherwise be
imported at *import time* and fail on corex are replaced in ``sys.modules`` with
ixformer-backed equivalents (e.g. ``vllm.vllm_flash_attn``, imported by the
community fa_utils ``is_cuda()`` branch since IluPlatform is cuda-like).  Each
stand-in mirrors the upstream module name under ``import_shims/``.

Finally registers post-import splat hooks (``_register_import_hooks`` ->
``patch_custom_ops``): ILU overrides for community modules whose functions
are called at *import time* (e.g. ``vllm._custom_ops`` used by ``w8a8_utils``
while it is imported in a spawned worker).  Registration is cheap — the target
modules are not imported here.

Env injection (``ilu_envs.apply()``) is intentionally NOT done here; it needs
``vllm.envs`` and is handled by ``register_platform()`` instead.
"""

import os
import types

_applied = False


def _patch_torch_accelerator() -> None:
    import torch

    if not hasattr(torch, "accelerator"):
        torch.accelerator = types.ModuleType("torch.accelerator")
    accel = torch.accelerator
    patch_map = {
        "empty_cache": torch.cuda.empty_cache,
        "device_count": torch.cuda.device_count,
        "set_device_index": torch.cuda.set_device,
        "current_device_index": torch.cuda.current_device,
        "memory_stats": torch.cuda.memory_stats,
        "memory_reserved": torch.cuda.memory_reserved,
        "max_memory_allocated": torch.cuda.max_memory_allocated,
        "reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
        "synchronize": torch.cuda.synchronize,
        "device_index": torch.cuda.device,
    }
    for name, func in patch_map.items():
        if not hasattr(accel, name):
            setattr(accel, name, func)


def _stub_symmetric_memory() -> None:
    """Pre-inject a stub for ``torch.distributed._symmetric_memory``.

    On corex torch this module is either missing or incomplete (importable
    but lacks ``empty``).  Community ``parallel_state.py`` has a bare
    ``import torch.distributed._symmetric_memory`` at module scope that would
    fail with ``ImportError`` on corex.  By inserting a lightweight stub into
    ``sys.modules`` early, the import succeeds harmlessly and
    ``_disable_unsupported_symmetric_memory`` (below) ensures the actual usage
    is gated off via ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``.
    """
    import sys

    key = "torch.distributed._symmetric_memory"
    if key not in sys.modules:
        sys.modules[key] = types.ModuleType(key)


def _disable_unsupported_symmetric_memory() -> None:
    try:
        import torch.distributed._symmetric_memory as torch_symm_mem
    except Exception:
        torch_symm_mem = None
    if torch_symm_mem is None or not hasattr(torch_symm_mem, "empty"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"


def _install_import_shims() -> None:
    # Inject module *stand-in* shims into sys.modules BEFORE any deep
    # ``import vllm``.  Unlike the attribute-level shims above, these replace a
    # whole community module that would otherwise be imported at import-time and
    # fail / pull NV-specific code on corex (e.g. vllm.vllm_flash_attn, imported
    # by the community fa_utils ``is_cuda()`` branch — IluPlatform is cuda-like,
    # so that branch runs).  Each stand-in lives in import_shims/ mirroring the
    # upstream module name; install_all() auto-discovers and injects them.
    from vllm_iluvatar.ilu_patch.import_shims import install_all

    install_all()


def _register_import_hooks() -> None:
    # Register post-import splat hooks (e.g. vllm._custom_ops) so ILU overrides
    # apply the moment the target community module finishes importing — early
    # enough for import-time uses in spawned subprocesses, where general_plugins
    # would run too late.  This only REGISTERS the hooks; the target modules are
    # NOT imported here, so no heavy deps / platform resolution / ixformer are
    # forced.  Each patch_<target>.py calls import_hook.register(...) at import.
    import vllm_iluvatar.ilu_patch.patch_custom_ops  # noqa: F401


def _load_native_torch_ops() -> None:
    # Import the native ``vllm_iluvatar._C`` extension so its TORCH_LIBRARY static
    # registrations populate ``torch.ops._C.{rms_norm, weak_ref_tensor,
    # get_cuda_view_from_cpu_tensor}`` BEFORE unmodified community code
    # (vllm/kernels/vllm_c.py, vllm/utils/torch_utils.py) first calls them.
    # Non-fatal in non-strict mode: a missing build degrades gracefully (the
    # community call would then raise with a clear torch error), and dev hosts
    # without the compiled extension still import the plugin.
    from vllm_iluvatar.ilu_util.torch_ops_c import load_native_ops

    load_native_ops(strict=False)


def apply() -> None:
    """Apply all early compat shims.  Idempotent and defensive."""
    global _applied
    if _applied:
        return
    _applied = True
    try:
        _patch_torch_accelerator()
        _stub_symmetric_memory()
        _disable_unsupported_symmetric_memory()
        _install_import_shims()
        _register_import_hooks()
        _load_native_torch_ops()
    except Exception:
        # Compat shims must never break interpreter / vLLM startup.
        pass


# Apply on import. This module is loaded exactly once via
# ``vllm_iluvatar._load_import_compat_patch()`` (main process at
# register_platform time, and spawned workers via the installed .pth), so the
# early shims + import-hook registration run before any deep ``import vllm``.
apply()
