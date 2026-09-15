# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar vLLM plugin package (plugin-based, zero community-source edits).

Registered with vLLM via two entry points (pyproject.toml):

  vllm.platform_plugins -> vllm_iluvatar:register_platform
  vllm.general_plugins  -> vllm_iluvatar:register_general
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from vllm_iluvatar.version import __git_id__, __version__

# Env flag that gates the installed startup-patch .pth hook. The .pth is inert
# unless this is "1"; ``register_platform`` sets it so spawned worker
# subprocesses inherit it and apply the early shims before unpickling.
STARTUP_PATCH_ENV = "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH"

__all__ = [
    "__version__",
    "__git_id__",
    "register_platform",
    "register_general",
    "STARTUP_PATCH_ENV",
]


def _restore_vllm_log_stream(stream) -> None:
    """Point vllm's root-logger StreamHandler(s) at ``stream``.

    The compat patch may run before vllm configures its logger; vllm's default
    log stream is stdout. We temporarily route stdout to stderr while the patch
    runs (see ``_load_import_compat_patch``), which makes vllm capture stderr as
    its handler stream. After the patch we restore the handler to the real stdout
    so normal (non-helper) processes keep vllm's default stream.
    """
    try:
        import logging

        for h in logging.getLogger("vllm").handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = stream
    except Exception:
        pass


def _load_import_compat_patch() -> None:
    """Apply early corex/torch compat shims + import-hook registration (once).

    Loads ``ilu_patch/patch_import_compat.py`` directly via a file spec (instead
    of a normal ``import vllm_iluvatar.ilu_patch.patch_import_compat``) so this
    can run at interpreter-startup time — from the installed .pth in spawned
    workers and from ``register_platform`` in the main process — without
    importing the whole ``vllm_iluvatar.ilu_patch`` package eagerly.  Idempotent
    via ``sys.modules``.

    stdout is redirected to stderr while the patch runs. The patch imports vllm,
    which triggers platform-plugin activation logs; vllm's default log stream is
    stdout, so in a helper subprocess that talks to its parent via stdout (e.g.
    py-cpuinfo's ``cpuinfo.py --json`` spawned by ``cpuinfo.get_cpu_info()`` in
    vllm's usage-reporting thread) those logs would corrupt the JSON pipe and
    raise ``JSONDecodeError``. Routing them to stderr during the patch keeps the
    pipe clean; vllm's handler stream is restored to the real stdout afterwards.
    """
    module_name = "vllm_iluvatar._early_import_compat"
    if module_name in sys.modules:
        return

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        patch_path = (
            Path(__file__).resolve().parent / "ilu_patch" / "patch_import_compat.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, patch_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.stdout = real_stdout
        _restore_vllm_log_stream(real_stdout)

    # Wire the vllm_iluvatar.* logger tree to vLLM's handler/level so that ILU
    # modules using bare ``logging.getLogger(__name__)`` still emit through vLLM's
    # configured stream/formatter. Runs once per process after vllm.logger is up.
    try:
        from vllm_iluvatar.ilu_logger import configure_ilu_logging

        configure_ilu_logging()
    except Exception:
        pass


def register_platform() -> str:
    """vllm.platform_plugins entry point for Iluvatar hardware.

    Called during ``current_platform`` first access.  This build targets
    Iluvatar corex + ixformer, so a missing ixformer is a hard error rather
    than a silent fall-through to the builtin CUDA platform.

    Enables the startup hook for spawned workers, applies the early compat
    shims now (current_platform is resolved before parallel_state import),
    injects the Iluvatar env config, then returns the IluPlatform class path.
    """
    try:
        import ixformer  # noqa: F401 — presence means Iluvatar hardware
    except ImportError as e:
        raise RuntimeError(
            "Iluvatar platform plugin is enabled but `ixformer` is not "
            "importable. This vLLM build requires Iluvatar corex + ixformer. "
            "Install ixformer or disable the iluvatar plugin."
        ) from e

    os.environ.setdefault(STARTUP_PATCH_ENV, "1")
    _load_import_compat_patch()

    # Inject Iluvatar env defaults + canonical/legacy readers (needs vllm.envs,
    # so it is done here rather than at .pth-time startup).
    from vllm_iluvatar.envs import apply as _apply_ilu_envs

    _apply_ilu_envs()

    # Backend registration reads Iluvatar env gates, so env readers must be
    # installed before overriding any official community backend enum member.
    from vllm_iluvatar.platforms.ilu_platform import register_attention_backends

    register_attention_backends()

    from vllm_iluvatar.ilu_patch.patch_platform_import_hooks import (
        register_platform_import_hooks,
    )

    register_platform_import_hooks()

    return "vllm_iluvatar.platforms.ilu_platform.IluPlatform"


def register_general() -> None:
    """vllm.general_plugins entry point for Iluvatar hardware.

    Called by load_general_plugins() after all modules are initialised.  Applies
    late-phase patches that need a fully initialised vLLM (the patch package's
    import side effects).  ixformer presence is already enforced by
    register_platform(); the guard here is purely defensive.

    NOTE: the custom_ops override is applied earlier via the import-hook
    (register_platform -> _load_import_compat_patch), NOT here. This entry
    applies the late-phase (general_plugins) work that only needs an initialised
    vLLM, in order:

      1. apply_general_patches()    — runtime monkey-patches auto-discovered from
                                      the mirror tree (vllm_iluvatar.ilu_patch).
      2. apply_oot_registrations()  — official OOT extension points
                                      (CustomOp.register_oot / register_model)
                                      auto-discovered from the model_executor
                                      mirror tree (vllm_iluvatar.ilu_register).
      3. apply_kv_transfer_registrations()
                                    — KVConnectorFactory registrations for
                                      Ilu NIXL connectors.

    Patches run first so any monkey-patched symbols are in place before the OOT
    modules import. Both finish before the model is loaded, which is when
    register_oot / register_model take effect.
    """
    try:
        import ixformer  # noqa: F401
    except ImportError:
        return

    from vllm_iluvatar.distributed.kv_transfer import (
        apply_kv_transfer_registrations,
    )
    from vllm_iluvatar.ilu_patch import apply_general_patches
    from vllm_iluvatar.ilu_register import apply_oot_registrations

    apply_general_patches()
    apply_oot_registrations()
    apply_kv_transfer_registrations()
