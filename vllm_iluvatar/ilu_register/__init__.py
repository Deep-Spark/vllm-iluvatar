# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT registration layer (official extension points).

This is the third late-phase mechanism, kept deliberately separate from
``ilu_patch`` (runtime monkey-patches) and the startup import-hook
(``vllm._custom_ops`` splat):

  * import-hook     -> import-time ``vllm._custom_ops`` overrides
  * ilu_patch       -> runtime monkey-patches (with fingerprint / signature)
  * ilu_register    -> official OOT extension points: CustomOp.register_oot and
                       ModelRegistry.register_model (NO fingerprint — these are
                       replacement subclasses / registry entries, not method
                       copies)

``apply_oot_registrations()`` is invoked from ``register_general`` after
``apply_general_patches()``. It drives two distinct OOT registration paths, kept
separate because their mechanisms differ:

  1. Op OOT registration (``_register_ops``): import every module under
     ``_OOT_OP_PACKAGES``. Registration happens purely via module-level
     ``@<Base>.register_oot`` / ``@CustomOp.register`` decorators — no hook.
  2. Model registration (``_register_models``): import every module under
     ``_OOT_MODEL_PACKAGES``, then call its ``register_model()`` hook, which does
     ``ModelRegistry.register_model``.

Afterwards it runs a *prefix-compatible* signature check over every OOT subclass
that came from our packages (see ``ilu_util.signature.validate_oot_override``).

To add an op override: drop a module under ``_OOT_OP_PACKAGES`` with
``@<Base>.register_oot`` classes (no hook needed).
To add a model override: drop a module under ``_OOT_MODEL_PACKAGES`` exposing a
``register_model()`` function. No edit here is needed in either case.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from vllm.logger import init_logger

logger = init_logger(__name__)

# Packages mirror the upstream tree so OOT overrides can be diffed against the
# community source file-by-file. Two lists, one per registration mechanism.
#
# Op packages register via import-time decorators (no hook).
_OOT_OP_PACKAGES = [
    "vllm_iluvatar.model_executor.layers",
    "vllm_iluvatar.model_executor.layers.fused_moe",
    "vllm_iluvatar.model_executor.layers.fused_moe.router",
    "vllm_iluvatar.model_executor.layers.mamba.gdn",
    "vllm_iluvatar.model_executor.layers.rotary_embedding",
    # Quantization configs register via @register_quantization_config at import.
    "vllm_iluvatar.model_executor.layers.quantization",
    "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors",
]
# Model packages register via a module-level ``register_model()`` hook.
# Sub-packages are not auto-discovered (``_iter_package_modules`` skips them),
# so multi-file model packages must be listed explicitly.
_OOT_MODEL_PACKAGES = [
    "vllm_iluvatar.model_executor.models",
    "vllm_iluvatar.models.deepseek_v32",
    "vllm_iluvatar.models.deepseek_v4",
    "vllm_iluvatar.models.glm_moe_dsa",
    "vllm_iluvatar.models.minimax_m3",
    "vllm_iluvatar.models.qwen4_exp",
]

__all__ = ["apply_oot_registrations"]

_applied = False


def apply_oot_registrations() -> None:
    """Auto-discover and apply all OOT registrations (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True

    _register_ops()
    _register_models()
    _validate_oot_overrides()


def _iter_package_modules(pkg_name: str):
    """Yield imported top-level modules of *pkg_name* (skips sub-packages)."""
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:
        logger.exception("ILU: failed to import OOT package %s", pkg_name)
        return
    for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if info.ispkg:
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception:
            logger.exception("ILU: failed to import OOT module %s", info.name)
            continue


def _register_ops() -> None:
    """Op OOT registration: import triggers @register_oot / @CustomOp.register."""
    for pkg_name in _OOT_OP_PACKAGES:
        for _module in _iter_package_modules(pkg_name):
            pass  # import side effect (decorators) is the registration


def _register_models() -> None:
    """Model registration: import then call each module's register_model() hook."""
    for pkg_name in _OOT_MODEL_PACKAGES:
        for module in _iter_package_modules(pkg_name):
            hook = getattr(module, "register_model", None)
            if callable(hook):
                hook()


def _validate_oot_overrides() -> None:
    """Run the prefix-compatible signature check on our OOT subclasses.

    Reuses the shared ``VLLM_ILU_PATCH_CHECK_SIGNATURE`` / ``VLLM_ILU_PATCH_VERBOSE``
    switches. No fingerprint check — OOT subclasses are replacements, not copies.
    """
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_iluvatar.ilu_patch.general_patch import (
        ENV_CHECK_SIGNATURE,
        ENV_VERBOSE,
        _enabled,
    )
    from vllm_iluvatar.ilu_util.signature import validate_oot_override

    if not _enabled(ENV_CHECK_SIGNATURE):
        logger.warning(
            "ILU: OOT signature check disabled via %s; skipping", ENV_CHECK_SIGNATURE
        )
        return

    prefixes = tuple(_OOT_OP_PACKAGES + _OOT_MODEL_PACKAGES)
    log = logger.info if _enabled(ENV_VERBOSE) else logger.debug

    for ilu_cls in list(op_registry_oot.values()):
        if not getattr(ilu_cls, "__module__", "").startswith(prefixes):
            continue
        # The immediate non-object base is the community class being overridden.
        base = next((b for b in ilu_cls.__mro__[1:] if b is not object), None)
        if base is None:
            continue
        for attr, value in list(vars(ilu_cls).items()):
            if not inspect.isfunction(value):
                continue
            if getattr(base, attr, None) is None:
                continue
            validate_oot_override(base, ilu_cls, attr, strict=True)
            log(
                "ILU OOT signature ok: %s.%s vs %s",
                ilu_cls.__name__,
                attr,
                base.__name__,
            )
