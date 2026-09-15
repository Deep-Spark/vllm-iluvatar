# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for ILU general-plugins patches.

Centralises the cross-cutting concerns every general-plugins ``register()``
otherwise repeats: idempotent application, env-gated safety checks, source
fingerprint helpers, and a uniform "applied" announcement.

Env switches (read at register-time through ``vllm.envs.<NAME>``; the readers +
defaults are registered in vllm_iluvatar/envs.py. register_general runs after
register_platform -> envs.apply(), so the readers are in place by then. A direct
os.environ read is used as a fallback if vllm.envs is unavailable):

  VLLM_ILU_PATCH_CHECK_SIGNATURE   (default ON)  strict signature validation
  VLLM_ILU_PATCH_CHECK_FINGERPRINT (default ON)  full-method-copy drift guard
  VLLM_ILU_PATCH_VERBOSE           (default OFF) announce applied patches at
                                                 INFO instead of DEBUG

Disable a switch with any of: 0 / false / no / off (case-insensitive).

Convention: each general-plugins patch wiring module lives under
``ilu_patch.general_patches`` and exposes ``register()`` delegating to
``register_once(__name__, _apply, detail=...)``; the patch-specific work lives in
``_apply()``. Override implementations stay in upstream-mirrored modules such as
``vllm_iluvatar.distributed.parallel_state``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import textwrap
from collections.abc import Callable

from vllm.logger import init_logger

from vllm_iluvatar.ilu_util.signature import validate_patch_signatures

logger = init_logger(__name__)

ENV_CHECK_SIGNATURE = "VLLM_ILU_PATCH_CHECK_SIGNATURE"
ENV_CHECK_FINGERPRINT = "VLLM_ILU_PATCH_CHECK_FINGERPRINT"
ENV_VERBOSE = "VLLM_ILU_PATCH_VERBOSE"

_TRUE = ("1", "true", "yes", "on")

# Fallback defaults used only when vllm.envs is unavailable (e.g. accessed before
# envs.apply() ran, or in a vllm-less unit test). The authoritative parsing +
# defaults live in vllm_iluvatar/envs.py (_ILU_GUARD_ENV_VARS).
_FALLBACK_DEFAULTS = {
    ENV_CHECK_SIGNATURE: True,
    ENV_CHECK_FINGERPRINT: True,
    ENV_VERBOSE: False,
}


def _enabled(env: str) -> bool:
    """Read a plugin control flag through vllm.envs (the project's env registry).

    These vars are registered in vllm_iluvatar/envs.py and surfaced via
    ``vllm.envs.<NAME>``. Falls back to a direct os.environ read if vllm.envs is
    not importable yet or the var is not registered.
    """
    try:
        import vllm.envs as envs

        return bool(getattr(envs, env))
    except (ImportError, AttributeError):
        default = _FALLBACK_DEFAULTS.get(env, False)
        raw = os.environ.get(env, "1" if default else "0")
        return raw.strip().lower() in _TRUE


def normalize_src(src: str) -> str:
    """Dedent + strip + drop trailing whitespace so cosmetic differences (indent,
    line endings) do not trip the fingerprint."""
    return "\n".join(
        line.rstrip() for line in textwrap.dedent(src).strip().splitlines()
    )


def source_sha(src: str) -> str:
    return hashlib.sha256(normalize_src(src).encode("utf-8")).hexdigest()


def check_signatures(
    community_module: object,
    ilu_module: object,
    func_names: list[str],
    *,
    strict: bool = True,
) -> None:
    """Env-gated wrapper around validate_patch_signatures (default ON)."""
    if not _enabled(ENV_CHECK_SIGNATURE):
        logger.warning(
            "ILU: signature check disabled via %s; skipping for %s",
            ENV_CHECK_SIGNATURE,
            func_names,
        )
        return
    validate_patch_signatures(community_module, ilu_module, func_names, strict=strict)


def check_fingerprint(
    symbol: str,
    current_src: str,
    baseline_src: str,
    *,
    resync_hint: str,
) -> None:
    """Env-gated source drift guard for full-method copies (default ON).

    Raises RuntimeError if the live community source no longer matches the
    recorded baseline, so a silent behavioural divergence fails fast at startup.
    """
    if not _enabled(ENV_CHECK_FINGERPRINT):
        logger.warning(
            "ILU: fingerprint check disabled via %s; skipping drift guard for %s",
            ENV_CHECK_FINGERPRINT,
            symbol,
        )
        return
    if source_sha(current_src) != source_sha(baseline_src):
        raise RuntimeError(
            f"ILU patch: community {symbol} source changed vs baseline. {resync_hint}"
        )


def rebind_module_func(module: object, name: str, new_obj: object) -> None:
    """Rebind a module-level function override, propagating to copied refs.

    ``setattr`` the override onto ``module``, then walk ``sys.modules`` and
    replace any attribute ``name`` that still holds the *original* object (by
    identity ``is``) within the same top-level package. This fixes
    ``from module import name`` consumers that copied the reference into their
    own namespace before the patch ran (a plain ``setattr`` on ``module`` would
    miss them). Identity matching means homonymous-but-distinct symbols in
    sibling modules are never touched.

    Use this for *module-level free functions* whose upstream consumers use
    ``from module import name``. Class methods only need ``setattr`` on the class
    (instances resolve through it at call time); import-time-CALLED symbols
    cannot be fixed here at all (the branch already ran) and must be handled via
    an early shim / import hook instead.
    """
    old = getattr(module, name, None)
    setattr(module, name, new_obj)
    if old is None:
        # New symbol injection: nothing previously imported it by reference.
        return
    top = getattr(module, "__name__", "").split(".", 1)[0]
    if not top:
        return
    for mod in list(sys.modules.values()):
        mod_name = getattr(mod, "__name__", "")
        if not isinstance(mod_name, str) or not mod_name.startswith(top):
            continue
        if getattr(mod, name, None) is old:
            setattr(mod, name, new_obj)


_APPLIED: set[str] = set()


def register_once(
    name: str,
    apply_fn: Callable[[], None],
    *,
    detail: str | None = None,
) -> None:
    """Apply a patch exactly once per process, then announce it.

    The announcement level is controlled by VLLM_ILU_PATCH_VERBOSE (default
    DEBUG; INFO when enabled). ``detail`` is an optional human-readable summary
    (e.g. "broadcast wrapper, broadcast_tensor_dict copy").
    """
    if name in _APPLIED:
        return
    apply_fn()
    _APPLIED.add(name)
    log = logger.info if _enabled(ENV_VERBOSE) else logger.debug
    if detail:
        log("ILU patch applied: %s (%s)", name, detail)
    else:
        log("ILU patch applied: %s", name)
