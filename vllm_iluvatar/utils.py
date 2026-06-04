"""Plugin-wide bootstrap helpers.

The single public symbol is :func:`adapt_patch`. It orchestrates every
runtime monkey-patch and ``sys.modules`` redirect required to overlay the
iluvatar fork onto community vLLM. All patches are idempotent so the function
can be called multiple times (vLLM is allowed to load plugins more than once
per process tree per the ``load_general_plugins`` docstring).
"""

from __future__ import annotations

_PATCHED = False


def adapt_patch() -> None:
    """Apply every plugin patch exactly once.

    Order matters:

    1. ``patch_envs`` runs FIRST so brand-new ``VLLM_*`` knobs and changed
       defaults are visible before anything reads them.
    2. ``install_module_redirects`` installs the mirror-tree finder so any
       subsequent ``import`` of an overridden / new ``vllm.*`` module gets
       our copy.
    3. Attribute-level patches run AFTER, targeting modules that may already
       be imported (``vllm._custom_ops``, kv-cache utils, model config) and
       registering the ilu attention backends / quant remaps.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # 0. Make custom-op registration idempotent BEFORE any redirect triggers a
    #    re-import of a module that registers torch custom ops at import time
    #    (e.g. vllm.distributed.parallel_state -> vllm::all_reduce).
    from .patch.patch_op_registration import apply as _patch_op_registration

    _patch_op_registration()

    # 1. Environment variables (must precede any env read).
    from .patch.patch_envs import apply as _patch_envs

    _patch_envs()

    # 2. torch.accelerator backfill for older corex torch.
    from .patch.patch_torch_accelerator import apply as _patch_torch_accel

    _patch_torch_accel()

    # 3. Mirror-tree sys.modules redirects (before downstream imports).
    #    NOTE: vllm.v1.core.kv_cache_utils is handled by the override finder
    #    (full internal version) rather than a partial attribute patch.
    from .patch.patch_sys_modules import install_module_redirects

    install_module_redirects()

    # 4. Attribute-level patches and registrations on live objects.
    from .patch.patch_attention_registry import apply as _patch_attn_registry
    from .patch.patch_custom_ops import apply as _patch_custom_ops
    from .patch.patch_model_config import apply as _patch_model_config
    from .patch.patch_platform import apply as _patch_platform
    from .patch.patch_quantization import apply as _patch_quantization

    _patch_custom_ops()
    _patch_model_config()
    _patch_attn_registry()
    _patch_platform()
    _patch_quantization()
