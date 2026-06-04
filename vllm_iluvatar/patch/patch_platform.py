"""Patch the live CUDA platform for iluvatar.

``current_platform`` is a singleton *instance* whose class was imported and
instantiated before the plugin loads, so we cannot use the override finder for
``vllm/platforms/cuda.py`` / ``interface.py`` -- the running instance would
keep using the community class. Instead we mutate the platform classes
in-place (``setattr`` on the class objects), which is picked up by the live
instance because method lookup happens on the class at call time.

Mirrors the internal fork's changes to ``vllm/platforms/cuda.py`` and
``vllm/platforms/interface.py``:

* ``_get_backend_priorities`` -> route to the ILU attention backends
  (``ILU_FLASHMLA_SPARSE`` / ``ILU_MLA`` / ``ILU_FLASH_ATTN``). The ``use_sparse``
  selection is threaded through a context var so we don't have to re-implement
  the long ``get_attn_backend_cls`` body.
* ``get_device_capability`` -> report ``(9, 0)`` (iluvatar reports a
  Hopper-class capability so the stock capability gates pass).
* ``opaque_attention_op`` / ``use_custom_op_collectives`` -> ``False``.
* ``Platform.import_kernels`` -> no-op (``vllm._C`` / ``vllm._moe_C`` are not
  built on iluvatar).
* ``Platform.has_device_capability`` -> ``True``.
"""

from __future__ import annotations

import contextvars

_PATCHED = False
_use_sparse: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ilu_use_sparse", default=False
)


def _ilu_backend_priorities(
    use_mla, device_capability, num_heads=None, kv_cache_dtype=None, use_sparse=None
):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as E

    if use_sparse is None:
        use_sparse = _use_sparse.get()
    if use_mla:
        if use_sparse:
            return (E.ILU_FLASHMLA_SPARSE,)
        return [E.ILU_MLA]
    return [E.ILU_FLASH_ATTN]


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    # --- platform base (interface.py) -----------------------------------
    try:
        from vllm.platforms.interface import Platform

        Platform.import_kernels = classmethod(lambda cls: None)  # type: ignore
        Platform.has_device_capability = classmethod(  # type: ignore
            lambda cls, capability, device_id=0: True
        )
    except Exception:
        logger.exception("vllm-iluvatar: failed to patch Platform base")

    # --- cuda platform (cuda.py) ----------------------------------------
    try:
        import vllm.platforms.cuda as cuda
        from vllm.platforms.interface import DeviceCapability

        # Module-level backend priority function (referenced by name from
        # get_attn_backend_cls, so replacing the global is enough).
        cuda._get_backend_priorities = _ilu_backend_priorities

        base = cuda.CudaPlatformBase

        # Thread use_sparse from attn_selector_config into the priority fn.
        orig_get_attn = base.get_attn_backend_cls

        def _wrapped_get_attn(
            cls, selected_backend, attn_selector_config, num_heads=None
        ):
            tok = _use_sparse.set(
                bool(getattr(attn_selector_config, "use_sparse", False))
            )
            try:
                return orig_get_attn.__func__(
                    cls, selected_backend, attn_selector_config, num_heads
                )
            finally:
                _use_sparse.reset(tok)

        base.get_attn_backend_cls = classmethod(_wrapped_get_attn)
        base.opaque_attention_op = classmethod(lambda cls: False)
        base.use_custom_op_collectives = classmethod(lambda cls: False)

        # Force a Hopper-class capability on the concrete subclasses.
        def _cap(cls, device_id=0):
            return DeviceCapability(major=9, minor=0)

        for sub_name in ("NvmlCudaPlatform", "NonNvmlCudaPlatform"):
            sub = getattr(cuda, sub_name, None)
            if sub is not None:
                sub.get_device_capability = classmethod(_cap)

        logger.info(
            "vllm-iluvatar: patched CUDA platform (ilu attention routing, "
            "device capability 9.0, opaque/collectives off)"
        )
    except Exception:
        logger.exception("vllm-iluvatar: failed to patch CUDA platform")
