"""Remap quantization methods to their iluvatar implementations.

The internal fork rewires ``get_quantization_config`` in
``vllm.model_executor.layers.quantization`` so that several method names
resolve to ilu config classes:

* ``auto_gptq`` / ``gptq`` / ``gptq_marlin`` -> ``IluAutoGPTQConfig``
* ``awq_marlin``                            -> ``IluAWQMarlinConfig``
* ``ilu_deepseek_v4_w4a8`` (NEW)            -> ``IluDeepseekV4W4A8Config``

We wrap ``get_quantization_config`` instead of overlaying the whole package
``__init__`` (which eagerly imports many submodules and is hard to evict
safely). The ilu config classes are imported lazily so the override finder
serves our mirror-tree copies.
"""

from __future__ import annotations

_PATCHED = False

# method name -> (module path, class name); module paths are stock vllm
# paths resolved through the override finder.
_ILU_REMAP = {
    "auto_gptq": (
        "vllm.model_executor.layers.quantization.ilu_auto_gptq",
        "IluAutoGPTQConfig",
    ),
    "gptq": (
        "vllm.model_executor.layers.quantization.ilu_auto_gptq",
        "IluAutoGPTQConfig",
    ),
    "gptq_marlin": (
        "vllm.model_executor.layers.quantization.ilu_auto_gptq",
        "IluAutoGPTQConfig",
    ),
    "awq_marlin": (
        "vllm.model_executor.layers.quantization.ilu_awq_marlin",
        "IluAWQMarlinConfig",
    ),
    "ilu_deepseek_v4_w4a8": (
        "vllm.model_executor.layers.quantization.ilu_deepseek_v4_w4a8",
        "IluDeepseekV4W4A8Config",
    ),
}


def _resolve(name: str):
    import importlib

    mod_path, cls_name = _ILU_REMAP[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import vllm.model_executor.layers.quantization as q
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    # Make new method names pass validation.
    methods = getattr(q, "QUANTIZATION_METHODS", None)
    if isinstance(methods, list) and "ilu_deepseek_v4_w4a8" not in methods:
        methods.append("ilu_deepseek_v4_w4a8")

    orig = q.get_quantization_config

    if getattr(orig, "_ilu_wrapped", False):
        return

    def get_quantization_config(quantization: str):
        if quantization in _ILU_REMAP:
            try:
                return _resolve(quantization)
            except Exception:  # pragma: no cover - fall back to stock
                logger.exception(
                    "vllm-iluvatar: failed to resolve ilu quant '%s'; "
                    "falling back to upstream",
                    quantization,
                )
        return orig(quantization)

    get_quantization_config._ilu_wrapped = True  # type: ignore[attr-defined]
    q.get_quantization_config = get_quantization_config

    logger.info(
        "vllm-iluvatar: remapped quant methods %s",
        ", ".join(sorted(_ILU_REMAP)),
    )
