"""vllm-iluvatar plugin entry point.

vLLM discovers this module via the ``vllm.general_plugins`` entry point
and calls :func:`register_model` exactly once per process.

The function MUST be cheap to import: heavy ilu/ixformer modules are only
touched lazily once :func:`adapt_patch` and the registry/quant registration
fire.
"""

from __future__ import annotations


def register_model() -> None:
    """vLLM general_plugins entry point.

    Order matters:

    1. Apply monkey-patches and ``sys.modules`` redirects FIRST so any
       subsequent ``import`` of a patched module sees the new version.
    2. Register the new model architecture so the model registry can resolve
       checkpoints whose ``config.json`` says
       ``"architectures": ["IluDeepseekV4ForCausalLM"]``.
    3. Import the quantization config module so its
       ``@register_quantization_config("ilu_deepseek_v4_w4a8")`` decorator
       wires the new quant method into ``QUANTIZATION_METHODS``.
    """
    from .utils import adapt_patch

    adapt_patch()

    from .models import register_model as _register_model

    _register_model()

    # Importing the module triggers the @register_quantization_config decorator.
    import vllm_iluvatar.quantization  # noqa: F401
