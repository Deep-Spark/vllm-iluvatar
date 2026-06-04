"""vllm-iluvatar plugin entry point.

vLLM discovers this module via the ``vllm.general_plugins`` entry point and
calls :func:`register_model` exactly once per process.

The function MUST be cheap to import: heavy ilu/ixformer modules are only
touched lazily once :func:`adapt_patch`, the model registry and the quant
remaps fire.
"""

from __future__ import annotations


def register_model() -> None:
    """vLLM general_plugins entry point.

    Order matters:

    1. Apply env injection, ``sys.modules`` redirects and monkey-patches
       FIRST so any subsequent ``import`` of an overridden module sees the
       iluvatar version, and the ilu attention backends / quant methods are
       registered.
    2. Register the new model architecture(s) so the registry can resolve
       checkpoints whose ``config.json`` declares them.
    """
    from .utils import adapt_patch

    adapt_patch()

    from .models import register_model as _register_model

    _register_model()
