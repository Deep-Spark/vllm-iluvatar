"""Patch ``vllm.config.model.ModelConfig`` for IluDeepseekV4ForCausalLM.

Why
---
Stock vLLM hardcodes the tokenizer-mode auto-detection inside
``ModelConfig.__post_init__`` (around the ``if arch == "DeepseekV4ForCausalLM"``
branch). We need the same ``tokenizer_mode = "deepseek_v4"`` default when
the checkpoint's ``architectures`` list contains
``"IluDeepseekV4ForCausalLM"``.

How
---
Wrap ``ModelConfig.__post_init__``: call the original, then post-fix
``self.tokenizer_mode`` if needed. The quant-method override list is left
alone -- ``IluDeepseekV4W4A8Config.override_quantization_method`` already
handles that classpath via the standard ``register_quantization_config``
hook (the new method appears in ``QUANTIZATION_METHODS`` /
``current_platform.supported_quantization`` and therefore in the probe
iteration).
"""

from __future__ import annotations

_ILU_V4_ARCHS = {"IluDeepseekV4ForCausalLM"}


def apply() -> None:
    from vllm.config.model import ModelConfig
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(ModelConfig.__post_init__, "_ilu_patched", False):
        return

    original_post_init = ModelConfig.__post_init__

    def patched_post_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_post_init(self, *args, **kwargs)

        # Stock vLLM has already resolved ``self._architecture``; mirror the
        # ``"DeepseekV4ForCausalLM" -> "deepseek_v4"`` rule for the ilu
        # architecture. Only override when the user did not explicitly set a
        # tokenizer_mode (i.e. it is still on the post-init default that
        # mirrors the upstream branch they did NOT hit).
        arch = getattr(self, "_architecture", None)
        if arch in _ILU_V4_ARCHS and self.tokenizer_mode != "deepseek_v4":
            self.tokenizer_mode = "deepseek_v4"
            logger.info(
                "vllm-iluvatar: defaulting tokenizer_mode='deepseek_v4' " "for %s",
                arch,
            )

    patched_post_init._ilu_patched = True  # type: ignore[attr-defined]
    ModelConfig.__post_init__ = patched_post_init  # type: ignore[assignment]
