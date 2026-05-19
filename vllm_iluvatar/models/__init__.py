"""Register the ilu DeepSeek-V4 architecture with vLLM's ModelRegistry."""

from __future__ import annotations


def register_model() -> None:
    """Add ``IluDeepseekV4ForCausalLM`` to ``ModelRegistry``.

    Uses a fully-qualified string path so the model module (which pulls
    in ixformer, the new attention layer and the W4A8 quant config) is
    only imported when a checkpoint with this architecture is actually
    loaded.
    """
    from vllm import ModelRegistry
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    name = "IluDeepseekV4ForCausalLM"
    if name in ModelRegistry.get_supported_archs():
        logger.info("vllm-iluvatar: %s already registered; skipping", name)
        return

    ModelRegistry.register_model(
        name,
        "vllm_iluvatar.models.ilu_deepseek_v4:IluDeepseekV4ForCausalLM",
    )
    logger.info("vllm-iluvatar: registered model architecture %s", name)
