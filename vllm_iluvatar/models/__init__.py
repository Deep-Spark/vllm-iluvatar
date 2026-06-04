"""Register iluvatar model architectures with vLLM's ModelRegistry.

Community vLLM already registers the stock architectures (Qwen3, DeepseekV2/V3,
GLM4Moe, GptOss, Step3p5, ...) by name; because the override finder redirects
those model modules to our mirror-tree copies, those archs automatically use
the iluvatar implementations -- no re-registration needed.

Only genuinely NEW architecture names have to be registered here. The internal
fork adds exactly one::

    IluDeepseekV4ForCausalLM -> vllm.models.deepseek_v4:IluDeepseekV4ForCausalLM

The target module path is a stock ``vllm.*`` path served by the override
finder, so the registration uses the full internal implementation.
"""

from __future__ import annotations

# arch name -> "module:ClassName" (lazy; resolved on first checkpoint load).
_ILU_MODELS = {
    "IluDeepseekV4ForCausalLM": ("vllm.models.deepseek_v4:IluDeepseekV4ForCausalLM"),
}


def register_model() -> None:
    from vllm import ModelRegistry
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    supported = set(ModelRegistry.get_supported_archs())
    registered = []
    for name, target in _ILU_MODELS.items():
        if name in supported:
            continue
        ModelRegistry.register_model(name, target)
        registered.append(name)

    if registered:
        logger.info(
            "vllm-iluvatar: registered model architectures %s",
            ", ".join(registered),
        )
    else:
        logger.info("vllm-iluvatar: model architectures already registered")
