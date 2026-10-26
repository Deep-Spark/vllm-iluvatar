# SPDX-License-Identifier: Apache-2.0
"""Check quantized checkpoint coverage even when DefaultModelLoader skips it."""

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)

logger = init_logger(__name__)


def verify_quantized_weights_loaded(model, loaded: set[str], quant_config) -> None:
    if quant_config is None or quant_config.get_name() != "compressed-tensors":
        return
    parameters = dict(model.named_parameters())
    covered = set(loaded)
    # vLLM 0.31 MoERunner delegates load_weights directly to RoutedExperts:
    # it returns "w13_weight", but named_parameters includes the child's
    # "routed_experts." path. Resolve only aliases proven by actual modules.
    for prefix, module in model.named_modules():
        if isinstance(module, MoERunnerInterface):
            for name, _ in module.routed_experts.named_parameters():
                alias = f"{prefix}.{name}" if prefix else name
                canonical = (
                    f"{prefix}.routed_experts.{name}"
                    if prefix
                    else f"routed_experts.{name}"
                )
                if alias in loaded:
                    covered.add(canonical)
    missing = sorted(set(parameters) - covered)
    if missing:
        raise RuntimeError(
            f"Qwen4Exp quantized checkpoint left {len(missing)} parameter(s) "
            f"unloaded: {missing[:16]}"
        )
    logger.info(
        "Qwen4Exp verified checkpoint coverage for all %d parameters",
        len(parameters),
    )
