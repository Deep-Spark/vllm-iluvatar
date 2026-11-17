"""Iluvatar quantization config for raw DeepSeek-V4.1-Flash FP8 checkpoints."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp8.deepseek_v41_linear import (
    IluDeepseekV41Bf16DequantLinearMethod,
    IluDeepseekV41FP8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v4_fp8 import (
    IluDeepseekV4FP8Config,
)

# Run the dense linears as fp8 rather than dequantizing them to bf16 at load.
# This is the default: it is what the checkpoint ships and it keeps the weights
# at half the footprint. The trade is measured -- prefill about 1.3x, decode
# about 0.5x -- so set this to False to put the compute-bound decode path back
# on bf16 GEMMs.
FP8_ALL_DENSE = True
logger = init_logger(f"vllm.{__name__}")


@register_quantization_config("deepseek_v41_fp8")
class IluDeepseekV41FP8Config(IluDeepseekV4FP8Config):
    """DeepSeek-V4.1 raw-FP8: MXFP4 experts on Iluvatar kernels, dense layers
    dequantized to bf16.

    ``expert_dtype`` / ``is_scale_e8m0`` are inherited and still resolve off
    ``hf_config``, where the V4.1 config class hoists ``expert_dtype`` out of
    ``quantization_config``. For this checkpoint that yields ``"fp4"`` and
    therefore ``is_scale_e8m0 == True``, matching the e8m0 scales in the shards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        block = getattr(self, "weight_block_size", None)
        if block != [32, 32]:
            raise ValueError(
                "DeepSeek V4.1 dense FP8 requires "
                f"weight_block_size=[32, 32], got {block}"
            )
        logger.info_once("DeepSeek V4.1 dense FP8 group_n/group_k = 32/32")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        """Claim ``deepseek_v41`` checkpoints; leave ``deepseek_v4`` to the V4
        config.

        The released V4.1 checkpoint declares ``quant_method: "fp8"``, same as
        V4, so the model type is the only discriminator.
        """
        if not isinstance(hf_quant_cfg, dict):
            return None
        if hf_quant_cfg.get("quant_method") not in ("fp8", "deepseek_v41_fp8"):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v41" or user_quant == "deepseek_v41_fp8":
            return "deepseek_v41_fp8"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                from vllm.model_executor.layers.linear import (
                    UnquantizedLinearMethod,
                )

                return UnquantizedLinearMethod()
            # wo_a is block-diagonal over o_groups. One group per rank
            # (TP >= 8 on this model) is an ordinary GEMM and stays fp8; fewer
            # ranks leave several groups per rank, and that contraction reads a
            # bf16 weight.
            from vllm.distributed import get_tensor_model_parallel_world_size

            grouped_wo_a = prefix.endswith(".wo_a") and (
                get_tensor_model_parallel_world_size() < 8
            )
            if FP8_ALL_DENSE and not grouped_wo_a:
                return IluDeepseekV41FP8LinearMethod(self)
            return IluDeepseekV41Bf16DequantLinearMethod(self)
        # RoutedExperts -> V4's MXFP4 branch; everything else -> community.
        return super().get_quant_method(layer, prefix)
