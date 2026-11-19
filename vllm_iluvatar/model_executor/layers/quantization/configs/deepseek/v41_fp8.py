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

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.deepseek_v41_linear import (
    IluDeepseekV41Bf16DequantLinearMethod,
    IluDeepseekV41FP8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v4_fp8 import (
    IluDeepseekV4FP8Config,
)

# Retain the existing BF16 developer fallback. Normal compute selection is
# owned by IluFp8LinearMethod, including ivcore11 INT8 execution.
FP8_ALL_DENSE = True
logger = init_logger(f"vllm.{__name__}")


@register_quantization_config("deepseek_v41_fp8")
class IluDeepseekV41FP8Config(IluDeepseekV4FP8Config):
    """DeepSeek-V4.1 FP8 sources with architecture-selected dense and MXFP4 compute.

    ``expert_dtype`` / ``is_scale_e8m0`` are inherited and still resolve off ``hf_config``, where
    the V4.1 config class hoists ``expert_dtype`` out of ``quantization_config``. For this
    checkpoint that yields ``"fp4"`` and therefore ``is_scale_e8m0 == True``, matching the e8m0
    scales in the shards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        block = getattr(self, "weight_block_size", None)
        if block != [32, 32]:
            raise ValueError(
                f"DeepSeek V4.1 dense FP8 requires weight_block_size=[32, 32], got {block}"
            )
        logger.info_once("DeepSeek V4.1 dense FP8 group_n/group_k = 32/32")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        """Claim ``deepseek_v41`` checkpoints; leave ``deepseek_v4`` to the V4 config.

        The released V4.1 checkpoint declares ``quant_method: "fp8"``, same as V4, so the model
        type is the only discriminator.
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
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            # wo_a is block-diagonal over o_groups. One group per rank (TP >= 8 on
            # this model) uses ordinary Linear selection; fewer ranks leave
            # several groups per rank and require the model-owned BMM policy.
            from vllm.distributed import get_tensor_model_parallel_world_size

            grouped_wo_a = prefix.endswith(".wo_a") and (
                get_tensor_model_parallel_world_size() < 8
            )
            if grouped_wo_a:
                from vllm_iluvatar import envs
                from vllm_iluvatar.ilu_util.arch import cuda_arch

                # Preserve ivcore11 policy. CQ deployments already enable the
                # old switch, so FP8-source INT8 on ivcore30 needs its own opt-in.
                # Ordinary Linear execution preferences never control this BMM.
                arch = cuda_arch()
                use_grouped_int8 = (
                    arch == "ivcore11" and envs.VLLM_ILU_DS41_GROUPED_INT8_O_PROJ
                ) or (arch == "ivcore30" and envs.VLLM_ILU_DS41_IVCORE30_FP8_WOA_INT8)
                if use_grouped_int8:
                    from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
                        IluDeepseekV41Fp8GroupedInt8Method,
                    )

                    return IluDeepseekV41Fp8GroupedInt8Method(self)
            from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.config import (
                resolve_ilu_runtime_method,
            )

            original = (
                IluDeepseekV41FP8LinearMethod(self)
                if FP8_ALL_DENSE and not grouped_wo_a
                else IluDeepseekV41Bf16DequantLinearMethod(self)
            )
            runtime = resolve_ilu_runtime_method(
                base_quant_config=self,
                original_method=original,
                original_disposition="quantized",
                layer=layer,
                prefix=prefix,
            )
            return runtime if runtime is not None else original
        # RoutedExperts -> V4's MXFP4 branch; everything else -> community.
        return super().get_quant_method(layer, prefix)
