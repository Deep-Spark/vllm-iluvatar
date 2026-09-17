"""Iluvatar quantization config for the raw DeepSeek-V4.1-Flash FP8 checkpoint.

Corresponds to: vllm_iluvatar/model_executor/layers/quantization/ilu_deepseek_v4_fp8.py
Target   : a new quantization name ``deepseek_v41_fp8``
Baseline : ``IluDeepseekV4FP8Config``
Why      : two independent reasons.

           1. V4's config never matches. ``DeepseekV4FP8Config.override_quantization_method``
              hard-codes ``model_type == "deepseek_v4"`` (``vllm/models/deepseek_v4/
              quant_config.py:149-152``), so a ``deepseek_v41`` checkpoint falls back to the
              generic ``fp8`` config. That loses both of the fields that matter: ``is_scale_e8m0``
              comes back ``None`` (so scale parameters would be built fp32 while the checkpoint
              ships e8m0) and ``expert_dtype`` comes back ``None`` (so the fp4 experts would not
              route to MXFP4).

           2. V4.1's dense weights are block **32x32**, and no fp8 GEMM in the stack accepts a
              32-wide K-group -- ixmm exports only ``group128`` / ``group256`` entry points and
              rejects 32 outright, on its latest master. So dense layers cannot execute as fp8 at
              all and must be dequantized to bf16.

How      : register a separate ``deepseek_v41_fp8`` name and subclass the V4 Iluvatar config, so
           V4's detection path is untouched (it keeps matching only ``deepseek_v4``). MoE keeps
           inheriting V4's MXFP4 branch, because V4.1's experts are also fp4 with group size 32.
           Dense ``LinearBase`` layers get ``IluDeepseekV41Bf16DequantLinearMethod`` instead of
           ``Fp8LinearMethod``.
Scope    : DeepSeek-V4.1-Flash raw-FP8 checkpoints on Iluvatar.
Remove   : if a 32-group block-fp8 GEMM appears in ixmm/ixformer, at which point the dense path
           can go back to a real fp8 kernel and only the detection override needs to stay.

Cost of the bf16 fallback, measured on the real checkpoint: median relative error 2.3e-03
against an fp32 dequant reference (i.e. ordinary bf16 accuracy), and dense fp8 weights grow
6.08 GiB -> 12.16 GiB, which at TP8 is +0.76 GiB per card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import BlockQuantScaleParameter

from vllm_iluvatar.model_executor.layers.quantization.ilu_deepseek_v4_fp8 import (
    IluDeepseekV4FP8Config,
)

if TYPE_CHECKING:
    pass


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Expand a 2D block-scaled fp8 weight into a dense ``out_dtype`` weight.

    ``weight`` is ``[out, in]`` fp8 and ``scale`` is ``[ceil(out/bn), ceil(in/bk)]``, one scale per
    ``bn x bk`` block. The e8m0 scales are powers of two, so widening them to fp32 is exact and
    the only error introduced is the final cast to ``out_dtype``.
    """
    bn, bk = block_size
    out_features, in_features = weight.shape
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)

    if out_features % bn == 0 and in_features % bk == 0:
        # Reshape into blocks and broadcast, which avoids materializing a full-size scale tensor.
        w = w.view(out_features // bn, bn, in_features // bk, bk)
        w = w * s.view(out_features // bn, 1, in_features // bk, 1)
        w = w.view(out_features, in_features)
    else:
        # Ragged tail: fall back to expanding the scales and cropping.
        s = s.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
        w = w * s[:out_features, :in_features]
    return w.to(out_dtype)


class IluDeepseekV41Bf16DequantLinearMethod(LinearMethodBase):
    """Loads block-fp8 weights in their checkpoint layout, then dequantizes to bf16.

    This deliberately does not subclass ``Fp8LinearMethod``: that class's ``create_weights``
    ends in ``init_fp8_linear_kernel``, which selects an fp8 GEMM for the declared group shape,
    and there is no kernel for a 32-wide group. Building the two parameters directly keeps the
    checkpoint contract (fp8 weight + e8m0 block scale under the DeepSeek-native
    ``weight_scale_inv`` name) without asking for a kernel that does not exist.

    The dequantized weight replaces ``weight`` in place and the scale parameter is dropped, so
    nothing downstream sees the fp8 form and memory does not hold both.
    """

    def __init__(self, quant_config: IluDeepseekV41FP8Config) -> None:
        self.quant_config = quant_config
        block_size = quant_config.weight_block_size
        assert block_size is not None, (
            "deepseek_v41_fp8 expects weight_block_size from the checkpoint's quantization_config"
        )
        self.block_size = (block_size[0], block_size[1])

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.weight_block_size = list(self.block_size)
        # Remember the compute dtype: params_dtype is what the rest of the model runs in.
        layer.ilu_dequant_dtype = params_dtype

        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                output_size_per_partition, input_size_per_partition, weight_loader
            ),
        )
        # ``weight_scale_inv`` is the DeepSeek-native name, which is what the V4 WeightsMapper
        # rewrites ``.scale`` into, so the checkpoint keys land here unchanged.
        layer.register_parameter(
            "weight_scale_inv",
            create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                list(self.block_size),
                weight_loader,
                scale_dtype=torch.float8_e8m0fnu,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dequantized = dequantize_block_fp8(
            layer.weight.data,
            layer.weight_scale_inv.data,
            self.block_size,
            out_dtype=getattr(layer, "ilu_dequant_dtype", torch.bfloat16),
        )
        layer.weight = torch.nn.Parameter(dequantized, requires_grad=False)
        # The scale has been folded in; keeping it would waste memory and let stale state be read.
        if hasattr(layer, "weight_scale_inv"):
            del layer.weight_scale_inv

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)


@register_quantization_config("deepseek_v41_fp8")
class IluDeepseekV41FP8Config(IluDeepseekV4FP8Config):
    """DeepSeek-V4.1 raw-FP8: MXFP4 experts on Iluvatar kernels, dense layers dequantized to bf16.

    ``expert_dtype`` / ``is_scale_e8m0`` are inherited and still resolve off ``hf_config``, where
    the V4.1 config class hoists ``expert_dtype`` out of ``quantization_config``. For this
    checkpoint that yields ``"fp4"`` and therefore ``is_scale_e8m0 == True``, matching the e8m0
    scales in the shards.
    """

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
            return IluDeepseekV41Bf16DequantLinearMethod(self)
        # RoutedExperts -> V4's MXFP4 branch; everything else -> community.
        return super().get_quant_method(layer, prefix)
