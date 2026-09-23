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

           2. the checkpoint's dense linears are block-fp8 and both ways of running them are
              worth keeping, so the choice needs a home. ``attn.wo_a`` stays bf16
              unconditionally, because ``_o_proj`` reads its raw ``.weight``.

How      : register a separate ``deepseek_v41_fp8`` name and subclass the V4 Iluvatar config, so
           V4's detection path is untouched (it keeps matching only ``deepseek_v4``). MoE keeps
           inheriting V4's MXFP4 branch, because V4.1's experts are also fp4 with group size 32.
           Dense linears get ``IluDeepseekV41FP8LinearMethod``, a thin ``Fp8LinearMethod``
           subclass that lands on ``IluFp8BlockScaledMMKernel`` exactly as V4 does, or --
           with ``FP8_ALL_DENSE = False`` -- ``IluDeepseekV41Bf16DequantLinearMethod``.
Scope    : DeepSeek-V4.1-Flash raw-FP8 checkpoints on Iluvatar.
Remove   : the ``K % 64`` fallback inside ``IluDeepseekV41FP8LinearMethod`` goes away if
           ``fp8_per_channel_mm`` ever accepts a K that is only 32-aligned. The bf16
           method itself stays, as the other half of the ``FP8_ALL_DENSE`` choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

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

# Run the dense linears as fp8 rather than dequantizing them to bf16 at load. This is
# the default: it is what the checkpoint ships and it keeps the weights at half the
# footprint. The trade is measured -- prefill about 1.3x, decode about 0.5x -- so set
# this to False to put the compute-bound decode path back on bf16 GEMMs.
FP8_ALL_DENSE = True


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


class IluDeepseekV41FP8LinearMethod(Fp8LinearMethod):
    """Community block-fp8 dense linear, falling back to bf16 on K the kernel cannot take.

    ``IluFp8BlockScaledMMKernel``'s load-time per-channel requant and ``fp8_per_channel_mm``
    both require ``K % 64 == 0``. Every V4.1 attention linear satisfies it (5120 / 5120 /
    1280 / 1024 at TP8), but ``ffn.shared_experts.w2`` is row-parallel over
    ``moe_intermediate_size``, leaving ``K = 2304 / 8 = 288`` at TP8.

    The split has to happen here rather than in ``get_quant_method``: that runs from
    ``LinearBase.__init__`` before ``tp_size`` or any per-partition size exists, so
    ``create_weights`` is the first place the real ``K`` is known. Both paths register the
    same two parameters (fp8 ``weight`` + e8m0 block ``weight_scale_inv``), so the choice
    is invisible to the weight loader either way.
    """

    def __init__(self, quant_config: "IluDeepseekV41FP8Config") -> None:
        super().__init__(quant_config)
        self._bf16: IluDeepseekV41Bf16DequantLinearMethod | None = None

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
        if input_size_per_partition % 64 or input_size_per_partition < 64:
            self._bf16 = IluDeepseekV41Bf16DequantLinearMethod(self.quant_config)
            self._bf16.create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            return
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._bf16 is not None:
            self._bf16.process_weights_after_loading(layer)
            return
        super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._bf16 is not None:
            return self._bf16.apply(layer, x, bias)
        return super().apply(layer, x, bias)


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
            # wo_a is block-diagonal over o_groups. One group per rank (TP >= 8 on
            # this model) is an ordinary GEMM and stays fp8; fewer ranks leave
            # several groups per rank, and that contraction reads a bf16 weight.
            from vllm.distributed import get_tensor_model_parallel_world_size

            grouped_wo_a = prefix.endswith(".wo_a") and (
                get_tensor_model_parallel_world_size() < 8
            )
            if FP8_ALL_DENSE and not grouped_wo_a:
                return IluDeepseekV41FP8LinearMethod(self)
            return IluDeepseekV41Bf16DequantLinearMethod(self)
        # RoutedExperts -> V4's MXFP4 branch; everything else -> community.
        return super().get_quant_method(layer, prefix)
