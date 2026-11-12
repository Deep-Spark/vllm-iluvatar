# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 block-FP8 source methods and grouped BF16 fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm import envs
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.linear import (
    IluFp8LinearMethod,
)

if TYPE_CHECKING:
    from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v41_fp8 import (
        IluDeepseekV41FP8Config,
    )


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Expand a 2D block-scaled fp8 weight into a dense ``out_dtype`` weight.

    ``weight`` is ``[out, in]`` fp8 and ``scale`` is
    ``[ceil(out/bn), ceil(in/bk)]``, one scale per ``bn x bk`` block. The e8m0
    scales are powers of two, so widening them to fp32 is exact and the only
    error introduced is the final cast to ``out_dtype``.
    """
    bn, bk = block_size
    out_features, in_features = weight.shape
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)

    if out_features % bn == 0 and in_features % bk == 0:
        # Reshape into blocks and broadcast, which avoids materializing a
        # full-size scale tensor.
        w = w.view(out_features // bn, bn, in_features // bk, bk)
        w = w * s.view(out_features // bn, 1, in_features // bk, 1)
        w = w.view(out_features, in_features)
    else:
        # Ragged tail: fall back to expanding the scales and cropping.
        s = s.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
        w = w * s[:out_features, :in_features]
    return w.to(out_dtype)


class IluDeepseekV41Bf16DequantLinearMethod(LinearMethodBase):
    """Loads block-fp8 weights in their checkpoint layout, then dequantizes
    them to bf16.

    This deliberately does not subclass ``Fp8LinearMethod``: that class's
    ``create_weights`` ends in ``init_fp8_linear_kernel``, which selects an fp8
    GEMM for the declared group shape, and there is no kernel for a 32-wide
    group. Building the two parameters directly keeps the checkpoint contract
    (fp8 weight + e8m0 block scale under the DeepSeek-native
    ``weight_scale_inv`` name) without asking for a kernel that does not exist.

    The dequantized weight replaces ``weight`` in place and the scale parameter
    is dropped, so nothing downstream sees the fp8 form and memory does not
    hold both at once.
    """

    def __init__(self, quant_config: IluDeepseekV41FP8Config) -> None:
        self.quant_config = quant_config
        block_size = quant_config.weight_block_size
        assert block_size is not None, (
            "deepseek_v41_fp8 expects weight_block_size from the checkpoint's "
            "quantization_config"
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
        # Remember the compute dtype: params_dtype is what the rest of the
        # model runs in.
        layer.ilu_dequant_dtype = params_dtype

        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                output_size_per_partition,
                input_size_per_partition,
                weight_loader,
            ),
        )
        # ``weight_scale_inv`` is the DeepSeek-native name, which is what the
        # V4 WeightsMapper rewrites ``.scale`` into, so the checkpoint keys
        # land here unchanged.
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
        if envs.VLLM_ILU_V41_PREFILL_FP8_O_PROJ and hasattr(
            layer, "bmm_geom"
        ):
            geom = layer.bmm_geom
            groups = int(geom.n_local_groups)
            channels = int(geom.o_lora_rank)
            width = int(geom.heads_per_group) * int(geom.head_dim)
            assert dequantized.shape == (groups * channels, width)
            assert dequantized.dtype is torch.bfloat16
            # Powers-of-two scales preserve checkpoint FP8 mantissas.
            block_scales = layer.weight_scale_inv.data.float()
            scale = block_scales.amax(dim=1).repeat_interleave(self.block_size[0])
            scale = scale[: dequantized.shape[0]].contiguous()
            fp8_weight = (dequantized.float() / scale[:, None]).to(torch.float8_e4m3fn)
            layer.register_buffer(
                "prefill_bmm_b_fp8",
                fp8_weight.view(groups, channels, width),
                persistent=False,
            )
            layer.register_buffer(
                "prefill_bmm_bs",
                scale.view(groups, channels),
                persistent=False,
            )
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


class IluDeepseekV41FP8LinearMethod(IluFp8LinearMethod):
    """OOT FP8-source linear; FP8 execution falls back to BF16 for unsupported K.

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

    def __init__(self, quant_config: IluDeepseekV41FP8Config) -> None:
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
        if self.int8_scheme is None and (
            input_size_per_partition % 64 or input_size_per_partition < 64
        ):
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
