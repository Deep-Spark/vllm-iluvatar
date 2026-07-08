# SPDX-License-Identifier: Apache-2.0
"""Iluvatar AWQ / AWQ-Marlin quantization config.

OOT plugin version of the fork-local ``vllm/.../quantization/ilu_auto_awq.py``.
Registration is done with the official ``@register_quantization_config``
decorator at import time, so community ``vllm`` does not need method literal or
``get_quantization_config`` edits for the Iluvatar AWQ path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import ixformer.inference.functions as ixfops
import torch
from torch.nn import Parameter
from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQConfig,
    AutoAWQLinearMethod,
    BaseAWQLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_marlin_supports_layer,
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    verify_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.scalar_type import scalar_types
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)

@register_quantization_config("auto_awq")
@register_quantization_config("awq")
@register_quantization_config("awq_marlin")
class IluAutoAWQConfig(AutoAWQConfig):
    """Iluvatar AWQ config registered out-of-tree.

    Uses Iluvatar-owned method names so community ``awq`` / ``auto_awq`` /
    ``awq_marlin`` keep their upstream behavior unless a checkpoint or user
    explicitly asks for the ``ilu_*`` methods.
    """
    def __repr__(self) -> str:
        return (
            f"IluAutoAWQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "auto_awq"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IluAutoAWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config,
            ["modules_to_not_convert"],
            None,
        )
        # Ensure full_config uses "awq" as quant_method for MoE fallback compatibility.
        # MoeWNA16Config only accepts "gptq" or "awq", so we normalize here.
        full_config = config.copy()
        full_config["quant_method"] = "awq"
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            full_config,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            # Check if Marlin is supported and not using batch invariant mode
            # (Marlin kernels are not batch invariant)
            use_marlin = (
                not envs.VLLM_BATCH_INVARIANT
                and current_platform.is_cuda()
                and check_marlin_supported(
                    self.quant_type, self.group_size, self.zero_point
                )
            )

            if use_marlin:
                # tile-misaligned shapes are fixed by padding at weight prep
                if not check_marlin_supports_layer(
                    layer, self.group_size, allow_tile_padding=True
                ):
                    logger.warning_once(
                        "Layer '%s' is not supported by AutoAWQMarlin. "
                        "Falling back to unoptimized AWQ kernels.",
                        prefix,
                    )
                    return AutoAWQLinearMethod(self)
                quant_method = IluAWQMarlinLinearMethod(self)
                quant_method.input_dtype = get_marlin_input_dtype(prefix)
                return quant_method

            return AutoAWQLinearMethod(self)

        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix,
                getattr(self, "modules_to_not_convert", []),
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=True
            ):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AutoAWQMoEMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )

            return IluAutoAWQMoEMethod(self, layer.moe_config)

        return None


class IluAWQMarlinLinearMethod(BaseAWQLinearMethod):
    """Linear AWQ method backed by Iluvatar W4A16 kernels."""

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: IluAutoAWQConfig) -> None:
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        verify_marlin_supported(
                quant_type=self.quant_config.quant_type,
                group_size=self.quant_config.group_size,
                has_zp=self.quant_config.zero_point,
            )

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
        super().create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        output_size_per_partition = sum(output_partition_sizes)
        group_size = (
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = input_size_per_partition // group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.output_size_per_partition = (
            layer.qweight.data.shape[1] * self.quant_config.pack_factor
        )
        align_size = 64 * 8 // self.quant_config.weight_bits
        if layer.output_size_per_partition % align_size != 0:
            padded_output_size = (
                (layer.output_size_per_partition + align_size - 1)
                // align_size
                * align_size
            )
            layer.output_padding_size = (
                padded_output_size - layer.output_size_per_partition
            )
            device = layer.qweight.device
            packed_output = (
                layer.output_size_per_partition // self.quant_config.pack_factor
            )
            padded_packed_output = padded_output_size // self.quant_config.pack_factor

            pad_qweight = torch.zeros(
                layer.input_size_per_partition,
                padded_packed_output,
                dtype=torch.int32,
                device=device,
            )
            pad_qzeros = torch.zeros(
                layer.num_groups,
                padded_packed_output,
                dtype=torch.int32,
                device=device,
            )
            pad_scales = torch.zeros(
                layer.num_groups,
                padded_output_size,
                dtype=layer.scales.data.dtype,
                device=device,
            )
            pad_qweight[..., :packed_output] = layer.qweight.data
            pad_qzeros[..., :packed_output] = layer.qzeros.data
            pad_scales[..., : layer.output_size_per_partition] = layer.scales.data
            replace_parameter(layer, "qweight", pad_qweight)
            replace_parameter(layer, "qzeros", pad_qzeros)
            replace_parameter(layer, "scales", pad_scales)

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = ops.awq_gemm(
            reshaped_x,
            qweight,
            scales,
            qzeros,
            pack_factor,
            group_size=self.quant_config.group_size,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class IluAutoAWQMoEMethod(FusedMoEMethodBase):
    """Iluvatar AWQ MoE method using ixformer routed expert kernels."""

    def __init__(
        self,
        quant_config: IluAutoAWQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("IluAutoAWQMoEMethod only supports 4-bit weights.")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.input_dtype = self.input_dtype
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full",
            intermediate_size_per_partition,
        )
        self.is_k_full = intermediate_size_per_partition == intermediate_size_full

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size
        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ixformer kernels consume the loaded AWQ layout directly.
        return

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> FusedMoEQuantConfig | None:
        return

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert layer.activation.value == "silu", "Only SiLU activation is supported."
        use_ep = layer.expert_map is not None
        attn_metadata = None
        try:
            from vllm.forward_context import get_forward_context

            attn_metadata = get_forward_context().attn_metadata
        except AssertionError:
            attn_metadata = None

        if attn_metadata:
            if isinstance(attn_metadata, dict):
                only_decode = not use_ep and all(
                    t.num_decodes > 0 and t.num_prefills == 0
                    for t in attn_metadata.values()
                )
            else:
                only_decode = (
                    not use_ep
                    and attn_metadata.num_decodes > 0
                    and attn_metadata.num_prefills == 0
                )
        else:
            only_decode = False

        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min(
                (layer.ep_rank + 1) * layer.local_num_experts,
                layer.global_num_experts,
            )

        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for fused "
                "Iluvatar AWQ MoE method."
            )

        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts
        if use_ep:
            hidden_size = x.shape[1]
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
                expand_tokens,
            ) = ixfops.moe_compute_token_index_ep(
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_eid,
                end_expert_id=end_eid,
            )
            if expert_sizes_cpu.sum() == 0:
                return torch.zeros(
                    (num_tokens, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            expand_tokens = num_tokens * layer.top_k
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
            ) = ixfops.moe_compute_token_index(
                topk_ids=topk_ids,
                num_experts=num_experts,
            )

        expand_hidden_states = ixfops.moe_expand_input(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
        )

        if only_decode:
            pt_output_1 = ixfops.moe_w4a16_group_gemv(
                input=expand_hidden_states,
                weight=layer.w13_qweight,
                w_scales=layer.w13_scales,
                quant_type="awq",
                w_zeros=layer.w13_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=None,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )
            pt_output_2 = ixfops.silu_and_mul(pt_output_1)

            pt_output_3 = ixfops.moe_w4a16_group_gemv(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )
            return ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=_get_ilu_output_scale(layer),
                extra_residual=shared_experts_input,
            )

        expert_sizes_cpu = expert_sizes_gpu.cpu()
        pt_output_1 = ixfops.moe_w4a16_group_gemm(
            input=expand_hidden_states,
            weight=layer.w13_qweight,
            w_scales=layer.w13_scales,
            quant_type="awq",
            tokens_per_experts=expert_sizes_cpu,
            w_zeros=layer.w13_qzeros,
            group_size=self.quant_config.group_size,
            dst_to_src=None,
            format="NN",
            tokens_per_experts_gpu=expert_sizes_gpu,
        )
        pt_output_2 = ixfops.silu_and_mul(pt_output_1)

        if use_ep:
            pt_output_3 = torch.empty(
                (num_tokens * layer.top_k, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )
            ixfops.moe_w4a16_group_gemm(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                tokens_per_experts=expert_sizes_cpu,
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                output=pt_output_3,
            )
            reduce_mask = src_to_dst == -1
            return ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=_get_ilu_output_scale(layer),
                mask=reduce_mask,
                extra_residual=shared_experts_input,
            )

        pt_output_3 = ixfops.moe_w4a16_group_gemm(
            input=pt_output_2,
            weight=layer.w2_qweight,
            w_scales=layer.w2_scales,
            quant_type="awq",
            tokens_per_experts=expert_sizes_cpu,
            w_zeros=layer.w2_qzeros,
            group_size=self.quant_config.group_size,
            dst_to_src=sorted_token_ids,
            format="NN",
            tokens_per_experts_gpu=expert_sizes_gpu,
        )
        return ixfops.moe_output_reduce_sum(
            input=pt_output_3.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
        )
