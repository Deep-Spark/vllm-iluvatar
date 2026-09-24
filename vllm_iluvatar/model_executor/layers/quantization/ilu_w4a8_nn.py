# SPDX-License-Identifier: Apache-2.0
"""ivcore30 W4A8 NN routed experts through the ixformer ixmm bridge.

The legacy W4A8 method supplies quant-config and activation helpers.
This class owns the fixed NN loading layout and execution and selects its
own standard/batched experts. FP8 checkpoint conversion subclasses this
method and overrides loading, following the FP4/FP8 method organization.
"""

import ixformer.inference.functions as ixfops
import torch
from compressed_tensors.quantization import QuantizationStrategy
from ixformer.inference.backend_registry import registry
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8 import (
    IluCompressedTensorsW4A8Int8MoEMethod,
    _get_ilu_output_scale,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
    replace_with_padded,
    unpad_tensor,
)


def require_ivcore30_nn() -> None:
    if not is_ivcore30():
        raise RuntimeError("IluW4A8NNMoEMethod requires ivcore30")
    if "ixmm" not in registry.available_backends("moe_w4a8_group_gemm_ixmm"):
        raise RuntimeError(
            "W4A8 NN requires the ixformer ixmm backend; install ixmm and "
            "build ixformer with HAVE_IXMM for ivcore30"
        )


def validate_nn_pack_version(pack_version: object) -> None:
    """Accept the historical checkpoint marker without selecting a kernel."""
    if type(pack_version) is not int or pack_version != 1:
        raise ValueError(
            f"Unsupported NN checkpoint pack_version={pack_version!r}; "
            "ivcore30 NN accepts legacy pack_version=1 or an omitted version. "
            "Convert weights and scales before loading other formats."
        )


class IluW4A8NNMoEMethod(IluCompressedTensorsW4A8Int8MoEMethod):
    """Fixed ixmm NN layout with device-resident routing in every batch phase."""

    def __init__(
        self, weight_quant, input_quant, moe, layer_name=None, ilu_params=None
    ):
        require_ivcore30_nn()
        params = ilu_params or {}
        if params.get("gemm_format") != "NN":
            raise ValueError("ivcore30 NN requires gemm_format=NN")
        validate_nn_pack_version(params.get("pack_version", 1))
        if (
            weight_quant.num_bits != 4
            or weight_quant.strategy != QuantizationStrategy.CHANNEL
            or weight_quant.group_size not in (None, -1)
            or not weight_quant.symmetric
            or input_quant.num_bits != 8
            or input_quant.strategy != QuantizationStrategy.TOKEN
            or not input_quant.dynamic
            or not input_quant.symmetric
            or moe.has_bias
        ):
            raise ValueError(
                "ivcore30 NN requires symmetric per-channel INT4 weights, "
                "dynamic symmetric per-token INT8 activations and no bias"
            )
        # Own the NN contract; share only version-independent parent helpers.
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.has_bias = False
        self.group_size = -1
        self.weight_symmetric = True
        self.gemm_format = "NN"
        self.static_input_scales = False
        self._interleave_buf = None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self.gemm1_info = W4A8GroupGemmInfo(
            "NN", hidden_size, 2 * intermediate_size_per_partition
        )
        self.gemm2_info = W4A8GroupGemmInfo(
            "NN", intermediate_size_per_partition, hidden_size
        )
        if hidden_size != self.gemm1_info.get_pad_k():
            raise ValueError("W4A8 NN does not support hidden K padding")
        # Load the on-disk shapes first, then pad. Scales already include the
        # complete INT4 dequant factor; there are no auxiliary INT8 scales.
        for prefix, info, shard_dim in (
            ("w13", self.gemm1_info, 1),
            ("w2", self.gemm2_info, 0),
        ):
            weight = torch.nn.Parameter(
                torch.empty((num_experts, *info.get_weight_shape()), dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_parameter(f"{prefix}_weight", weight)
            set_weight_attrs(weight, dict(extra_weight_attrs))
            weight.shard_dim = shard_dim
            scale = torch.nn.Parameter(
                torch.empty((num_experts, 1, info.get_n()), dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter(f"{prefix}_weight_scale", scale)
            scale.shard_dim = shard_dim
            set_weight_attrs(
                scale,
                {
                    **extra_weight_attrs,
                    "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
                },
            )
            for suffix in ("i8_weight_scale", "i8_weight_zero", "bias", "input_scale"):
                setattr(layer, f"{prefix}_{suffix}", None)

    def process_weights_after_loading(self, layer):
        # NN checkpoint scales already include their INT4 dequant factor.
        for prefix, info in (("w13", self.gemm1_info), ("w2", self.gemm2_info)):
            experts = getattr(layer, f"{prefix}_weight").shape[0]
            replace_with_padded(
                layer,
                f"{prefix}_weight",
                (experts, *info.get_pad_weight_shape()),
                pad_value=0,
                log_prefix=f"[w4a8_nn][{prefix}][weight]",
            )
            replace_with_padded(
                layer,
                f"{prefix}_weight_scale",
                (experts, 1, info.get_pad_n()),
                pad_value=1,
                log_prefix=f"[w4a8_nn][{prefix}][weight_scale]",
            )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        runtime_context = getattr(self, "_runtime_context", None)
        if runtime_context is not None:
            runtime_context.assert_validation_complete()
        residual = self._valid_extra_residual(x, shared_experts_input)
        return self._apply_group_gemm(layer, x, topk_weights, topk_ids, residual)

    def select_gemm_impl(self, prepare_finalize, layer):
        from .ilu_w4a8_nn_experts import (
            IluMoEBatchedExpertsW4A8NN,
            IluMoEStandardExpertsW4A8NN,
        )

        classes = {
            FusedMoEActivationFormat.BatchedExperts: IluMoEBatchedExpertsW4A8NN,
            FusedMoEActivationFormat.Standard: IluMoEStandardExpertsW4A8NN,
        }
        experts_cls = classes.get(prepare_finalize.activation_format)
        if experts_cls is None:
            raise NotImplementedError(
                f"Unsupported NN activation format: {prepare_finalize.activation_format}"
            )
        return experts_cls(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            gemm1_info=self.gemm1_info,
            gemm2_info=self.gemm2_info,
            runner_scaling_factor=_get_ilu_output_scale(layer),
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            max_num_tokens=prepare_finalize.max_num_tokens_per_rank(),
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )

    def apply_fused_moe(self, layer, x, topk_weights, topk_ids):
        return self.apply(layer, x, topk_weights, topk_ids, None, None)

    def _apply_group_gemm(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """NN prefill/decode for EP and non-EP inputs."""
        dtype = x.dtype
        start = (
            layer.moe_config.ep_rank * layer.local_num_experts
            if layer.expert_map is not None
            else 0
        )
        end = min(start + layer.local_num_experts, layer.global_num_experts)
        rows = topk_ids.numel()
        group_layout = torch.empty(
            (end - start, 2), device=topk_ids.device, dtype=torch.int32
        )
        packed_to_expert_row = torch.empty(
            (rows,), device=topk_ids.device, dtype=torch.int32
        )
        (
            src_to_dst,
            _,
            weights_packed,
            row_offsets,
            _,
            valid_rows,
        ) = ixfops.moe_compute_token_index_ep_packed(
            topk_ids=topk_ids.to(torch.int32),
            num_experts=layer.global_num_experts,
            start_expert_id=start,
            end_expert_id=end,
            topk_weights=topk_weights,
            graph_safe=True,
            group_layout=group_layout,
            packed_to_expert_row=packed_to_expert_row,
        )
        if layer.w13_input_scale is not None:
            raise NotImplementedError("NN group GEMM requires dynamic INT8 input")
        quantized, input_scale = ixfops.scaled_int8_quant(x)
        act = torch.empty(
            (rows, self.gemm1_info.pad_k), dtype=torch.int8, device=x.device
        )
        a_scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
        ixfops.moe_expand_input_ep(
            hidden_states=quantized,
            src_to_expert_row=src_to_dst,
            expand_tokens=rows,
            topk=layer.top_k,
            output=act,
            output_scale=a_scale,
            input_scale=input_scale,
        )
        act = self._group_gemm(
            self.gemm1_info,
            act,
            a_scale,
            layer.w13_weight,
            layer.w13_weight_scale,
            group_layout,
            rows,
            dtype,
        )
        act, a_scale = self._activation_and_quant(
            layer=layer,
            input=act,
            sorted_token_ids=None,
            output_format=0,
            check_valid=True,
            expand_tokens_gpu=valid_rows,
        )
        out = self._group_gemm(
            self.gemm2_info,
            act,
            a_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            group_layout,
            rows,
            dtype,
        )
        out = unpad_tensor(out, self.gemm2_info.get_n(), contiguous=False)
        return ixfops.moe_output_reduce_sum_ep(
            input=out,
            token_row_offsets=row_offsets,
            weights=weights_packed,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
            packed_to_expert_row=packed_to_expert_row,
        )

    def _group_gemm(
        self,
        gemm_info: W4A8GroupGemmInfo,
        act: torch.Tensor,
        a_scale: torch.Tensor,
        weight: torch.Tensor,
        w_scales: torch.Tensor,
        group_layout: torch.Tensor,
        rows: int,
        dtype: torch.dtype,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ixformer ixmm NN GEMM over device-resident expert ranges."""
        if (
            gemm_info.format_str != "NN"
            or self.group_size != -1
            or dtype not in (torch.bfloat16, torch.float16)
        ):
            raise ValueError(
                "ixformer group GEMM requires NN per-channel weights, BF16/FP16 output"
            )
        if output is None:
            output = torch.empty(
                (rows, gemm_info.get_pad_n()), device=act.device, dtype=dtype
            )
        return ixfops.moe_w4a8_group_gemm_ixmm(
            a=act,
            weight=weight,
            scale_a=a_scale.reshape(-1).contiguous(),
            weight_scale=w_scales.reshape(weight.shape[0], -1),
            group_layout=group_layout,
            output=output,
            backend="ixmm",
        )
