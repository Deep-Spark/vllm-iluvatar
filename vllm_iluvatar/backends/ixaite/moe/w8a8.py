# SPDX-License-Identifier: Apache-2.0
# Source: vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/ilu_compressed_tensors_moe_w8a8_int8.py at 47fb804; independent ixaite lifecycle.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
from vllm_iluvatar.backends.ixaite.moe.decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W8A8GroupGemmInfo,
    replace_with_padded,
)

logger = init_logger(__name__)

_activation_dynamic_scaled_int8 = None
_activation_swigluoai_dynamic_scaled_int8 = None
_moe_compute_token_index = None
_moe_compute_token_index_ep = None
_moe_expand_input_dynamic_scaled_int8 = None
_moe_output_reduce_sum = None
_moe_w8a8_group_gemm = None
_moe_w8a8_group_gemv = None


# Missing grouped compute rejects W8A8 before creating checkpoint tensors.
def _bind_w8a8():
    global \
        _activation_dynamic_scaled_int8, \
        _activation_swigluoai_dynamic_scaled_int8, \
        _moe_compute_token_index, \
        _moe_compute_token_index_ep, \
        _moe_expand_input_dynamic_scaled_int8, \
        _moe_output_reduce_sum, \
        _moe_w8a8_group_gemm, \
        _moe_w8a8_group_gemv
    _moe_w8a8_group_gemm, reason = load_ixaite_python_op(
        "moe_w8a8_group_gemm", "moe_w8a8_group_gemm"
    )
    if _moe_w8a8_group_gemm is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_w8a8_group_gemm is unavailable: {reason}"
        )
    _moe_w8a8_group_gemv, reason = load_ixaite_python_op(
        "moe_w8a8_group_gemv", "moe_w8a8_group_gemv"
    )
    if _moe_w8a8_group_gemv is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_w8a8_group_gemv is unavailable: {reason}"
        )
    _activation_dynamic_scaled_int8, reason = load_ixaite_python_op(
        "activation_dynamic_scaled_int8", "activation_dynamic_scaled_int8"
    )
    if _activation_dynamic_scaled_int8 is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE activation_dynamic_scaled_int8 is unavailable: {reason}"
        )
    _activation_swigluoai_dynamic_scaled_int8, reason = load_ixaite_python_op(
        "activation_swigluoai_dynamic_scaled_int8",
        "activation_swigluoai_dynamic_scaled_int8",
    )
    _moe_compute_token_index, reason = load_ixaite_python_op(
        "moe_compute_token_index", "moe_compute_token_index"
    )
    if _moe_compute_token_index is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_compute_token_index is unavailable: {reason}"
        )
    _moe_compute_token_index_ep, reason = load_ixaite_python_op(
        "moe_compute_token_index", "moe_compute_token_index_ep"
    )
    if _moe_compute_token_index_ep is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_compute_token_index_ep is unavailable: {reason}"
        )
    _moe_expand_input_dynamic_scaled_int8, reason = load_ixaite_python_op(
        "moe_expand_input_dynamic_scaled_int8", "moe_expand_input_dynamic_scaled_int8"
    )
    if _moe_expand_input_dynamic_scaled_int8 is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_expand_input_dynamic_scaled_int8 is unavailable: {reason}"
        )
    _moe_output_reduce_sum, reason = load_ixaite_python_op(
        "moe_output_reduce_sum", "moe_output_reduce_sum"
    )
    if _moe_output_reduce_sum is None:
        raise RuntimeError(
            f"ixaite W8A8 MoE moe_output_reduce_sum is unavailable: {reason}"
        )


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


def _apply_decode_w8a8_int8_nn(
    *,
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    src_to_dst: torch.Tensor,
    expert_sizes_gpu: torch.Tensor,
    expand_tokens: int,
    shared_experts_input: torch.Tensor | None,
    dtype: torch.dtype,
    group_size: int,
    num_tokens: int,
) -> torch.Tensor:
    """Decode-only W8A8 NN path using GEMV and avoiding D2H sync."""
    act, a_scale = _moe_expand_input_dynamic_scaled_int8(
        hidden_states=x,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=layer.top_k,
        src_to_dst=src_to_dst,
        topk_ids=None,
        smooth_scales=layer.w13_input_scale,
    )

    assert act.shape[-1] == layer.w13_weight.shape[-2], (
        f"w8a8 moe gemv(NN) requires k % 64 == 0, "
        f"got act k={act.shape[-1]} != weight k={layer.w13_weight.shape[-2]}"
    )
    act = _moe_w8a8_group_gemv(
        input=act,
        weight=layer.w13_weight,
        i_scales=a_scale,
        w_scales=layer.w13_weight_scale,
        output_dtype=dtype,
        tokens_per_experts=expert_sizes_gpu,
        dst_to_src=None,
        bias=layer.w13_bias,
        format=0,
        group_size=group_size,
    )

    if layer.activation.value == "swigluoai":
        act, a_scale = _activation_swigluoai_dynamic_scaled_int8(
            input=act,
            bias=None,
            smooth_scales=layer.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=None,
        )
    else:
        act, a_scale = _activation_dynamic_scaled_int8(
            input=act,
            bias=None,
            smooth_scales=layer.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=None,
            act_type="swiglu",
        )

    assert act.shape[-1] == layer.w2_weight.shape[-2], (
        f"w8a8 moe gemv(NN) requires k % 64 == 0, "
        f"got act k={act.shape[-1]} != weight k={layer.w2_weight.shape[-2]}"
    )
    act = _moe_w8a8_group_gemv(
        input=act,
        weight=layer.w2_weight,
        i_scales=a_scale,
        w_scales=layer.w2_weight_scale,
        output_dtype=dtype,
        tokens_per_experts=expert_sizes_gpu,
        dst_to_src=sorted_token_ids,
        bias=layer.w2_bias,
        format=0,
        group_size=group_size,
    )

    return _moe_output_reduce_sum(
        input=act.view(num_tokens, layer.top_k, -1),
        topk_weight=topk_weights,
        scaling_factor=_get_ilu_output_scale(layer),
        extra_residual=shared_experts_input,
    )


class IluCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
    """W8A8 Int8 MoE quantization using compressed tensors."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        ilu_params: dict[str, Any] | None = None,
    ):
        _bind_w8a8()
        super().__init__(moe)
        ilu_params = ilu_params or {}
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        self.group_size = (
            -1 if self.weight_quant.group_size is None else self.weight_quant.group_size
        )
        self.gemm_format = ilu_params.get("gemm_format", "")
        assert self.gemm_format == "NN", (
            f"W8A8 INT8 MoE only supports NN format, but got {self.gemm_format}"
        )
        self.padding_dim = -2  # NN layout: weight is [E, k, n], k on dim -2

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

        # Select Int8 MoE backend.
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.int8
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # [1] Build GEMM metadata from the raw (unpadded) sizes. Padding is
        # deferred to process_weights_after_loading() so checkpoint loading is
        # exact (mirrors the W4A8 path). k is the contraction dim, n the output.
        self.gemm1_info = W8A8GroupGemmInfo(
            self.gemm_format,
            k=hidden_size,
            n=w13_num_shards * intermediate_size_per_partition,
        )
        self.gemm2_info = W8A8GroupGemmInfo(
            self.gemm_format,
            k=intermediate_size_per_partition,
            n=hidden_size,
        )
        w13_shape = (num_experts, *self.gemm1_info.get_weight_shape())
        w2_shape = (num_experts, *self.gemm2_info.get_weight_shape())
        logger.debug_once(
            f"[W8A8GroupGemmInfo] create gemm13 shape:{w13_shape} "
            f"gemm2 shape:{w2_shape}"
        )

        # [2] Register core expert weights (w13 / w2).
        # ILU shard_dim override: the community weight_loader defaults the w13
        # shard dim to 0 and w2 to 1, which assumes a TN-style [out, in] layout.
        # w13 is ColumnParallel (shard inter/output) and w2 is RowParallel (shard
        # inter/input). Under NN, w13 is [E, hidden, 2*inter] (inter on dim 1) and
        # w2 is [E, inter, hidden] (inter on dim 0), so the community default would
        # narrow the wrong dimension and produce a 2x size mismatch in
        # _load_w13/_load_w2. Weight scales keep the default: their trailing ",1"
        # axis leaves the sharded dim on dim 0, matching the community default.
        w13_weight = torch.nn.Parameter(
            torch.zeros(w13_shape, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w13_weight.shard_dim = 1

        w2_weight = torch.nn.Parameter(
            torch.zeros(w2_shape, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w2_weight.shard_dim = 0

        # [3] Bias (if enabled). Stored unpadded on the output (n) axis; padded
        # after loading.
        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    (num_experts, self.gemm1_info.get_n()),
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    (num_experts, self.gemm2_info.get_n()),
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias = None
            layer.w2_bias = None

        # [4] Register weight scales (w13_weight_scale / w2_weight_scale).
        # Per-channel on the output (n) axis: [num_experts, n, 1].
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                (num_experts, self.gemm1_info.get_n(), 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                (num_experts, self.gemm2_info.get_n(), 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # [5] Input scales (if static). Per-channel on the input (k) axis.
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(
                    (num_experts, self.gemm1_info.get_k()),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = torch.nn.Parameter(
                torch.ones(
                    (num_experts, self.gemm2_info.get_k()),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        def _log_prefix(branch: str, tensor_kind: str) -> str:
            return f"[w8a8][process_weights_after_loading][{branch}][{tensor_kind}]"

        # [1] Pad core expert weights to the kernel-aligned layout.
        w13_target_shape = (
            layer.w13_weight.shape[0],
            *self.gemm1_info.get_pad_weight_shape(),
        )
        w2_target_shape = (
            layer.w2_weight.shape[0],
            *self.gemm2_info.get_pad_weight_shape(),
        )
        replace_with_padded(
            layer,
            "w13_weight",
            w13_target_shape,
            pad_value=0,
            log_prefix=_log_prefix("w13", "weight"),
        )
        replace_with_padded(
            layer,
            "w2_weight",
            w2_target_shape,
            pad_value=0,
            log_prefix=_log_prefix("w2", "weight"),
        )

        # [2] Pad main weight scales (per-channel on the n axis).
        replace_with_padded(
            layer,
            "w13_weight_scale",
            (layer.w13_weight_scale.shape[0], self.gemm1_info.get_pad_n(), 1),
            pad_value=1,
            log_prefix=_log_prefix("w13", "weight_scale"),
        )
        replace_with_padded(
            layer,
            "w2_weight_scale",
            (layer.w2_weight_scale.shape[0], self.gemm2_info.get_pad_n(), 1),
            pad_value=1,
            log_prefix=_log_prefix("w2", "weight_scale"),
        )

        # [3] Pad bias (if enabled) on the n axis.
        if self.has_bias:
            replace_with_padded(
                layer,
                "w13_bias",
                (layer.w13_bias.shape[0], self.gemm1_info.get_pad_n()),
                pad_value=0,
                log_prefix=_log_prefix("w13", "bias"),
            )
            replace_with_padded(
                layer,
                "w2_bias",
                (layer.w2_bias.shape[0], self.gemm2_info.get_pad_n()),
                pad_value=0,
                log_prefix=_log_prefix("w2", "bias"),
            )

        # [4] Pad optional static input scales (per-channel on the k axis).
        if self.static_input_scales:
            replace_with_padded(
                layer,
                "w13_input_scale",
                (layer.w13_input_scale.shape[0], self.gemm1_info.get_pad_k()),
                pad_value=1,
                log_prefix=_log_prefix("w13", "input_scale"),
            )
            replace_with_padded(
                layer,
                "w2_input_scale",
                (layer.w2_input_scale.shape[0], self.gemm2_info.get_pad_k()),
                pad_value=1,
                log_prefix=_log_prefix("w2", "input_scale"),
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        # Assign the value of shared_experts_output to variable shared_experts_input for fusion
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_metadata = get_forward_context().attn_metadata
        use_ep = layer.expert_map is not None
        # unsupported ep now
        only_decode = not use_ep and _is_only_decode(attn_metadata)
        _ensure_cuda_graph_safe_dispatch(only_decode, attn_metadata)

        if use_ep:
            ep_rank = layer.moe_config.ep_rank
            start_eid = ep_rank * layer.local_num_experts
            end_eid = min(
                (ep_rank + 1) * layer.local_num_experts, layer.global_num_experts
            )

        dtype = x.dtype
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
            ) = _moe_compute_token_index_ep(
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
            ) = _moe_compute_token_index(
                topk_ids=topk_ids,
                num_experts=num_experts,
            )

        if only_decode:
            final_hidden_states = _apply_decode_w8a8_int8_nn(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                src_to_dst=src_to_dst,
                expert_sizes_gpu=expert_sizes_gpu,
                expand_tokens=expand_tokens,
                shared_experts_input=shared_experts_input,
                dtype=dtype,
                group_size=self.group_size,
                num_tokens=num_tokens,
            )
        else:
            expert_sizes_cpu = expert_sizes_gpu.cpu()
            # expand + reorder + quant
            act, a_scale = _moe_expand_input_dynamic_scaled_int8(
                hidden_states=x,
                dst_to_src=sorted_token_ids,
                dst_tokens=expand_tokens,
                topk=layer.top_k,
                src_to_dst=src_to_dst,
                topk_ids=None,
                smooth_scales=layer.w13_input_scale,
            )

            assert act.shape[-1] == layer.w13_weight.shape[self.padding_dim], (
                f"w8a8 moe gemm requires k % 64 == 0, "
                f"got act k={act.shape[-1]} != weight k="
                f"{layer.w13_weight.shape[self.padding_dim]}"
            )

            # w8a8 group gemm 1
            act = _moe_w8a8_group_gemm(
                input=act,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_cpu,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm_format,
            )

            # act + quant
            if layer.activation.value == "swigluoai":
                act, a_scale = _activation_swigluoai_dynamic_scaled_int8(
                    input=act,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                )
            else:
                act, a_scale = _activation_dynamic_scaled_int8(
                    input=act,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                    act_type="swiglu",
                )

            assert act.shape[-1] == layer.w2_weight.shape[self.padding_dim], (
                f"w8a8 moe gemm requires k % 64 == 0, "
                f"got act k={act.shape[-1]} != weight k="
                f"{layer.w2_weight.shape[self.padding_dim]}"
            )

            # w8a8 group gemm 2 + reorder
            if use_ep:
                out = torch.empty(
                    (num_tokens * layer.top_k, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
                _moe_w8a8_group_gemm(
                    input=act,
                    weight=layer.w2_weight,
                    i_scales=a_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                    output=out,
                )

                reduce_mask = src_to_dst == -1
                final_hidden_states = _moe_output_reduce_sum(
                    input=out.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    mask=reduce_mask,
                    extra_residual=shared_experts_input,
                )
            else:
                act = _moe_w8a8_group_gemm(
                    input=act,
                    weight=layer.w2_weight,
                    i_scales=a_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                )

                # mul + reduce_sum
                final_hidden_states = _moe_output_reduce_sum(
                    input=act.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    extra_residual=shared_experts_input,
                )

        return final_hidden_states
