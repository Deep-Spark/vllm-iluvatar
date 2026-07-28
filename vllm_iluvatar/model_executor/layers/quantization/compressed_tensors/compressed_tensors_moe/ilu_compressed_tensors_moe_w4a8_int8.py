# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar W4A8 MoE method for compressed-tensors models (e.g. DeepSeek-R1).

Ported from iluvatar_v0.17.0 CompressedTensorsW4A8MoEMethod.

Triggered by VLLM_W8A8_MOE_USE_W4A8=1.
The checkpoint is described as W8A8 but we run it through ixformer's
native W4A8 kernels (INT4-packed weights + dynamic INT8 activations).

Key env vars:
  - VLLM_W4A8_FORMAT: "TN" (default) or "NN"
  - VLLM_W4A8_VERSION: 2 (default) or 1
"""

import ixformer.inference.functions as ixfops  # type: ignore
import torch
import vllm.envs as envs
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    FusedMoeWeightScaleSupported,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoEExpertsModular,
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
    pad_tensor,
    replace_with_padded,
    unpad_tensor,
)

logger = init_logger(__name__)


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


def int4_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    g1_alphas: torch.Tensor | None = None,
    g2_alphas: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """Construct the modular MoE quant config for INT8 activations + INT4 weights."""
    return FusedMoEQuantConfig.make(
        torch.int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=True,
        weight_dtype="int4",
    )


class IluCompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
    """
    Iluvatar W4A8 MoE method using ixformer native kernels.

    Weights are stored as INT4-packed-in-INT8 (pack_factor=2), halving
    the weight memory compared to full INT8.  Inference uses ixformer's
    moe_w4a8_group_gemm / moe_w4a8_group_gemv kernels directly.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = (
            weight_quant.group_size if (weight_quant.group_size is not None) else -1
        )
        # if weight_quant.num_bits != 4:
        #     raise ValueError("This method only supports 4-bit weights (num_bits=4).")

        self.weight_symmetric = self.weight_quant.symmetric
        self.gemm_format = envs.VLLM_W4A8_FORMAT
        self.version = envs.VLLM_W4A8_VERSION
        assert self.version == 2, (
            "IluCompressedTensorsW4A8Int8MoEMethod only supports version 2"
        )
        assert self.gemm_format in ["TN", "NN"], (
            "IluCompressedTensorsW4A8Int8MoEMethod only supports format TN or NN"
        )
        if not envs.VLLM_WNA16_MOE_USE_W4A8 and not (
            (
                self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                or self.weight_quant.strategy == QuantizationStrategy.GROUP
            )
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        ):
            raise ValueError(
                "For INT4 pack2 Fused MoE layers, only per-channel or group "
                "scales for weights and per-token scales for activations are "
                f"supported. Found {self.weight_quant}, {self.input_quant}"
            )
        if envs.VLLM_WNA16_MOE_USE_W4A8:
            self.static_input_scales = False
        else:
            self.static_input_scales = not self.input_quant.dynamic

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_dtype = torch.int8
        base_attrs = dict(extra_weight_attrs)

        # [1] Build GEMM metadata and raw (unpadded) tensor shapes.
        self.gemm1_info = W4A8GroupGemmInfo(
            self.gemm_format,
            k=hidden_size,
            n=2 * intermediate_size_per_partition,
        )
        self.gemm2_info = W4A8GroupGemmInfo(
            self.gemm_format,
            k=intermediate_size_per_partition,
            n=hidden_size,
        )
        # Keep create_weights on original shapes so checkpoint loading is exact.
        # Padding is deferred to process_weights_after_loading().
        w13_shape = (num_experts, *self.gemm1_info.get_weight_shape())
        w2_shape = (num_experts, *self.gemm2_info.get_weight_shape())
        logger.debug_once(
            f"[W4A8GroupGemmInfo] create gemm13 shape:{w13_shape} "
            f"gemm2 shape:{w2_shape}"
        )

        # [2] Register core expert weights (w13 / w2).
        w13_weight = torch.nn.Parameter(
            torch.empty(w13_shape, dtype=weight_dtype), requires_grad=False
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, base_attrs)
        if self.gemm1_info.format_str == "NN":
            w13_weight.shard_dim = 1

        w2_weight = torch.nn.Parameter(
            torch.empty(w2_shape, dtype=weight_dtype), requires_grad=False
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, base_attrs)
        if self.gemm2_info.format_str == "NN":
            w2_weight.shard_dim = 0

        # [3] Register weight scales (w13_weight_scale / w2_weight_scale).
        scale_groups_w13 = (
            1
            if self.version == 2 or self.group_size == -1
            else self.gemm1_info.get_k() // self.group_size
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scale_groups_w13,
                self.gemm1_info.get_n(),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w13_weight_scale.shard_dim = 1

        scale_groups_w2 = (
            1
            if self.version == 2 or self.group_size == -1
            else self.gemm2_info.get_k() // self.group_size
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scale_groups_w2,
                self.gemm2_info.get_n(),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        w2_weight_scale.shard_dim = 0

        weight_scale_attrs = {
            **base_attrs,
            "quant_method": (
                FusedMoeWeightScaleSupported.CHANNEL.value
                if self.version == 2 or self.group_size == -1
                else FusedMoeWeightScaleSupported.GROUP.value
            ),
        }
        set_weight_attrs(w13_weight_scale, weight_scale_attrs)
        set_weight_attrs(w2_weight_scale, weight_scale_attrs)

        # [4] Register version-2 INT8->INT4 auxiliary tensors
        #     (i8 scales / i8 zeros).
        aux_scale_attrs = {
            **base_attrs,
            "quant_method": (
                FusedMoeWeightScaleSupported.CHANNEL.value
                if self.group_size == -1
                else FusedMoeWeightScaleSupported.GROUP.value
            ),
        }
        if self.version == 2:
            if self.group_size != -1:
                w13_i8_weight_scale = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        self.gemm1_info.get_k() // self.group_size,
                        self.gemm1_info.get_n(),
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w13_i8_weight_scale", w13_i8_weight_scale)
                w13_i8_weight_scale.shard_dim = 1
                set_weight_attrs(w13_i8_weight_scale, aux_scale_attrs)

            if not self.weight_symmetric:
                w13_i8_weight_zero = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        (
                            1
                            if self.group_size == -1
                            else self.gemm1_info.get_k() // self.group_size
                        ),
                        self.gemm1_info.get_n(),
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w13_i8_weight_zero", w13_i8_weight_zero)
                w13_i8_weight_zero.shard_dim = 1
                set_weight_attrs(w13_i8_weight_zero, aux_scale_attrs)

            if self.group_size != -1:
                w2_i8_weight_scale = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        self.gemm2_info.get_k() // self.group_size,
                        self.gemm2_info.get_n(),
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w2_i8_weight_scale", w2_i8_weight_scale)
                w2_i8_weight_scale.shard_dim = 0
                set_weight_attrs(w2_i8_weight_scale, aux_scale_attrs)

            if not self.weight_symmetric:
                w2_i8_weight_zero = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        (
                            1
                            if self.group_size == -1
                            else self.gemm2_info.get_k() // self.group_size
                        ),
                        self.gemm2_info.get_n(),
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w2_i8_weight_zero", w2_i8_weight_zero)
                w2_i8_weight_zero.shard_dim = 0
                set_weight_attrs(w2_i8_weight_zero, aux_scale_attrs)

        if not (self.version == 2 and self.group_size != -1):
            layer.w13_i8_weight_scale = None
            layer.w2_i8_weight_scale = None
        if not (self.version == 2 and not self.weight_symmetric):
            layer.w13_i8_weight_zero = None
            layer.w2_i8_weight_zero = None

        # [5] Register bias and optional static input scales.
        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    self.gemm1_info.get_n(),
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, base_attrs)

            w2_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    self.gemm2_info.get_n(),
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, base_attrs)
        else:
            layer.w13_bias = None
            layer.w2_bias = None

        if self.static_input_scales:
            input_scale_attrs = {
                **base_attrs,
                "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
            }
            w13_input_scale = torch.nn.Parameter(
                torch.empty(num_experts, self.gemm1_info.get_k(), dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, input_scale_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    self.gemm2_info.get_k(),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, input_scale_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def _log_prefix(branch: str, tensor_kind: str) -> str:
            return f"[w4a8][process_weights_after_loading][{branch}][{tensor_kind}]"

        # [1] Pad core expert weights.
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

        # [2] Pad main weight scales.
        replace_with_padded(
            layer,
            "w13_weight_scale",
            (
                layer.w13_weight_scale.shape[0],
                (
                    1
                    if self.version == 2 or self.group_size == -1
                    else self.gemm1_info.get_pad_k() // self.group_size
                ),
                self.gemm1_info.get_pad_n(),
            ),
            pad_value=1,
            log_prefix=_log_prefix("w13", "weight_scale"),
        )
        replace_with_padded(
            layer,
            "w2_weight_scale",
            (
                layer.w2_weight_scale.shape[0],
                (
                    1
                    if self.version == 2 or self.group_size == -1
                    else self.gemm2_info.get_pad_k() // self.group_size
                ),
                self.gemm2_info.get_pad_n(),
            ),
            pad_value=1,
            log_prefix=_log_prefix("w2", "weight_scale"),
        )

        # [3] Pad version-2 auxiliary i8 scales / zeros.
        if self.version == 2 and self.group_size != -1:
            replace_with_padded(
                layer,
                "w13_i8_weight_scale",
                (
                    layer.w13_i8_weight_scale.shape[0],
                    self.gemm1_info.get_pad_k() // self.group_size,
                    self.gemm1_info.get_pad_n(),
                ),
                pad_value=1,
                log_prefix=_log_prefix("w13", "i8_weight_scale"),
            )
            replace_with_padded(
                layer,
                "w2_i8_weight_scale",
                (
                    layer.w2_i8_weight_scale.shape[0],
                    self.gemm2_info.get_pad_k() // self.group_size,
                    self.gemm2_info.get_pad_n(),
                ),
                pad_value=1,
                log_prefix=_log_prefix("w2", "i8_weight_scale"),
            )

        if self.version == 2 and not self.weight_symmetric:
            replace_with_padded(
                layer,
                "w13_i8_weight_zero",
                (
                    layer.w13_i8_weight_zero.shape[0],
                    (
                        1
                        if self.group_size == -1
                        else self.gemm1_info.get_pad_k() // self.group_size
                    ),
                    self.gemm1_info.get_pad_n(),
                ),
                pad_value=0,
                log_prefix=_log_prefix("w13", "i8_weight_zero"),
            )
            replace_with_padded(
                layer,
                "w2_i8_weight_zero",
                (
                    layer.w2_i8_weight_zero.shape[0],
                    (
                        1
                        if self.group_size == -1
                        else self.gemm2_info.get_pad_k() // self.group_size
                    ),
                    self.gemm2_info.get_pad_n(),
                ),
                pad_value=0,
                log_prefix=_log_prefix("w2", "i8_weight_zero"),
            )

        # [4] Pad bias (if enabled).
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

        # [5] Pad optional static input scales.
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

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            g1_alphas=layer.w13_i8_weight_scale,
            g2_alphas=layer.w2_i8_weight_scale,
            w1_zp=layer.w13_i8_weight_zero,
            w2_zp=layer.w2_i8_weight_zero,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shared_experts_input = self._valid_extra_residual(
            x=x,
            shared_experts_input=shared_experts_input,
        )
        if shared_experts_input is None:
            shared_experts = None
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        use_ep = layer.expert_map is not None
        only_decode = _is_only_decode(attn_metadata)
        _ensure_cuda_graph_safe_dispatch(only_decode, attn_metadata)

        if use_ep:
            return self._apply_ep(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
                run_decode=only_decode,
            )

        # ---- Decode-optimized path ----
        if only_decode:
            return self._apply_decode(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        # ---- General path (prefill or mixed) ----
        return self._apply_general(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

    def _apply_ep(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        run_decode: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ep_rank = layer.moe_config.ep_rank
        start_eid = ep_rank * layer.local_num_experts
        end_eid = min(
            (ep_rank + 1) * layer.local_num_experts,
            layer.global_num_experts,
        )

        dtype = x.dtype
        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts
        hidden_size = x.shape[1]

        if run_decode:
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
                expand_tokens,
                dense_expert_sizes_gpu,
            ) = ixfops.moe_compute_token_index_ep(
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_eid,
                end_expert_id=end_eid,
                graph_safe=run_decode,
                return_dense_expert_sizes=True,
            )
        else:
            dense_expert_sizes_gpu = None
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
                graph_safe=run_decode,
            )

        # only prefill stage has expert_sizes_cpu info
        if not run_decode and expert_sizes_cpu.sum() == 0:
            return torch.zeros(
                (num_tokens, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )

        # for decode, we need to expand the tokens to the full number of tokens
        if run_decode:
            expand_tokens = num_tokens * layer.top_k

        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
            output_format=1 if run_decode else 0,
            check_valid=run_decode,
        )

        if run_decode:
            act = ixfops.moe_w4a8_group_gemv(
                input=act,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                w_i8scales=layer.w13_i8_weight_scale,
                w_i8zeros=layer.w13_i8_weight_zero,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm1_info.format,
                group_size=self.group_size,
                dense_tokens_per_experts=dense_expert_sizes_gpu,
            )
        else:
            act = ixfops.moe_w4a8_group_gemm(
                input=act,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_cpu,
                w_i8scales=layer.w13_i8_weight_scale,
                w_i8zeros=layer.w13_i8_weight_zero,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm1_info.format,
                group_size=self.group_size,
                version=self.version,
            )

        act, a_scale = self._activation_and_quant(
            layer=layer,
            input=act,
            sorted_token_ids=sorted_token_ids,
            output_format=1 if run_decode else 0,
            check_valid=run_decode,
        )
        out = torch.empty(
            (num_tokens * layer.top_k, self.gemm2_info.get_pad_n()),
            device=x.device,
            dtype=x.dtype,
        )
        if run_decode:
            ixfops.moe_w4a8_group_gemv(
                input=act,
                weight=layer.w2_weight,
                i_scales=a_scale,
                w_scales=layer.w2_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                w_i8scales=layer.w2_i8_weight_scale,
                w_i8zeros=layer.w2_i8_weight_zero,
                dst_to_src=sorted_token_ids,
                bias=layer.w2_bias,
                format=self.gemm2_info.format,
                group_size=self.group_size,
                output=out,
                dense_tokens_per_experts=dense_expert_sizes_gpu,
            )
        else:
            ixfops.moe_w4a8_group_gemm(
                input=act,
                weight=layer.w2_weight,
                i_scales=a_scale,
                w_scales=layer.w2_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_cpu,
                w_i8scales=layer.w2_i8_weight_scale,
                w_i8zeros=layer.w2_i8_weight_zero,
                dst_to_src=sorted_token_ids,
                bias=layer.w2_bias,
                format=self.gemm2_info.format,
                group_size=self.group_size,
                version=self.version,
                output=out,
            )
        out = unpad_tensor(out, self.gemm2_info.get_n(), contiguous=False)

        reduce_mask = src_to_dst == -1
        return ixfops.moe_output_reduce_sum(
            input=out.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            mask=reduce_mask,
            extra_residual=shared_experts_input,
        )

    def _apply_decode(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts

        expand_tokens = num_tokens * layer.top_k
        (
            src_to_dst,
            sorted_token_ids,
            expert_sizes_gpu,
            dense_expert_sizes_gpu,
        ) = ixfops.moe_compute_token_index_for_decode(
            topk_ids=topk_ids,
            num_experts=num_experts,
            return_dense_expert_sizes=True,
        )

        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
            output_format=1,
        )

        act = ixfops.moe_w4a8_group_gemv(
            input=act,
            weight=layer.w13_weight,
            i_scales=a_scale,
            w_scales=layer.w13_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_gpu,
            w_i8scales=layer.w13_i8_weight_scale,
            w_i8zeros=layer.w13_i8_weight_zero,
            dst_to_src=None,
            bias=layer.w13_bias,
            format=self.gemm1_info.format,
            group_size=self.group_size,
            dense_tokens_per_experts=dense_expert_sizes_gpu,
        )

        act, a_scale = self._activation_and_quant(
            layer=layer,
            input=act,
            sorted_token_ids=sorted_token_ids,
            output_format=1,
        )

        act = ixfops.moe_w4a8_group_gemv(
            input=act,
            weight=layer.w2_weight,
            i_scales=a_scale,
            w_scales=layer.w2_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_gpu,
            w_i8scales=layer.w2_i8_weight_scale,
            w_i8zeros=layer.w2_i8_weight_zero,
            dst_to_src=sorted_token_ids,
            bias=layer.w2_bias,
            format=self.gemm2_info.format,
            group_size=self.group_size,
            dense_tokens_per_experts=dense_expert_sizes_gpu,
        )
        act = unpad_tensor(act, self.gemm2_info.get_n(), contiguous=False)

        return ixfops.moe_output_reduce_sum(
            input=act.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
        )

    def _apply_general(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts

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

        expert_sizes_cpu = expert_sizes_gpu.cpu()
        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
        )

        act = ixfops.moe_w4a8_group_gemm(
            input=act,
            weight=layer.w13_weight,
            i_scales=a_scale,
            w_scales=layer.w13_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_cpu,
            w_i8scales=layer.w13_i8_weight_scale,
            w_i8zeros=layer.w13_i8_weight_zero,
            dst_to_src=None,
            bias=layer.w13_bias,
            format=self.gemm1_info.format,
            group_size=self.group_size,
            version=self.version,
        )

        act, a_scale = self._activation_and_quant(
            layer=layer,
            input=act,
            sorted_token_ids=sorted_token_ids,
        )

        act = ixfops.moe_w4a8_group_gemm(
            input=act,
            weight=layer.w2_weight,
            i_scales=a_scale,
            w_scales=layer.w2_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_cpu,
            w_i8scales=layer.w2_i8_weight_scale,
            w_i8zeros=layer.w2_i8_weight_zero,
            dst_to_src=sorted_token_ids,
            bias=layer.w2_bias,
            format=self.gemm2_info.format,
            group_size=self.group_size,
            version=self.version,
        )
        act = unpad_tensor(act, self.gemm2_info.get_n(), contiguous=False)

        result = ixfops.moe_output_reduce_sum(
            input=act.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
        )
        return result

    # expand + quant + pad
    def _expand_and_quant(
        self,
        layer,
        input,
        expand_tokens,
        sorted_token_ids,
        src_to_dst,
        output_format=0,
        check_valid=False,
    ):
        output, scale = ixfops.moe_expand_input_dynamic_scaled_int8(
            hidden_states=input,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
            topk_ids=None,
            smooth_scales=layer.w13_input_scale,
            output_format=output_format,
            check_valid=check_valid,
        )

        if output.shape[-1] != self.gemm1_info.pad_k:
            output = pad_tensor(output, self.gemm1_info.pad_k)

        return output, scale

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        experts_kwargs = dict(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            quant_mode="w4a8",
            gemm1_info=self.gemm1_info,
            gemm2_info=self.gemm2_info,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
            runner_scaling_factor=_get_ilu_output_scale(layer),
            format=self.gemm1_info.format_str,
            version=self.version,
            group_size=self.group_size,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            max_num_tokens=prepare_finalize.max_num_tokens_per_rank(),
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            from .ilu_moe_batch_experts_w4a8 import IluMoEBatchedExpertsW4A8

            return IluMoEBatchedExpertsW4A8(**experts_kwargs)

        if prepare_finalize.activation_format == FusedMoEActivationFormat.Standard:
            from .ilu_moe_standard_experts_w4a8 import IluMoEStandardExpertsW4A8

            return IluMoEStandardExpertsW4A8(**experts_kwargs)

        raise NotImplementedError(
            f"Unsupported activation format: {prepare_finalize.activation_format}"
        )

    @staticmethod
    def _valid_extra_residual(
        x: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if shared_experts_input is None:
            return None
        if shared_experts_input.shape[0] == x.shape[0]:
            return shared_experts_input
        # Naive DP/EP gathers routed inputs across DP ranks before the W4A8
        # kernel, while shared experts are still combined separately per rank.
        return None

    # act + quant + pad
    def _activation_and_quant(
        self,
        layer,
        input,
        sorted_token_ids,
        output_format=None,
        check_valid=False,
    ):
        # unpad
        input = unpad_tensor(input, self.gemm1_info.get_n(), contiguous=True)

        # ---- DeepSeek-V4 swiglu + clamp(limit) preprocessing -----------------
        # V4's MoE expert activation is:
        #   gate = clamp(x[..., :d], max=swiglu_limit)
        #   up   = clamp(x[..., d:], min=-swiglu_limit, max=+swiglu_limit)
        #   y    = silu(gate) * up
        # i.e. SiluAndMulWithClamp followed by dynamic int8 quant.
        #
        # ixformer's ``activation_dynamic_scaled_int8`` doesn't clamp gate/up;
        # ``activation_swigluoai_dynamic_scaled_int8`` clamps but uses the
        # OpenAI formula ``(clamp_up + 1) * sigmoid(alpha * clamp_gate) *
        # clamp_gate`` which is *not* the DeepSeek-V4 formula. When
        # ``swiglu_limit`` is set and the activation is silu/swiglu, we use
        # the fused ``swiglu_limit_dynamic_scaled_int8`` op that bakes the
        # clamp into the same kernel as the dynamic int8 quant.
        swiglu_limit = getattr(layer, "swiglu_limit", None)
        act_value = layer.activation.value
        use_fused_swiglu_limit = swiglu_limit is not None and act_value in (
            "silu",
            "swiglu",
        )

        kwargs = dict(
            input=input,
            smooth_scales=layer.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=None,
        )
        if output_format is not None:
            kwargs["output_format"] = output_format
        kwargs["check_valid"] = check_valid

        # Explicit dispatch with whitelist — unknown activations must raise
        # rather than silently falling back to ``swiglu`` and producing wrong
        # results. ixformer's W4A8 MoE path supports exactly these gated
        # activations:
        #   - "silu" / "swiglu" : silu_and_mul; when swiglu_limit is set
        #                         we use the fused clamp+swiglu+quant op,
        #                         otherwise the plain swiglu kernel.
        #   - "swiglustep"      : silu(gate) * (up + 1)
        #   - "swigluoai"       : OpenAI/GPT-OSS variant with built-in clamp
        # Other ``MoEActivation`` values (gelu, relu2, *_no_mul) are NOT
        # implemented by the W4A8 kernel.
        if use_fused_swiglu_limit:
            output, scale = ixfops.swiglu_limit_dynamic_scaled_int8(
                limit=float(swiglu_limit), **kwargs
            )
        elif act_value == "swigluoai":
            kwargs["bias"] = None
            output, scale = ixfops.activation_swigluoai_dynamic_scaled_int8(**kwargs)
        elif act_value == "swiglustep":
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglustep"
            output, scale = ixfops.activation_dynamic_scaled_int8(**kwargs)
        elif act_value in ("silu", "swiglu"):
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglu"
            output, scale = ixfops.activation_dynamic_scaled_int8(**kwargs)
        else:
            raise ValueError(
                f"IluCompressedTensorsW4A8Int8MoEMethod: unsupported MoE "
                f"activation {act_value!r}. Supported activations: "
                f"'silu', 'swiglu', 'swiglustep', 'swigluoai'."
            )

        if output.shape[-1] != self.gemm2_info.pad_k:
            output = pad_tensor(output, self.gemm2_info.pad_k)

        return output, scale
