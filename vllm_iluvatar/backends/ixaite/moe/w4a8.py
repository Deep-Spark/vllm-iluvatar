# SPDX-License-Identifier: Apache-2.0
# Source: vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/ilu_compressed_tensors_moe_w4a8_int8.py at 47fb804; independent ixaite lifecycle.
"""
Iluvatar W4A8 MoE method for compressed-tensors models.

Selected when a config group has an ``iluvatar`` block and Compressed-Tensors
describes 4-bit weights with 8-bit activations. Layout is taken from
``iluvatar.gemm_format`` and ``iluvatar.pack_version``.
"""

from typing import Any

import torch
import vllm.envs as envs
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
from vllm_iluvatar.backends.ixaite.moe.decode_metadata import (
    _is_only_decode,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
    hidden_padding,
    replace_with_padded,
    unpad_tensor,
)

logger = init_logger(__name__)

_moe_w4a8_group_gemm = None
_moe_w4a8_group_gemv = None
_activation_dynamic_scaled_int8 = None
_moe_compute_token_index = None
_moe_compute_token_index_for_decode = None
_moe_compute_token_index_ep = None
_moe_expand_input_dynamic_scaled_int8 = None
_moe_output_reduce_sum = None


# Bind complete W4A8 execution before weights or graph workspaces exist.
def configure_moe_providers():
    global \
        _moe_w4a8_group_gemm, \
        _moe_w4a8_group_gemv, \
        _activation_dynamic_scaled_int8, \
        _moe_compute_token_index, \
        _moe_compute_token_index_for_decode, \
        _moe_compute_token_index_ep, \
        _moe_expand_input_dynamic_scaled_int8, \
        _moe_output_reduce_sum
    _moe_w4a8_group_gemm, reason = load_ixaite_python_op(
        "moe_w4a8_group_gemm", "moe_w4a8_group_gemm"
    )
    if _moe_w4a8_group_gemm is None:
        raise RuntimeError(f"ixaite moe_w4a8_group_gemm is unavailable: {reason}")
    _moe_w4a8_group_gemv, reason = load_ixaite_python_op(
        "moe_w4a8_group_gemv", "moe_w4a8_group_gemv"
    )
    if _moe_w4a8_group_gemv is None:
        raise RuntimeError(f"ixaite moe_w4a8_group_gemv is unavailable: {reason}")
    _activation_dynamic_scaled_int8, reason = load_ixaite_python_op(
        "activation_dynamic_scaled_int8", "activation_dynamic_scaled_int8"
    )
    if _activation_dynamic_scaled_int8 is None:
        raise RuntimeError(
            f"ixaite activation_dynamic_scaled_int8 is unavailable: {reason}"
        )
    _moe_compute_token_index, reason = load_ixaite_python_op(
        "moe_compute_token_index", "moe_compute_token_index"
    )
    if _moe_compute_token_index is None:
        raise RuntimeError(f"ixaite moe_compute_token_index is unavailable: {reason}")
    _moe_compute_token_index_for_decode, reason = load_ixaite_python_op(
        "moe_compute_token_index", "moe_compute_token_index_for_decode"
    )
    if _moe_compute_token_index_for_decode is None:
        raise RuntimeError(
            f"ixaite moe_compute_token_index_for_decode is unavailable: {reason}"
        )
    _moe_compute_token_index_ep, reason = load_ixaite_python_op(
        "moe_compute_token_index", "moe_compute_token_index_ep"
    )
    if _moe_compute_token_index_ep is None:
        raise RuntimeError(
            f"ixaite moe_compute_token_index_ep is unavailable: {reason}"
        )
    _moe_expand_input_dynamic_scaled_int8, reason = load_ixaite_python_op(
        "moe_expand_input_dynamic_scaled_int8", "moe_expand_input_dynamic_scaled_int8"
    )
    if _moe_expand_input_dynamic_scaled_int8 is None:
        raise RuntimeError(
            f"ixaite moe_expand_input_dynamic_scaled_int8 is unavailable: {reason}"
        )

    _moe_output_reduce_sum, reason = load_ixaite_python_op(
        "moe_output_reduce_sum", "moe_output_reduce_sum"
    )
    if _moe_output_reduce_sum is None:
        raise RuntimeError(f"ixaite moe_output_reduce_sum is unavailable: {reason}")


# Output scaling belongs after routed expert reduction.
def _get_ilu_output_scale(layer):
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
    W4A8 MoE method with independently bound ixaite kernels.

    Weights are stored as INT4-packed-in-INT8 (pack_factor=2), halving
    the weight memory compared to full INT8.  Inference uses ixaite
    moe_w4a8_group_gemm / moe_w4a8_group_gemv kernels directly.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        ilu_params: dict[str, Any] | None = None,
    ):
        if ilu_params is None or (
            ilu_params.get("gemm_format"),
            ilu_params.get("pack_version"),
        ) != ("TN", 2):
            raise RuntimeError("ixaite W4A8 MoE requires TN, pack_version=2")
        if (
            weight_quant.strategy != QuantizationStrategy.CHANNEL
            or weight_quant.group_size not in (None, -1)
            or not weight_quant.symmetric
            or not input_quant.symmetric
        ):
            raise RuntimeError(
                "ixaite W4A8 MoE requires symmetric channel weight scales"
            )
        if envs.VLLM_ILU_MOE_EP_LL_MOE_ALGO == "FUSED":
            raise RuntimeError("ixaite does not support vendor fused MoE prefill")
        if _moe_w4a8_group_gemm is None:
            configure_moe_providers()
        super().__init__(moe)
        ilu_params = ilu_params or {}
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
        self.gemm_format = ilu_params.get("gemm_format", "")
        self.version = ilu_params.get("pack_version", -1)
        assert self.version == 2, (
            "IluCompressedTensorsW4A8Int8MoEMethod only supports version 2"
        )
        assert self.gemm_format in ["TN", "NN"], (
            "IluCompressedTensorsW4A8Int8MoEMethod only supports format TN or NN"
        )
        if not (
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
        self.static_input_scales = not self.input_quant.dynamic

        # Reuse decode outputs across graph capture and replay.
        self._decode_ws: dict[tuple, torch.Tensor] = {}
        # Prefill token-index buffers + async histogram D2H (overlap expand).
        self._prefill_index_ws: dict[str, torch.Tensor] | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if getattr(layer, "swiglu_limit", None) is not None or layer.activation.value not in (
            "silu",
            "swiglu",
            "swiglustep",
            "swigluoai",
            "swigluoai_uninterleave",
        ):
            raise RuntimeError(
                "ixaite W4A8 MoE does not support this activation contract"
            )
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
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        use_ep = layer.expert_map is not None
        only_decode = _is_only_decode(attn_metadata, capture_fallback=False)

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

    # The vendor fused pipeline has no ixaite equivalent.
    def apply_fused_moe(self, *args, **kwargs):
        raise RuntimeError("ixaite does not support vendor fused MoE prefill")

    def _prefill_token_index_workspace(
        self, topk_ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse index/histogram buffers; grow when the prefill token count rises."""
        numel = int(topk_ids.numel())
        ws = self._prefill_index_ws
        need = (
            ws is None
            or int(ws["src_to_dst"].numel()) < numel
            or int(ws["expert_sizes_gpu"].numel()) != num_experts
            or ws["src_to_dst"].device != topk_ids.device
        )
        if need:
            src_to_dst = topk_ids.new_empty([numel])
            sorted_token_ids = torch.empty_like(src_to_dst)
            expert_sizes_gpu = topk_ids.new_empty([num_experts])
            self._prefill_index_ws = {
                "src_to_dst": src_to_dst,
                "sorted_token_ids": sorted_token_ids,
                "expert_sizes_gpu": expert_sizes_gpu,
            }
            ws = self._prefill_index_ws
        return (
            ws["src_to_dst"][:numel],
            ws["sorted_token_ids"][:numel],
            ws["expert_sizes_gpu"],
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

        # Fixed capacity and GPU counts support empty experts without host synchronization.
        (
            src_to_dst,
            sorted_token_ids,
            expert_sizes_gpu,
            _,
            _,
            dense_expert_sizes_gpu,
        ) = _moe_compute_token_index_ep(
            topk_ids=topk_ids,
            num_experts=num_experts,
            start_expert_id=start_eid,
            end_expert_id=end_eid,
            graph_safe=True,
            return_dense_expert_sizes=True,
        )
        expand_tokens = num_tokens * layer.top_k

        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
            output_format=0,
            check_valid=True,
        )

        if run_decode:
            act = _moe_w4a8_group_gemv(
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
                output_format=0,
                check_valid=True,
                topk_ids=topk_ids,
            )
        else:
            act = _moe_w4a8_group_gemm(
                input=act,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                tokens_per_experts_gpu=expert_sizes_gpu,
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
                output_format=0,
                check_valid=False,
                topk_ids=topk_ids,
            )
        out = torch.empty(
            (num_tokens * layer.top_k, self.gemm2_info.get_pad_n()),
            device=x.device,
            dtype=x.dtype,
        )
        if run_decode:
            _moe_w4a8_group_gemv(
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
            _moe_w4a8_group_gemm(
                input=act,
                weight=layer.w2_weight,
                i_scales=a_scale,
                w_scales=layer.w2_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                tokens_per_experts_gpu=expert_sizes_gpu,
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
        return _moe_output_reduce_sum(
            input=out.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            mask=reduce_mask,
            extra_residual=shared_experts_input,
        )

    def _decode_workspace(
        self,
        tag: str,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (tag, shape, str(device), dtype)
        buf = self._decode_ws.get(key)
        if buf is None:
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._decode_ws[key] = buf
        return buf

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
        ) = _moe_compute_token_index_for_decode(
            topk_ids=topk_ids,
            num_experts=num_experts,
            return_dense_expert_sizes=True,
        )
        # ixformer gemv requires a device token table. A D2H here is also
        # illegal during CUDA graph capture; ixaite copies to host inside
        # the op if it needs CPU counts.

        # ixaite GEMV consumes row-major INT8, unlike the vendor format-1 shuffle.
        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
            output_format=0,
        )

        gemv1_out = self._decode_workspace(
            "gemv1",
            (expand_tokens, layer.w13_weight.shape[1]),
            device=x.device,
            dtype=dtype,
        )
        act = _moe_w4a8_group_gemv(
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
            output=gemv1_out,
            dense_tokens_per_experts=dense_expert_sizes_gpu,
        )
        act, a_scale = self._activation_and_quant(
            layer=layer,
            input=act,
            sorted_token_ids=sorted_token_ids,
            output_format=0,
            topk_ids=topk_ids,
        )

        gemv2_out = self._decode_workspace(
            "gemv2",
            (expand_tokens, layer.w2_weight.shape[1]),
            device=x.device,
            dtype=dtype,
        )
        act = _moe_w4a8_group_gemv(
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
            output=gemv2_out,
            dense_tokens_per_experts=dense_expert_sizes_gpu,
        )
        act = unpad_tensor(act, self.gemm2_info.get_n(), contiguous=False)

        return _moe_output_reduce_sum(
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
        src_to_dst, sorted_token_ids, expert_sizes_gpu = (
            self._prefill_token_index_workspace(topk_ids, num_experts)
        )
        _moe_compute_token_index(
            topk_ids=topk_ids,
            num_experts=num_experts,
            src_dst=src_to_dst,
            dst_src=sorted_token_ids,
            expert_sizes_gpu=expert_sizes_gpu,
        )
        # The device histogram avoids D2H and remains valid during capture.
        act, a_scale = self._expand_and_quant(
            layer=layer,
            input=x,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
        )

        act = _moe_w4a8_group_gemm(
            input=act,
            weight=layer.w13_weight,
            i_scales=a_scale,
            w_scales=layer.w13_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_gpu,
            tokens_per_experts_gpu=expert_sizes_gpu,
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
            topk_ids=topk_ids,
        )

        act = _moe_w4a8_group_gemm(
            input=act,
            weight=layer.w2_weight,
            i_scales=a_scale,
            w_scales=layer.w2_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_gpu,
            tokens_per_experts_gpu=expert_sizes_gpu,
            w_i8scales=layer.w2_i8_weight_scale,
            w_i8zeros=layer.w2_i8_weight_zero,
            dst_to_src=sorted_token_ids,
            bias=layer.w2_bias,
            format=self.gemm2_info.format,
            group_size=self.group_size,
            version=self.version,
        )
        act = unpad_tensor(act, self.gemm2_info.get_n(), contiguous=False)

        result = _moe_output_reduce_sum(
            input=act.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
        )
        return result

    # expand + quant + pad
    # The existing ixaite path uses separate W13, activation and quantization.

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
        output, scale = _moe_expand_input_dynamic_scaled_int8(
            hidden_states=input,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
            topk_ids=None,
            smooth_scales=layer.w13_input_scale,
            output_format=output_format,
            output_padding=hidden_padding(input.shape[-1], self.gemm1_info.pad_k),
            check_valid=check_valid,
        )

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
            raise RuntimeError("ixaite does not support batched expert dispatch")

        if prepare_finalize.activation_format == FusedMoEActivationFormat.Standard:
            from .standard_w4a8 import IluMoEStandardExpertsW4A8

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
        expand_tokens_gpu=None,
        topk_ids=None,
    ):
        # unpad
        input = unpad_tensor(input, self.gemm1_info.get_n(), contiguous=True)

        act_value = layer.activation.value

        kwargs = dict(
            input=input,
            smooth_scales=layer.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=topk_ids,
        )
        if output_format is not None:
            kwargs["output_format"] = output_format
        kwargs["check_valid"] = check_valid
        # SwiGLU halves gemm1.N; fuse K-align pad into the quant kernel
        # instead of a follow-up pad_tensor launch.
        kwargs["output_padding"] = hidden_padding(
            self.gemm1_info.get_n() // 2, self.gemm2_info.pad_k
        )

        # Unsupported clamp contracts were rejected before weight creation.
        # ixaite activation_dynamic_scaled_int8 swigluoai reads packed
        # [gates; ups] directly (MiniMax swigluoai_uninterleave).
        if act_value == "swiglustep":
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglustep"
            if expand_tokens_gpu is not None:
                kwargs["expand_tokens_gpu"] = expand_tokens_gpu
            output, scale = _activation_dynamic_scaled_int8(**kwargs)
        elif act_value in ("swigluoai", "swigluoai_uninterleave"):
            kwargs["bias"] = None
            kwargs["act_type"] = "swigluoai"
            if expand_tokens_gpu is not None:
                kwargs["expand_tokens_gpu"] = expand_tokens_gpu
            output, scale = _activation_dynamic_scaled_int8(**kwargs)
        elif act_value in ("silu", "swiglu"):
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglu"
            if expand_tokens_gpu is not None:
                kwargs["expand_tokens_gpu"] = expand_tokens_gpu
            output, scale = _activation_dynamic_scaled_int8(**kwargs)
        else:
            raise ValueError(
                f"IluCompressedTensorsW4A8Int8MoEMethod: unsupported MoE "
                f"activation {act_value!r}. Supported activations: "
                f"'silu', 'swiglu', 'swiglustep', 'swigluoai', "
                f"'swigluoai_uninterleave'."
            )

        return output, scale
