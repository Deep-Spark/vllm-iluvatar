# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPTQ / GPTQ-Marlin quantization config.

OOT plugin version of the fork-local ``vllm/.../quantization/ilu_auto_gptq.py``.
The module is auto-imported by ``vllm_iluvatar.ilu_register`` and registers via
the official ``@register_quantization_config`` decorator.
"""

from copy import deepcopy

import torch
from vllm import _custom_ops as ops

import vllm.model_executor.layers.fused_moe  # noqa
from vllm.logger import init_logger
from vllm_iluvatar.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)

from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types
from vllm.forward_context import get_forward_context

import ixformer.inference.functions as ixfops

logger = init_logger(__name__)


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


#[B,K//8,N] ->[B,K,N]
# less memmory
def unpack_k_batch_opt(packed_w: torch.Tensor, num_bits: int = 4, chunk_size: int = 2) -> torch.Tensor:
    """
    Memory-efficient unpacking for 3D tensor.
    Converts [B, K // pack_factor, N] int32 tensor → [B, K, N] int8 tensor,
    without broadcasting huge intermediate tensors (avoids OOM).

    Args:
        packed_w: torch.int32 tensor of shape [B, K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4 or 2).
        chunk_size: How many bit groups to unpack at once (tradeoff between speed and memory).

    Returns:
        unpacked: torch.int8 tensor of shape [B, K, N].
    """
    B, k_packed, N = packed_w.shape
    pack_factor = 32 // num_bits
    K = k_packed * pack_factor
    mask = (1 << num_bits) - 1

    # Allocate output tensor once
    unpacked = torch.empty((B, K, N), dtype=torch.int8, device=packed_w.device)

    # Process bit chunks iteratively to save memory
    for i in range(0, pack_factor, chunk_size):
        # Precompute shifts for this chunk
        shift_vals = num_bits * torch.arange(i, min(i + chunk_size, pack_factor), device=packed_w.device)
        # [chunk_size, 1, 1, 1]
        shifts = shift_vals.view(-1, 1, 1, 1)
        # Compute small chunk only
        chunk = ((packed_w.unsqueeze(0) >> shifts) & mask).to(torch.int8)

        # chunk: [chunk_size, B, k_packed, N]
        # write into output
        for j in range(chunk.shape[0]):
            unpacked[:, (i + j)::pack_factor, :] = chunk[j]

        del chunk  # release memory early

    return unpacked

# more memmory
def unpack_k_batch(packed_w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    Efficient vectorized unpacking for 3D tensor (batch version).
    Converts [B, K // pack_factor, N] int32 tensor → [B, K, N] int8 tensor.

    Args:
        packed_w: torch.int32 tensor of shape [B, K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4).

    Returns:
        unpacked: torch.int8 tensor of shape [B, K, N].
    """
    B, k_packed, n = packed_w.shape
    pack_factor = 32 // num_bits
    k = k_packed * pack_factor

    mask = (1 << num_bits) - 1

    # [pack_factor, 1, 1, 1]
    shifts = (num_bits * torch.arange(pack_factor, device=packed_w.device)).view(-1, 1, 1, 1)

    # [1, B, k_packed, N]
    packed_expanded = packed_w.unsqueeze(0)

    # Extract each group of num_bits using bitwise ops
    unpacked_groups = ((packed_expanded >> shifts) & mask).to(torch.int8)

    # [pack_factor, B, k_packed, N] → [B, K, N]
    unpacked = unpacked_groups.permute(1, 2, 0, 3).reshape(B, k, n)

    return unpacked


#[B,K,N] ->[B,K,N//8]
# less memmory
def pack_n_batch_opt(x: torch.Tensor, pack_num: int = 8, order_map=None, chunk_size: int = 2) -> torch.Tensor:
    """
    Memory-efficient batch packing with correct bit order.
    [B, K, N] int4 -> [B, K, N//pack_num] int32.
    """
    B, K, N = x.shape
    assert N % pack_num == 0, "N must be divisible by pack_num"
    cols = N // pack_num
    unit = 32 // pack_num

    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    shifts = unit * torch.arange(pack_num, device=x.device)  # always 0..unit*(pack_num-1)
    packed = torch.zeros((B, K, cols), dtype=torch.int32, device=x.device)
    x_reshape = x.view(B, K, cols, pack_num) & 0xF

    # process in chunks for memory efficiency
    for start in range(0, pack_num, chunk_size):
        end = min(start + chunk_size, pack_num)
        idx_chunk = order_map[start:end]
        shift_chunk = shifts[start:end]

        vals = torch.gather(x_reshape, 3, idx_chunk.view(1,1,1,-1).expand(B,K,cols,-1)).to(torch.int32)
        for j in range(vals.shape[-1]):
            packed.add_(vals[..., j] << shift_chunk[j])

    return packed
## more memmory
def pack_n_batch(x: torch.Tensor, pack_num: int = 8, order_map=None) -> torch.Tensor:
    """
    Efficient vectorized batch packing: [B, K, N] int4 -> [B, K, N//pack_num] int32.

    Args:
        x: torch.int32 tensor of shape [B, K, N], each element 0-15 (int4).
        pack_num: Number of 4-bit elements per packed int32 (default=8).
        order_map: Optional order of elements within each packed int32.

    Returns:
        torch.int32 tensor of shape [B, K, N//pack_num].
    """
    
    B, K, N = x.shape
    assert N % pack_num == 0, "N must be divisible by pack_num"
    cols = N // pack_num

    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    unit = 32 // pack_num  # number of bits per element

    # reshape to [B, K, cols, pack_num]
    pack_num_int = int(pack_num)

    x_reshape = x.view(B, K, cols, pack_num_int)

    # reorder according to order_map
    x_reorder = torch.gather(
        x_reshape, 3, order_map.view(1, 1, 1, -1).expand(B, K, cols, -1)
    )

    # mask low 4 bits
    x_reorder = x_reorder & 0xF

    # bit shifts [pack_num] -> [1,1,1,pack_num] broadcastable
    shifts = (unit * torch.arange(pack_num_int, device=x.device)).view(1, 1, 1, -1)

    # shift and sum along last dimension to combine bits
    packed = (x_reorder << shifts).sum(dim=-1).to(torch.int32)

    return packed

def get_moe_quant_method(
    config: "IluAutoGPTQConfig",
    layer: RoutedExperts,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    assert isinstance(layer, RoutedExperts)
    # False = skip module, None = no override, else = Positive match
    if (
        get_dynamic_override(  # noqa: E712
            cloned_config,  # noqa: E712
            layer_name=prefix,
        )
        == False
    ):  # noqa: E712
        return UnquantizedFusedMoEMethod(layer.moe_config)

    if prefix:
        # Dynamic per module/layer rules may override base config
        override_config(cloned_config, prefix=prefix)

    return moe_method_cls(cloned_config, layer.moe_config)



@register_quantization_config("auto_gptq")
@register_quantization_config("gptq")
@register_quantization_config("gptq_marlin")
class IluAutoGPTQConfig(AutoGPTQConfig):
    """Config class for AutoGPTQ quantization using Marlin kernels."""

    def __repr__(self) -> str:
        return (
            f"IluAutoGPTQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            moe_quant_method = get_moe_quant_method(
                self, layer, prefix, IluAutoGPTQMoEMethod
            )
            if moe_quant_method is None:
                return None
            moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            return moe_quant_method

        quant_method = get_linear_quant_method(
            self, layer, prefix, IluAutoGPTQLinearMethod
        )
        if quant_method is None:
            return None
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        return quant_method


class IluAutoGPTQLinearMethod(LinearMethodBase):
    """Linear method for AutoGPTQ using Marlin kernels.

    Args:
        quant_config: The AutoGPTQ quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: IluAutoGPTQConfig) -> None:
        self.quant_config = quant_config
        self.input_dtype = None
        self.quant_type = self.quant_config.quant_type

        # Verify supported on platform.
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
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
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        input_dtype = self.input_dtype

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype if input_dtype is None else input_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for IluAutoGPTQLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # Activation order
        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class IluAutoGPTQMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: IluAutoGPTQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits != 4:
            raise ValueError("IluAutoGPTQMoEMethod only supports int4 now.")   
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        self.use_marlin = True

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )
        if self.input_dtype == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(layer.w13_qweight, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_qweight, inplace=True)
            layer.w13_scales.data = layer.w13_scales.data * 512
            layer.w2_scales.data = layer.w2_scales.data * 512

        
        if self.quant_config.desc_act:
            raise NotImplementedError(
                "GPTQMarlinMoEMethod now not support  desc_act. please fix it")   
        w13_qweight_unpacked = unpack_k_batch(layer.w13_qweight)
        w13_qweight_repacked = pack_n_batch(w13_qweight_unpacked,self.quant_config.pack_factor,order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        replace_parameter(layer, "w13_qweight", w13_qweight_repacked)
        
        w13_zp = torch.full_like(layer.w13_scales, self.quant_config.quant_type.bias, dtype=torch.int32)
        w13_zp_pack = pack_n_batch(w13_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]).contiguous()
        replace_parameter(layer, "w13_qzeros", w13_zp_pack)
        
        w2_qweight_unpacked = unpack_k_batch(layer.w2_qweight)
        w2_qweight_repacked = pack_n_batch(w2_qweight_unpacked,self.quant_config.pack_factor,order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        replace_parameter(layer, "w2_qweight", w2_qweight_repacked)
        
        w2_zp = torch.full_like(layer.w2_scales, self.quant_config.quant_type.bias, dtype=torch.int32)
        w2_zp_pack = pack_n_batch(w2_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]).contiguous()
        replace_parameter(layer, "w2_qzeros", w2_zp_pack)

    

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: RoutedExperts,
    ):
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

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
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata:
            if isinstance(attn_metadata, dict):
                only_decode = (use_ep == False and all(t.num_decodes > 0 and t.num_prefills ==0 for t in list(attn_metadata.values())))
            else:
                only_decode = use_ep == False and attn_metadata.num_decodes > 0 and attn_metadata.num_prefills == 0
        else:
            only_decode = False
        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min((layer.ep_rank + 1) * layer.local_num_experts, layer.global_num_experts)
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused Marlin MoE method.")
       
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

        # expand + reorder
        # TODO use kernel
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

                # mul + reduce_sum
                # final_hidden_states: (num_tokens, k)
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=_get_ilu_output_scale(layer),
                extra_residual=shared_experts_input,
            )
        else:
            expert_sizes_cpu = expert_sizes_gpu.cpu()
            # w4a16 group gemm 1
            # pt_output_1: (expand_tokens, 2n) dtype
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

            # act
            pt_output_2 = ixfops.silu_and_mul(pt_output_1)

            # w4a16 group gemm 2 + reorder
            # pt_output_3: (expand_tokens, k) dtype
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
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    mask=reduce_mask,
                    extra_residual=shared_experts_input,
                )
            else:
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

                # mul + reduce_sum
                # final_hidden_states: (num_tokens, k)
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    extra_residual=shared_experts_input,
                )
        return final_hidden_states
