# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import ixformer.inference.functions as ixf_ops
import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)


def unpack_rows(packed_w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Efficient vectorized unpacking.
    Converts [K // pack_factor, N] int32 tensor → [K, N] int8 tensor.
    Args:
        packed_w: torch.int32 tensor of shape [K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4).

    Returns:
        unpacked: torch.int8 tensor of shape [K, N].
    """
    pack_factor = 32 // num_bits
    k_packed, n = packed_w.shape
    k = k_packed * pack_factor

    mask = (1 << num_bits) - 1

    # [pack_factor, 1, 1]
    shifts = (num_bits * torch.arange(pack_factor, device=packed_w.device)).view(
        -1, 1, 1
    )

    # [pack_factor, k_packed, n]
    packed_expanded = packed_w.unsqueeze(0)

    # Extract each group of num_bits using bitwise ops
    unpacked_groups = ((packed_expanded >> shifts) & mask).to(torch.int8)
    # [pack_factor, k_packed, n] → [k, n]
    unpacked = unpacked_groups.permute(1, 0, 2).reshape(k, n)

    return unpacked


def pack_cols(x: torch.Tensor, pack_num: int = 8, order_map=None) -> torch.Tensor:
    """
    Efficient vectorized version: pack int4 values (0–15) into int32.
    Each int32 element contains `pack_num` 4-bit values.

    Args:
        x: Tensor of shape [rows, cols * pack_num], dtype=int32.
           Represents unpacked int4 values.
        pack_num: Number of 4-bit elements to pack into each int32.
        order_map: Index mapping defining the order of 4-bit packing,
                   must match the unpack order used in `unpack_tensor`.

    Returns:
        Tensor of shape [rows, cols], dtype=int32 — packed result.
    """
    # Default sequential order if none provided
    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    # Number of bits per packed element (e.g., 32 / 8 = 4 bits)
    unit = 32 // pack_num
    rows, cols_pack = x.shape
    assert cols_pack % pack_num == 0, "Number of columns must be a multiple of pack_num"
    cols = cols_pack // pack_num

    # Reshape input into groups of `pack_num` int4 values
    # Shape: [rows, cols, pack_num]
    x_reshape = x.view(rows, cols, pack_num)

    # Reorder elements according to order_map
    # order_map is broadcasted to match shape [rows, cols, pack_num]
    x_reorder = torch.gather(
        x_reshape, 2, order_map.view(1, 1, -1).expand(rows, cols, -1)
    )

    # Keep only the lower 4 bits of each value
    x_reorder = x_reorder & 0xF

    # Compute bit shifts for each position (e.g., [0, 4, 8, 12, 16, 20, 24, 28])
    shifts = (unit * torch.arange(pack_num, device=x.device)).view(1, 1, -1)

    # Shift and combine (bitwise OR) along the last dimension
    # Using sum() is safe since bits don't overlap between 4-bit slots
    res = (x_reorder << shifts).sum(dim=-1).to(torch.int32)

    return res


class IluMarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        return check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert (
                c.weight_type == scalar_types.uint4b8
            ), "W8A8 is not supported by marlin kernel."

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )
        assert (
            c.weight_type.size_bits == 4
        ), f"MarlinLinearKernel now only support uint4, uint4b8, \
                        now quant weight_type {c.weight_typ}"

        # device = getattr(layer, self.w_q_name).device

        # row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        # self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        # self.workspace = marlin_make_workspace_new(device)

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        # if self.w_gidx_name is None:
        #     self.w_gidx_name = "g_idx"
        # if self.w_zp_name is None:
        #     self.w_zp_name = "w_zp"
        if c.has_g_idx:
            assert self.w_gidx_name is not None
            perm = torch.argsort(getattr(layer, self.w_gidx_name)).to(torch.int)

            self.act_perm = lambda x: x[:, perm]

        def transform_w_q(x):
            #
            assert x.data.ndim == 2
            if x._packed_dim == 1:  # CompressedTensorsWNA16
                # [oc,    ic // 8] - > [oc,    ic]
                x_unpacked = unpack_quantized_values_into_int32(
                    x.data, c.weight_type, packed_dim=1
                )
                if c.has_g_idx:
                    x_unpacked = x_unpacked[:, perm]
                # [oc,    ic] -> [ic,    oc]
                x_unpacked = x_unpacked.t().contiguous()

            elif x._packed_dim == 0:  # GPTQMarlinLinearMethod

                # [ic // 8, oc]  -> [ic , oc]
                x_unpacked = unpack_rows(x.data, c.weight_type.size_bits)
                if c.has_g_idx:
                    x_unpacked = x_unpacked[perm:]
                    raise NotImplementedError(
                        f"GPTQMarlinLinearMethod has_g_idx not test, \
                        Please check whether the model's inference results are correct, and annotate/modify the statement. "
                    )
            else:
                raise NotImplementedError(
                    f"transform_w_q pack_dim {x._packed_dim} not implement"
                )

            # [ic,    oc]-> [ic, oc//8]
            x_packed = pack_cols(x_unpacked, order_map=[0, 2, 4, 6, 1, 3, 5, 7])
            x.data = x_packed.contiguous()
            x._packed_dim = 1
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        def transform_w_zp(x):
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            x_unpacked = unpack_cols(
                x.clone().t(),
                c.weight_type.size_bits,
                grouped_k,
                c.partition_weight_shape[1],
            )
            x_packed = pack_cols(x_unpacked, order_map=[0, 2, 4, 6, 1, 3, 5, 7])
            x.data = x_packed.contiguous()
            return x

        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        else:
            w_zp = torch.full_like(
                getattr(layer, self.w_s_name), c.weight_type.bias, dtype=torch.int32
            )
            w_zp_pack = pack_cols(w_zp, order_map=[0, 2, 4, 6, 1, 3, 5, 7]).contiguous()
            weight_zero_point = torch.nn.Parameter(w_zp_pack, requires_grad=False)
            if hasattr(layer, self.w_zp_name):
                replace_parameter(
                    layer, self.w_zp_name, weight_zero_point
                )  # GPTQMarlinLinearMethod
            else:
                layer.register_parameter(
                    "weight_zero_point", weight_zero_point
                )  # CompressedTensorsWNA16
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        pack_factor = 32 // c.weight_type.size_bits

        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
        x_2d = x.reshape(-1, x.shape[-1])

        if c.has_g_idx:
            x_2d = self.act_perm(x_2d)
        # TODO optimize bias
        out = ops.custom_gptq_marlin_gemm(
            input=x_2d,
            qweight=w_q,
            scales=w_s,
            qzeros=w_zp,
            pack_factor=pack_factor,
            group_size=c.group_size,
            bias=bias,
        )
        out = out.reshape(out_shape)
        return out
