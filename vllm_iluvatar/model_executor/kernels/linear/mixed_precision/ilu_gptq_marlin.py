# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPTQ-Marlin-style WNA16 linear kernel.

OOT plugin version of the fork-local
``vllm/model_executor/kernels/linear/mixed_precision/ilu_gptq_marlin.py``.  vLLM
does not currently expose a kernel-class registry for ``MPLinearKernel``, so
plugin quantization methods call
``vllm_iluvatar.model_executor.kernels.linear.choose_mp_linear_kernel`` directly
to select this implementation without editing community kernel lists.
"""

from __future__ import annotations

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    check_marlin_supports_shape,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types



def unpack_rows(packed_w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Unpack a row-packed ``[K // pack_factor, N]`` int32 tensor."""

    pack_factor = 32 // num_bits
    k_packed, n = packed_w.shape
    mask = (1 << num_bits) - 1
    shifts = (num_bits * torch.arange(pack_factor, device=packed_w.device)).view(
        -1,
        1,
        1,
    )
    unpacked_groups = ((packed_w.unsqueeze(0) >> shifts) & mask).to(torch.int8)
    return unpacked_groups.permute(1, 0, 2).reshape(k_packed * pack_factor, n)


def pack_cols(
    x: torch.Tensor,
    pack_num: int = 8,
    order_map: list[int] | None = None,
) -> torch.Tensor:
    """Pack int4 columns into int32 using the order expected by ixformer."""

    if order_map is None:
        order_map = list(range(pack_num))
    order = torch.tensor(order_map, device=x.device)

    unit = 32 // pack_num
    rows, cols_pack = x.shape
    assert cols_pack % pack_num == 0, (
        "Number of columns must be a multiple of pack_num"
    )
    cols = cols_pack // pack_num
    x_reshape = x.view(rows, cols, pack_num)
    x_reorder = torch.gather(
        x_reshape,
        2,
        order.view(1, 1, -1).expand(rows, cols, -1),
    )
    x_reorder = x_reorder & 0xF
    shifts = (unit * torch.arange(pack_num, device=x.device)).view(1, 1, -1)
    return (x_reorder << shifts).sum(dim=-1).to(torch.int32)


class IluGPTQMarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Iluvatar platform intentionally follows vLLM's CUDA-like path.
        if not current_platform.is_cuda():
            return False, "IluGPTQMarlin only supported on CUDA-like platforms"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by IluGPTQMarlin, "
                f"supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by IluGPTQMarlin, "
                f"supported group sizes are: {MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        return check_marlin_supports_shape(
            c.partition_weight_shape[1],
            c.partition_weight_shape[0],
            c.full_weight_shape[0],
            c.group_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        assert c.weight_type.size_bits == 4, (
            "IluGPTQMarlinLinearKernel only supports 4-bit weights, got "
            f"{c.weight_type}"
        )

        if c.has_g_idx:
            assert self.w_gidx_name is not None
            perm = torch.argsort(getattr(layer, self.w_gidx_name)).to(torch.int)
            self.act_perm = lambda x: x[:, perm]

        def transform_w_q(x):
            assert x.data.ndim == 2
            if x._packed_dim == 1:
                x_unpacked = unpack_quantized_values_into_int32(
                    x.data,
                    c.weight_type,
                    packed_dim=1,
                )
                if c.has_g_idx:
                    x_unpacked = x_unpacked[:, perm]
                x_unpacked = x_unpacked.t().contiguous()
            elif x._packed_dim == 0:
                x_unpacked = unpack_rows(x.data, c.weight_type.size_bits)
                if c.has_g_idx:
                    raise NotImplementedError(
                        "GPTQMarlinLinearMethod with g_idx is not validated for "
                        "IluGPTQMarlinLinearKernel."
                    )
            else:
                raise NotImplementedError(
                    f"transform_w_q packed_dim {x._packed_dim} is not implemented"
                )

            x.data = pack_cols(
                x_unpacked,
                order_map=[0, 2, 4, 6, 1, 3, 5, 7],
            ).contiguous()
            x._packed_dim = 1
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        def transform_w_zp(x):
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size
                if c.group_size != -1
                else 1
            )
            x_unpacked = unpack_cols(
                x.clone().t(),
                c.weight_type.size_bits,
                grouped_k,
                c.partition_weight_shape[1],
            )
            x.data = pack_cols(
                x_unpacked,
                order_map=[0, 2, 4, 6, 1, 3, 5, 7],
            ).contiguous()
            return x

        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        else:
            assert self.w_zp_name is not None
            w_zp = torch.full_like(
                getattr(layer, self.w_s_name),
                c.weight_type.bias,
                dtype=torch.int32,
            )
            w_zp_pack = pack_cols(
                w_zp,
                order_map=[0, 2, 4, 6, 1, 3, 5, 7],
            ).contiguous()
            weight_zero_point = torch.nn.Parameter(w_zp_pack, requires_grad=False)
            if hasattr(layer, self.w_zp_name):
                replace_parameter(layer, self.w_zp_name, weight_zero_point)
            else:
                layer.register_parameter("weight_zero_point", weight_zero_point)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _w_gidx = self._get_weight_params(layer)
        pack_factor = 32 // c.weight_type.size_bits
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
        x_2d = x.reshape(-1, x.shape[-1])

        if c.has_g_idx:
            x_2d = self.act_perm(x_2d)

        out = ops.custom_gptq_marlin_gemm(
            input=x_2d,
            qweight=w_q,
            scales=w_s,
            qzeros=w_zp,
            pack_factor=pack_factor,
            group_size=c.group_size,
            bias=bias,
        )
        return out.reshape(out_shape)
