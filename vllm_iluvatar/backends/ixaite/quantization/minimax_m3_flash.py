# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Upstream reference: vLLM v0.27.1 (6e448d0),
# vllm/model_executor/layers/quantization/fp8.py and modelopt.py.
# Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. for the ixaite backend.
"""MiniMax-M3.1-Flash QuantizationConfig for the ixaite provider.

Claims HF ``quant_method=mxfp8`` + ``moe_quant_algo=NVFP4``. Dense linears
and shared experts convert MXFP8 → channel W8A8 at load; routed MoE experts
convert NVFP4 → FlyDSL W4A8. Native MXFP8 / NVFP4 GEMM apply paths raise —
ivcore11 has no MX/NVFP4 hardware and must not fall through to CUDA / Triton.
"""

from __future__ import annotations

from typing import Any

import torch
from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.model_executor.utils import set_weight_attrs

METHOD_NAME = "ilu_minimax_m3_flash"
_MXFP8_BLOCK = [1, 32]
_NVFP4_GROUP = 16
# Flash NVFP4 shards at TP<8 are 2-4x larger; holding packed+converted for
# both w13 and w2 at once OOMs 32 GB BI-V150S cards. Serve load target is
# TP=8, with expert/half streaming convert below.
RECOMMENDED_LOAD_TP = 8


def _drop_layer_attrs(layer: torch.nn.Module, *names: str) -> None:
    for name in names:
        if hasattr(layer, name):
            delattr(layer, name)


def stream_convert_nvfp4_moe_weights(
    layer: torch.nn.Module,
    *,
    reference: bool | None = None,
    on_phase: Any | None = None,
) -> None:
    """Convert fused NVFP4 packs to FlyDSL W4A8 one expert/half at a time.

    Drops each family's packed NVFP4 attributes from ``layer`` before the
    converted W4A8 buffers are installed, and finishes w13 before touching
    w2 packs, so peak memory never holds both packed families plus both
    converted families. ``on_phase`` is invoked with a short phase name for
    tests that assert the drop order.
    """
    from ixaite.ops.minimax_weight_convert import nvfp4_to_w4a8_flydsl

    packed_w13 = layer.w13_weight_packed.data
    # A GPU host still reports CUDA available while these unit-test tensors
    # stay on CPU. The device kernel rejects that; use the reference then.
    use_reference = packed_w13.device.type != "cuda" if reference is None else reference

    def _convert_matrix(
        packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = global_scale
        if g.ndim > 0:
            g = g.reshape(-1)[0]
        try:
            return nvfp4_to_w4a8_flydsl(packed, scale, g, reference=use_reference)
        except ValueError as exc:
            # FlyDSL first-launch / multi-process JIT races have been observed
            # to leave logical INT4 buffers partially unwritten; fall back to
            # the pure-torch reference so load can finish.
            if use_reference:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "ixaite NVFP4→W4A8 FlyDSL convert failed (%s); "
                "retrying with reference=True",
                exc,
            )
            return nvfp4_to_w4a8_flydsl(packed, scale, g, reference=True)

    def _phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name, layer)

    scale_w13 = layer.w13_weight_scale.data
    global_w13 = layer.w13_weight_global_scale.data
    # Detach NVFP4 w13 storage from the module before allocating W4A8.
    _drop_layer_attrs(
        layer,
        "w13_weight_packed",
        "w13_weight_scale",
        "w13_weight_global_scale",
        "w13_input_global_scale",
    )
    _phase("w13_nvfp4_dropped")

    experts = int(packed_w13.shape[0])
    half = int(packed_w13.shape[1]) // 2
    pw0, ps0 = _convert_matrix(
        packed_w13[0, :half],
        scale_w13[0, :half],
        global_w13[0, 0] if global_w13.ndim == 2 else global_w13[0],
    )
    w13_pack = torch.empty(
        experts, 2 * half, *pw0.shape[1:], device=pw0.device, dtype=pw0.dtype
    )
    w13_scale = torch.empty(
        experts, 2 * half, *ps0.shape[1:], device=ps0.device, dtype=ps0.dtype
    )
    w13_pack[0, :half] = pw0
    w13_scale[0, :half] = ps0
    del pw0, ps0
    for e in range(experts):
        for h in (0, 1):
            if e == 0 and h == 0:
                continue
            sl = slice(h * half, (h + 1) * half)
            g = global_w13[e, h] if global_w13.ndim == 2 else global_w13[e]
            pw, ps = _convert_matrix(packed_w13[e, sl], scale_w13[e, sl], g)
            w13_pack[e, sl] = pw
            w13_scale[e, sl] = ps
            del pw, ps
    del packed_w13, scale_w13, global_w13
    layer.w13_weight = torch.nn.Parameter(w13_pack, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    del w13_pack, w13_scale
    _phase("w13_w4a8_installed")

    packed_w2 = layer.w2_weight_packed.data
    scale_w2 = layer.w2_weight_scale.data
    global_w2 = layer.w2_weight_global_scale.data
    _drop_layer_attrs(
        layer,
        "w2_weight_packed",
        "w2_weight_scale",
        "w2_weight_global_scale",
        "w2_input_global_scale",
    )
    _phase("w2_nvfp4_dropped")

    pw0, ps0 = _convert_matrix(packed_w2[0], scale_w2[0], global_w2[0])
    w2_pack = torch.empty(experts, *pw0.shape, device=pw0.device, dtype=pw0.dtype)
    w2_scale = torch.empty(experts, *ps0.shape, device=ps0.device, dtype=ps0.dtype)
    w2_pack[0] = pw0
    w2_scale[0] = ps0
    del pw0, ps0
    for e in range(1, experts):
        pw, ps = _convert_matrix(packed_w2[e], scale_w2[e], global_w2[e])
        w2_pack[e] = pw
        w2_scale[e] = ps
        del pw, ps
    del packed_w2, scale_w2, global_w2
    layer.w2_weight = torch.nn.Parameter(w2_pack, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    _phase("w2_w4a8_installed")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_minimax_mixed_hf_cfg(hf_quant_cfg: Any) -> bool:
    if not isinstance(hf_quant_cfg, dict):
        return False
    method = str(hf_quant_cfg.get("quant_method", "")).lower()
    moe_algo = str(hf_quant_cfg.get("moe_quant_algo", "")).upper()
    return method == "mxfp8" and moe_algo == "NVFP4"


def _mro_named(layer: torch.nn.Module, name: str) -> bool:
    return any(base.__name__ == name for base in type(layer).__mro__)


def _ignored(prefix: str, ignored_layers: list[str]) -> bool:
    return any(prefix == ign or prefix.endswith(ign) for ign in ignored_layers)


class IluMxfp8ToW8A8LinearMethod(LinearMethodBase):
    """Load MXFP8 (e4m3 + UE8M0 1×32), convert to channel INT8 for ixaite.w8a8."""

    use_mxfp8 = True

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
        del input_size, output_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = list(_MXFP8_BLOCK)
        layer.params_dtype = params_dtype

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # MiniMax Flash stores UE8M0 scales as uint8 under weight_scale_inv.
        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            _MXFP8_BLOCK,
            weight_loader,
            scale_dtype=torch.uint8,
        )
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from ixaite.ops.minimax_weight_convert import mxfp8_to_w8a8

        weight = layer.weight.data
        scale_u8 = layer.weight_scale_inv.data
        if weight.numel() == 0 or scale_u8.numel() == 0:
            return

        # CPU tensors stay on the reference even when this host has a GPU.
        # is_available() is true there, and the device kernel rejects CPU storage.
        reference = weight.device.type != "cuda"
        weight_kn, scale_n = mxfp8_to_w8a8(weight, scale_u8, reference=reference)
        # IxaiteCompressedTensorsW8A8Int8.process expects [N, K] then may
        # transpose to NN; keep that contract when the scheme can bind.
        layer.weight = torch.nn.Parameter(
            weight_kn.t().contiguous(), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            scale_n.view(-1, 1).contiguous(), requires_grad=False
        )
        if hasattr(layer, "weight_scale_inv"):
            delattr(layer, "weight_scale_inv")

        if reference:
            return

        from vllm_iluvatar.backends.ixaite.quantization.w8a8 import (
            IxaiteCompressedTensorsW8A8Int8,
        )

        scheme = IxaiteCompressedTensorsW8A8Int8(
            strategy=QuantizationStrategy.CHANNEL,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        scheme.params_dtype = getattr(layer, "params_dtype", torch.bfloat16)
        layer.scheme = scheme
        scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer.weight, "dtype", None) != torch.int8:
            raise RuntimeError(
                "Iluvatar MiniMax MXFP8 path refused native apply: weight is "
                f"{getattr(layer.weight, 'dtype', None)}, expected int8 after "
                "load-time convert to W8A8"
            )
        scheme = getattr(layer, "scheme", None)
        if scheme is None:
            raise RuntimeError(
                "Iluvatar MiniMax MXFP8 path converted weights but has no "
                "ixaite W8A8 scheme bound (requires CUDA + ixaite.w8a8)"
            )
        return scheme.apply_weights(layer, x, bias=bias)


class IluNvfp4ToW4A8MoEMethod(FusedMoEMethodBase):
    """Load fused NVFP4 expert packs, convert to FlyDSL W4A8, then W4A8 apply."""

    group_size = _NVFP4_GROUP

    def __init__(self, moe: Any) -> None:
        super().__init__(moe)
        self._w4a8: Any | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        experts = int(num_experts)
        hidden = int(hidden_size)
        intermediate = int(intermediate_size_per_partition)
        gs = self.group_size

        w13 = torch.nn.Parameter(
            torch.empty(experts, 2 * intermediate, hidden // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13)
        set_weight_attrs(w13, extra_weight_attrs)

        w2 = torch.nn.Parameter(
            torch.empty(experts, hidden, intermediate // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2)
        set_weight_attrs(w2, extra_weight_attrs)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value
        w13_scale = torch.nn.Parameter(
            torch.empty(
                experts, 2 * intermediate, hidden // gs, dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, scale_attrs)

        w2_scale = torch.nn.Parameter(
            torch.empty(experts, hidden, intermediate // gs, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, scale_attrs)

        g_attrs = dict(extra_weight_attrs)
        g_attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value
        w13_g = torch.nn.Parameter(
            torch.empty(experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_g)
        set_weight_attrs(w13_g, g_attrs)
        w2_g = torch.nn.Parameter(
            torch.empty(experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_g)
        set_weight_attrs(w2_g, g_attrs)

        w13_in = torch.nn.Parameter(
            torch.empty(experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_global_scale", w13_in)
        set_weight_attrs(w13_in, g_attrs)
        w2_in = torch.nn.Parameter(
            torch.empty(experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_in)
        set_weight_attrs(w2_in, g_attrs)

    def _bind_w4a8_after_convert(self, layer: torch.nn.Module) -> None:
        """Reshape FlyDSL channel scales and bind the ixaite W4A8 apply path."""
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationType,
        )

        from vllm_iluvatar.backends.ixaite.moe.w4a8 import (
            IluCompressedTensorsW4A8Int8MoEMethod,
        )
        from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
            W4A8GroupGemmInfo,
        )

        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name, None)
            if scale is None:
                continue
            data = scale.data
            if data.ndim == 2:
                data = data.unsqueeze(1).contiguous()
            elif data.ndim == 3 and data.shape[-1] == 1 and data.shape[1] != 1:
                # FlyDSL 3-D path stores [E, N, 1]; W4A8 apply wants [E, 1, N].
                data = data.transpose(1, 2).contiguous()
            setattr(
                layer,
                name,
                torch.nn.Parameter(data, requires_grad=False),
            )

        layer.w13_bias = None
        layer.w2_bias = None
        layer.w13_i8_weight_scale = None
        layer.w2_i8_weight_scale = None
        layer.w13_i8_weight_zero = None
        layer.w2_i8_weight_zero = None
        if getattr(layer, "w13_input_scale", None) is None:
            layer.w13_input_scale = None
        if getattr(layer, "w2_input_scale", None) is None:
            layer.w2_input_scale = None

        # TN pack: w13 [E, 2I, H/2], w2 [E, H, I/2]
        hidden = int(layer.w2_weight.shape[1])
        intermediate = int(layer.w13_weight.shape[1]) // 2
        weight_quant = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
            group_size=None,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
        method = IluCompressedTensorsW4A8Int8MoEMethod(
            weight_quant=weight_quant,
            input_quant=input_quant,
            moe=self.moe,
            ilu_params={"gemm_format": "TN", "pack_version": 2},
        )
        method.gemm1_info = W4A8GroupGemmInfo("TN", k=hidden, n=2 * intermediate)
        method.gemm2_info = W4A8GroupGemmInfo("TN", k=intermediate, n=hidden)
        method.process_weights_after_loading(layer)
        self._w4a8 = method

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        stream_convert_nvfp4_moe_weights(layer)
        # Unit tests convert on CPU. W4A8 apply needs a device-resident weight;
        # cuda.is_available() is true on a GPU host even for those tensors.
        weight = getattr(layer, "w13_weight", None)
        if weight is not None and weight.device.type == "cuda":
            self._bind_w4a8_after_convert(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        w13 = getattr(layer, "w13_weight", None)
        w2 = getattr(layer, "w2_weight", None)
        if w13 is None or w2 is None or w13.dtype != torch.int8:
            return None
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4",
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = getattr(layer, "w13_weight", None)
        if weight is None or weight.dtype != torch.int8:
            raise RuntimeError(
                "Iluvatar MiniMax NVFP4 MoE path refused native apply: "
                "experts must be converted to FlyDSL W4A8 (int8 packed) at load"
            )
        if self._w4a8 is None:
            # Converted packs are FlyDSL W4A8; shared experts stay on the
            # separate MXFP8→W8A8 linear path.
            raise RuntimeError(
                "Iluvatar MiniMax NVFP4 MoE experts are converted to W4A8; "
                "bind ixaite moe_w4a8_group_gemm apply before serving "
                "(shared experts remain a separate W8A8 linear)"
            )
        return self._w4a8.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


class IxaiteMiniMaxM3FlashConfig(QuantizationConfig):
    """MXFP8 dense + NVFP4 MoE → ixaite W8A8 / W4A8 on Iluvatar."""

    def __init__(self, ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        self.ignored_layers = list(ignored_layers or [])

    def get_name(self) -> str:
        return METHOD_NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IxaiteMiniMaxM3FlashConfig:
        return cls(ignored_layers=list(config.get("ignored_layers", []) or []))

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        del user_quant, hf_config
        if _is_minimax_mixed_hf_cfg(hf_quant_cfg):
            return METHOD_NAME
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if _ignored(prefix, self.ignored_layers):
            if _mro_named(layer, "LinearBase"):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            return None

        if _mro_named(layer, "RoutedExperts"):
            return IluNvfp4ToW4A8MoEMethod(layer.moe_config)

        if _mro_named(layer, "LinearBase"):
            return IluMxfp8ToW8A8LinearMethod()

        return None


__all__ = [
    "METHOD_NAME",
    "RECOMMENDED_LOAD_TP",
    "IxaiteMiniMaxM3FlashConfig",
    "IluMxfp8ToW8A8LinearMethod",
    "IluNvfp4ToW4A8MoEMethod",
    "stream_convert_nvfp4_moe_weights",
]
