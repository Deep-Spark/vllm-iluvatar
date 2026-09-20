#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reusable offline dequantization and quantization methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuantResult:
    weight: torch.Tensor
    weight_scale: torch.Tensor | None = None
    i8_weight_scale: torch.Tensor | None = None
    i8_weight_zero: torch.Tensor | None = None


DequantMethod = Callable[
    [torch.Tensor, torch.Tensor | None, Mapping[str, Any]], torch.Tensor
]
QuantMethod = Callable[[torch.Tensor, Mapping[str, Any]], QuantResult]

DEQUANT_METHODS: dict[str, DequantMethod] = {}
QUANT_METHODS: dict[str, QuantMethod] = {}


def register_dequant(name: str):
    def decorator(method: DequantMethod) -> DequantMethod:
        if name in DEQUANT_METHODS:
            raise ValueError(f"dequant method {name!r} is already registered")
        DEQUANT_METHODS[name] = method
        return method

    return decorator


def register_quant(name: str):
    def decorator(method: QuantMethod) -> QuantMethod:
        if name in QUANT_METHODS:
            raise ValueError(f"quant method {name!r} is already registered")
        QUANT_METHODS[name] = method
        return method

    return decorator


def _require_scale(
    scale: torch.Tensor | None, method: str, weight: torch.Tensor
) -> torch.Tensor:
    if scale is None:
        raise ValueError(
            f"{method} requires a companion scale for {tuple(weight.shape)}"
        )
    return scale


def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    elif scale.dtype == torch.int8:
        scale = scale.view(torch.uint8).view(torch.float8_e8m0fnu)
    return scale.float()


def _block_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int,
    decode_e8m0: bool,
) -> torch.Tensor:
    if weight.dim() == 3:
        if scale.dim() != 3 or scale.shape[0] != weight.shape[0]:
            raise ValueError(
                "3-D block dequant expects per-expert scale "
                f"[E, *, *], got {weight.shape}/{scale.shape}"
            )
        return torch.stack(
            [
                _block_dequant(
                    expert_weight,
                    expert_scale,
                    block_size=block_size,
                    decode_e8m0=decode_e8m0,
                )
                for expert_weight, expert_scale in zip(
                    weight.unbind(0), scale.unbind(0), strict=True
                )
            ],
            dim=0,
        )
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            f"block dequant expects 2-D or 3-D weight/scale, got {weight.shape}/{scale.shape}"
        )
    rows, cols = weight.shape
    scale_rows, scale_cols = scale.shape
    padded = torch.zeros(
        scale_rows * block_size,
        scale_cols * block_size,
        dtype=weight.dtype,
        device=weight.device,
    )
    padded[:rows, :cols] = weight
    blocks = padded.view(scale_rows, block_size, scale_cols, block_size).permute(
        0, 2, 1, 3
    )
    scale_fp32 = _decode_e8m0(scale) if decode_e8m0 else scale.float()
    value = blocks.float() * scale_fp32[..., None, None]
    return value.permute(0, 2, 1, 3).reshape(padded.shape)[:rows, :cols]


@register_dequant("identity")
def dequant_identity(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    del scale, options
    return weight


@register_dequant("to_float32")
def dequant_to_float32(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    del scale, options
    return weight.float()


@register_dequant("fp8_block")
def dequant_fp8_block(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    if scale is None and options.get("allow_unscaled", False):
        return weight.float()
    return _block_dequant(
        weight,
        _require_scale(scale, "fp8_block", weight),
        block_size=int(options.get("block_size", 128)),
        decode_e8m0=False,
    )


@register_dequant("fp8_block_e8m0")
def dequant_fp8_block_e8m0(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    return _block_dequant(
        weight,
        _require_scale(scale, "fp8_block_e8m0", weight),
        block_size=int(options.get("block_size", 128)),
        decode_e8m0=True,
    )


@register_dequant("fp8_static")
def dequant_fp8_static(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    if scale is None and options.get("allow_unscaled", False):
        return weight.float()
    scale = _require_scale(scale, "fp8_static", weight).float()
    block_size = int(options.get("block_size", -1))
    if block_size == -1:
        return weight.float() * scale
    return _block_dequant(weight, scale, block_size=block_size, decode_e8m0=False)


_FP4_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@register_dequant("mxfp4_block32")
def dequant_mxfp4_block32(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    del options
    scale = _require_scale(scale, "mxfp4_block32", weight)
    if weight.dtype == torch.int8:
        packed = weight.view(torch.uint8)
    elif weight.dtype == torch.uint8:
        packed = weight
    else:
        raise TypeError(f"MXFP4 weight must be int8/uint8, got {weight.dtype}")

    out_dim, packed_dim = packed.shape
    in_dim = packed_dim * 2
    if in_dim % 32 or scale.shape != (out_dim, in_dim // 32):
        raise ValueError(
            f"MXFP4 shape mismatch: weight={weight.shape}, scale={scale.shape}"
        )
    lut = _FP4_E2M1_MAGNITUDES.to(packed.device)

    def decode(nibble: torch.Tensor) -> torch.Tensor:
        sign = ((nibble & 0x08) != 0).float()
        magnitude = lut[(nibble & 0x07).long()]
        return magnitude * (1.0 - 2.0 * sign)

    low = decode(packed & 0x0F)
    high = decode((packed >> 4) & 0x0F)
    unpacked = torch.stack((low, high), dim=-1).reshape(out_dim, in_dim)
    return unpacked * _decode_e8m0(scale).repeat_interleave(32, dim=1)


@register_dequant("int4_group")
def dequant_int4_group(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    options: Mapping[str, Any],
) -> torch.Tensor:
    """Dequantize Compressed-Tensors INT4 ``weight_packed`` tensors."""
    from compressed_tensors.compressors.pack_quantized.helpers import (
        unpack_from_int32,
    )

    scale = _require_scale(scale, "int4_group", weight).float()
    if weight.dim() != 2 or weight.dtype != torch.int32:
        raise ValueError(
            "int4_group expects a 2-D int32 weight_packed tensor, "
            f"got {weight.shape}/{weight.dtype}"
        )
    num_bits = int(options.get("num_bits", 4))
    group_size = int(options.get("group_size", 32))
    if num_bits <= 0 or 32 % num_bits:
        raise ValueError(f"num_bits must divide 32, got {num_bits}")
    out_dim, packed_dim = weight.shape
    in_dim = packed_dim * (32 // num_bits)
    if in_dim % group_size or scale.shape != (out_dim, in_dim // group_size):
        raise ValueError(
            "INT4 group shape mismatch: "
            f"weight={weight.shape}, scale={scale.shape}, group_size={group_size}"
        )
    unpacked = unpack_from_int32(
        weight, num_bits, torch.Size((out_dim, in_dim))
    ).float()
    return (
        unpacked.view(out_dim, in_dim // group_size, group_size) * scale.unsqueeze(-1)
    ).reshape(out_dim, in_dim)


def _int8_per_channel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dim() != 2:
        raise ValueError(f"per-channel quantization expects 2-D input: {value.shape}")
    abs_max = value.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.float()


# Dense W4A8 kernel ``alpha`` (``w4a8.cpp``). Stored scale is ``true_scale / alpha``
# so GEMM ``(x @ q) * i_scale * w_scale * 18`` reconstructs the int4 range.
_W4A8_KERNEL_ALPHA = 18.0
_INT4_QMAX = 7.0
_PACK_ALIGN = 32


def _int4_per_channel(
    value: torch.Tensor, scale_div: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel INT4 into int8 ``[N, K]``; caller kernel-packs."""
    if value.dim() != 2:
        raise ValueError(f"per-channel quantization expects 2-D input: {value.shape}")
    if scale_div <= 0:
        raise ValueError(f"scale_div must be positive, got {scale_div}")
    abs_max = value.abs().amax(dim=-1, keepdim=True)
    qscale = abs_max / _INT4_QMAX
    qscale = torch.where(qscale > 0, qscale, torch.ones_like(qscale))
    quantized = torch.round(value / qscale).clamp(-8, 7).to(torch.int8)
    return quantized, (qscale / scale_div).float()


def _pack_int4_nn(weight: torch.Tensor) -> torch.Tensor:
    """Kernel-pack INT4-in-int8 ``[N, K]`` into NN ``[K, N/2]``."""
    import ixformer.inference.functions as ixfop

    n, k = weight.shape
    if n % _PACK_ALIGN or k % _PACK_ALIGN:
        raise ValueError(
            "int4_per_channel packed_nn requires N and K to be multiples of "
            f"{_PACK_ALIGN} so TP / fused shards stay sliceable in the packed "
            f"domain, got [N={n}, K={k}]."
        )
    packed, _, _ = ixfop.w4a8_gemm_quant_repack_int4(
        # group_size == -1 path internally uses x / 18 -> int4, so q*18 keeps
        # the already-quantized int4 values after packing.
        weight.to(torch.float32) * _W4A8_KERNEL_ALPHA,
        -1,
        2,
        "NN",
    )
    return packed.contiguous()


def _fp8_per_channel(
    value: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dim() != 2:
        raise ValueError(f"per-channel quantization expects 2-D input: {value.shape}")
    supported_dtypes = {torch.float8_e4m3fn}
    fnuz_dtype = getattr(torch, "float8_e4m3fnuz", None)
    if fnuz_dtype is not None:
        supported_dtypes.add(fnuz_dtype)
    if dtype not in supported_dtypes:
        raise ValueError(f"unsupported per-channel FP8 dtype: {dtype}")

    fp8_max = torch.finfo(dtype).max
    abs_max = value.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = (value / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized, scale.float()


@register_quant("identity")
def quant_identity(value: torch.Tensor, options: Mapping[str, Any]) -> QuantResult:
    del options
    return QuantResult(weight=value)


@register_quant("cast")
def quant_cast(value: torch.Tensor, options: Mapping[str, Any]) -> QuantResult:
    dtype_name = str(options["dtype"])
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype {dtype_name!r}")
    return QuantResult(weight=value.to(dtype))


@register_quant("int8_per_channel")
def quant_int8_per_channel(
    value: torch.Tensor, options: Mapping[str, Any]
) -> QuantResult:
    if options.get("round_through_bfloat16", False):
        value = value.to(torch.bfloat16).float()
    transport = bool(options.get("transport", False))
    if transport:
        value = value.transpose(0, 1)
    weight, scale = _int8_per_channel(value)
    return QuantResult(weight=weight, weight_scale=scale)


@register_quant("int4_per_channel")
def quant_int4_per_channel(
    value: torch.Tensor, options: Mapping[str, Any]
) -> QuantResult:
    """Dense Linear W4A8: symmetric per-channel INT4, kernel-packed ``[K, N/2]``.

    Scale is divided by 18 to cancel the dense W4A8 kernel ``alpha``. N and K
    must be 32-aligned so TP and fused QKV / gate_up can slice in the packed
    domain. Do not use ``int4_moe`` here: that path packs along K for MoE TN.
    """
    if options.get("round_through_bfloat16", False):
        value = value.to(torch.bfloat16).float()
    transport = bool(options.get("transport", False))
    if transport:
        value = value.transpose(0, 1)
    group_size = int(options.get("group_size", -1))
    if group_size not in (-1,):
        raise ValueError(
            f"int4_per_channel only supports group_size=-1, got {group_size}"
        )
    if not bool(options.get("symmetric", True)):
        raise ValueError("int4_per_channel only supports symmetric quantization")
    weight_layout = str(options.get("weight_layout", "packed_nn"))
    if weight_layout != "packed_nn":
        raise ValueError(
            "int4_per_channel only exports packed_nn ([K, N/2] int8), "
            f"got weight_layout={weight_layout!r}"
        )
    scale_div = float(options.get("scale_div", _W4A8_KERNEL_ALPHA))
    weight, scale = _int4_per_channel(value, scale_div)
    return QuantResult(weight=_pack_int4_nn(weight), weight_scale=scale)


@register_quant("fp8_per_channel")
def quant_fp8_per_channel(
    value: torch.Tensor, options: Mapping[str, Any]
) -> QuantResult:
    """Symmetric max-abs FP8 quantization with one scale per output channel."""
    if options.get("round_through_bfloat16", False):
        value = value.to(torch.bfloat16).float()
    if options.get("transport", False):
        value = value.transpose(0, 1)

    dtype_name = str(options.get("dtype", "float8_e4m3fn"))
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype {dtype_name!r}")
    weight, scale = _fp8_per_channel(value, dtype)
    return QuantResult(weight=weight, weight_scale=scale)


@register_quant("int4_moe")
def quant_int4_moe(value: torch.Tensor, options: Mapping[str, Any]) -> QuantResult:
    import ixformer.inference.functions as ixfop

    transport = bool(options.get("transport", False))
    if transport:
        value = value.transpose(0, 1)
    int8_weight, scale = _int8_per_channel(value)
    group_size = int(options.get("group_size", -1))
    pack_version = int(options.get("pack_version", 2))
    gemm_format = str(options.get("gemm_format", "TN"))
    symmetric = bool(options.get("symmetric", True))
    packed, i8_scale, i8_zero = ixfop.quant_repack_int4(
        int8_weight.unsqueeze(0),
        group_size,
        pack_version,
        gemm_format,
        not symmetric,
    )
    if pack_version == 1:
        if group_size == -1:
            scale = scale * i8_scale.view(-1, 1)
        else:
            scale = scale.view(1, -1) * i8_scale.view(
                value.shape[1] // group_size, value.shape[0]
            )
        i8_scale = None
        if i8_zero is not None or not symmetric:
            raise ValueError("pack_version=1 requires symmetric quantization")
    elif options.get("scale_layout") == "row":
        scale = scale.contiguous().view(1, -1)

    return QuantResult(
        weight=packed.squeeze(0),
        weight_scale=scale.float(),
        i8_weight_scale=None if i8_scale is None else i8_scale.squeeze(0),
        i8_weight_zero=None if i8_zero is None else i8_zero.squeeze(0),
    )


@register_quant("pack_int4_moe")
def quant_pack_int4_moe(value: torch.Tensor, options: Mapping[str, Any]) -> QuantResult:
    if value.dim() != 3:
        raise ValueError(
            "pack_int4_moe expects stacked expert weights with shape "
            f"(num_experts, dim0, dim1), got {value.shape}"
        )

    packed = bool(options.get("packed", False))
    chunk_dim = int(options.get("chunk_dim", -2))
    gemm_format = str(options.get("gemm_format", "TN").upper())

    def merge_optional(
        chunk_results: list[QuantResult], field: str, dim: int
    ) -> torch.Tensor | None:
        tensors = [getattr(result, field) for result in chunk_results]
        if all(tensor is None for tensor in tensors):
            return None
        if any(tensor is None for tensor in tensors):
            raise ValueError(
                f"pack_int4_moe produced inconsistent {field} results across chunks"
            )
        return torch.cat([tensor for tensor in tensors if tensor is not None], dim=dim)

    results: list[QuantResult] = []
    for expert_weight in value.unbind(dim=0):
        if not packed:
            results.append(quant_int4_moe(expert_weight, options))
            continue
        if expert_weight.shape[chunk_dim] % 2:
            raise ValueError(
                "pack_int4_moe requires an even last dimension when "
                f"packed=true, got {expert_weight.shape}"
            )
        chunk_results = [
            quant_int4_moe(chunk, options)
            for chunk in expert_weight.chunk(2, dim=chunk_dim)
        ]
        scale_dim = (
            -2
            if int(options.get("pack_version", 2)) != 1
            and options.get("scale_layout") == "row"
            else -1
        )
        weight_dim = -2 if gemm_format[0] == "T" else -1
        results.append(
            QuantResult(
                weight=torch.cat(
                    [result.weight for result in chunk_results], dim=weight_dim
                ),
                weight_scale=merge_optional(chunk_results, "weight_scale", scale_dim),
                i8_weight_scale=merge_optional(chunk_results, "i8_weight_scale", -2),
                i8_weight_zero=merge_optional(chunk_results, "i8_weight_zero", -2),
            )
        )

    def stack_optional(field: str) -> torch.Tensor | None:
        tensors = [getattr(result, field) for result in results]
        if all(tensor is None for tensor in tensors):
            return None
        if any(tensor is None for tensor in tensors):
            raise ValueError(f"pack_int4_moe produced inconsistent {field} results")
        return torch.stack([tensor for tensor in tensors if tensor is not None], dim=0)

    return QuantResult(
        weight=torch.stack([result.weight for result in results], dim=0),
        weight_scale=stack_optional("weight_scale"),
        i8_weight_scale=stack_optional("i8_weight_scale"),
        i8_weight_zero=stack_optional("i8_weight_zero"),
    )


@register_quant("pack_int8_moe")
def quant_pack_int8_moe(value: torch.Tensor, options: Mapping[str, Any]) -> QuantResult:
    """Per-expert symmetric INT8 for stacked MoE weights.

    Each expert is quantized with ``int8_per_channel`` on the logical ``(N, K)``
    layout. Scale is ``(E, N, 1)``. Weight is transposed to ``(E, K, N)`` for
    ``moe_w8a8_group_gemv`` (NN-only).
    """
    if value.dim() != 3:
        raise ValueError(
            "pack_int8_moe expects stacked expert weights with shape "
            f"(num_experts, dim0, dim1), got {value.shape}"
        )
    gemm_format = str(options.get("gemm_format", "NN")).upper()
    if gemm_format != "NN":
        raise ValueError(f"pack_int8_moe gemm_format must be NN, got {gemm_format!r}")
    results = [
        quant_int8_per_channel(expert_weight, options)
        for expert_weight in value.unbind(dim=0)
    ]
    scales = [result.weight_scale for result in results]
    if any(scale is None for scale in scales):
        raise ValueError("pack_int8_moe requires a weight_scale from every expert")
    weight = torch.stack([result.weight for result in results], dim=0)
    weight = weight.transpose(1, 2).contiguous()
    return QuantResult(
        weight=weight,
        weight_scale=torch.stack(
            [scale for scale in scales if scale is not None], dim=0
        ),
    )
