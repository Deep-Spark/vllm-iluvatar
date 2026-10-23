# SPDX-License-Identifier: Apache-2.0
"""Small-weight checks for W4A8 expert packing and device transfers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_METHODS_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "iluvatar-compressed-tensors"
    / "methods.py"
)
_SPEC = importlib.util.spec_from_file_location("ilu_ct_moe_pack_methods", _METHODS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
methods = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = methods
_SPEC.loader.exec_module(methods)


class _RejectCudaToCpu(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten._to_copy.default:
            destination = kwargs.get("device")
            if (
                args[0].device.type == "cuda"
                and destination is not None
                and torch.device(destination).type == "cpu"
            ):
                raise AssertionError("in-memory packing must not copy CUDA data to CPU")
        return func(*args, **kwargs)


def _options(packed: bool) -> dict:
    return {
        "group_size": -1,
        "symmetric": True,
        "gemm_format": "TN",
        "pack_version": 2,
        "scale_layout": "row",
        "packed": packed,
        "chunk_dim": -2,
    }


@pytest.mark.parametrize("packed", [False, True], ids=["down", "gate_up"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pack_int4_moe_keeps_outputs_on_input_device(packed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    rows = 128 if packed else 64
    value = torch.randn(2, rows, 64, generator=torch.Generator().manual_seed(7))
    value = value.to(device)
    expected = methods.quant_pack_int4_moe(value, _options(packed))

    # Final device checks alone would miss the old CUDA -> CPU -> CUDA detour.
    with _RejectCudaToCpu():
        result = methods.quant_pack_int4_moe(value, _options(packed))

    for field in ("weight", "weight_scale", "i8_weight_scale", "i8_weight_zero"):
        actual, reference = getattr(result, field), getattr(expected, field)
        if reference is None:
            assert actual is None
        else:
            assert actual.device == value.device
            assert torch.equal(actual, reference)


@pytest.mark.parametrize("packed", [False, True], ids=["down", "gate_up"])
def test_stream_pack_int4_moe_offloads_outputs_without_changing_values(packed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    rows = 128 if packed else 64
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(2, rows, 64, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.randint(
        124, 130, (2, rows, 2), dtype=torch.uint8, generator=generator
    )
    options = _options(packed)
    value = methods.dequant_fp8_block_e8m0(weight, scale, {"block_size": 32})
    expected = methods.quant_pack_int4_moe(value.cuda(), options)

    result = methods.pack_int4_moe_stream_dequant(
        weight,
        scale,
        dequant_method="fp8_block_e8m0",
        dequant_options={"block_size": 32},
        quant_options=options,
        work_device="cuda",
    )

    for field in ("weight", "weight_scale", "i8_weight_scale", "i8_weight_zero"):
        actual, reference = getattr(result, field), getattr(expected, field)
        if reference is None:
            assert actual is None
        else:
            assert actual.device.type == "cpu"
            assert torch.equal(actual, reference.cpu())
