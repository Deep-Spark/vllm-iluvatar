# SPDX-License-Identifier: Apache-2.0
"""Native NN MoE must not fill or copy routing buffers on the normal path."""

import pytest
import torch
from conftest import requires_corex30
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
    ilu_w16a16_fused_experts,
)

pytestmark = requires_corex30


def _case(tokens=19, routing="mixed"):
    torch.manual_seed(93)
    x = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda") / 8
    w13 = torch.randn(4, 128, 192, dtype=torch.bfloat16, device="cuda") / 8
    w2 = torch.randn(4, 96, 128, dtype=torch.bfloat16, device="cuda") / 8
    ids = torch.randint(8, (tokens, 4), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(tokens, 4, device="cuda"), -1)
    expert_map = torch.tensor(
        [2, -1, 0, -1, 3, -1, 1, -1], device="cuda", dtype=torch.int32
    )
    if routing == "remote":
        ids.fill_(1)
    elif routing == "duplicate":
        ids.fill_(2)
    return x, w13, w2, weights, ids, expert_map


def _run(case, activation, masked):
    x, w13, w2, weights, ids, expert_map = case
    return ilu_w16a16_fused_experts(
        x,
        w13,
        w2,
        weights,
        ids,
        activation,
        expert_map,
        step5_native_activation=masked,
    )


class _Audit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "activation,masked", [("silu", False), ("swiglustep", False), ("swiglustep", True)]
)
@torch.inference_mode()
def test_no_fill_cast_copy_or_cpu_read(activation, masked):
    case = _case()
    with _Audit() as audit:
        result = _run(case, activation, masked)
    forbidden = (
        "aten.to.",
        "_to_copy",
        "copy_",
        "zero_",
        "zeros",
        "fill_",
        "cumsum",
        "stack",
        "aten.item.",
        "_local_scalar_dense",
    )
    assert not [name for name in audit.calls if any(s in name for s in forbidden)]
    expected = _run(case, activation, masked)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "activation,masked", [("silu", False), ("swiglustep", False), ("swiglustep", True)]
)
@pytest.mark.parametrize("routing", ["mixed", "remote", "duplicate"])
@torch.inference_mode()
def test_unwritten_capacity_tail_does_not_affect_valid_rows(
    activation, masked, routing, monkeypatch
):
    case = _case(routing=routing)
    expected = _run(case, activation, masked)
    empty = torch.empty
    poison_calls = []

    def poisoned(*args, **kwargs):
        result = empty(*args, **kwargs)
        if result.shape == (case[4].numel(), 192) and result.dtype == torch.bfloat16:
            result.fill_(float("nan"))
            poison_calls.append(True)
        return result

    monkeypatch.setattr(torch, "empty", poisoned)
    actual = _run(case, activation, masked)
    assert poison_calls
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_strided_routing_compatibility(ids_dtype, weights_dtype):
    case = list(_case())
    ids = torch.empty(19, 8, dtype=ids_dtype, device="cuda")[:, ::2]
    weights = torch.empty(19, 8, dtype=weights_dtype, device="cuda")[:, ::2]
    ids.copy_(case[4])
    weights.copy_(case[3])
    case[3], case[4] = weights, ids
    actual = _run(case, "swiglustep", True)
    case[3], case[4] = weights.float().contiguous(), ids.int().contiguous()
    assert torch.equal(actual, _run(case, "swiglustep", True))
