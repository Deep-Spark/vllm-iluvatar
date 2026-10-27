# SPDX-License-Identifier: Apache-2.0
"""Producer compatibility and fallbacks without initializing a GPU context."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import query_quant as module
from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention


def linear(*, static=False, symmetric=True):
    scheme = object.__new__(module.IluCompressedTensorsW8A8Int8)
    scheme.is_static_input_scheme = static
    scheme.input_symmetric = symmetric
    return SimpleNamespace(scheme=scheme)


@pytest.mark.parametrize(
    "width,key,expected",
    [
        (1280, 512, True),
        (640, 512, True),
        (2048, 512, True),
        (384, 512, False),
        (1281, 512, False),
        (1280, 256, False),
    ],
)
def test_capability_follows_generic_native_shape_contract(width, key, expected):
    assert module.can_fuse_query_quant([linear(), linear()], width, key) is expected


@pytest.mark.parametrize(
    "other", [None, object(), linear(static=True), linear(symmetric=False)]
)
def test_every_projection_must_accept_the_same_dynamic_symmetric_activation(other):
    assert not module.can_fuse_query_quant([linear(), other], 1280, 512)
    assert not module.can_fuse_query_quant([], 1280, 512)


@pytest.mark.parametrize(
    "enabled,dtype,rows",
    [
        (False, torch.bfloat16, 6),
        (True, torch.float32, 6),
        (True, torch.bfloat16, 0),
        (True, torch.bfloat16, 6),
    ],
)
def test_cpu_empty_and_disabled_inputs_keep_existing_norm_path(enabled, dtype, rows):
    packed = torch.zeros(rows, 1792, dtype=dtype)
    model = SimpleNamespace(
        q_lora_rank=1280,
        head_dim=512,
        _fuse_query_quant=enabled,
        q_norm=SimpleNamespace(weight=torch.ones(1280)),
        kv_norm=SimpleNamespace(weight=torch.ones(512)),
        eps=1e-6,
    )
    expected = (object(), object())
    with (
        patch(
            "vllm_iluvatar.models.deepseek_v41.attention.ixfops.rms_norm_qk",
            return_value=expected,
        ) as old,
        patch.object(
            module.ixfops,
            "rms_norm_qk_dynamic_int8",
            side_effect=AssertionError("unexpected fusion"),
        ),
    ):
        actual = IluDeepseekV41Attention._split_qkv_and_norm(model, packed)
    assert actual is expected and old.call_count == 1
    assert not torch.cuda.is_initialized()
