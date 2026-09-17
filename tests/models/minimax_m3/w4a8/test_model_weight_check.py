# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.minimax_m3.w4a8.model import (
    _assert_m3_parameters_loaded,
)


class _DenseAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.self_attn = nn.Module()
        self.self_attn.attn = nn.Module()
        for name in (
            "q_scale",
            "k_scale",
            "v_scale",
            "q_zero_point",
            "k_zero_point",
            "v_zero_point",
        ):
            self.self_attn.attn.register_parameter(name, nn.Parameter(torch.ones(1)))


def test_load_check_skips_runtime_attention_quantization_parameters() -> None:
    model = nn.Module()
    model.layer = _DenseAttentionLayer()

    _assert_m3_parameters_loaded(model, {"layer.weight"}, kv_cache_dtype="auto")


def test_load_check_requires_kv_scales_when_fp8_cache() -> None:
    model = nn.Module()
    model.layer = _DenseAttentionLayer()

    with pytest.raises(ValueError, match=r"k_scale"):
        _assert_m3_parameters_loaded(model, {"layer.weight"}, kv_cache_dtype="fp8")


def test_load_check_still_rejects_missing_checkpoint_parameters() -> None:
    model = nn.Module()
    model.layer = _DenseAttentionLayer()

    with pytest.raises(ValueError, match=r"layer\.weight"):
        _assert_m3_parameters_loaded(model, set())
