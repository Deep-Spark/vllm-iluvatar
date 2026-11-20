# SPDX-License-Identifier: Apache-2.0
"""GLM KDA cache planning and layer state honor the SSM dtype override."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.glm5next.kda import IluGlm5NextLinearAttention
from vllm_iluvatar.models.glm5next.model import (
    IluGlm5NextForCausalLM,
    IluGlm5NextForConditionalGeneration,
)


def _config(conv_dtype, ssm_dtype):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(
            mamba_cache_dtype=conv_dtype,
            mamba_ssm_cache_dtype=ssm_dtype,
            mamba_block_size=4480,
            mamba_page_size_padded=None,
            mamba_cache_mode="align",
            use_kda_recoverssm=False,
        ),
        num_speculative_tokens=5,
    )


def _layer(config):
    layer = IluGlm5NextLinearAttention.__new__(IluGlm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.model_config = config.model_config
    layer.cache_config = config.cache_config
    layer.tp_size = 2
    layer.num_heads = 64
    layer.head_dim = 128
    layer.conv_size = 4
    layer.num_spec = config.num_speculative_tokens
    return layer


@pytest.mark.parametrize(
    ("conv_dtype", "ssm_dtype", "expected"),
    [
        ("auto", "auto", (torch.bfloat16, torch.float32)),
        ("bfloat16", "float32", (torch.bfloat16, torch.float32)),
        ("auto", "bfloat16", (torch.bfloat16, torch.bfloat16)),
        ("float32", "bfloat16", (torch.float32, torch.bfloat16)),
    ],
)
def test_kda_state_dtype_and_page_size(conv_dtype, ssm_dtype, expected):
    config = _config(conv_dtype, ssm_dtype)
    layer = _layer(config)
    spec = layer.get_kv_cache_spec(config)

    assert layer.get_state_dtype() == expected
    assert spec.dtypes == expected
    for model in (IluGlm5NextForCausalLM, IluGlm5NextForConditionalGeneration):
        assert model.get_mamba_state_dtype_from_config(config) == expected

    conv_elements = 3 * 32 * 128 * (3 + 5)
    recurrent_elements = 32 * 128 * 128
    conv_bytes = 2 if expected[0] == torch.bfloat16 else 4
    state_bytes = 2 if expected[1] == torch.bfloat16 else 4
    assert prod(spec.shapes[0]) == conv_elements
    assert prod(spec.shapes[1]) == recurrent_elements
    assert spec.state_content_size_bytes == (
        conv_elements * conv_bytes + recurrent_elements * state_bytes
    )


def test_kda_rejects_unsupported_ssm_dtype():
    config = _config("auto", "float16")
    for get_dtype in (
        _layer(config).get_state_dtype,
        lambda: IluGlm5NextForCausalLM.get_mamba_state_dtype_from_config(config),
        lambda: IluGlm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(
            config
        ),
    ):
        with pytest.raises(ValueError, match="KDA recurrent state supports"):
            get_dtype()
