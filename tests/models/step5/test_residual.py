# SPDX-License-Identifier: Apache-2.0
"""Configuration and PP receive buffers preserve residual precision."""

from types import SimpleNamespace

import torch

from vllm_iluvatar.models.step5.config import Step5Config, Step5VLConfig
from vllm_iluvatar.models.step5.model import Step5Model, Step5RMSNorm


def test_custom_config_preserves_explicit_embedding_tie_policy():
    assert Step5Config().tie_word_embeddings is False
    assert Step5Config(tie_word_embeddings=True).tie_word_embeddings is True
    assert Step5VLConfig(text_config={}).tie_word_embeddings is False
    assert (
        Step5VLConfig(text_config={"tie_word_embeddings": True}).tie_word_embeddings
        is True
    )


def test_fp32_residual_pp_receive_buffer():
    model = Step5Model.__new__(Step5Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=32)
    model.fp32_residual_connection = True
    buffers = model.make_empty_intermediate_tensors(
        2, torch.bfloat16, torch.device("cpu")
    )
    source = torch.linspace(0.99, 1.01, 64).view(2, 32)
    buffers["hidden_states"].copy_(source)
    assert buffers["hidden_states"].dtype == torch.float32
    torch.testing.assert_close(buffers["hidden_states"], source, atol=0, rtol=0)
    model.fp32_residual_connection = False
    assert (
        model.make_empty_intermediate_tensors(2, torch.bfloat16, torch.device("cpu"))[
            "hidden_states"
        ].dtype
        == torch.bfloat16
    )


def test_norm_preserves_fp32_residual_storage():
    layer = Step5RMSNorm(32, 1e-5)
    residual = torch.randn(4, 32)
    saved = residual.clone()
    result = layer(residual)
    torch.testing.assert_close(residual, saved, atol=0, rtol=0)
    expected = saved * torch.rsqrt(saved.square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(result, expected)
    assert result.data_ptr() != residual.data_ptr()
