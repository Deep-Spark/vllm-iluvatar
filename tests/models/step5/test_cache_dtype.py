# SPDX-License-Identifier: Apache-2.0
"""Step5 cache dtype follows the active attention path."""

from types import SimpleNamespace

import pytest
from vllm.config import CacheConfig

from vllm_iluvatar.models.step5.model import IluStep3p5Attention, Step5Attention


class ObservedCache(Exception):
    pass


@pytest.mark.parametrize(
    ("layer_type", "sparse", "fp8", "expected"),
    [
        ("full_attention", "0", "1", "auto"),
        ("full_attention", "1", "1", "fp8"),
        ("sliding_attention", "0", "1", "fp8"),
        ("sliding_attention", "0", "0", "auto"),
    ],
)
def test_step5_cache_dtype(monkeypatch, layer_type, sparse, fp8, expected):
    def observe_cache_config(self, **kwargs):
        raise ObservedCache(kwargs["cache_config"].cache_dtype)

    monkeypatch.setenv("VLLM_ILU_STEP5_SPARSE_ATTENTION", sparse)
    monkeypatch.setenv("VLLM_ILU_STEP5_SPARSE_FP8", fp8)
    monkeypatch.setattr(IluStep3p5Attention, "__init__", observe_cache_config)
    config = SimpleNamespace(
        layer_types=[layer_type],
        sparse_config={"enabled": True},
        partial_rotary_factors=[1.0],
        num_attention_heads=64,
        num_attention_groups=4,
        head_dim=192,
        hidden_size=4096,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        use_head_wise_attn_gate=True,
    )
    with pytest.raises(ObservedCache) as observed:
        Step5Attention(
            config=config,
            cache_config=CacheConfig(cache_dtype="auto", block_size=16),
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
    assert observed.value.args == (expected,)
