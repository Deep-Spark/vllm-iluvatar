# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for compressed-tensors padding helpers."""

from __future__ import annotations

import pytest

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    hidden_padding,
)


def test_hidden_padding_zero_when_already_aligned():
    assert hidden_padding(2048, 2048) == 0
    assert hidden_padding(384, 384) == 0


def test_hidden_padding_qwen3_tp4_down_proj():
    # Qwen3-30B-A3B TP=4: intermediate/TP = 192, W4A8 TN pad_k = 256.
    assert hidden_padding(192, 256) == 64


def test_hidden_padding_rejects_shrink():
    with pytest.raises(AssertionError, match="pad_k"):
        hidden_padding(256, 192)
