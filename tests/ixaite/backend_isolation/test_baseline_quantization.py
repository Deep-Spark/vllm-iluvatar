# SPDX-License-Identifier: Apache-2.0
"""Protect baseline quantization options while adding the private route."""

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

from tests.ixaite.quantization.config_fixtures import _TARGET, _make_config
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    IluQuantConfig,
)


# Baseline retains upstream's opt-in contract when vendor metadata is absent.
def test_standard_w8a8_keeps_community_scheme():
    scheme = IluQuantConfig.from_config(_make_config(iluvatar={})).get_scheme(
        MagicMock(spec=LinearBase), _TARGET
    )
    assert type(scheme) is CompressedTensorsW8A8Int8


# Baseline continues to require explicit vendor execution options.
def test_partial_w8a8_options_are_not_completed_by_baseline():
    config = IluQuantConfig.from_config(_make_config(iluvatar={"gemm_mode": "NORMAL"}))
    with pytest.raises(AssertionError):
        config.get_scheme(MagicMock(spec=LinearBase), _TARGET)


# Baseline keeps upstream's validation of explicit vendor quantization semantics.
def test_explicit_w8a8_strategy_is_not_ignored():
    config = IluQuantConfig.from_config(
        _make_config(iluvatar={"mixed_strategy": "TENSOR"})
    )
    with pytest.raises(ValueError, match="channel-wise"):
        config.get_scheme(MagicMock(spec=LinearBase), _TARGET)
