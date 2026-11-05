# SPDX-License-Identifier: Apache-2.0
"""Explicit all-mode capability and compatibility of the implicit cache default."""

from types import SimpleNamespace

import pytest

from vllm_iluvatar.backends.ixaite.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm_iluvatar.backends.ixaite.models.qwen3_5_apc import IluQwen3_5Config


@pytest.mark.parametrize(
    "requested,expected", [("none", "align"), ("align", "align"), ("all", "all")]
)
def test_dense_cache_mode_contract(requested, expected):
    """The explicit all setting survives validation; the old default remains align."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode=requested,
            mamba_ssm_cache_dtype="auto",
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_ssm_dtype="float32")
        ),
        speculative_config=None,
    )
    IluQwen3_5Config.verify_and_update_config(config)
    assert config.cache_config.mamba_cache_mode == expected
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"
    assert Qwen3_5ForConditionalGeneration.supports_mamba_prefix_caching


def test_speculative_all_is_explicitly_rejected():
    """An unsupported combination must fail before cache allocation or model loading."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, mamba_cache_mode="all"
        ),
        speculative_config=object(),
    )
    with pytest.raises(NotImplementedError, match="non-speculative"):
        IluQwen3_5Config.verify_and_update_config(config)
