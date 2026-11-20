# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash dense Linear dispatch off the raw FP8 checkpoint.

The V4 config only overrode routed experts, so Linear fell through to
community ``Fp8LinearMethod`` and never consulted ``fp8_linear_backend()``.
These cover the ivcore11 takeover and that ivcore30 keeps falling through.
Constructing ``IluFp8LinearMethod`` builds a W8A8 scheme that reads the local
arch, so the dense branch is asserted by dispatch, not by instantiation.
"""

from types import SimpleNamespace

import pytest
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v4_fp8 import (
    IluDeepseekV4FP8Config,
)

# DeepSeek-V4-Flash: attention / shared experts are block-FP8 [128, 128] with
# UE8M0 scales; the routed experts are MXFP4 and keep their own branch.
V4_CONFIG = dict(
    ignored_layers=[], packed_modules_mapping={}, weight_block_size=[128, 128]
)


def _select(monkeypatch, architecture, prefix, preference="auto", **overrides):
    monkeypatch.setattr(arch, "cuda_arch", lambda: architecture)
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": preference}}}
        ),
    )
    config = SimpleNamespace(**{**V4_CONFIG, **overrides})
    return IluDeepseekV4FP8Config._int8_linear_method(
        config, object.__new__(LinearBase), prefix
    )


@pytest.mark.parametrize("preference", ["auto", "w8a8_int8"])
def test_ivcore11_wo_a_uses_grouped_int8(monkeypatch, preference):
    """``wo_a`` is contracted per o_group, so it needs the grouped loader."""
    from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
        IluDeepseekV41Fp8GroupedInt8Method,
        IluV4W8A8BMMLinearMethod,
    )

    method = _select(
        monkeypatch, "ivcore11", "model.layers.0.attn.wo_a", preference=preference
    )
    assert type(method) is IluDeepseekV41Fp8GroupedInt8Method
    assert method.apply_with_rope.__func__ is IluV4W8A8BMMLinearMethod.apply_with_rope


@pytest.mark.parametrize("preference", ["auto", "fp8", "w8a8_int8"])
def test_only_ivcore11_is_claimed(monkeypatch, preference):
    """None keeps the FP8 execution path: community method + block kernel.

    The takeover is architecture-gated, not backend-gated: ivcore30 accepts an
    explicit ``w8a8_int8`` preference for other models, and that must not drag
    this checkpoint off its working FP8 path.
    """
    for prefix in ("model.layers.0.attn.wo_a", "model.layers.0.attn.wq_b"):
        assert _select(monkeypatch, "ivcore30", prefix, preference=preference) is None


def test_ignored_layer_stays_unquantized(monkeypatch):
    method = _select(
        monkeypatch,
        "ivcore11",
        "model.layers.0.ffn.gate",
        ignored_layers=["model.layers.0.ffn.gate"],
    )
    assert type(method) is UnquantizedLinearMethod
