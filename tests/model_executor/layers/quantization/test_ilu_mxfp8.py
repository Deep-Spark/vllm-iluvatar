# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar MXFP8 routed-expert method and config.

The failure this file exists to prevent is the one that actually happened: the
community MXFP8 MoE selector lands on ``MARLIN`` (``MarlinExperts`` gates only on
"CUDA SM75+", and ``IluPlatform._enum`` is CUDA), and the layer then dies at load
time in ``prepare_mxfp8_moe_layer_for_marlin`` -> ``gptq_marlin_repack``, an op
the Iluvatar build does not define.

Covered here:

* ``__init__`` never calls ``select_mxfp8_moe_backend``, and leaves
  ``experts_cls`` unset so community ``is_monolithic()`` cannot dereference it.
* the UE8M0 scale is *decoded*, not cast -- a plain ``.to(float32)`` reads the
  exponent byte as an integer (0x7F -> 127.0 instead of 1.0), which would scale
  weights by ~1e38 rather than erroring.
* the config only swaps ``FusedMoEMethodCls``; Linear keeps reaching the
  community method, which is what calls ``init_mxfp8_linear_kernel()``. (The
  ivcore11 INT8 takeover in front of it lives in ``test_ilu_mxfp8_linear.py``.)

The load lifecycle itself -- contract selection, paired streaming, conversion --
is shared with block-FP8 and covered in ``test_ilu_mxfp8_moe.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
    _e8m0_to_fp32 as e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.moe import (
    IluMxFp8MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs.mxfp8 import (
    IluMxFp8Config,
)

_BLOCK = 32
_H = 512
_I = 256
_E = 2


def _make_method():
    quant_config = MagicMock()
    quant_config.is_checkpoint_mxfp8_serialized = True
    return IluMxFp8MoEMethod(quant_config=quant_config, moe_config=MagicMock())


def test_init_skips_community_backend_selection(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as oracle

    def _boom(*args, **kwargs):
        raise AssertionError("select_mxfp8_moe_backend must not be called")

    monkeypatch.setattr(oracle, "select_mxfp8_moe_backend", _boom)

    method = _make_method()

    assert method.moe_kernel is None
    # The OCP grid is fixed, so nothing refines it per partition.
    assert method.weight_block_size == [1, _BLOCK]
    assert method.block_quant is True
    assert method.weight_scale_name == "weight_scale"
    assert method.weight_scale_refine is None
    # Community ``FusedMoEMethodBase.is_monolithic`` dereferences ``experts_cls``;
    # leaving it unset (not None) keeps that path unreachable.
    assert not hasattr(method, "experts_cls")
    assert method.is_monolithic is False
    assert method.supports_eplb is False


def test_the_method_is_the_block_fp8_lifecycle_with_another_source():
    """Only the source differs; anything else overridden is a regression."""
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        IluFp8MoEMethod,
    )

    assert IluMxFp8MoEMethod.compute_source == "MXFP8"
    defined = {
        name
        for name in IluMxFp8MoEMethod.__dict__
        if not name.startswith(("__", "_abc"))
    }
    assert defined == {"compute_source", "_make_loader"}
    for name in ("create_weights", "process_weights_after_loading", "apply"):
        assert getattr(IluMxFp8MoEMethod, name) is getattr(IluFp8MoEMethod, name)


def test_e8m0_scale_is_decoded_not_cast():
    raw = torch.tensor([[0x7F, 0x80, 0x7E, 0x81]], dtype=torch.uint8)

    decoded = e8m0_to_fp32(raw)

    assert torch.equal(decoded, torch.tensor([[1.0, 2.0, 0.5, 4.0]]))
    # The silent-failure mode: ``.to(float32)`` yields 127/128/126/129.
    assert not torch.equal(decoded, raw.to(torch.float32))
    # A float8_e8m0fnu view of the same bytes must decode identically.
    assert torch.equal(e8m0_to_fp32(raw.view(torch.float8_e8m0fnu)), decoded)


def test_both_mxfp8_tagging_conventions_are_claimed():
    """MiniMax-M3 tags ``quant_method: "mxfp8"``; NVIDIA ModelOpt resolves to
    ``modelopt_mxfp8``. Each registry name has its own ILU config.

    ``_registration`` is imported explicitly because that is what makes the
    answer independent of which other module happened to be imported first:
    the registration is a side effect of importing the config module, and
    nothing else in this file's import graph pulls in the ModelOpt one.
    """
    from vllm.model_executor.layers.quantization import get_quantization_config

    from vllm_iluvatar.model_executor.layers.quantization import (  # noqa: F401
        _registration,
    )
    from vllm_iluvatar.model_executor.layers.quantization.configs.modelopt.mxfp8 import (  # noqa: E501
        IluModelOptMxFp8Config,
    )

    assert get_quantization_config("mxfp8") is IluMxFp8Config
    assert get_quantization_config("modelopt_mxfp8") is IluModelOptMxFp8Config


def test_config_only_swaps_the_moe_method():
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
        build_linear_method,
    )

    assert IluMxFp8Config.FusedMoEMethodCls is IluMxFp8MoEMethod
    # v0.31.0: ``LinearMethodCls`` is dropped; Linear dispatch
    # goes through the community ``build_linear_method`` -> generic
    # ``ModelOptLinearMethod``
    # (whose ``create_weights`` calls ``init_mxfp8_linear_kernel``). The ILU
    # config must not override that path -- only MoE is swapped.
    assert "LinearMethodCls" not in IluMxFp8Config.__dict__
    quant_config = IluMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert quant_config.get_name() == "mxfp8"
    assert isinstance(
        build_linear_method(quant_config, quant_config.quant_method, "lm_head"),
        ModelOptLinearMethod,
    )
