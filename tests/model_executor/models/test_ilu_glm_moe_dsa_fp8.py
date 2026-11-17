# SPDX-License-Identifier: Apache-2.0
"""CPU tests for GLM-5.2 official-FP8 weight streaming helpers."""

import pytest
import torch

from vllm_iluvatar.models.glm_moe_dsa.fp8_model import (
    _consumer_quant_method,
    _consumes_int8_tuple,
    dequant_indexer_wk,
)

_WK = "model.layers.0.self_attn.indexer.wk.weight"
_SCALE = "model.layers.0.self_attn.indexer.wk.weight_scale_inv"


def _fp8_ones(n: int = 128, k: int = 256) -> torch.Tensor:
    return torch.ones(n, k, dtype=torch.float8_e4m3fn)


def _block_scale(n: int = 128, k: int = 256, value: float = 0.5) -> torch.Tensor:
    return torch.full(((n + 127) // 128, (k + 127) // 128), value, dtype=torch.float32)


def test_dequant_indexer_wk_folds_scale_in_either_order():
    weight = _fp8_ones()
    scale = _block_scale()
    other = torch.tensor(1.0)

    weight_first = dict(
        dequant_indexer_wk(
            [
                (_WK, weight),
                (_SCALE, scale),
                ("other", other),
            ]
        )
    )
    scale_first = dict(
        dequant_indexer_wk(
            [
                (_SCALE, scale),
                (_WK, weight),
                ("other", other),
            ]
        )
    )

    assert set(weight_first) == {_WK, "other"}
    assert set(scale_first) == {_WK, "other"}
    assert weight_first["other"] is other
    torch.testing.assert_close(weight_first[_WK], scale_first[_WK])
    torch.testing.assert_close(
        weight_first[_WK],
        torch.full((128, 256), 0.5, dtype=torch.bfloat16),
    )


def test_dequant_indexer_wk_passes_through_bf16_wk():
    bf16 = torch.ones(4, 4, dtype=torch.bfloat16)
    out = dict(dequant_indexer_wk([(_WK, bf16)]))
    assert out[_WK] is bf16


def test_dequant_indexer_wk_rejects_unpaired_scale():
    with pytest.raises(ValueError, match="unpaired indexer.wk"):
        list(dequant_indexer_wk([(_SCALE, _block_scale())]))


class _Scheme:
    pass


class _RoutedExperts:
    def __init__(self, quant_method):
        self.quant_method = quant_method


class _MoERunner:
    """Shaped like ``FusedMoEFactory``'s return value."""

    def __init__(self, quant_method):
        self.routed_experts = _RoutedExperts(quant_method)

    @property
    def _quant_method(self):
        return self.routed_experts.quant_method


def test_moe_runner_keeps_its_method_on_routed_experts():
    """The structural fact ``_consumer_quant_method`` exists to handle.

    A ``MoERunner`` exposes the method only as ``_quant_method``; reading
    ``quant_method`` off the runner yields ``None`` and would disarm the fused
    RMSNorm+INT8 path for every MoE layer.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    assert isinstance(getattr(MoERunner, "_quant_method", None), property)
    assert not isinstance(getattr(MoERunner, "quant_method", None), property)


def test_consumer_quant_method_resolves_through_moe_runner():
    from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.w4a8.nn_moe import (
        IluW4A8NNMoEMethod,
    )

    method = object.__new__(IluW4A8NNMoEMethod)
    assert _consumer_quant_method(_MoERunner(method)) is method
    assert _consumes_int8_tuple(_MoERunner(method))


def test_consumes_int8_tuple_false_for_other_moe_method():
    assert not _consumes_int8_tuple(_MoERunner(object()))


def test_consumer_quant_method_refuses_to_guess():
    with pytest.raises(RuntimeError, match="exposes no quant_method"):
        _consumer_quant_method(_Scheme())


def test_consumes_int8_tuple_accepts_w8a8_linear_scheme():
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
        IluCompressedTensorsW8A8Int8,
    )

    linear = _Scheme()
    linear.scheme = object.__new__(IluCompressedTensorsW8A8Int8)
    # No quant_method on purpose: the scheme alone must settle it.
    assert _consumes_int8_tuple(linear)
