# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar compressed-tensors linear scheme selection."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

from compressed_tensors.quantization import QuantizationStrategy


def _install_fake_ixformer(monkeypatch):
    ixformer = types.ModuleType("ixformer")
    inference = types.ModuleType("ixformer.inference")
    functions = types.ModuleType("ixformer.inference.functions")
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _load_ilu_compressed_tensors(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    return importlib.import_module(
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors"
    )


def test_w8a8_community_scheme_is_replaced_with_ilu_scheme(monkeypatch):
    ilu_ct = _load_ilu_compressed_tensors(monkeypatch)
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
        CompressedTensorsW8A8Int8,
    )

    community_scheme = CompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    ilu_ct.set_original_get_scheme_from_parts(lambda *args, **kwargs: community_scheme)

    weight_quant = SimpleNamespace(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    input_quant = SimpleNamespace(dynamic=True, symmetric=True)

    scheme = ilu_ct.IluCompressedTensorsConfig()._get_scheme_from_parts(
        weight_quant=weight_quant,
        input_quant=input_quant,
    )

    assert scheme.__class__.__name__ == "IluCompressedTensorsW8A8Int8"
    assert scheme.is_static_input_scheme is False
    assert scheme.input_symmetric is True


def test_w4a8_community_scheme_is_replaced_with_ilu_scheme(monkeypatch):
    ilu_ct = _load_ilu_compressed_tensors(monkeypatch)
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int import (  # noqa: E501
        CompressedTensorsW4A8Int,
    )

    community_scheme = CompressedTensorsW4A8Int(
        num_bits=4,
        strategy=QuantizationStrategy.CHANNEL,
        group_size=-1,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    ilu_ct.set_original_get_scheme_from_parts(lambda *args, **kwargs: community_scheme)

    weight_quant = SimpleNamespace(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=4,
        group_size=-1,
    )
    input_quant = SimpleNamespace(dynamic=True, symmetric=True)

    scheme = ilu_ct.IluCompressedTensorsConfig()._get_scheme_from_parts(
        weight_quant=weight_quant,
        input_quant=input_quant,
    )

    assert scheme.__class__.__name__ == "IluCompressedTensorsW4A8Int"
    assert scheme.group_size == -1
    assert scheme.input_symmetric is True


def test_unrelated_community_scheme_is_returned_unchanged(monkeypatch):
    ilu_ct = _load_ilu_compressed_tensors(monkeypatch)
    community_scheme = object()
    ilu_ct.set_original_get_scheme_from_parts(lambda *args, **kwargs: community_scheme)

    scheme = ilu_ct.IluCompressedTensorsConfig()._get_scheme_from_parts(
        weight_quant=SimpleNamespace(),
        input_quant=SimpleNamespace(),
    )

    assert scheme is community_scheme
