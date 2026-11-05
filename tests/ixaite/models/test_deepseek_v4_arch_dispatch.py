# SPDX-License-Identifier: Apache-2.0
"""0731 compressed-tensors must dispatch to W4A8, not FP8 OOT."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch import (
    IluDeepseekV4Arch,
    IluDeepSeekV4MTPArch,
)
from vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch import (
    register_model as register_arch,
)
from vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint import (
    classify_deepseek_v4_checkpoint,
)
from vllm_iluvatar.models.deepseek_v4.fp8_model import (
    register_model as register_fp8,
)

_0731_HF = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "expert_dtype": "fp4",
    "iluvatar_quant": {"version": 2},
    "compression_config": {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_w4a8": {
                "weights": {"num_bits": 4, "type": "int"},
                "input_activations": {"num_bits": 8, "type": "int"},
            },
            "group_w8a8": {
                "weights": {"num_bits": 8, "type": "int"},
                "input_activations": {"num_bits": 8, "type": "int"},
            },
        },
    },
}


class _Sentinel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _vllm_config(hf):
    if isinstance(hf, dict):
        hf_obj = SimpleNamespace(**hf)
    else:
        hf_obj = hf
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_obj),
        quant_config=None,
    )


def test_arch_class_is_text_generation_model():
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    assert is_text_generation_model(IluDeepseekV4Arch)
    assert is_text_generation_model(IluDeepSeekV4MTPArch)
    assert classify_deepseek_v4_checkpoint(_0731_HF) == "w4a8"


def test_converted_ilu_method_is_w4a8():
    hf = {"quantization_config": {"quant_method": "ilu_deepseek_v4_w4a8"}}
    assert classify_deepseek_v4_checkpoint(hf) == "w4a8"


def test_fp8_quant_method_is_fp8():
    hf = {"quantization_config": {"quant_method": "fp8"}, "expert_dtype": "fp4"}
    assert classify_deepseek_v4_checkpoint(hf) == "fp8"


def test_empty_hf_is_unknown():
    assert classify_deepseek_v4_checkpoint({}) == "unknown"


def test_arch_constructs_w4a8_for_0731(monkeypatch):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.model.IluDeepseekV4ForCausalLM",
        _Sentinel,
    )
    out = IluDeepseekV4Arch(vllm_config=_vllm_config(_0731_HF))
    assert type(out) is _Sentinel


def test_mtp_arch_constructs_w4a8_for_0731(monkeypatch):
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.mtp.IluDeepSeekV4MTP",
        _Sentinel,
    )
    out = IluDeepSeekV4MTPArch(vllm_config=_vllm_config(_0731_HF))
    assert type(out) is _Sentinel


# FP8 keeps its baseline class and remains an explicit ixaite capability boundary.
def test_ixaite_arch_rejects_fp8():
    hf = {"quantization_config": {"quant_method": "fp8"}}
    with pytest.raises(RuntimeError, match="FP8 requires unavailable"):
        IluDeepseekV4Arch(vllm_config=_vllm_config(hf))


def test_unknown_quant_fails_closed():
    with pytest.raises(ValueError, match="cannot choose FP8 vs W4A8"):
        IluDeepseekV4Arch(vllm_config=_vllm_config({}))


# Each backend binds its own community architecture without changing checkpoint files.
def test_arch_registration_preserves_baseline_and_ixaite_choices(monkeypatch):
    calls = []
    import vllm
    monkeypatch.setattr(vllm.ModelRegistry, "register_model",
                        lambda arch, path: calls.append((arch, path)))
    register_fp8()
    assert [path for _, path in calls] == [
        "vllm_iluvatar.models.deepseek_v4.fp8_model:IluDeepseekV4Fp8ForCausalLM",
        "vllm_iluvatar.models.deepseek_v4.fp8_mtp:IluDeepseekV4Fp8MTP",
    ]
    calls.clear()
    register_arch()
    assert [path for _, path in calls] == [
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch:IluDeepseekV4Arch",
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch:IluDeepSeekV4MTPArch",
    ]
