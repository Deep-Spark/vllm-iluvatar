# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Iluvatar Engram architecture whitelist."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_iluvatar.ilu_patch.general_patches.patch_engram_config import (
    _as_upstream_engram_arch,
    _reject_ilu_engram_ubatch,
    register,
)


def _model(architecture: str, layer_ids):
    return SimpleNamespace(
        architecture=architecture,
        hf_text_config=SimpleNamespace(engram_layer_ids=layer_ids),
    )


def test_ilu_v41_arch_is_an_engram_model():
    register()
    from vllm.config.engram import model_has_engram_layers

    assert model_has_engram_layers(
        _model("IluDeepseekV41ForCausalLM", [1, 14])
    )
    assert model_has_engram_layers(_model("DeepseekV41ForCausalLM", [1]))
    assert not model_has_engram_layers(_model("IluDeepseekV41ForCausalLM", []))
    assert not model_has_engram_layers(_model("IluDeepseekV4ForCausalLM", [1]))


def test_ilu_engram_rejects_ubatch():
    cfg = SimpleNamespace(
        model_config=_model("IluDeepseekV41ForCausalLM", [1]),
        speculative_config=None,
        parallel_config=SimpleNamespace(use_ubatching=True),
    )
    with pytest.raises(ValueError, match="does not support DBO"):
        _reject_ilu_engram_ubatch(cfg)


class _Arch:
    def __init__(self, name: str) -> None:
        self._architecture = name

    @property
    def architecture(self) -> str:
        return self._architecture


def test_ilu_arch_is_upstream_only_while_the_dp_group_is_built():
    model = _Arch("IluDeepseekV41ForCausalLM")
    with _as_upstream_engram_arch(model):
        assert model.architecture == "DeepseekV41ForCausalLM"
    assert model.architecture == "IluDeepseekV41ForCausalLM"


def test_upstream_arch_is_left_unchanged():
    model = _Arch("DeepseekV41ForCausalLM")
    with _as_upstream_engram_arch(model):
        assert model.architecture == "DeepseekV41ForCausalLM"
    assert model.architecture == "DeepseekV41ForCausalLM"


def test_ilu_engram_ubatch_check_ignores_other_arches():
    cfg = SimpleNamespace(
        model_config=_model("DeepseekV41ForCausalLM", [1]),
        speculative_config=None,
        parallel_config=SimpleNamespace(use_ubatching=True),
    )
    _reject_ilu_engram_ubatch(cfg)
