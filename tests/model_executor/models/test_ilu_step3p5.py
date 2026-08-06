# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Step3.5 OOT model registration."""

from __future__ import annotations

import inspect

from vllm_iluvatar.model_executor.models import step3p5 as ilu_step3p5
from vllm_iluvatar.model_executor.models import step3p5_mtp as ilu_step3p5_mtp


def test_step3p5_oot_model_constructors_expose_vllm_config():
    for model_cls in (
        ilu_step3p5.IluStep3p5ForCausalLM,
        ilu_step3p5_mtp.IluStep3p5MTP,
    ):
        parameters = inspect.signature(model_cls.__init__).parameters
        assert parameters["vllm_config"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["prefix"].default == ""


def test_step3p5_causal_lm_refreshes_pp_intermediate_factory():
    source = inspect.getsource(ilu_step3p5.IluStep3p5ForCausalLM.__init__)

    assert "self.model.make_empty_intermediate_tensors" in source


def test_step3p5_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_step3p5.register_model()
    ilu_step3p5_mtp.register_model()

    assert calls == [
        (
            "Step3p5ForCausalLM",
            "vllm_iluvatar.model_executor.models.step3p5:IluStep3p5ForCausalLM",
        ),
        (
            "Step3p5MTP",
            "vllm_iluvatar.model_executor.models.step3p5_mtp:IluStep3p5MTP",
        ),
    ]
