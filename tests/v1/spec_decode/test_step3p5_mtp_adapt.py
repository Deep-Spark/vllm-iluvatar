# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Step-3.5 / 3.7 MTP-3 draft-layer config and proposer."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch
from vllm_iluvatar.model_executor.models.step3p5_mtp_layer_config import (
    _extend_mtp_layer_config,
    _restore_mtp_layer_config,
)
from vllm_iluvatar.v1.spec_decode.step3p5 import (
    _ensure_draft_vllm_config,
    _ForceMtpLayer0,
    make_build_per_group_and_layer_attn_metadata,
    make_create_draft_vllm_config,
    make_propose,
    make_sample_draft_tokens_for_step,
)


def test_extend_mtp_layer_types_for_draft_index():
    config = SimpleNamespace(
        num_hidden_layers=45,
        layer_types=["full_attention"] * 45,
        attention_other_setting={"attention_type": "sliding_attention"},
        partial_rotary_factors=[0.5] * 45,
    )
    orig_lt, orig_pr = _extend_mtp_layer_config(config, 47)
    assert config.layer_types[47] == "sliding_attention"
    assert len(config.layer_types) == 48
    assert config.partial_rotary_factors[47] == 0.5
    _restore_mtp_layer_config(config, orig_lt, orig_pr)
    assert config.layer_types == ["full_attention"] * 45
    assert config.partial_rotary_factors == [0.5] * 45


def test_extend_mtp_layer_types_skips_backbone_index():
    types = ["full_attention"] * 45
    config = SimpleNamespace(
        num_hidden_layers=45,
        layer_types=list(types),
        attention_other_setting={"attention_type": "sliding_attention"},
        partial_rotary_factors=[0.5] * 45,
    )
    _extend_mtp_layer_config(config, 3)
    assert config.layer_types == types


def test_create_draft_vllm_config_forces_pp1(monkeypatch):
    seen = {}

    def fake_replace(cfg, **kwargs):
        return SimpleNamespace(parallel_config=kwargs["parallel_config"])

    import vllm.config as vllm_config

    monkeypatch.setattr(vllm_config, "replace", fake_replace)

    def orig(self):
        seen["pp"] = self.vllm_config.parallel_config.pipeline_parallel_size
        return "draft-cfg"

    wrapped = make_create_draft_vllm_config(orig)
    proposer = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=4)
        ),
    )
    assert wrapped(proposer) == "draft-cfg"
    assert seen["pp"] == 1
    assert proposer.vllm_config.parallel_config.pipeline_parallel_size == 4


def test_create_draft_vllm_config_keeps_pp1(monkeypatch):
    def fail_replace(*_args, **_kwargs):
        raise AssertionError("PP=1 must not rebuild parallel config")

    import vllm.config as vllm_config

    monkeypatch.setattr(vllm_config, "replace", fail_replace)

    def orig(self):
        assert self.vllm_config.parallel_config.pipeline_parallel_size == 1
        return "draft-cfg"

    wrapped = make_create_draft_vllm_config(orig)
    proposer = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1)
        ),
    )
    assert wrapped(proposer) == "draft-cfg"


def test_ensure_draft_vllm_config_is_lazy():
    class Dummy:
        def __init__(self):
            self.draft_vllm_config = None
            self.calls = 0

        def _create_draft_vllm_config(self):
            self.calls += 1
            return "draft-cfg"

    proposer = Dummy()
    _ensure_draft_vllm_config(proposer)
    _ensure_draft_vllm_config(proposer)
    assert proposer.draft_vllm_config == "draft-cfg"
    assert proposer.calls == 1


def test_make_propose_wraps_orig_and_forces_layer_zero():
    seen: dict[str, object] = {}

    class DummyModel:
        def __call__(self, **kwargs):
            seen["model_kwargs"] = kwargs
            return kwargs

    dummy_model = DummyModel()

    def orig(self, *args, **kwargs):
        seen["cfg"] = self.vllm_config
        seen["model_type"] = type(self.model)
        self.model(spec_step_idx=99, hidden_states="h")
        return "ok"

    wrapped = make_propose(orig)
    proposer = SimpleNamespace(
        model=dummy_model,
        vllm_config="target",
        draft_vllm_config="draft",
    )
    assert wrapped(proposer) == "ok"
    assert proposer.model is dummy_model
    assert proposer.vllm_config == "target"
    assert seen["cfg"] == "draft"
    assert seen["model_type"] is _ForceMtpLayer0
    assert seen["model_kwargs"] == {"spec_step_idx": 0, "hidden_states": "h"}


def test_sample_and_attn_metadata_wrappers_force_layer_zero():
    sample_seen = []
    build_seen = []

    def orig_sample(self, hidden, meta, spec_step_idx):
        sample_seen.append(spec_step_idx)
        return hidden, None

    def orig_build(self, cm, draft_index=0):
        build_seen.append(draft_index)
        return cm

    sample = make_sample_draft_tokens_for_step(orig_sample)
    build = make_build_per_group_and_layer_attn_metadata(orig_build)
    proposer = SimpleNamespace()
    sample(proposer, "h", "m", 2)
    build(proposer, "cm", draft_index=2)
    assert sample_seen == [0]
    assert build_seen == [0]


def test_step3p5_proposer_patch_does_not_wrap_eagle():
    import inspect

    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer

    from vllm_iluvatar.ilu_patch.general_patches import patch_step3p5_proposer

    source = inspect.getsource(patch_step3p5_proposer._apply)
    assert "Step3p5MTPProposer" in source
    assert "EagleProposer" not in source

    eagle_propose = EagleProposer.propose
    patch_step3p5_proposer._apply()
    assert EagleProposer.propose is eagle_propose
    assert Step3p5MTPProposer.propose is not eagle_propose


def test_refresh_draft_quant_config_noop_without_spec():
    from vllm_iluvatar.ilu_patch.general_patches.patch_step3p5_shared_head import (
        _refresh_draft_quant_config,
    )

    assert _refresh_draft_quant_config(None) is None
    cfg = SimpleNamespace(speculative_config=None, quant_config="target-qc")
    assert _refresh_draft_quant_config(cfg) is cfg


def test_refresh_draft_quant_config_replaces_shared_target_config(monkeypatch):
    from vllm_iluvatar.ilu_patch.general_patches import patch_step3p5_shared_head as mod

    draft_qc = object()
    calls = {}

    def fake_get_draft_quant_config(vllm_config):
        calls["cfg"] = vllm_config
        return draft_qc

    def fake_replace(cfg, **kwargs):
        return SimpleNamespace(
            speculative_config=cfg.speculative_config,
            quant_config=kwargs["quant_config"],
        )

    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_draft_quant_config",
        fake_get_draft_quant_config,
    )
    monkeypatch.setattr("vllm.config.replace", fake_replace)

    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object()),
        quant_config="target-qc",
    )
    refreshed = mod._refresh_draft_quant_config(cfg)
    assert calls["cfg"] is cfg
    assert refreshed.quant_config is draft_qc


def test_shared_head_rebind_restores_only_flagged_draft(monkeypatch):
    from vllm_iluvatar.ilu_patch.general_patches.patch_step3p5_shared_head import (
        _wrap_load_eagle_model,
    )

    target_head = object()

    def make_draft(has_own_shared_head):
        heads = [object(), object(), object()]
        layers = [
            SimpleNamespace(shared_head=SimpleNamespace(head=head)) for head in heads
        ]
        model = SimpleNamespace(layers=layers)
        draft = SimpleNamespace(model=model)
        if has_own_shared_head:
            draft.has_own_shared_head = True
        return draft, heads

    fake_utils = ModuleType("fakepkg.eagle")
    consumer = ModuleType("fakepkg.consumer")

    def community_loader(_target_model, _vllm_config):
        loaded = fake_utils.get_model()
        for layer in loaded.model.layers:
            layer.shared_head.head = target_head
        return loaded

    fake_utils.load_eagle_model = community_loader
    consumer.load_eagle_model = community_loader
    monkeypatch.setitem(sys.modules, fake_utils.__name__, fake_utils)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    general_patch.rebind_module_func(
        fake_utils,
        "load_eagle_model",
        _wrap_load_eagle_model(community_loader, fake_utils),
    )
    assert consumer.load_eagle_model is fake_utils.load_eagle_model

    for flagged in (True, False):
        draft, original_heads = make_draft(flagged)
        fake_utils.get_model = lambda *_args, _draft=draft, **_kwargs: _draft
        loaded = consumer.load_eagle_model(None, None)
        actual = [layer.shared_head.head for layer in loaded.model.layers]
        assert actual == (original_heads if flagged else [target_head] * 3)


def test_draft_logits_wrapper_passes_step_only_for_flagged_model():
    from vllm_iluvatar.ilu_patch.general_patches.patch_step3p5_draft_logits import (
        _wrap_sample_draft,
    )

    class DraftStep:
        def reshape(self, *_shape):
            return self

        def __getitem__(self, _index):
            return self

        def item(self):
            return 2

    calls = []

    def original(speculator, *args, **kwargs):
        del args, kwargs
        return speculator.model.compute_logits("hidden")

    wrapped = _wrap_sample_draft(original)
    for flagged in (True, False):

        def compute_logits(hidden_states, spec_step_idx=0, *, _flagged=flagged):
            calls.append((_flagged, hidden_states, spec_step_idx))
            return spec_step_idx

        model = SimpleNamespace(compute_logits=compute_logits)
        if flagged:
            model.ilu_pass_spec_step_idx = True
        speculator = SimpleNamespace(model=model)
        result = wrapped(
            speculator,
            None,
            None,
            None,
            None,
            None,
            DraftStep(),
        )
        assert result == (2 if flagged else 0)

    assert calls == [(True, "hidden", 2), (False, "hidden", 0)]