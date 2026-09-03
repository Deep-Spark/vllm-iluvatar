# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar ``SpeculativeConfig.hf_config_override`` override.

Two layers are covered:

1. The pure ``make_hf_config_override`` wrapper factory
   (``vllm_iluvatar.config.speculative``) — exercised with a faithful fake
   community ``orig`` and a minimal ``PretrainedConfig`` stand-in, so the ilu
   MTP draft-arch remap logic is verified without importing vLLM.
2. The general-patch wiring (``patch_speculative_config``) — verifies the
   wrapper is bound back onto the real ``SpeculativeConfig`` as a staticmethod
   and that ``register()`` is idempotent.

No GPU required.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.config.speculative import (
    make_create_draft_parallel_config,
    make_hf_config_override,
    make_post_init,
)


class _FakeHFConfig:
    """Minimal ``PretrainedConfig`` stand-in: ``architectures`` + ``update()``."""

    def __init__(self, architectures):
        self.architectures = list(architectures) if architectures is not None else None

    def update(self, updates):
        for key, value in updates.items():
            setattr(self, key, value)


class _FakeModelConfig:
    def __init__(self, architectures):
        self.architectures = list(architectures)
        self.hf_config = _FakeHFConfig(architectures)


class _FakeSpeculativeConfig:
    def __init__(self, *, method="dspark", target_architectures=None):
        self.method = method
        self.target_model_config = _FakeModelConfig(target_architectures or [])
        self.draft_model_config = _FakeModelConfig(["DSparkDraftModel"])
        self.update_arch_calls = 0

    def update_arch_(self):
        self.update_arch_calls += 1
        self.draft_model_config.architectures = list(
            self.draft_model_config.hf_config.architectures
        )


def _make_orig(records, *, out_architectures=None):
    """Build a fake community ``hf_config_override``.

    Records each call and optionally rewrites ``architectures`` to simulate the
    public MTP routing (e.g. ``glm_moe_dsa`` -> ``DeepSeekMTPModel``), so we can
    assert the ilu remap keys on the *initial* arch, not the post-orig one.
    """

    def orig(hf_config):
        records.append(hf_config)
        if out_architectures is not None:
            hf_config.update({"architectures": list(out_architectures)})
        return hf_config

    return orig


# --- factory: ilu remap ----------------------------------------------------


def test_remaps_glm_after_community_routing():
    records = []
    orig = _make_orig(records, out_architectures=["DeepSeekMTPModel"])
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(["IluGlmMoeDsaForCausalLM"])
    result = wrapped(cfg)

    # Delegated exactly once, to the same object, and remap keyed on the
    # initial arch even though orig overwrote architectures.
    assert records == [cfg]
    assert result is cfg
    assert result.architectures == ["IluGlmMoeDsaMTPModel"]


def test_remaps_deepseek_v32_after_community_routing():
    orig = _make_orig([], out_architectures=["DeepSeekMTPModel"])
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(["DeepseekV32ForCausalLM"])
    result = wrapped(cfg)

    assert result.architectures == ["DeepseekV32MTPModel"]


def test_remaps_deepseek_v4_after_community_routing():
    orig = _make_orig([], out_architectures=["DeepSeekV4MTPModel"])
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(["IluDeepseekV4ForCausalLM"])
    result = wrapped(cfg)

    assert result.architectures == ["IluDeepSeekV4MTPModel"]


# --- factory: non-ilu passthrough ------------------------------------------


def test_non_ilu_arch_is_left_to_community_result():
    orig = _make_orig([], out_architectures=["EagleModel"])
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(["LlamaForCausalLM"])
    result = wrapped(cfg)

    assert result.architectures == ["EagleModel"]


def test_remap_keys_on_initial_arch_not_orig_output():
    # Initial arch is non-ilu; even if orig happens to emit an ilu-target-like
    # arch, the wrapper must NOT remap (it keys on the captured initial arch).
    orig = _make_orig([], out_architectures=["IluGlmMoeDsaForCausalLM"])
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(["LlamaForCausalLM"])
    result = wrapped(cfg)

    assert result.architectures == ["IluGlmMoeDsaForCausalLM"]


# --- factory: empty / missing architectures --------------------------------


def test_empty_architectures_delegates_without_remap():
    records = []
    orig = _make_orig(records, out_architectures=None)
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig([])
    result = wrapped(cfg)

    assert records == [cfg]
    assert result is cfg


def test_none_architectures_does_not_crash():
    orig = _make_orig([], out_architectures=None)
    wrapped = make_hf_config_override(orig)

    cfg = _FakeHFConfig(None)
    result = wrapped(cfg)

    assert result is cfg


def test_preserves_orig_metadata():
    def orig(hf_config):
        return hf_config

    orig.__name__ = "hf_config_override"
    wrapped = make_hf_config_override(orig)

    assert wrapped.__wrapped__ is orig
    assert wrapped.__name__ == "hf_config_override"


# --- factory: DSpark post-init remap ----------------------------------------


def test_post_init_remaps_dspark_draft_for_ilu_deepseek_v4():
    calls = []

    def orig(self):
        calls.append(self)

    wrapped = make_post_init(orig)
    cfg = _FakeSpeculativeConfig(target_architectures=["IluDeepseekV4ForCausalLM"])

    wrapped(cfg)

    assert calls == [cfg]
    assert cfg.draft_model_config.hf_config.architectures == ["IluDSparkDraftModel"]
    assert cfg.draft_model_config.architectures == ["IluDSparkDraftModel"]
    assert cfg.update_arch_calls == 1


def test_post_init_leaves_non_ilu_dspark_draft_unchanged():
    wrapped = make_post_init(lambda self: None)
    cfg = _FakeSpeculativeConfig(target_architectures=["DeepseekV32ForCausalLM"])

    wrapped(cfg)

    assert cfg.draft_model_config.hf_config.architectures == ["DSparkDraftModel"]
    assert cfg.update_arch_calls == 0


def test_post_init_skips_non_dspark_methods():
    wrapped = make_post_init(lambda self: None)
    cfg = _FakeSpeculativeConfig(
        method="mtp",
        target_architectures=["IluDeepseekV4ForCausalLM"],
    )

    wrapped(cfg)

    assert cfg.draft_model_config.hf_config.architectures == ["DSparkDraftModel"]
    assert cfg.update_arch_calls == 0


# --- factory: draft PP=1 ---------------------------------------------------


def test_draft_parallel_config_forces_pp1_on_two_arg_orig():
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        pipeline_parallel_size: int
        tensor_parallel_size: int

    def orig(target, tp):
        return _Cfg(
            pipeline_parallel_size=target.pipeline_parallel_size,
            tensor_parallel_size=tp,
        )

    wrapped = make_create_draft_parallel_config(orig)
    out = wrapped(SimpleNamespace(pipeline_parallel_size=8), 4)
    assert out.pipeline_parallel_size == 1
    assert out.tensor_parallel_size == 4


def test_draft_parallel_config_passes_pp_to_three_arg_orig():
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        pipeline_parallel_size: int
        tensor_parallel_size: int

    def orig(target, tp, speculative_draft_pipeline_parallel_size):
        return _Cfg(
            pipeline_parallel_size=speculative_draft_pipeline_parallel_size,
            tensor_parallel_size=tp,
        )

    wrapped = make_create_draft_parallel_config(orig)
    out = wrapped(SimpleNamespace(pipeline_parallel_size=8), 4, 8)
    assert out.pipeline_parallel_size == 1


def test_draft_parallel_config_keeps_pp1_when_target_is_pp1():
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        pipeline_parallel_size: int
        tensor_parallel_size: int

    def orig(target, tp):
        return _Cfg(
            pipeline_parallel_size=target.pipeline_parallel_size,
            tensor_parallel_size=tp,
        )

    wrapped = make_create_draft_parallel_config(orig)
    out = wrapped(SimpleNamespace(pipeline_parallel_size=1), 4)
    assert out.pipeline_parallel_size == 1
    assert out.tensor_parallel_size == 4


# --- wiring: general patch binds onto the real class -----------------------


def test_patch_binds_wrapper_as_staticmethod():
    import inspect

    from vllm.config.speculative import SpeculativeConfig
    from vllm.engine.arg_utils import EngineArgs

    from vllm_iluvatar.ilu_patch.general_patches import patch_speculative_config

    patch_speculative_config.register()

    static = inspect.getattr_static(SpeculativeConfig, "hf_config_override")
    assert isinstance(static, staticmethod), (
        "hf_config_override must stay a staticmethod"
    )
    assert hasattr(SpeculativeConfig.hf_config_override, "__wrapped__"), (
        "hf_config_override not wrapped by the ilu override"
    )
    assert hasattr(SpeculativeConfig.__post_init__, "__wrapped__"), (
        "__post_init__ not wrapped by the ilu override"
    )
    draft_pp = inspect.getattr_static(
        SpeculativeConfig, "create_draft_parallel_config"
    )
    assert isinstance(draft_pp, staticmethod), (
        "create_draft_parallel_config must stay a staticmethod"
    )
    assert hasattr(SpeculativeConfig.create_draft_parallel_config, "__wrapped__"), (
        "create_draft_parallel_config not wrapped by the ilu override"
    )
    assert hasattr(EngineArgs.create_speculative_config, "__wrapped__"), (
        "create_speculative_config not wrapped by the ilu override"
    )


def test_patch_register_is_idempotent():
    from vllm.config.speculative import SpeculativeConfig
    from vllm.engine.arg_utils import EngineArgs

    from vllm_iluvatar.ilu_patch.general_patches import patch_speculative_config

    patch_speculative_config.register()
    first = SpeculativeConfig.hf_config_override
    first_post_init = SpeculativeConfig.__post_init__
    first_draft_pp = SpeculativeConfig.create_draft_parallel_config
    first_create = EngineArgs.create_speculative_config
    patch_speculative_config.register()
    patch_speculative_config.register()

    assert SpeculativeConfig.hf_config_override is first
    assert SpeculativeConfig.__post_init__ is first_post_init
    assert SpeculativeConfig.create_draft_parallel_config is first_draft_pp
    assert EngineArgs.create_speculative_config is first_create
