# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
import subprocess
import sys
from dataclasses import asdict

import pytest
from pydantic import TypeAdapter, ValidationError
from vllm.config import SpeculativeConfig


def _config(**overrides):
    # Ngram configuration needs no remote model metadata or checkpoint.
    kwargs = dict(
        method="ngram",
        num_speculative_tokens=5,
        rejection_sample_method="fly",
        draft_sample_method="probabilistic",
    )
    kwargs.update(overrides)
    return SpeculativeConfig(**kwargs)


def test_standard_remains_the_default_verification_method():
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=5)

    assert config.rejection_sample_method == "standard"
    assert config.fly_window_size is None
    assert config.fly_min_target_probability == 0.0


@pytest.mark.parametrize(("tokens", "window"), [(2, 1), (5, 4), (9, 6)])
def test_fly_default_window_is_bounded_by_the_draft(tokens, window):
    config = _config(num_speculative_tokens=tokens)

    assert config.fly_window_size == window
    assert config.fly_entropy_threshold == 0.3
    assert config.fly_entropy_top_k == 3


def test_fly_keeps_explicit_window_and_probabilistic_draft_sampling():
    config = _config(fly_window_size=2)

    assert config.fly_window_size == 2
    assert config.num_speculative_tokens == 5
    assert config.draft_sample_method == "probabilistic"


@pytest.mark.parametrize("floor", [0.0, 0.01, 1.0])
def test_fly_target_probability_floor_preserves_explicit_configuration(floor):
    config = _config(fly_window_size=2, fly_min_target_probability=floor)

    assert config.fly_min_target_probability == floor
    assert config.fly_window_size == 2
    assert config.draft_sample_method == "probabilistic"


@pytest.mark.parametrize("floor", [-0.01, 1.01, float("nan"), float("inf")])
def test_fly_rejects_invalid_target_probability_floor(floor):
    with pytest.raises((ValueError, ValidationError)):
        _config(fly_min_target_probability=floor)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rejection_sample_method": "standard"}, "requires FLy verification"),
        ({"draft_sample_method": "greedy"}, "requires probabilistic drafting"),
    ],
)
def test_fly_target_probability_floor_requires_probabilistic_fly(overrides, message):
    with pytest.raises((ValueError, ValidationError), match=message):
        _config(fly_min_target_probability=0.01, **overrides)


def test_configuration_is_extended_before_cli_and_engine_schema_capture_it():
    from vllm.config import VllmConfig
    from vllm.config.speculative import SpeculativeConfig as DirectConfig
    from vllm.engine.arg_utils import EngineArgs, get_kwargs

    from vllm_iluvatar.config.speculative import SpeculativeConfig as PluginConfig

    assert SpeculativeConfig is DirectConfig is PluginConfig
    assert (
        EngineArgs.create_speculative_config.__globals__["SpeculativeConfig"]
        is PluginConfig
    )
    assert "fly_min_target_probability" in get_kwargs(SpeculativeConfig)
    field_type = VllmConfig.__pydantic_fields__["speculative_config"].annotation
    config = TypeAdapter(field_type).validate_python(
        dict(
            method="ngram",
            num_speculative_tokens=5,
            draft_sample_method="probabilistic",
            rejection_sample_method="fly",
            fly_window_size=2,
            fly_min_target_probability=0.01,
        )
    )
    assert isinstance(config, PluginConfig)
    assert config.fly_min_target_probability == 0.01


def test_config_serialization_and_pickle_preserve_a5_gate():
    config = _config(fly_window_size=2, fly_min_target_probability=0.01)
    adapter = TypeAdapter(SpeculativeConfig)
    serialized = adapter.dump_json(config)

    assert adapter.validate_json(serialized) == config
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize("startup_patch", ["0", "1"])
def test_worker_unpickles_a5_without_preimporting_community_config(
    tmp_path, startup_patch
):
    path = tmp_path / "a5.pkl"
    path.write_bytes(
        pickle.dumps(_config(fly_window_size=2, fly_min_target_probability=0.01))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import pathlib, pickle, sys; "
                "cfg = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes()); "
                "from vllm.config import SpeculativeConfig; "
                "assert type(cfg) is SpeculativeConfig; "
                "assert cfg.rejection_sample_method == 'fly'; "
                "assert cfg.fly_window_size == 2; "
                "assert cfg.fly_min_target_probability == 0.01"
            ),
            str(path),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH": startup_patch,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_standard_configuration_preserves_community_fields_and_hash():
    from vllm_iluvatar.config.speculative import CommunitySpeculativeConfig

    base = CommunitySpeculativeConfig(method="ngram", num_speculative_tokens=5)
    plugin = SpeculativeConfig(method="ngram", num_speculative_tokens=5)
    plugin_fields = asdict(plugin)
    for field, value in asdict(base).items():
        assert plugin_fields[field] == value
    assert plugin.compute_hash() == base.compute_hash()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_speculative_tokens": 1}, "at least 2"),
        ({"fly_window_size": 5}, "smaller than"),
        ({"fly_window_size": 6}, "smaller than"),
        ({"fly_window_size": 0}, "greater than or equal to 1"),
        ({"fly_entropy_top_k": 0}, "greater than or equal to 1"),
        ({"fly_entropy_threshold": -0.1}, "greater than or equal to 0"),
    ],
)
def test_fly_rejects_invalid_configuration(overrides, message):
    with pytest.raises((ValueError, ValidationError), match=message):
        _config(**overrides)
