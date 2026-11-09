"""CPU contracts for opt-in fused weights and replay chunk coverage."""

import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/ixmm_moe.py"
)


@pytest.fixture
def fusion():
    spec = importlib.util.spec_from_file_location("isolated_ixmm_moe", PATH)
    mod = importlib.util.module_from_spec(spec)
    env = NS(VLLM_ILU_MOE_IXMM_FUSION=False)
    with patch.dict(sys.modules, {spec.name: mod, "vllm": NS(envs=env)}):
        spec.loader.exec_module(mod)
        yield mod


def test_disabled_prepare_does_not_inspect_or_modify_weights(fusion):
    fusion.prepare(object(), object())
    assert not fusion._states


@pytest.mark.parametrize("name", ["draft.layers.40.experts", "other_model.experts"])
def test_draft_retains_original_weights(fusion, name):
    fusion.envs.VLLM_ILU_MOE_IXMM_FUSION = True
    fusion.prepare(NS(_ixmm_layer_name=name), NS())
    assert not fusion._states


@pytest.mark.parametrize(
    "counts",
    [
        [0] * 24,
        [1] + [0] * 23,
        [0] * 23 + [1],
        [15] * 24,
        [16] * 24,
        [17] * 24,
        [1023] + [0] * 23,
        [1024] + [0] * 23,
        [0] * 5 + [4096] + [0] * 18,
        [1537] * 24,
        [8192] * 24,
        [0, 31, 32, 33, 63, 64, 65] + [0] * 17,
    ],
)
def test_batches_preserve_every_expert_row(fusion, counts):
    restored = []
    end = 0
    for lo, hi, groups in fusion.batches(counts):
        assert lo == end and 0 < hi - lo <= 16384 and 0 < len(groups) <= 24
        cursor = 0
        for start, stop, expert in groups:
            assert start == cursor and 0 < stop - start <= 1023
            restored.extend([expert] * (stop - start))
            cursor = stop
        assert cursor == hi - lo
        end = hi
    assert end == sum(counts)
    assert restored == [e for e, n in enumerate(counts) for _ in range(n)]


def test_negative_count_fails(fusion):
    with pytest.raises(RuntimeError, match="negative"):
        list(fusion.batches([-1] + [0] * 23))


def test_registry_does_not_retain_layer_weights(fusion):
    w1 = torch.empty(3)
    w2 = torch.empty(4)
    state = fusion.FusedWeights(
        weakref.ref(w1), weakref.ref(w2), torch.ones(1), torch.ones(1)
    )
    fusion._states[w1.data_ptr()] = state
    assert fusion.state_for(w1, w2) is state
    with pytest.raises(RuntimeError, match="unsupported"):
        fusion.reject_unhandled(w1, w2)
    with pytest.raises(RuntimeError, match="stale"):
        fusion.state_for(w1, torch.empty(4))
    del state
    gc.collect()
    assert fusion.state_for(w1, w2) is None


@pytest.mark.parametrize("tp,ep", [(2, 16), (1, 8), (1, 32)])
def test_unsupported_parallelism_fails_before_conversion(fusion, tp, ep):
    fusion.envs.VLLM_ILU_MOE_IXMM_FUSION = True
    with patch.dict(
        sys.modules,
        {"vllm.distributed": NS(get_tensor_model_parallel_world_size=lambda: tp)},
    ):
        with pytest.raises(RuntimeError, match="TP1/EP16"):
            fusion.prepare(
                NS(_ixmm_layer_name="model.layers.21.mlp"),
                NS(moe_config=NS(ep_size=ep)),
            )


def test_fusion_flag_uses_registered_vllm_env(monkeypatch):
    import types

    env_path = PATH.parents[5] / "envs.py"
    spec = importlib.util.spec_from_file_location("isolated_ilu_envs", env_path)
    definitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definitions)
    vllm_envs = types.ModuleType("vllm.envs")
    vllm_envs.environment_variables = {}
    vllm_envs.__getattr__ = lambda name: vllm_envs.environment_variables[name]()
    with patch.dict(sys.modules, {"vllm": NS(envs=vllm_envs), "vllm.envs": vllm_envs}):
        definitions.apply()
        spec = importlib.util.spec_from_file_location("registered_ixmm_moe", PATH)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: mod}):
            spec.loader.exec_module(mod)
            for value, expected in (("0", False), ("1", True)):
                monkeypatch.setenv("VLLM_ILU_MOE_IXMM_FUSION", value)
                assert mod.enabled() is expected
