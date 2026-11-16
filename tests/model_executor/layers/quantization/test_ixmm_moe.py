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


@pytest.mark.parametrize("tp,ep", [(3, 16), (8, 16), (1, 8), (1, 32)])
def test_unsupported_parallelism_fails_before_conversion(fusion, tp, ep):
    fusion.envs.VLLM_ILU_MOE_IXMM_FUSION = True
    with patch.dict(
        sys.modules,
        {"vllm.distributed": NS(get_tensor_model_parallel_world_size=lambda: tp)},
    ):
        with pytest.raises(RuntimeError, match="target EP16"):
            fusion.prepare(
                NS(_ixmm_layer_name="model.layers.21.mlp"),
                NS(moe_config=NS(ep_size=ep)),
            )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_supported_parallelism_still_checks_weight_format(fusion, tp):
    fusion.envs.VLLM_ILU_MOE_IXMM_FUSION = True
    with patch.dict(
        sys.modules,
        {"vllm.distributed": NS(get_tensor_model_parallel_world_size=lambda: tp)},
    ):
        with pytest.raises(RuntimeError, match="NN V2"):
            fusion.prepare(
                NS(_ixmm_layer_name="model.layers.21.mlp", gemm_format="TN"),
                NS(moe_config=NS(ep_size=16)),
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


def test_local_decode_writes_output_for_combine(fusion, monkeypatch):
    activations = torch.zeros((24, 2, 5120), dtype=torch.int8)
    output = torch.zeros_like(activations, dtype=torch.bfloat16)
    counts = torch.tensor([2] + [0] * 23, dtype=torch.int32)
    scales = torch.ones(24, 2)
    state = NS(local_output=True, scale1=None, scale2=None)
    layout = torch.empty(24, 3, dtype=torch.int32)
    workspace = torch.empty(0)

    def run_local(x, w1, w2, sx, sw1, sw2, expert_layout, scratch, out):
        assert x.shape == out.shape == (48, 5120)
        assert expert_layout is layout and scratch is workspace
        assert sx.data_ptr() == scales.data_ptr()
        out[:2].fill_(7)

    ops = NS(moe_ixmm_local=run_local, moe_ixmm_layout=lambda c, dst: None)
    monkeypatch.setattr(fusion, "state_for", lambda w1, w2: state)
    monkeypatch.setattr(fusion, "scratch", lambda *args, **kw: (workspace, layout))
    monkeypatch.setattr(fusion, "kernels", lambda: ops)
    assert fusion.decode(None, output, activations, torch.empty(0), None, scales,
                         NS(expert_num_tokens=counts))
    assert torch.all(output[0] == 7)

@pytest.mark.parametrize("tp,direct,local", [
    (1, "1", False), (2, "1", False), (2, "0", True), (4, "0", True), (4, "1", True),
])
def test_selects_compatible_output(fusion, monkeypatch, tp, direct, local):
    monkeypatch.setenv("IXFORMER_NVSM_W2_DIRECT_COMBINE", direct)
    assert fusion._uses_local_output(tp) is local
