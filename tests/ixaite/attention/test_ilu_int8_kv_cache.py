# SPDX-License-Identifier: Apache-2.0
"""Quantized cache views and lifecycle must work independently of Attention math."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.envs as envs

from vllm_iluvatar.backends.ixaite.attention import kv_cache


@pytest.mark.parametrize("mode", ["", "KINT8", "KVINT8"])
def test_registration_selects_private_cache_policy(monkeypatch, mode):
    from vllm.v1.core import kv_cache_utils as community

    from vllm_iluvatar.backends.ixaite import register
    from vllm_iluvatar.ilu_patch import general_patch
    from vllm_iluvatar.ilu_patch.general_patches import patch_kv_cache_utils
    from vllm_iluvatar.v1.attention import ilu_static_quant
    from vllm_iluvatar.v1.core import kv_cache_utils as common
    from vllm_iluvatar.v1.worker import gpu_model_runner

    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", mode)
    shared_apply, rebind, wrapped = Mock(), Mock(), object()
    wrap = Mock(return_value=wrapped)
    baseline_loader = ilu_static_quant.load_static_v_scales
    monkeypatch.setattr(patch_kv_cache_utils, "_apply", shared_apply)
    monkeypatch.setattr(general_patch, "rebind_module_func", rebind)
    monkeypatch.setattr(common, "make_get_kv_cache_config_from_groups", wrap)
    monkeypatch.setattr(gpu_model_runner, "load_static_v_scales", baseline_loader)
    register._apply_kv_cache()
    shared_apply.assert_called_once_with()
    if mode:
        wrap.assert_called_once_with(kv_cache.get_kv_cache_config_from_groups)
        rebind.assert_called_once_with(
            community, "get_kv_cache_config_from_groups", wrapped
        )
    else:
        wrap.assert_not_called()
        rebind.assert_not_called()
    assert gpu_model_runner.load_static_v_scales is (
        kv_cache.load_ixaite_static_v_scales if mode == "KVINT8" else baseline_loader
    )
    assert ilu_static_quant.load_static_v_scales is baseline_loader


@pytest.fixture
def calibrated_runner(monkeypatch, tmp_path):
    import ixaite
    from vllm.distributed import parallel_state

    ixaite.initialize("ivcore11")
    from ixaite.ops.attention.int8_kv_cache import Int8KVCache

    static = torch.arange(1, 257).reshape(2, 128).float() / 100
    path = tmp_path / "scales.pt"
    torch.save({"0": static}, path)
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KVINT8")
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", str(path))
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        kv_cache, "load_ixaite_python_op", lambda *args: (Int8KVCache, None)
    )
    layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 2, 2, 16, 128, dtype=torch.int8),
        kv_cache_scale=[torch.zeros(2, 2, 16), static.clone()],
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layers.0": layer}),
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 2),
        shared_kv_cache_layers={},
        _update_states=Mock(),
    )
    return runner, layer


@pytest.mark.parametrize("alias_name", ["shared", "model.layers.1.self_attn.attn"])
@pytest.mark.parametrize("alias_first", [False, True])
def test_bind_calibrated_shared_aliases(calibrated_runner, alias_name, alias_first):
    runner, layer = calibrated_runner
    alias = SimpleNamespace(
        kv_cache=layer.kv_cache, kv_cache_scale=layer.kv_cache_scale
    )
    entries = [("layers.0", layer), (alias_name, alias)]
    runner.compilation_config.static_forward_context = dict(
        reversed(entries) if alias_first else entries
    )
    runner.shared_kv_cache_layers[alias_name] = "layers.0"
    kv_cache.bind_int8_kv_caches(runner)
    assert layer._ixaite_int8_cache is alias._ixaite_int8_cache
    assert len(runner._ixaite_int8_caches) == 1


def test_binding_rejects_different_calibrated_values(calibrated_runner):
    runner, layer = calibrated_runner
    layer.kv_cache_scale[1].mul_(2)
    with pytest.raises(ValueError, match="do not match calibrated values"):
        kv_cache.bind_int8_kv_caches(runner)


def test_binding_checks_model_total_heads(calibrated_runner):
    runner, _ = calibrated_runner
    runner.model_config.get_total_num_kv_heads = lambda: 4
    with pytest.raises(ValueError, match="file heads do not match"):
        kv_cache.bind_int8_kv_caches(runner)


def test_binding_rejects_shared_scale_mismatch(calibrated_runner):
    runner, layer = calibrated_runner
    alias = SimpleNamespace(
        kv_cache=layer.kv_cache,
        kv_cache_scale=[layer.kv_cache_scale[0], layer.kv_cache_scale[1] * 2],
    )
    runner.compilation_config.static_forward_context["shared"] = alias
    with pytest.raises(ValueError, match="must use the same static V scales"):
        kv_cache.bind_int8_kv_caches(runner)


# Both the legacy planes-first view and block-major backing must remain zero-copy.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("mode", ["KINT8", "KVINT8"])
@pytest.mark.parametrize("block_major", [False, True])
@pytest.mark.parametrize("pages_per_block", [1, 2])
def test_bind_write_copy_zero_and_replay(
    monkeypatch, tmp_path, mode, block_major, pages_per_block
):
    import ixaite
    from vllm.distributed import parallel_state

    ixaite.initialize("ivcore11")
    from ixaite.ops.attention.int8_kv_cache import Int8KVCache

    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", mode)
    planes = 3 if mode == "KINT8" else 2
    pages = 4 * pages_per_block
    shape = (planes, pages, 4, 16, 128)
    if block_major:
        raw = torch.zeros(
            pages, planes, 4, 16, 128, dtype=torch.int8, device="cuda"
        ).permute(1, 0, 2, 3, 4)
    else:
        raw = torch.zeros(shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(pages, 4, 16, device="cuda")
    static = torch.full((4, 128), 0.02, device="cuda") if mode == "KVINT8" else None
    if static is not None:
        path = tmp_path / "scales.pt"
        torch.save({"0": static.cpu()}, path)
        monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", str(path))
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    layer = SimpleNamespace(kv_cache=raw, kv_cache_scale=[scales, static])
    alias = SimpleNamespace(kv_cache=raw, kv_cache_scale=layer.kv_cache_scale)
    original = Mock(return_value="updated")
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layers.0": layer, "shared": alias}
        ),
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 4),
        shared_kv_cache_layers={"shared": "layers.0"},
        kv_cache_config=SimpleNamespace(num_blocks=4),
        _update_states=original,
    )
    kv_cache.bind_int8_kv_caches(runner)
    cache = layer._ixaite_int8_cache
    assert isinstance(cache, Int8KVCache)
    assert cache is alias._ixaite_int8_cache
    assert len(runner._ixaite_int8_caches) == 1
    assert cache.key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert cache.value.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key = torch.randn(17, 4, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.arange(17, dtype=torch.int64, device="cuda")
    cache.write(key, value, slots)
    saved = tuple(
        t[:pages_per_block].clone() for t in (cache.key, cache.value, cache.key_scale)
    )
    for tensor in (cache.key, cache.value, cache.key_scale):
        tensor[2 * pages_per_block : 3 * pages_per_block].fill_(17)
    output = SimpleNamespace(new_block_ids_to_zero=[2], kv_cache_block_copies=[(0, 1)])
    assert runner._update_states(output) == "updated"
    for tensor, expected in zip((cache.key, cache.value, cache.key_scale), saved):
        torch.testing.assert_close(
            tensor[pages_per_block : 2 * pages_per_block], expected, atol=0, rtol=0
        )
        assert (
            torch.count_nonzero(tensor[2 * pages_per_block : 3 * pages_per_block]) == 0
        )
    forwarded = original.call_args.args[0]
    assert not forwarded.kv_cache_block_copies and not forwarded.new_block_ids_to_zero
    assert output.kv_cache_block_copies == [(0, 1)]

    # Replaying against the real packed views checks padded stride specializations.
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache.write(key, value, slots)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        cache.write(key, value, slots)
    for _ in range(2):
        key.normal_()
        value.normal_()
        slots.copy_(
            torch.arange(17, device="cuda", dtype=torch.int64) + 32 * pages_per_block
        )
        graph.replay()
        expected_k, expected_s = ixaite.scaled_int8_quant_for_attn(key, 2)
        torch.testing.assert_close(
            cache.key[slots // 16, :, slots % 16], expected_k, atol=0, rtol=0
        )
        torch.testing.assert_close(
            cache.key_scale[slots // 16, :, slots % 16], expected_s, atol=0, rtol=0
        )
        expected_v = (
            value
            if static is None
            else ixaite.scaled_int8_quant_for_attn(value, 0, scale=static)[0]
        )
        torch.testing.assert_close(
            cache.value[slots // 16, :, slots % 16], expected_v, atol=0, rtol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("mode", ["KINT8", "KVINT8"])
def test_updates_reuse_indices_and_refresh_each_schedule(monkeypatch, mode):
    import ixaite

    ixaite.initialize("ivcore11")
    from ixaite.ops.attention.int8_kv_cache import Int8KVCache

    caches = []
    for pages_per_block in (1, 2, 1):
        shape = (4 * pages_per_block, 2, 16, 128)
        key = torch.empty(shape, dtype=torch.int8, device="cuda")
        value = torch.empty_like(
            key, dtype=torch.bfloat16 if mode == "KINT8" else torch.int8
        )
        scale = torch.empty(shape[:3], device="cuda")
        for block in range(4):
            for tensor in (key, value, scale):
                tensor[block * pages_per_block : (block + 1) * pages_per_block].fill_(
                    block + 1
                )
        static = torch.ones(2, 128, device="cuda") if mode == "KVINT8" else None
        caches.append(Int8KVCache(key, value, scale, static))
    runner = SimpleNamespace(
        _ixaite_int8_caches=caches, kv_cache_config=SimpleNamespace(num_blocks=4)
    )
    tensor_call = Mock(wraps=torch.tensor)
    arange_call = Mock(wraps=torch.arange)
    monkeypatch.setattr(
        kv_cache,
        "torch",
        SimpleNamespace(tensor=tensor_call, arange=arange_call, int64=torch.int64),
    )
    schedules = (
        ([3], [(0, 1), (1, 2)], [1, 1, 2, 0]),
        ([0], [(2, 3)], [0, 1, 2, 2]),
    )
    for zero_ids, copies, expected in schedules:
        tensor_call.reset_mock()
        arange_call.reset_mock()
        kv_cache.apply_int8_cache_updates(
            runner,
            SimpleNamespace(
                new_block_ids_to_zero=zero_ids, kv_cache_block_copies=copies
            ),
        )
        # Two distinct layouts each build one zero index and one copy index.
        assert tensor_call.call_count == arange_call.call_count == 4
        for cache, pages_per_block in zip(caches, (1, 2, 1)):
            for tensor in (cache.key, cache.value, cache.key_scale):
                for block, fill in enumerate(expected):
                    assert torch.all(
                        tensor[block * pages_per_block : (block + 1) * pages_per_block]
                        == fill
                    )
            if cache.value_scale is not None:
                assert torch.all(cache.value_scale == 1)


# Scope the exact byte budget calculation to ixaite; baseline retains its formula.
@pytest.mark.parametrize("mode,planes", [("KINT8", 3), ("KVINT8", 2)])
@pytest.mark.parametrize("include_scale_budget", [False, True])
def test_planning_counts_fixed_and_token_scales(
    monkeypatch, mode, planes, include_scale_budget
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_iluvatar.v1.core import kv_cache_utils as planning

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", mode)
    payload = planes * 16 * 4 * 128
    scale_page = 4 * 16 * 4
    fixed = 4 * 4 * 128 if mode == "KVINT8" else 0
    monkeypatch.setattr(
        FullAttentionSpec,
        "scale_page_size_bytes",
        property(lambda self: scale_page),
        raising=False,
    )
    monkeypatch.setattr(
        FullAttentionSpec,
        "fixed_size_bytes",
        property(lambda self: fixed),
        raising=False,
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
    )
    tensor = SimpleNamespace(size=10 * payload, shared_by=["layers.0"])
    config = SimpleNamespace(num_blocks=10, kv_cache_tensors=[tensor])
    monkeypatch.setattr(
        planning, "origin_get_kv_cache_config_from_groups", lambda *args: config
    )
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["layers.0"])
    budget = 10 * payload + (10 * scale_page + fixed if include_scale_budget else 0)
    result = kv_cache.get_kv_cache_config_from_groups(
        SimpleNamespace(), [group], budget
    )
    assert result.num_blocks == (budget - fixed) // (payload + scale_page)
    actual_bytes = tensor.size + tensor.scale_size + tensor.fixed_size
    assert actual_bytes <= budget
    assert actual_bytes + payload + scale_page > budget


@pytest.mark.parametrize("tp_rank", range(4))
def test_static_scale_loader_uses_layer_and_replicated_heads(
    monkeypatch, tmp_path, tp_rank
):
    from vllm.distributed import parallel_state
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_iluvatar.v1.worker import gpu_model_runner

    path = tmp_path / "scales.pt"
    full = torch.arange(1, 257).reshape(2, 128).float() / 100
    torch.save({"3": full, "7": full * 2}, path)
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", str(path))
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KVINT8")
    monkeypatch.setattr(
        gpu_model_runner, "load_static_v_scales", kv_cache.load_ixaite_static_v_scales
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=tp_rank, world_size=4),
    )
    for layer, factor in (("3", 1), ("7", 2)):
        scales = kv_cache.load_ixaite_static_v_scales(1, 128, "cpu", layer)
        assert list(scales) == [layer]
        torch.testing.assert_close(
            scales[layer],
            full[tp_rank // 2 : tp_rank // 2 + 1] * factor,
            atol=0,
            rtol=0,
        )
        _, allocated_scales = gpu_model_runner._reshape_attention_kv_cache_with_scale(
            torch.zeros(2 * 2 * 16 * 128, dtype=torch.uint8),
            FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
            ),
            (2, 2, 1, 16, 128),
            (0, 1, 2, 3, 4),
            2,
            None,
            f"model.layers.{layer}.self_attn.attn",
            [
                torch.zeros(2 * 16 * 4, dtype=torch.uint8),
                torch.zeros(128 * 4, dtype=torch.uint8),
            ],
        )
        torch.testing.assert_close(allocated_scales[1], scales[layer], atol=0, rtol=0)
