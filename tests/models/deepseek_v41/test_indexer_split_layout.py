"""Allocation, paired-transfer and E25 boundary invariants for split keys."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
    MemoryRegistrationPlanner,
)
from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
    SPLIT_INDEXER_FORMAT,
    IndexerScaleState,
    plan_indexer_scales,
    reserve_indexer_scale_memory,
)
from vllm_iluvatar.v1.worker.prefill_state.cache_rows import (
    copy_indexer_row_to_host,
    restore_indexer_row,
)


def planned(count=3):
    spec = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        cache_dtype_str=SPLIT_INDEXER_FORMAT,
    )
    page = spec.page_size_bytes
    names = ["key0", "key1", "alias0"]
    config = KVCacheConfig(
        num_blocks=count,
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=count * page * 2,
                layers=names[:2],
                offset=0,
                layer_stride=count * page,
                block_stride=page,
            ),
            KVCacheTensor(
                size=count * page * 2,
                layers=names[2:],
                offset=0,
                layer_stride=count * page,
                block_stride=page,
            ),
        ],
    )
    return config, spec


def test_aliases_reserve_only_real_indexer_scales_and_bind_kernel_pages():
    config, _ = planned()
    layout = plan_indexer_scales(config)
    assert layout.nbytes == 2 * 3 * 512 * 4
    context = {
        name: SimpleNamespace(kv_cache_scales=None) for name in layout.placements
    }
    state = IndexerScaleState(layout, "cpu", context)
    try:
        cache = torch.empty(3 * 8, 64, 128, dtype=torch.uint8)
        state.bind("key0", cache)
        assert context["key0"].kv_cache_scales.shape == (24, 64)
        assert state.scales["key0"].data_ptr() == state.scales["alias0"].data_ptr()
        state.scales["alias0"][0, 7] = 3
        assert state.scales["key0"][0, 7] == 3
    finally:
        state.release()
    assert all(layer.kv_cache_scales is None for layer in context.values())


def test_partial_aliases_are_rejected_before_sideplane_allocation():
    config, _ = planned()
    shifted = replace(config.kv_cache_tensors[-1], offset=128)
    config.kv_cache_tensors[-1] = shifted
    with pytest.raises(ValueError, match="partial aliases"):
        plan_indexer_scales(config)


def test_scale_bytes_are_deducted_before_final_block_count():
    calls = []

    def allocate(_cfg, _specs, budgets):
        calls.append(budgets)
        return [planned(budgets[0] // (2 * 65536))[0]]

    config, spec = planned()
    budget = 12 * 2 * 65536
    result = reserve_indexer_scale_memory(allocate, None, [{"key": spec}], [budget])[0]
    layout = plan_indexer_scales(result)
    assert len(calls) == 2 and result.num_blocks == 11
    assert result.kv_cache_tensors[0].size + layout.nbytes <= budget


@pytest.mark.parametrize("pp", [False, True])
def test_real_1m_allocator_budgets_only_owning_and_forwarded_index_keys(
    pp, monkeypatch
):
    """Use production grouping and the requested PP8 cut, without GPU allocation."""
    import inspect

    from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32
    from test_compressed_kv_v2_planning import (
        INDEX_SOURCES,
        RATIOS,
        SOURCES,
        config,
        specs_for_rank,
    )
    from vllm.v1.core import kv_cache_utils

    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        CacheBudget,
        plan_scale_layout,
    )
    from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan

    cfg = config()
    if pp:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "3,3,3,3,3,2,2,21")
        plans = [
            build_rank_plan(40, SOURCES, INDEX_SOURCES, RATIOS, rank, 8, last_layer=20)
            for rank in range(8)
        ]
        ranks = [
            specs_for_rank(
                cfg, I4G32, plan.start_layer, plan.end_layer, plan.forwarder
            )[0]
            for plan in plans
        ]
    else:
        ranks = [specs_for_rank(cfg, I4G32, 0, 40)[0]]
    for specs in ranks:
        for name, spec in list(specs.items()):
            if name.endswith(".indexer.k_cache"):
                specs[name] = replace(
                    spec, head_size=128, cache_dtype_str=SPLIT_INDEXER_FORMAT
                )
    budget = CacheBudget.create(8 * 1024**3, I4G32)
    original = inspect.unwrap(kv_cache_utils.get_kv_cache_configs)
    configs = reserve_indexer_scale_memory(
        original, cfg, ranks, [budget.payload_bytes] * len(ranks)
    )
    assert len({item.num_blocks for item in configs}) == 1
    for local in configs:
        main = plan_scale_layout(local, I4G32)
        index = plan_indexer_scales(local)
        budget.validate(main.payload_bytes, main.scale_bytes + index.nbytes)
        assert all(name.endswith(".indexer.k_cache") for name in index.placements)


def test_nixl_scale_registration_covers_exactly_the_paired_manager_blocks():
    config, spec = planned()
    layout = plan_indexer_scales(config)
    context = {
        name: SimpleNamespace(kv_cache_scales=None) for name in layout.placements
    }
    state = IndexerScaleState(layout, "cpu", context)
    raw = torch.empty(config.kv_cache_tensors[0].size, dtype=torch.uint8)
    payload = raw[: 3 * 65536].view(24, 64, 128)
    state.bind("key0", payload)
    planner = MemoryRegistrationPlanner(config)
    region = planner.describe(
        layer_name="key0", tensor=payload, spec=spec, num_blocks=3
    )
    paired = planner.describe_indexer_scale_plane(
        layer_name="key0", scales=state.scales["key0"], payload=region
    )
    assert paired.kernel_blocks == region.kernel_blocks == 8
    assert paired.block_stride == 512 * 4
    assert paired.payload_segments == ((0, 512 * 4),)
    assert paired.registration_nbytes == layout.nbytes
    state.bind("key1", payload)
    with pytest.raises(ValueError, match="payload ownership"):
        planner.describe_indexer_scale_plane(
            layer_name="key0", scales=state.scales["key1"], payload=region
        )
    state.release()


@pytest.mark.parametrize("row", [0, 63])
def test_e25_split_boundary_roundtrip_is_byte_exact_and_leaves_neighbors(row):
    values = torch.randint(0, 256, (4, 64, 128), dtype=torch.uint8)
    scales = torch.randn(4, 64, dtype=torch.float32)
    original = values.clone(), scales.clone()
    host = torch.empty(132, dtype=torch.uint8)
    copy_indexer_row_to_host(host, values, 2, row, scales)
    values[2, row].zero_()
    scales[2, row].zero_()
    restore_indexer_row(values, 2, row, host, scales)
    assert torch.equal(values, original[0])
    assert torch.equal(scales.view(torch.uint8), original[1].view(torch.uint8))
