"""Real V2 planning/grouping with DS41 cache specs and no model/service.

1M admission is preserved. Large resulting tensors use the meta device for
CPU geometry checks; small real GPU allocations are checked separately.
"""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from vllm.config import CacheConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_iluvatar.models.deepseek_v41.attention import (
    IluDeepseekV41Attention,
    IluDeepseekV41IndexerCache,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PREFILL_WORKSPACE_BYTES,
    WORKSPACE_BYTES,
    CacheBudget,
    CompressedKVState,
    is_quantized_main_spec,
    layer_specs,
    plan_scale_layout,
    retained_mapper_bytes_for_config,
    workspace_bytes_for_config,
)
from vllm_iluvatar.models.deepseek_v41.compressor import IluCompressorStateCache
from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import IluDeepseekV4SWACache

SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
RATIOS = tuple([0, 0] + [2] * 18 + [1] * 20)


def config():
    cache = CacheConfig(block_size=64, cache_dtype="bfloat16")
    cache.kv_cache_layout = "LBNHC"
    return SimpleNamespace(
        cache_config=cache,
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(
            max_model_len=1048576, original_max_model_len=1048576
        ),
        speculative_config=None,
        num_prefill_lookahead_tokens=1,
        max_in_flight_tokens=1024,
    )


def bare(cls):
    result = cls.__new__(cls)
    torch.nn.Module.__init__(result)
    return result


def specs_for_rank(cfg, fmt, start, end, forwarder=None):
    specs, context = {}, {}
    for layer_id in range(start, end):
        prefix = f"model.layers.{layer_id}.attn"
        swa = bare(IluDeepseekV4SWACache)
        swa.cache_config = cfg.cache_config
        swa._block_size = None
        swa.dtype, swa.head_dim, swa.window_size, swa.bounded_replay = (
            torch.bfloat16,
            512,
            128,
            True,
        )
        specs[prefix + ".swa_cache"] = swa.get_kv_cache_spec(cfg)
        context[prefix + ".swa_cache"] = swa
        if layer_id not in SOURCES and layer_id != forwarder:
            continue
        main = bare(IluDeepseekV41Attention)
        main.is_kv_source, main.is_kv_forwarder = (
            layer_id in SOURCES,
            layer_id == forwarder,
        )
        main.compressed_kv_format, main.compress_ratio, main.head_dim = (
            fmt,
            RATIOS[layer_id],
            512,
        )
        main.kv_cache_dtype = "bfloat16"
        main.kv_cache_torch_dtype = torch.bfloat16
        specs[prefix] = main.get_kv_cache_spec(cfg)
        context[prefix] = main
        indexer = bare(IluDeepseekV41IndexerCache)
        indexer.cache_config = cfg.cache_config
        indexer.dtype, indexer.head_dim, indexer.compress_ratio = (
            torch.uint8,
            132,
            RATIOS[layer_id],
        )
        specs[prefix + ".indexer.k_cache"] = indexer.get_kv_cache_spec(cfg)
        context[prefix + ".indexer.k_cache"] = indexer
        if layer_id in SOURCES and RATIOS[layer_id] > 1:
            ring = bare(IluCompressorStateCache)
            ring.block_size, ring.state_dim, ring.dtype = 8, 1024, torch.float32
            specs[prefix + ".compressor.state_cache"] = ring.get_kv_cache_spec(cfg)
            context[prefix + ".compressor.state_cache"] = ring
    return specs, context


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("producer", [False, True])
def test_real_v2_groups_and_generated_page_views(fmt, producer, monkeypatch):
    cfg = config()
    if producer:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "5,5,5,25")
        rank_plans = [
            build_rank_plan(40, SOURCES, INDEX_SOURCES, RATIOS, rank, 4, last_layer=20)
            for rank in range(4)
        ]
        assert [plan.forwarder for plan in rank_plans] == [None, 5, 10, 15]
        rank_sources = [
            specs_for_rank(cfg, fmt, plan.start_layer, plan.end_layer, plan.forwarder)
            for plan in rank_plans
        ]
    else:
        rank_sources = [specs_for_rank(cfg, fmt, 0, 40)]
        assert (
            len([name for name in rank_sources[0][0] if name.endswith(".swa_cache")])
            == 40
        )
        assert all(
            f"model.layers.{layer_id}.attn" in rank_sources[0][0]
            for layer_id in SOURCES
        )
    monkeypatch.setattr(
        kv_cache_utils,
        "_try_get_full_allocation_fallback_groups",
        lambda *_: pytest.fail("full-allocation fallback was reached"),
    )
    budget = CacheBudget.create(8 * 1024**3, fmt)
    configs = kv_cache_utils.get_kv_cache_configs(
        cfg,
        [specs for specs, _ in rank_sources],
        [budget.payload_bytes] * len(rank_sources),
    )
    assert len({cache.num_blocks for cache in configs}) == 1
    for cache, (_, context) in zip(configs, rank_sources):
        layout = plan_scale_layout(cache, fmt)
        budget.validate(layout.payload_bytes, layout.scale_bytes)
        state = CompressedKVState(layout, fmt, "meta", context)
        views = allocate_kv_cache(
            cache, torch.device("meta"), cfg.cache_config.get_resolved_kv_cache_layout()
        )
        actual_specs = layer_specs(cache.kv_cache_groups)
        for name, spec in actual_specs.items():
            if is_quantized_main_spec(spec):
                assert (
                    spec.block_size
                    == fmt.rows_per_page(512)
                    * RATIOS[int(name.split(".layers.")[1].split(".")[0])]
                )
                assert spec.page_size_bytes == 65536
                assert views[name].shape == (
                    cache.num_blocks,
                    1,
                    fmt.rows_per_page(512),
                    fmt.payload_bytes(512),
                )
                scale = state.scales[name]
                assert scale.shape == (
                    cache.num_blocks,
                    fmt.rows_per_page(512),
                    fmt.scale_groups(512),
                )
                assert scale.storage_offset() * 2 == views[
                    name
                ].storage_offset() // 65536 * fmt.scale_page_bytes(512)
            if name.endswith(".swa_cache"):
                assert spec.dtype is torch.bfloat16 and spec.block_size == 64
            if name.endswith(".compressor.state_cache"):
                assert spec.dtype is torch.float32 and spec.block_size == 16
        assert len({view.untyped_storage()._cdata for view in views.values()}) == 1
        assert (
            layout.scale_bytes
            == layout.payload_bytes // 65536 * fmt.scale_page_bytes(512)
        )
        assert (
            any(not group.layer_names for group in cache.kv_cache_groups)
            or not producer
        )
        state.release()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("workspace_bytes", [WORKSPACE_BYTES, PREFILL_WORKSPACE_BYTES])
def test_real_v2_block_override_cannot_allocate_sidecars_outside_budget(
    fmt, workspace_bytes
):
    cfg = config()
    specs, _ = specs_for_rank(cfg, fmt, 0, 40)
    budget = CacheBudget.create(8 * 1024**3, fmt, workspace_bytes=workspace_bytes)
    cfg.cache_config.num_gpu_blocks_override = budget.payload_bytes // 65536 + 1
    cache = kv_cache_utils.get_kv_cache_configs(cfg, [specs], [budget.payload_bytes])[0]
    layout = plan_scale_layout(cache, fmt)
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        budget.validate(layout.payload_bytes, layout.scale_bytes)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("producer", [False, True])
def test_role_workspace_fits_real_v2_1m_groups_and_meta_allocation(
    fmt, producer, monkeypatch
):
    cfg = config()
    cfg.kv_transfer_config = SimpleNamespace(
        kv_role="kv_producer" if producer else "kv_consumer"
    )
    cfg.max_in_flight_tokens = 8192 if producer else 4096
    cfg.scheduler_config.max_num_batched_tokens = 2048
    cfg.model_config.hf_text_config = SimpleNamespace(index_topk=512)
    cfg.scheduler_config.max_num_seqs = 32 if producer else 2
    if producer:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "5,5,5,25")
        plans = [
            build_rank_plan(40, SOURCES, INDEX_SOURCES, RATIOS, rank, 4, last_layer=20)
            for rank in range(4)
        ]
        sources = [
            specs_for_rank(cfg, fmt, plan.start_layer, plan.end_layer, plan.forwarder)
            for plan in plans
        ]
    else:
        # DSpark's three draft layers add only their existing BF16 SWA pools.
        specs, context = specs_for_rank(cfg, fmt, 0, 40)
        draft_specs, draft_context = specs_for_rank(cfg, fmt, 40, 43)
        specs.update(draft_specs)
        context.update(draft_context)
        sources = [(specs, context)]
    monkeypatch.setattr(
        kv_cache_utils,
        "_try_get_full_allocation_fallback_groups",
        lambda *_: pytest.fail("full-allocation fallback was reached"),
    )
    workspace_bytes = workspace_bytes_for_config(cfg)
    assert workspace_bytes == (512 if producer else 128) * 1024**2
    baseline = CacheBudget.create(8 * 1024**3, fmt, workspace_bytes=workspace_bytes)
    budget = CacheBudget.create(
        8 * 1024**3,
        fmt,
        workspace_bytes=workspace_bytes,
        retained_mapper_bytes=retained_mapper_bytes_for_config(cfg),
    )
    assert budget.payload_bytes == baseline.payload_bytes
    caches = kv_cache_utils.get_kv_cache_configs(
        cfg, [specs for specs, _ in sources], [budget.payload_bytes] * len(sources)
    )
    expected_blocks = (budget.total_bytes - workspace_bytes) // (
        65536 + fmt.scale_page_bytes(512)
    )
    for cache, (_, context) in zip(caches, sources):
        assert cache.num_blocks == expected_blocks
        assert cfg.model_config.max_model_len == 1048576
        capacity, _ = kv_cache_utils.get_kv_cache_capacity(cfg, cache)
        scheduler = kv_cache_utils.generate_scheduler_kv_cache_config([cache])
        assert kv_cache_utils.get_kv_cache_capacity(cfg, scheduler)[0] == capacity
        assert capacity >= cfg.model_config.max_model_len
        layout = plan_scale_layout(cache, fmt)
        budget.validate(layout.payload_bytes, layout.scale_bytes)
        state = CompressedKVState(
            layout, fmt, "meta", context, workspace_bytes=budget.arena_bytes
        )
        try:
            views = allocate_kv_cache(
                cache,
                torch.device("meta"),
                cfg.cache_config.get_resolved_kv_cache_layout(),
            )
            assert len({view.untyped_storage()._cdata for view in views.values()}) == 1
            assert state.workspace.storage.numel() == budget.arena_bytes
            state.workspace.reserve_metadata(120)
            assert (
                state.workspace.tensor.numel()
                == workspace_bytes - budget.retained_mapper_bytes - 256
            )
        finally:
            state.release()



def receipt_chain(fmt, tmp_path):
    from vllm_iluvatar.v1.worker.prefill_state.configuration import PrefixGeometry
    from vllm_iluvatar.v1.worker.prefill_state.scheduler_state import (
        PrefixSchedulerState, selected_groups,
    )
    cfg = config()
    specs, _ = specs_for_rank(cfg, fmt, 0, 40)
    available = CacheBudget.create(8 * 1024**3, fmt).payload_bytes if fmt else 8 * 1024**3
    cache = kv_cache_utils.get_kv_cache_configs(cfg, [specs], [available])[0]
    tracker = PrefixSchedulerState.__new__(PrefixSchedulerState)
    tracker.wire, tracker.nonce, tracker.engine_id = tmp_path, "local-run", "D"
    tracker.consumer, tracker.cache_config = True, cache
    tracker.geometry = PrefixGeometry(main_kv_format=fmt.name if fmt else "none")
    tracker.selected = selected_groups(cache, tracker.geometry)
    tracker.requests, tracker.serial = {}, 0
    n = 512
    blocks = SimpleNamespace(blocks=[
        [SimpleNamespace(block_id=g * 100 + j + 1,
                         block_hash="local-prefix" if j == 0 else None, is_null=False)
         for j in range((n + group.kv_cache_spec.block_size - 1)
                        // group.kv_cache_spec.block_size)]
        for g, group in enumerate(cache.kv_cache_groups)])
    request = SimpleNamespace(request_id="request", num_prompt_tokens=n,
        num_computed_tokens=384, replay_start=384, prompt_token_ids=list(range(n)),
        kv_transfer_params={"do_remote_prefill": True, "remote_engine_id": "P"})
    output = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(),
        num_scheduled_tokens={"request": 128},
        kv_connector_block_state=SimpleNamespace(get_block_ids=lambda rid:
            [[b.block_id for b in group] for group in blocks.blocks]))
    metadata = SimpleNamespace(reqs_to_recv={"request": object()})
    return tracker, request, blocks, output, metadata


@pytest.mark.parametrize("fmt", [None, I8G64, I4G32])
def test_real_v2_e25_receipt_keeps_current_prefix_and_compact_remote_positions(fmt, tmp_path):
    tracker, req, blocks, output, meta = receipt_chain(fmt, tmp_path)
    tracker.allocated(req, blocks, 384)
    peers = {"P": SimpleNamespace(addr="offline-peer")}
    tracker.add_metadata(meta, output, peers)
    assert not meta.source_full_ready_receipts
    tracker.completed(SimpleNamespace(finished_recving={"request"}, failed_recving=set(),
                                      invalid_block_ids=set()))
    tracker.add_metadata(meta, output, peers)
    receipt = meta.source_full_ready_receipts["request"]
    assert receipt["N"] == receipt["fullkv_valid_before_replay"] == 512
    assert receipt["original_receive_success"] and receipt["external"] == 384
    assert meta.source_state_endpoints["request"]["engine_id"] == "P"
    names = set()
    for g in tracker.selected:
        group = tracker.cache_config.kv_cache_groups[g]
        assert receipt["group_specs"][g] == dict(block_size=group.kv_cache_spec.block_size,
            layer_names=list(group.layer_names), dtype=str(group.kv_cache_spec.dtype))
        assert receipt["groups"][g] == [b.block_id for b in blocks.blocks[g]]
        assert receipt["received_positions"][g] == [dict(logical=j, block_id=b.block_id)
            for j,b in enumerate(blocks.blocks[g]) if j > 0]
        names.update(name for name in group.layer_names if name.endswith(".attn"))
    assert names == {f"model.layers.{i}.attn" for i in SOURCES}


@pytest.mark.parametrize("fmt", [None, I8G64, I4G32])
def test_real_v2_e25_failed_receive_and_preempt_do_not_reuse_ready_receipt(fmt, tmp_path):
    tracker, req, blocks, output, meta = receipt_chain(fmt, tmp_path)
    tracker.allocated(req, blocks, 384)
    tracker.completed(SimpleNamespace(finished_recving={"request"}, failed_recving={"request"},
                                      invalid_block_ids=set()))
    tracker.add_metadata(meta, output, {})
    assert not meta.source_full_ready_receipts
    output.preempted_req_ids.add("request")
    tracker.add_metadata(meta, output, {})
    assert not tracker.requests
    output.preempted_req_ids.clear()
    for group in blocks.blocks:
        for block in group:
            block.block_id += 500
    tracker.allocated(req, blocks, 384)
    tracker.add_metadata(meta, output, {})
    assert not meta.source_full_ready_receipts
    tracker.completed(SimpleNamespace(finished_recving={"request"}, failed_recving=set(),
                                      invalid_block_ids=set()))
    tracker.add_metadata(meta, output, {})
    receipt = meta.source_full_ready_receipts["request"]
    assert receipt["alloc_serial"] == 2
    for g, ids in receipt["groups"].items():
        assert ids == [b.block_id for b in blocks.blocks[g]]
