"""CPU-only topology, replay receipts and standard lifecycle regression tests."""

import os
import queue
import threading
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from vllm_iluvatar.v1.worker.prefill_state.configuration import (
    SESSION_KEY,
    ProducerTopology,
    config_pipeline_bounds,
    enabled,
    initialize_config,
    layer_partitions,
    session_paths,
)
from vllm_iluvatar.v1.worker.prefill_state.runtime import PrefixRuntime
from vllm_iluvatar.v1.worker.prefill_state.scheduler_state import PrefixSchedulerState

TOPOLOGIES = [
    (2, (5, 5, 5, 25)),
    (1, (3, 3, 3, 3, 3, 2, 2, 21)),
    (2, (3, 3, 3, 3, 3, 2, 2, 21)),
    (1, (2,) * 5 + (1,) * 10 + (20,)),
]


@pytest.fixture(autouse=True)
def restore_partition_environment():
    previous = os.environ.get("VLLM_PP_LAYER_PARTITION")
    yield
    if previous is None:
        os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
    else:
        os.environ["VLLM_PP_LAYER_PARTITION"] = previous


def config(tmp_path, role="kv_consumer", tp=2, pp=1):
    hf = NS(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        sliding_window=128,
        head_dim=512,
        hidden_size=5120,
        hc_mult=4,
        index_topk=512,
        candidate_topk_blocks=2048,
        index_head_dim=128,
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
    )
    extra = {"prefix_state_root": str(tmp_path)}
    transfer = NS(
        kv_role=role,
        kv_connector="OptNixlConnector",
        engine_id="engine",
        kv_connector_extra_config=extra,
        get_from_extra_config=lambda key, default=None: extra.get(key, default),
    )
    return NS(
        kv_transfer_config=transfer,
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=hf, hf_config=NS(text_config=hf)),
        parallel_config=NS(
            tensor_parallel_size=tp, pipeline_parallel_size=pp, data_parallel_size=1
        ),
        cache_config=NS(cache_dtype="bfloat16"),
    )


@pytest.mark.parametrize("tp,parts", TOPOLOGIES)
def test_model_connector_bounds_and_full_rank_coverage(
    tmp_path, monkeypatch, tp, parts
):
    from vllm_iluvatar.models.deepseek_v41.pp_plan import build_rank_plan

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, parts)))
    monkeypatch.delenv("VLLM_ILU_PD_PREFILL_FULL_DEPTH", raising=False)
    cfg = config(tmp_path, "kv_producer", tp, len(parts))
    initialize_config(cfg)
    topology = ProducerTopology.from_config(cfg)
    bounds = config_pipeline_bounds(cfg)
    assert topology.world_size == tp * len(parts)
    assert topology.boundary_pp == len(parts) - 1
    assert {topology.rank(pp, t) for pp in range(len(parts)) for t in range(tp)} == set(
        range(tp * len(parts))
    )
    for rank, (start, end) in enumerate(bounds):
        plan = build_rank_plan(
            40,
            (2, 8, 14, 20),
            (2, 8, 14, 20, 24, 28, 32, 36),
            tuple([0, 0] + [2] * 18 + [1] * 20),
            rank,
            len(parts),
            20,
            20,
        )
        assert (
            (plan.start_layer, plan.end_layer)
            == (start, min(end, 21))
            == topology.layer_bounds[rank]
        )
    for layer in range(21):
        a, b = topology.layer_bounds[topology.owner(layer)]
        assert a <= layer < b


def test_default_partitions_are_nonempty_and_reach_upstream_helpers(
    tmp_path, monkeypatch
):
    from vllm.distributed.utils import get_pp_indices

    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    cfg = config(tmp_path, "kv_producer", 1, 8)
    initialize_config(cfg)
    assert config_pipeline_bounds(cfg) == tuple(
        get_pp_indices(40, r, 8) for r in range(8)
    )
    assert layer_partitions(40, 16, 20) == (2,) * 5 + (1,) * 10 + (20,)
    assert all(a < b for a, b in ProducerTopology.from_config(cfg).layer_bounds)


def test_default_optout_and_configuration_owned_lifetime(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    cfg = config(tmp_path)
    assert enabled(cfg)
    initialize_config(cfg)
    nonce = cfg.kv_transfer_config.kv_connector_extra_config[SESSION_KEY]
    clone = deepcopy(cfg)
    initialize_config(clone)
    assert clone.kv_transfer_config.kv_connector_extra_config[SESSION_KEY] == nonce
    assert session_paths(cfg) == session_paths(clone)
    other = config(tmp_path)
    initialize_config(other)
    assert session_paths(other) != session_paths(cfg)
    cfg.kv_transfer_config.kv_connector_extra_config["enable_prefix_state_reuse"] = (
        False
    )
    assert not enabled(cfg)
    cfg.kv_transfer_config.kv_connector_extra_config["enable_prefix_state_reuse"] = (
        "false"
    )
    with pytest.raises(ValueError, match="boolean"):
        enabled(cfg)


def test_full_depth_producer_and_unsupported_model_do_not_attach(tmp_path, monkeypatch):
    cfg = config(tmp_path, "kv_producer", 2, 4)
    monkeypatch.setenv("VLLM_ILU_PD_PREFILL_FULL_DEPTH", "1")
    assert not enabled(cfg)
    cfg = config(tmp_path)
    cfg.model_config.hf_text_config.model_type = "llama"
    assert not enabled(cfg)


def cache_config():
    names = [
        "model.layers.2.attn",
        "model.layers.2.attn.indexer.k_cache",
        "model.layers.8.attn",
        "model.layers.8.attn.indexer.k_cache",
        "model.layers.14.attn",
        "model.layers.14.attn.indexer.k_cache",
        "model.layers.20.attn",
        "model.layers.20.attn.indexer.k_cache",
        "model.layers.0.attn.swa_cache",
    ]
    return NS(
        kv_cache_groups=[
            NS(layer_names=[name], kv_cache_spec=NS(block_size=64, dtype="bfloat16"))
            for name in names
        ]
    )


def request(n=257):
    return NS(
        request_id="r",
        num_prompt_tokens=n,
        prompt_token_ids=list(range(n)),
        num_computed_tokens=0,
        replay_start=0,
        kv_transfer_params={"do_remote_prefill": True, "remote_engine_id": "P"},
    )


def setup_receipt(tmp_path, n):
    cfg = config(tmp_path)
    initialize_config(cfg)
    tracker = PrefixSchedulerState(cfg, cache_config())
    req = request(n)
    blocks = NS(
        blocks=[
            [
                NS(
                    block_id=g * 10 + j + 1,
                    block_hash=None if j else "local",
                    is_null=False,
                )
                for j in range(5)
            ]
            for g in range(9)
        ]
    )
    tracker.allocated(req, blocks, n)
    table = tuple([b.block_id for b in group] for group in blocks.blocks)
    output = NS(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"r": 128 + n % 2},
        kv_connector_block_state=NS(get_block_ids=lambda rid: table),
    )
    meta = NS(reqs_to_recv={"r": NS()})
    peers = {"P": NS(addr="10.2.3.4:5600")}
    return tracker, req, blocks, output, meta, peers


@pytest.mark.parametrize("n", [256, 257, 76393])
def test_receipt_requires_original_success_and_current_replay_pages(tmp_path, n):
    tracker, req, blocks, out, meta, peers = setup_receipt(tmp_path, n)
    req.num_computed_tokens = req.replay_start = n - n % 2 - 128
    tracker.add_metadata(meta, out, peers)
    assert not meta.source_full_ready_receipts
    assert meta.source_state_endpoints["r"]["addr"] == "10.2.3.4:5600"
    tracker.completed(
        NS(finished_recving={"r"}, failed_recving=set(), invalid_block_ids=set())
    )
    tracker.add_metadata(meta, out, peers)
    receipt = meta.source_full_ready_receipts["r"]
    assert receipt["H"] == n - n % 2 and receipt["query_rows"] == 128 + n % 2
    assert receipt["groups"][0] == [b.block_id for b in blocks.blocks[0]]
    assert receipt["received_positions"][0][0] == {"logical": 1, "block_id": 2}
    req.num_computed_tokens += 1
    tracker.add_metadata(meta, out, peers)
    assert not meta.source_full_ready_receipts
    req.num_computed_tokens -= 1
    out.kv_connector_block_state = NS(get_block_ids=lambda rid: None)
    tracker.add_metadata(meta, out, peers)
    assert not meta.source_full_ready_receipts


@pytest.mark.parametrize(
    "failure", ["receive", "invalid_page", "preempt", "finish", "readmit"]
)
def test_receipt_failure_cancellation_and_readmission_fail_closed(tmp_path, failure):
    tracker, req, blocks, out, meta, peers = setup_receipt(tmp_path, 257)
    req.num_computed_tokens = req.replay_start = 128
    tracker.completed(
        NS(finished_recving={"r"}, failed_recving=set(), invalid_block_ids=set())
    )
    if failure == "receive":
        tracker.completed(
            NS(finished_recving={"r"}, failed_recving={"r"}, invalid_block_ids=set())
        )
    elif failure == "invalid_page":
        tracker.completed(
            NS(finished_recving=set(), failed_recving=set(), invalid_block_ids={1})
        )
    elif failure == "preempt":
        out.preempted_req_ids = {"r"}
    elif failure == "finish":
        out.finished_req_ids = {"r"}
    else:
        tracker.allocated(req, blocks, 257)
    tracker.add_metadata(meta, out, peers)
    assert not meta.source_full_ready_receipts


def test_block_lifecycle_tracks_replacements_and_new_admissions():
    class Runtime(PrefixRuntime):
        states = {}
        lock = threading.Lock()

    runner = NS(
        vllm_config=None, device="cpu", block_tables=NS(blocks_per_kv_block=[2, 1])
    )
    ctx = Runtime(runner, None)
    ctx.states = {"r": {"blocks": [[2, 3], [2]]}}
    out = NS(
        block_table_updates={"r": ([10, 11], [20])},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=NS(req_ids=["r"], new_block_ids=[([12], [21])]),
    )
    ctx.observe_scheduler_output(out)
    assert ctx.states["r"]["blocks"] == [[20, 21, 22, 23], [20]]
    out.block_table_updates = {}
    ctx.observe_scheduler_output(out)
    assert ctx.states["r"]["blocks"] == [[20, 21, 22, 23, 24, 25], [20, 21]]
    out.preempted_req_ids = {"r"}
    ctx.observe_scheduler_output(out)
    assert ctx.states["r"]["blocks"] == [[20, 21, 22, 23, 24, 25], [20, 21]]


def test_multimodal_embeddings_and_lora_do_not_use_token_only_identity():
    req = NS(
        prompt_token_ids=list(range(128)),
        prefill_token_ids=list(range(128)),
        prompt_len=128,
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
    )
    assert PrefixRuntime.request_eligible(req)
    for key, value in [
        ("mm_features", ["image"]),
        ("prompt_embeds", object()),
        ("lora_request", object()),
    ]:
        bad = deepcopy(req)
        setattr(bad, key, value)
        assert not PrefixRuntime.request_eligible(bad)


def test_native_connector_scheduler_delivers_receipts_without_scheduler_patches(
    tmp_path,
):
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.scheduler import (
        NixlConnectorScheduler,
    )

    cfg = config(tmp_path)
    initialize_config(cfg)
    tracker = PrefixSchedulerState(cfg, cache_config())
    req = request(257)
    blocks = NS(
        blocks=[
            [
                NS(block_id=g * 10 + j + 1, block_hash=None, is_null=False)
                for j in range(5)
            ]
            for g in range(9)
        ]
    )
    ids = tuple([b.block_id for b in group] for group in blocks.blocks)
    blocks.get_unhashed_block_ids_all_groups = lambda: ids
    scheduler = NixlConnectorScheduler.__new__(NixlConnectorScheduler)
    scheduler.prefix_state = tracker
    scheduler.is_consumer = True
    scheduler.engine_id = "D"
    scheduler.kv_group_nums = 9
    scheduler._reqs_to_recv = {}
    scheduler._reqs_not_processed_recv = set()
    scheduler._reqs_not_send_to_worker = set()
    scheduler._live_recv_reqs = {req.request_id}
    scheduler._scheduler_failed_recvs = set()
    scheduler._ownership_generations = {}
    scheduler._ownership_remote_engines = {}
    scheduler._ownership_states = {}
    scheduler._ownership_deadlines = {}
    scheduler._ownership_cancel_pending = set()
    scheduler._ownership_safe_releases = queue.SimpleQueue()
    scheduler._kv_lease_duration = 45
    scheduler.peer_node_metas = {"P": NS(addr="10.0.0.2:5600")}
    scheduler._get_or_connect_peer = lambda rid: scheduler.peer_node_metas[rid]
    scheduler.get_sw_clipped_blocks = lambda blocks: blocks
    scheduler._remote_block_starts = lambda blocks: [0] * 9
    sent = []
    scheduler._async_loop = NS(call_soon_threadsafe=lambda fn, arg: sent.append(arg))
    scheduler.send_to_peer_infos = NS(put_nowait=lambda arg: None)
    scheduler.update_state_after_alloc(req, blocks, 257)
    assert len(sent) == 1 and tracker.requests["r"]["external"] == 257
    assert not req.kv_transfer_params["do_remote_prefill"]
    output = NS(
        finished_req_ids=set(),
        preempted_req_ids=None,
        num_scheduled_tokens={"r": 129},
        kv_connector_block_state=NS(get_block_ids=lambda rid: ids),
    )
    meta = scheduler.build_connector_meta(output)
    assert meta.source_state_endpoints["r"]["engine_id"] == "P"
    assert not meta.source_full_ready_receipts
    # The native method normally returns early when there are no P send completions.
    scheduler.update_connector_output(
        NS(
            finished_sending=set(),
            finished_recving={"r"},
            failed_recving=set(),
            invalid_block_ids=set(),
        )
    )
    req.num_computed_tokens = req.replay_start = 128
    meta = scheduler.build_connector_meta(output)
    assert meta.source_full_ready_receipts["r"]["query_rows"] == 129
    assert meta.source_full_ready_receipts["r"]["groups"][0] == ids[0]
