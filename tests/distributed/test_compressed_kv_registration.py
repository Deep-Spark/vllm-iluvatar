"""Paired registration and descriptor planning without agents or services."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.layer_mapping import (
    LayerMappingPlanner,
    RegistrationView,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
    MemoryRegistrationPlanner,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.lifecycle import RequestTombstoneStore
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
    TransferNotificationStatus,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.utils import (
    NotifyPayload,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    NixlConnectorWorker,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.xfer_plan import (
    RegionGeometry,
    RegionPair,
    build_xfer_plan,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PAYLOAD_PAGE_BYTES,
    CompressedKVState,
    format_factors,
    plan_scale_layout,
)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("block_outer", [False, True])
def test_registration_bundles_payload_and_scale_pointers_for_every_owner(
    fmt, block_outer, monkeypatch
):
    page, count = PAYLOAD_PAGE_BYTES, 3
    names = ["model.layers.2.attn", "model.layers.8.attn"]
    spec = MLAAttentionSpec(
        block_size=fmt.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=fmt.payload_bytes(512),
        cache_dtype_str=fmt.name,
    )
    tensor = SimpleNamespace(
        layers=names,
        size=page * count * 2,
        offset=0,
        layer_stride=page if block_outer else page * count,
        block_stride=page * 2 if block_outer else page,
    )
    config = SimpleNamespace(
        num_blocks=count,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[SimpleNamespace(layer_names=names, kv_cache_spec=spec)],
    )
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    state = CompressedKVState(plan_scale_layout(config, fmt), fmt, "cpu", context)
    raw = torch.empty(tensor.size, dtype=torch.uint8)
    caches = {
        name: torch.as_strided(
            raw,
            (count, fmt.rows_per_page(512), fmt.payload_bytes(512)),
            (tensor.block_stride, fmt.payload_bytes(512), 1),
            index * tensor.layer_stride,
        )
        for index, name in enumerate(names)
    }
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker.use_mla, worker._has_mamba, worker.kv_buffer_device = True, False, "cpu"
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context)
    )
    worker._memory_registration_planner = MemoryRegistrationPlanner(config)
    worker._logical_num_blocks = count
    worker._layer_specs = dict.fromkeys(names, spec)
    worker._layer_group_id = dict.fromkeys(names, 0)
    worker._layer_backend_names = dict.fromkeys(names, "ILU_FLASHMLA_SPARSE_DSV41")
    worker.transfer_topo = SimpleNamespace(
        get_transfer_cache_regions=lambda cache, _: [cache]
    )
    module = __import__(NixlConnectorWorker.__module__, fromlist=["get_skip_layers"])
    monkeypatch.setattr(module, "get_skip_layers", set)
    monkeypatch.setattr(module, "host_kv_enabled", lambda: False)
    try:
        registrations, addresses = worker._register_kv_views(caches)
        assert len(registrations) == len(addresses) == 2
        for name in names:
            assert worker.layer_ptr_mapping[name] == [0, 1]
            assert worker.layer_num_blocks_mapping[name] == [count, count]
            assert worker.layer_group_mapping[name] == [0, 0]
            assert worker.layer_kernel_blocks_mapping[name] == [1, 1]
            assert worker.layer_block_tokens_mapping[name] == [
                spec.block_size,
                spec.block_size,
            ]
        state.scales[names[0]] = state.scales[names[1]]
        with pytest.raises(ValueError, match="mirror payload"):
            worker._register_kv_views(caches, state.scales)
    finally:
        state.release()


def test_one_transfer_plan_uses_both_planes_with_identical_block_ranges():
    data = RegionGeometry(0, 0, 65536, 65536, 7)
    scales = RegionGeometry(1, 0, 2048, 2048, 7)
    plan = build_xfer_plan(
        [RegionPair(2, region, region, 0, 0, True) for region in (data, scales)]
    )
    assert len(plan.per_layer[2]) == 1
    group = plan.per_layer[2][0]
    local, remote = group.descriptor_ids(
        [np.array([[1, 2, 3]], dtype=np.int64)], [np.array([[4, 5]], dtype=np.int64)]
    )
    assert local.tolist() == [2, 3, 9, 10]
    assert remote.tolist() == [4, 5, 11, 12]
    assert group.num_regions == 2


def test_descriptor_ids_remain_64_bit_above_4gib():
    count = (1 << 32) + 3
    plan = build_xfer_plan(
        [
            RegionPair(2, region, region, 0, 0, True)
            for region in (
                RegionGeometry(0, 0, 65536, 65536, count),
                RegionGeometry(1, 0, 2048, 2048, count),
            )
        ]
    )
    local, remote = plan.per_rank[0].descriptor_ids(
        [np.array([[count - 1]], dtype=np.int64)],
        [np.array([[count - 2]], dtype=np.int64)],
    )
    assert local.dtype == remote.dtype == np.int64
    assert local[1] == 2 * count - 1 and remote[1] == 2 * count - 2


def test_compatibility_includes_encoding_groups_scales_and_layout(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    assert format_factors() == {}
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i8g64")
    first = format_factors()
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i4g32")
    second = format_factors()
    assert first != second
    assert first["compressed_kv_encoding"] == second["compressed_kv_encoding"] == 1
    assert (
        first["compressed_kv_group_size"] == 64
        and second["compressed_kv_group_size"] == 32
    )
    assert (
        first["compressed_kv_scale_dtype"]
        == second["compressed_kv_scale_dtype"]
        == "float16"
    )
    assert first["compressed_kv_layout"] == second["compressed_kv_layout"]


def test_peer_missing_scale_pointer_is_rejected_before_transfer():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=512,
        cache_dtype_str="i8g64",
    )

    def view(count):
        return RegistrationView(
            ptr={"main": list(range(count))},
            block_stride={"main": [65536, 2048][:count]},
            desc={"main": [[(0, 65536)], [(0, 2048)]][:count]},
            num_blocks={"main": [3] * count},
            group={"main": [0] * count},
            kernel_blocks={"main": [1] * count},
            block_tokens={"main": [128] * count},
            backend={"main": "ILU_FLASHMLA_SPARSE_DSV41"},
        )

    planner = LayerMappingPlanner(
        local=view(2),
        layer_specs={"main": spec},
        layer_global_index={"main": 2},
        pp_layer_start=2,
        num_layers=1,
        local_tp_size=1,
        has_mamba=False,
    )
    with pytest.raises(ValueError, match="pointer count mismatch"):
        planner.select(local_start=0, local_end=1, remote=view(1), remote_tp_size=1)


def receiver():
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker._recv_requests = {
        "request": SimpleNamespace(engine_id="peer", generation="generation", local_block_ids=[[1, 2], [3]])
    }
    worker._recv_generations = {"request": "generation"}
    worker._recv_tombstones = RequestTombstoneStore(86400, 128)
    worker._accepting_work = True
    worker._recv_notification_status = {}
    worker._aborted_recv_requests = set()
    worker.remote_sessions = {"peer": SimpleNamespace(pending_recv={"request"})}
    worker.num_groups = 2
    completions = []
    worker._put_completion = completions.append
    return worker, completions


def notification(layer):
    return NotifyPayload(
        src_pp=0,
        pp_start=0,
        pp_end=1,
        pp_layer=layer,
        pp_layer_start=0,
        pp_layer_end=2,
        src_tp=0,
        tp_start=0,
        tp_end=1,
        count=2,
    )


def test_partial_and_duplicate_notifications_never_publish_ready():
    worker, completions = receiver()
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.SUCCESS, notification(0)
    )
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.SUCCESS, notification(0)
    )
    assert completions == [] and "request" in worker._recv_requests
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.SUCCESS, notification(1)
    )
    assert len(completions) == 1 and completions[0].failures == {}
    assert set(completions[0].recv_metadata) == {"request"}
    assert (
        worker._recv_requests
        == worker._recv_notification_status
        == worker._recv_generations
        == {}
    )
    assert worker.remote_sessions["peer"].pending_recv == set()


def test_failed_pair_is_retained_until_generation_is_fenced():
    worker, completions = receiver()
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.SUCCESS, notification(0)
    )
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.FAILED, notification(1)
    )
    assert completions == [] and "request" in worker._recv_requests
    # SDK forwards terminal producer proof in the generation-safe release map.
    worker._apply_start_load({}, {}, {}, set(), {}, {"request": ("generation", "producer fenced")})
    worker._apply_recv_notification(
        "request", "generation", TransferNotificationStatus.SUCCESS, notification(1)
    )
    assert len(completions) == 1 and "request" in completions[0].failures
    assert completions[0].recv_metadata == {}
    assert (
        worker._recv_requests
        == worker._recv_notification_status
        == worker._recv_generations
        == {}
    )
    assert worker.remote_sessions["peer"].pending_recv == set()
