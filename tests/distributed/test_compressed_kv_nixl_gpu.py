"""Local two-agent UCX/VRAM smoke; no engine, listener, or vLLM service."""

import os
import time
import uuid
from collections import defaultdict
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from nixl._api import nixl_agent, nixl_agent_config
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.layer_mapping import (
    LayerMappingPlanner,
    RegistrationView,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
    MemoryRegistrationPlanner,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
    TransferNotificationStatus,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.utils import (
    NotifyPayload,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    NixlConnectorWorker,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.xfer_plan import (
    build_xfer_plan,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PAYLOAD_PAGE_BYTES,
    plan_scale_layout,
)
from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
    IluDeepseekV4FlashMLABackend,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1"
    or os.environ.get("KV_COMPRESSED_KV_NIXL_TESTS") != "1",
    reason="requires an explicit single-GPU lease and both KV_COMPRESSED_KV_*_TESTS flags",
)


def byte_pattern(size, page_bytes, seed):
    indices = torch.arange(size, dtype=torch.int64)
    return ((indices + indices // page_bytes * 37 + seed) % 256).to(torch.uint8)


def buffers(fmt, block_outer):
    count, names = 5, ("model.layers.2.attn", "model.layers.8.attn")
    page = PAYLOAD_PAGE_BYTES
    spec = MLAAttentionSpec(
        block_size=fmt.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=fmt.payload_bytes(512),
        cache_dtype_str=fmt.name,
    )
    # Two owner slots and one untouched guard slot per allocation pattern.
    tensor = SimpleNamespace(
        layers=list(names),
        size=3 * page * count,
        offset=0,
        layer_stride=page if block_outer else page * count,
        block_stride=3 * page if block_outer else page,
    )
    config = SimpleNamespace(
        num_blocks=count,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[SimpleNamespace(layer_names=list(names), kv_cache_spec=spec)],
    )
    layout = plan_scale_layout(config, fmt)
    data = torch.full((tensor.size,), 0xB7, dtype=torch.uint8, device="cuda")
    scales = torch.empty(layout.scale_bytes // 2, dtype=torch.float16, device="cuda")
    scales.view(torch.uint8).fill_(0xC5)
    caches, scale_views = {}, {}
    for index, name in enumerate(names):
        caches[name] = torch.as_strided(
            data,
            (count, fmt.rows_per_page(512), fmt.payload_bytes(512)),
            (tensor.block_stride, fmt.payload_bytes(512), 1),
            index * tensor.layer_stride,
        )
        placement = layout.placements[name]
        scale_views[name] = torch.as_strided(
            scales,
            (count, placement.rows, placement.groups),
            (placement.block_stride_elements, placement.groups, 1),
            placement.offset_elements,
        )
    return SimpleNamespace(
        config=config,
        spec=spec,
        data=data,
        scales=scales,
        caches=caches,
        scale_views=scale_views,
        names=names,
    )


def make_worker(storage, fmt, name):
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker.nixl_wrapper = nixl_agent(
        name,
        nixl_agent_config(backends=["UCX"], num_threads=1, enable_listen_thread=False),
    )
    model = SimpleNamespace(
        model="paired-main-kv-smoke",
        dtype=torch.bfloat16,
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: 512,
        get_total_num_hidden_layers=lambda: 10,
    )
    context = {
        key: SimpleNamespace(compressed_kv_format=fmt, kv_cache_scales=value)
        for key, value in storage.scale_views.items()
    }
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        model_config=model,
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    worker.model_config = model
    worker.use_mla, worker._has_mamba, worker.kv_buffer_device = True, False, "cuda"
    worker.nixl_backends, worker.nixl_memory_type = ["UCX"], "VRAM"
    worker.engine_id = name
    worker.tp_rank = worker.pp_rank = worker.global_rank = 0
    worker.tp_world_size = worker.pp_world_size = worker.total_num_kv_heads = 1
    worker.num_blocks = worker._logical_num_blocks = storage.config.num_blocks
    worker.block_size, worker.kv_cache_layout = 64, "LBNHC"
    worker._mamba_ssm_size = (0, 0)
    worker.attn_backend = IluDeepseekV4FlashMLABackend
    worker.backend_name = worker.attn_backend.get_name()
    worker._registered_descs = []
    worker._layer_specs = dict.fromkeys(storage.names, storage.spec)
    worker._layer_group_id = dict.fromkeys(storage.names, 0)
    worker._layer_backend_names = dict.fromkeys(storage.names, worker.backend_name)
    worker._layer_global_index = dict(zip(storage.names, (2, 8)))
    worker._memory_registration_planner = MemoryRegistrationPlanner(storage.config)
    worker.src_xfer_side_handles = {}
    worker.dst_xfer_side_handles = defaultdict(list)
    worker._remote_agents = {}
    worker.enforce_handshake_compat = False
    worker.register_kv_caches(storage.caches, storage.scale_views)
    assert len(worker.seen_base_addresses) == 2  # one shared pool per plane
    return worker


def expected_after_mapping(source, destination, source_ids, destination_ids):
    data, scales = destination.data.cpu().clone(), destination.scales.cpu().clone()
    for name in source.names:
        for old, new in zip(source_ids, destination_ids, strict=True):
            view = destination.caches[name]
            torch.as_strided(data, view.shape, view.stride(), view.storage_offset())[
                new
            ].copy_(source.caches[name][old].cpu())
            view = destination.scale_views[name]
            torch.as_strided(scales, view.shape, view.stride(), view.storage_offset())[
                new
            ].copy_(source.scale_views[name][old].cpu())
    return data, scales.view(torch.uint8)


def wait_transfer(agent, handle):
    deadline = time.monotonic() + 10
    state = agent.transfer(handle)
    while state == "PROC" and time.monotonic() < deadline:
        time.sleep(0.001)
        state = agent.check_xfer_state(handle)
    assert state == "DONE", f"native UCX transfer failed/timed out: {state}"


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("block_outer", [False, True])
def test_native_paired_public_registration_nonidentity_blocks_and_ready(
    fmt, block_outer, monkeypatch
):
    assert torch.cuda.device_count() == 1
    free_before, _ = torch.cuda.mem_get_info()
    assert free_before - (256 << 20) >= (1 << 30)
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", fmt.name)
    worker_module = __import__(
        NixlConnectorWorker.__module__, fromlist=["get_skip_layers"]
    )
    # A standalone test has no model-configured skipped layers or host mirror.
    monkeypatch.setattr(worker_module, "get_skip_layers", set)
    monkeypatch.setattr(worker_module, "host_kv_enabled", lambda: False)
    monkeypatch.setattr(worker_module, "may_override_kv_cache", lambda caches: caches)
    source, destination = buffers(fmt, block_outer), buffers(fmt, block_outer)
    source.data.copy_(byte_pattern(source.data.numel(), PAYLOAD_PAGE_BYTES, 19))
    source.scales.view(torch.uint8).copy_(
        byte_pattern(source.scales.numel() * 2, fmt.scale_page_bytes(512), 71)
    )
    torch.cuda.synchronize()
    workers, xfers = [], []
    remote_name = reverse_name = None
    try:
        token = uuid.uuid4().hex
        workers.append(make_worker(source, fmt, "kv-paired-source-" + token))
        workers.append(make_worker(destination, fmt, "kv-paired-destination-" + token))
        sender, receiver = workers
        wire = msgspec.msgpack.encode(receiver.xfer_handshake_metadata)
        monkeypatch.setattr(
            worker_module, "zmq_comm2proxy", lambda *args, **kwargs: wire
        )
        remote = sender._fetch_remote_agent_metadata(
            None,
            engine_id=receiver.engine_id,
            global_remote_rank=0,
            handshake_decoder=msgspec.msgpack.Decoder(NixlHandshakePayload),
            decoder=msgspec.msgpack.Decoder(NixlAgentMetadata),
        )
        local = msgspec.msgpack.decode(
            sender.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
        )
        planner = LayerMappingPlanner(
            local=RegistrationView.from_agent_metadata(local),
            layer_specs=sender._layer_specs,
            layer_global_index=sender._layer_global_index,
            pp_layer_start=0,
            num_layers=10,
            local_tp_size=1,
            has_mamba=False,
        )
        pairs = []
        for layer in planner.select(
            local_start=0,
            local_end=10,
            remote=RegistrationView.from_agent_metadata(remote),
            remote_tp_size=1,
        ):
            geometry = sender._check_layer_xfer(
                layer, remote_tp_rank=0, remote_tp_size=1, remote_dp_rank=0
            )
            pairs.extend(sender._layer_region_pairs(layer, geometry))
        plan = build_xfer_plan(pairs)
        assert len(plan.local_regions) == len(plan.remote_regions) == 4
        remote_name = sender._ensure_remote_nixl_agent(remote, None)
        reverse_name = receiver.nixl_wrapper.add_remote_agent(local.agent_metadata)
        local_handle = sender._local_xfer_dlist(plan.local_regions)
        remote_handle = sender._remote_xfer_dlist(
            remote, remote_name, plan.remote_regions
        )
        source_ids, destination_ids = [3, 1], [1, 4]
        local_ids, remote_ids = plan.per_rank[0].descriptor_ids(
            [np.array([source_ids], dtype=np.int64)],
            [np.array([destination_ids], dtype=np.int64)],
        )
        assert len(local_ids) == len(remote_ids) == 8
        expected_data, expected_scales = expected_after_mapping(
            source, destination, source_ids, destination_ids
        )
        source_data, source_scale_bytes = (
            source.data.cpu(),
            source.scales.view(torch.uint8).cpu(),
        )
        original_scales = destination.scales.view(torch.uint8).cpu()
        completions = []
        request_id = "paired-byte-request"
        receiver._recv_requests = {
            request_id: SimpleNamespace(
                engine_id=sender.engine_id, local_block_ids=[destination_ids]
            )
        }
        receiver._recv_deadlines = {request_id: time.perf_counter() + 10}
        receiver._recv_notification_status = {}
        receiver._aborted_recv_requests = set()
        receiver.remote_sessions = {
            sender.engine_id: SimpleNamespace(pending_recv={request_id})
        }
        receiver.num_groups = 1

        def complete(receipt):
            # This is the actual connector receive-completion boundary.
            assert torch.equal(destination.data.cpu(), expected_data)
            assert torch.equal(
                destination.scales.view(torch.uint8).cpu(), expected_scales
            )
            completions.append(receipt)

        receiver._put_completion = complete
        receiver._poll_new_notifs()
        assert completions == []
        # Deliver just payload descriptors with no completion notification:
        # visible bytes alone must not publish a ready receipt.
        payload_regions = np.array(
            [region.length == PAYLOAD_PAGE_BYTES for region in plan.local_regions]
        )
        payload_mask = payload_regions[local_ids // source.config.num_blocks]
        first = sender.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            local_handle,
            local_ids[payload_mask],
            remote_handle,
            remote_ids[payload_mask],
            backends=["UCX"],
        )
        xfers.append(first)
        wait_transfer(sender.nixl_wrapper, first)
        receiver._poll_new_notifs()
        assert completions == []
        assert torch.equal(destination.data.cpu(), expected_data)
        assert torch.equal(destination.scales.view(torch.uint8).cpu(), original_scales)
        notification = NotifyPayload(0, 0, 1, 0, 0, 1, 0, 0, 1, 1)
        message = sender._build_notification(
            request_id, notification.encode(), TransferNotificationStatus.SUCCESS
        )
        paired = sender.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            local_handle,
            local_ids,
            remote_handle,
            remote_ids,
            notif_msg=message,
            backends=["UCX"],
        )
        xfers.append(paired)
        wait_transfer(sender.nixl_wrapper, paired)
        deadline = time.monotonic() + 10
        while not completions and time.monotonic() < deadline:
            receiver._poll_new_notifs()
            time.sleep(0.001)
        assert len(completions) == 1
        assert completions[0].failures == {}
        assert completions[0].done_recving == {request_id}
        assert receiver._recv_requests == receiver._recv_notification_status == {}
        assert torch.equal(source.data.cpu(), source_data)
        assert torch.equal(source.scales.view(torch.uint8).cpu(), source_scale_bytes)
        assert free_before - torch.cuda.mem_get_info()[0] < (256 << 20)
    finally:
        if workers:
            for handle in xfers:
                workers[0].nixl_wrapper.release_xfer_handle(handle)
            for handle in workers[0].src_xfer_side_handles.values():
                workers[0].nixl_wrapper.release_dlist_handle(handle)
            for handles in workers[0].dst_xfer_side_handles.values():
                for handle in handles:
                    workers[0].nixl_wrapper.release_dlist_handle(handle)
            if remote_name is not None:
                workers[0].nixl_wrapper.remove_remote_agent(remote_name)
            if reverse_name is not None:
                workers[1].nixl_wrapper.remove_remote_agent(reverse_name)
            for worker in workers:
                for registration in worker._registered_descs:
                    worker.nixl_wrapper.deregister_memory(
                        registration, backends=["UCX"]
                    )
