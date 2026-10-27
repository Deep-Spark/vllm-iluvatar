# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for pinned FP8 wire staging."""

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
import torch
import vllm.v1.core.kv_cache_utils as community
from conftest import requires_cuda
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_iluvatar.v1.core.kv_cache_utils as ilu_planning
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl import (
    fp8_staging as staging_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_worker import (
    IluNixlWorkerMixin,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_layout import (
    FP8_WIRE_SCRATCH_BUFFER_COUNT,
    get_fp8_wire_scratch_blocks,
    page_states,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_staging import (
    FP8DSMLAWireStaging,
    allocate_fp8_ds_mla_wire_caches,
    allocate_fp8_ds_mla_wire_scratch_caches,
    build_hma_copy_spans,
    copy_wire_rows_to_host,
    merge_byte_intervals,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
    IluNixlAgentMetadata,
    IluNixlTransferRegion,
)


def _config(**extra):
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.prefix_cache_retention_interval = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = (
        KVCacheLayout.BLHNC
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    transfer = config.kv_transfer_config = MagicMock()
    transfer.kv_role = "kv_producer"
    transfer.kv_connector = "IluNixlConnector"
    transfer.kv_connector_extra_config = {
        "enable_fp8_ds_mla_wire_staging": "True",
        **extra,
    }
    transfer.get_from_extra_config.side_effect = lambda key, default: (
        transfer.kv_connector_extra_config.get(key, default)
    )
    return config


def _single_group():
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        cache_dtype_str="bfloat16",
        alignment=512,
    )
    specs = {"layers.0.main": spec}
    uniform = UniformTypeKVCacheSpecs(
        block_size=spec.block_size,
        kv_cache_specs=specs,
    )
    return spec, specs, [KVCacheGroupSpec(list(specs), uniform)]


@pytest.mark.parametrize(
    ("configured_scratch_blocks", "scratch_blocks"),
    [(None, 4), ("3", 3)],
)
def test_planning_reserves_exactly_two_scratch_buffers_and_keeps_full_wire_ids(
    configured_scratch_blocks,
    scratch_blocks,
):
    extra = (
        {}
        if configured_scratch_blocks is None
        else {"fp8_wire_scratch_blocks": configured_scratch_blocks}
    )
    config = _config(**extra)
    spec, specs, groups = _single_group()
    available_memory = 32 * 1024**2
    patched = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )

    planned = patched(config, groups, available_memory, specs)
    wire = planned.wire_kv_cache_config
    wire_bytes_per_block = wire.kv_cache_tensors[0].block_stride
    expected = community.get_kv_cache_config_from_groups(
        config,
        groups,
        available_memory
        - FP8_WIRE_SCRATCH_BUFFER_COUNT * scratch_blocks * wire_bytes_per_block,
    )

    assert planned.num_blocks == expected.num_blocks
    assert (
        planned.num_blocks
        == (
            available_memory
            - FP8_WIRE_SCRATCH_BUFFER_COUNT * scratch_blocks * wire_bytes_per_block
        )
        // spec.page_size_bytes
    )
    assert wire.num_blocks == planned.num_blocks
    assert all(
        tensor.size == planned.num_blocks * wire_bytes_per_block
        for tensor in wire.kv_cache_tensors
    )


def test_planning_rejects_scratch_larger_than_available_memory():
    config = _config(fp8_wire_scratch_blocks="1000000")
    _, specs, groups = _single_group()
    patched = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )

    with pytest.raises(ValueError, match="scratch does not fit"):
        patched(config, groups, 1024, specs)


@pytest.mark.parametrize("value", [0, -1, True, "1.5"])
def test_scratch_block_validation(value):
    with pytest.raises(ValueError, match="positive integer"):
        get_fp8_wire_scratch_blocks(_config(fp8_wire_scratch_blocks=value))


def test_scratch_blocks_default_to_four():
    assert get_fp8_wire_scratch_blocks(_config()) == 4


def test_allocator_requests_pinned_full_cpu_and_two_gpu_scratch_backings(monkeypatch):
    config = _config()
    _, specs, groups = _single_group()
    planned = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )(config, groups, 16 * 1024**2, specs)
    wire = planned.wire_kv_cache_config
    stride = wire.kv_cache_tensors[0].block_stride
    real_zeros = torch.zeros
    calls = []

    def fake_zeros(size, *, dtype, device, pin_memory=False):
        calls.append((size, torch.device(device), pin_memory))
        return real_zeros(size, dtype=dtype)

    monkeypatch.setattr(staging_module.torch, "zeros", fake_zeros)
    cpu_caches = allocate_fp8_ds_mla_wire_caches(
        wire, torch.device("cpu"), pin_memory=True
    )
    scratch_buffers = allocate_fp8_ds_mla_wire_scratch_caches(
        wire,
        torch.device("cuda"),
        scratch_blocks=4,
    )

    assert calls == [
        (wire.num_blocks * stride, torch.device("cpu"), True),
        (4 * stride, torch.device("cuda"), False),
        (4 * stride, torch.device("cuda"), False),
    ]
    assert next(iter(cpu_caches.values())).shape[0] == wire.num_blocks
    assert [next(iter(caches.values())).shape[0] for caches in scratch_buffers] == [
        4,
        4,
    ]
    assert (
        len(
            {
                next(iter(caches.values())).untyped_storage().data_ptr()
                for caches in scratch_buffers
            }
        )
        == FP8_WIRE_SCRATCH_BUFFER_COUNT
    )


def _span_configs():
    compute_groups = [
        SimpleNamespace(layer_names=["layer.a", "layer.c"]),
        SimpleNamespace(layer_names=["layer.b"]),
    ]
    wire_groups = [
        SimpleNamespace(
            layer_names=["layer.a", "layer.b"],
            kv_cache_spec=SimpleNamespace(page_size_bytes=3),
        ),
        SimpleNamespace(
            layer_names=["layer.c"],
            kv_cache_spec=SimpleNamespace(page_size_bytes=2),
        ),
    ]
    tensors = [
        KVCacheTensor(
            size=16,
            layers=["layer.a", "layer.b"],
            layer_stride=3,
            offset=2,
            block_stride=8,
        ),
        KVCacheTensor(
            size=16,
            layers=["layer.c"],
            layer_stride=2,
            offset=5,
            block_stride=8,
        ),
    ]
    return (
        SimpleNamespace(kv_cache_groups=compute_groups),
        SimpleNamespace(kv_cache_groups=wire_groups, kv_cache_tensors=tensors),
    )


def test_hma_spans_merge_and_include_shared_wire_slot_for_each_group():
    compute_config, wire_config = _span_configs()

    assert merge_byte_intervals([(5, 7), (2, 5), (3, 4)]) == ((2, 7),)
    assert build_hma_copy_spans(compute_config, wire_config) == (
        ((2, 7),),
        ((2, 5),),
    )


def test_hma_spans_require_every_compute_layer_to_be_covered():
    compute_config, wire_config = _span_configs()
    wire_config.kv_cache_tensors[1].layers = ["other.layer"]

    with pytest.raises(ValueError, match="layer.c"):
        build_hma_copy_spans(compute_config, wire_config)


def test_hma_span_copy_updates_only_selected_regions_and_preserves_poison():
    scratch = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.uint8,
    )
    source_ids = [4, 1]
    spans = ((1, 3), (5, 7))
    poison = 0xEE
    span_copy = torch.full((6, 8), poison, dtype=torch.uint8)

    copy_wire_rows_to_host(scratch, span_copy, source_ids, spans)

    for dense_id, source_id in enumerate(source_ids):
        for start, end in spans:
            assert torch.equal(
                span_copy[source_id, start:end],
                scratch[dense_id, start:end],
            )
        assert span_copy[source_id, [0, 3, 4, 7]].tolist() == [poison] * 4
    assert span_copy[[0, 2, 3, 5]].eq(poison).all()


def test_hma_span_copy_rejects_empty_effective_spans():
    with pytest.raises(ValueError, match="no effective HMA byte spans"):
        copy_wire_rows_to_host(
            torch.zeros((1, 8), dtype=torch.uint8),
            torch.zeros((1, 8), dtype=torch.uint8),
            [0],
            (),
        )


def test_export_skips_empty_hma_span_groups():
    empty_group = SimpleNamespace(kv_cache_spec=SimpleNamespace(), layer_names=[])
    swa_group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=64, sliding_window=128),
        layer_names=["layer.swa"],
    )
    staging = object.__new__(FP8DSMLAWireStaging)
    staging.compute_config = SimpleNamespace(kv_cache_groups=[empty_group, swa_group])
    staging.compute_caches = {"layer": torch.empty(1)}
    staging.scratch_blocks = 4
    staging.hma_copy_spans = ((), ((0, 4),))
    calls = []
    staging._enqueue_group_chunk = lambda group_idx, *args, **kwargs: calls.append(
        group_idx
    )

    staging.export([[3, 5], [7]], 128)

    assert calls == [1]


def test_export_alternates_ab_scratch_and_orders_events_across_requests(
    monkeypatch,
):
    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            events.append(f"wait:{self.name}:{event.name}")

        def synchronize(self):
            events.append(f"sync:{self.name}")

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            events.append(f"record:{self.name}:{stream.name}")

    staging = object.__new__(FP8DSMLAWireStaging)
    staging.compute_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=64, sliding_window=128),
                layer_names=["layer"],
            )
        ]
    )
    staging.compute_caches = {"layer": torch.empty(1)}
    staging.scratch_blocks = 2
    staging.scratch_row_buffers = [
        torch.zeros((2, 4), dtype=torch.uint8),
        torch.zeros((2, 4), dtype=torch.uint8),
    ]
    staging.scratch_backings = staging.scratch_row_buffers
    staging.wire_rows = torch.zeros((16, 4), dtype=torch.uint8)
    staging.scratch_wire_cache_buffers = [
        {"layer": staging.scratch_row_buffers[0]},
        {"layer": staging.scratch_row_buffers[1]},
    ]
    staging.quant_done_events = [FakeEvent("quant-a"), FakeEvent("quant-b")]
    staging.copy_done_events = [FakeEvent("copy-a"), FakeEvent("copy-b")]
    staging.copy_stream = FakeStream("copy")
    staging.hma_copy_spans = (((0, 4),),)
    staging._next_scratch_chunk = 0
    calls = []

    def stage_chunk(
        group_idx,
        source_ids,
        destination_ids,
        destination_caches,
        num_tokens,
        total_block_count,
        chunk_offset,
    ):
        buffer_name = (
            "a" if destination_caches is staging.scratch_wire_cache_buffers[0] else "b"
        )
        events.append(f"quant:{buffer_name}")
        calls.append(
            (
                group_idx,
                source_ids.tolist(),
                destination_ids.tolist(),
                total_block_count,
                chunk_offset,
                buffer_name,
            )
        )
        for source_id, destination_id in zip(
            source_ids.tolist(), destination_ids.tolist()
        ):
            destination_caches["layer"][destination_id].fill_(source_id)

    staging._stage_group_chunk = stage_chunk
    quant_stream = FakeStream("quant")
    monkeypatch.setattr(
        staging_module.torch.cuda,
        "current_stream",
        lambda _device: quant_stream,
    )
    monkeypatch.setattr(
        staging_module.torch.cuda,
        "stream",
        lambda _stream: nullcontext(),
    )
    real_copy = staging_module.copy_wire_rows_to_host

    def tracked_copy(scratch_rows, *args, **kwargs):
        buffer_name = "a" if scratch_rows is staging.scratch_row_buffers[0] else "b"
        events.append(f"copy:{buffer_name}")
        real_copy(scratch_rows, *args, **kwargs)

    monkeypatch.setattr(staging_module, "copy_wire_rows_to_host", tracked_copy)
    staging.export_requests(
        [
            ([[7, 3, 9]], 768),
            ([[2, 8]], 512),
        ]
    )

    assert calls == [
        (0, [7, 3], [0, 1], 3, 0, "a"),
        (0, [9], [0], 3, 2, "b"),
        (0, [2, 8], [0, 1], 2, 0, "a"),
    ]
    assert events == [
        "quant:a",
        "record:quant-a:quant",
        "wait:copy:quant-a",
        "copy:a",
        "record:copy-a:copy",
        "quant:b",
        "record:quant-b:quant",
        "wait:copy:quant-b",
        "copy:b",
        "record:copy-b:copy",
        "wait:quant:copy-a",
        "quant:a",
        "record:quant-a:quant",
        "wait:copy:quant-a",
        "copy:a",
        "record:copy-a:copy",
        "sync:copy",
    ]
    for block_id in [7, 3, 9, 2, 8]:
        assert staging.wire_rows[block_id].tolist() == [block_id] * 4


def test_logical_valid_entries_include_chunk_offset():
    spec, _, _ = _single_group()

    first = FP8DSMLAWireStaging._valid_entries_for_chunk(
        spec, 700, 3, 0, 2, torch.device("cpu")
    )
    last = FP8DSMLAWireStaging._valid_entries_for_chunk(
        spec, 700, 3, 2, 1, torch.device("cpu")
    )

    assert first.tolist() == [64, 64]
    assert last.tolist() == [47]


def _indexer_specs(source_dtype, source_head_size):
    # DeepSeek-V4 indexer specs leave the v0.31 storage_block_size at None.
    source = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=source_head_size,
        dtype=source_dtype,
        tokens_per_state=4,
    )
    wire = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        alignment=576,
    )
    return source, wire


def test_page_states_reads_rows_with_or_without_storage_override():
    unset = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    # GLM-5.3 kpool sets storage_block_size as a token width.
    glm_like = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        storage_block_size=128,
    )
    swa = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
    )

    assert unset.storage_block_size is None
    assert page_states(unset) == 64
    assert page_states(glm_like) == 32
    assert not hasattr(swa, "storage_block_size")
    assert page_states(swa) == 64


def test_quantize_pages_accepts_sliding_window_spec(monkeypatch):
    source_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    source = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    destination = torch.full((2, 64 * 584), 0xEE, dtype=torch.uint8)
    calls = {}

    def fake_quantize(rows, dst, slot_mapping, *, block_size, **kwargs):
        calls["rows"] = rows.shape[0]
        calls["slots"] = slot_mapping.tolist()
        calls["block_size"] = block_size

    monkeypatch.setattr(staging_module, "quantize_and_insert_k_cache", fake_quantize)
    staging = object.__new__(FP8DSMLAWireStaging)
    staging._quantize_pages(
        source,
        destination,
        torch.tensor([1]),
        torch.tensor([0]),
        source_spec,
        num_tokens=40,
        total_block_count=1,
        chunk_offset=0,
    )

    assert calls["block_size"] == 64
    assert calls["rows"] == 40
    assert calls["slots"] == list(range(40))
    assert destination[0].eq(0).all()
    assert destination[1].eq(0xEE).all()


def test_int8_indexer_staging_transcodes_only_valid_rows(monkeypatch):
    source_spec, wire_spec = _indexer_specs(torch.uint8, 132)
    source = torch.zeros(
        (2, page_states(source_spec), 132),
        dtype=torch.uint8,
    )
    destination = torch.full(
        (2, wire_spec.page_size_bytes),
        0xEE,
        dtype=torch.uint8,
    )
    calls = {}

    class FakeKernel:
        def __getitem__(self, grid):
            calls["grid"] = grid
            return self

        def __call__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        staging_module,
        "_transcode_int8_indexer_wire_kernel",
        FakeKernel(),
    )
    staging = object.__new__(FP8DSMLAWireStaging)

    staging._stage_indexer_pages(
        source,
        destination,
        torch.tensor([1]),
        torch.tensor([0]),
        source_spec,
        wire_spec,
        num_tokens=188,
        total_block_count=1,
        chunk_offset=0,
    )

    assert calls["grid"] == (page_states(source_spec),)
    assert calls["args"][4].tolist() == [47]
    assert calls["args"][5:8] == (
        source.stride(0),
        destination.stride(0),
        page_states(source_spec),
    )
    assert calls["kwargs"] == {"HEAD_SIZE": 128}
    assert destination[0].eq(0).all()
    assert destination[1].eq(0xEE).all()


@requires_cuda
def test_int8_indexer_transcode_matches_torch_reference_for_partial_page(
    cuda_device,
):
    source_spec, wire_spec = _indexer_specs(torch.uint8, 132)
    tokens_per_page = page_states(source_spec)
    values = torch.linspace(
        -3.0,
        3.0,
        tokens_per_page * 128,
        dtype=torch.float32,
        device=cuda_device,
    ).reshape(tokens_per_page, 128)
    int8_scale = values.abs().amax(dim=1).clamp(min=1e-10) / 127.0
    int8_values = (
        (values / int8_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    )
    source = torch.zeros(
        (1, tokens_per_page, 132),
        dtype=torch.uint8,
        device=cuda_device,
    )
    source_flat = source.view(-1)
    source_flat[: tokens_per_page * 128].copy_(
        int8_values.contiguous().view(torch.uint8).reshape(-1)
    )
    source_flat[tokens_per_page * 128 : source_spec.real_page_size_bytes].copy_(
        int8_scale.contiguous().view(torch.uint8)
    )
    destination = torch.full(
        (1, wire_spec.page_size_bytes),
        0xEE,
        dtype=torch.uint8,
        device=cuda_device,
    )
    staging = object.__new__(FP8DSMLAWireStaging)

    staging._stage_indexer_pages(
        source,
        destination,
        torch.tensor([0], dtype=torch.int64, device=cuda_device),
        torch.tensor([0], dtype=torch.int64, device=cuda_device),
        source_spec,
        wire_spec,
        num_tokens=188,
        total_block_count=1,
        chunk_offset=0,
    )

    valid_rows = 47
    dequantized = int8_values.float() * int8_scale[:, None]
    amax = dequantized.abs().amax(dim=1).clamp(min=1e-4)
    wire_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    expected_fp8 = (
        (dequantized / wire_scale[:, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    destination_flat = destination.view(-1)
    actual_fp8 = destination_flat[: tokens_per_page * 128].reshape(
        tokens_per_page, 128
    )
    scale_start = tokens_per_page * 128
    actual_scale = destination_flat[
        scale_start : scale_start + tokens_per_page * 4
    ].view(torch.float32)

    assert torch.equal(actual_fp8[:valid_rows], expected_fp8[:valid_rows])
    assert torch.equal(actual_scale[:valid_rows], wire_scale[:valid_rows])
    assert actual_fp8[valid_rows:].eq(0).all()
    assert actual_scale[valid_rows:].eq(0).all()
    assert destination_flat[source_spec.real_page_size_bytes :].eq(0).all()


def test_bf16_indexer_staging_keeps_quantization_path(monkeypatch):
    source_spec, wire_spec = _indexer_specs(torch.bfloat16, 128)
    source = torch.zeros(
        (1, page_states(source_spec), 128),
        dtype=torch.bfloat16,
    )
    destination = torch.empty(
        (1, wire_spec.page_size_bytes),
        dtype=torch.uint8,
    )
    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            calls.append(("grid", grid))
            return self

        def __call__(self, *args, **kwargs):
            calls.append(("launch", args, kwargs))

    monkeypatch.setattr(staging_module, "_quantize_indexer_wire_kernel", FakeKernel())
    staging = object.__new__(FP8DSMLAWireStaging)
    staging._stage_indexer_pages(
        source,
        destination,
        torch.tensor([0]),
        torch.tensor([0]),
        source_spec,
        wire_spec,
        num_tokens=256,
        total_block_count=1,
        chunk_offset=0,
    )

    assert calls[0] == ("grid", (page_states(source_spec),))
    assert calls[1][0] == "launch"


def _agent_metadata(memory_type: str | None = None) -> IluNixlAgentMetadata:
    kwargs = {} if memory_type is None else {"memory_type": memory_type}
    return IluNixlAgentMetadata(
        engine_id="producer",
        agent_metadata=b"agent",
        kv_caches_base_addr=[1000],
        device_id=0,
        num_blocks=2,
        block_lens=[64],
        block_strides=[128],
        kv_cache_layout="NHD",
        block_size=256,
        ssm_sizes=(0, 0),
        attn_backend_name="MLA",
        physical_blocks_per_logical_kv_block=1,
        transfer_regions=[
            IluNixlTransferRegion(
                region_id=0,
                block_len=64,
                shared_by=["layers.0.main"],
                layer_indices=[0],
                group_ids=[0],
                cache_kinds=["main_mla"],
                identities=["0:main_mla:group0"],
                block_stride=128,
            )
        ],
        **kwargs,
    )


def test_metadata_defaults_to_vram_and_remote_descriptor_uses_dram():
    old_payload = asdict(_agent_metadata())
    old_payload.pop("memory_type")
    decoded = msgspec.msgpack.Decoder(IluNixlAgentMetadata).decode(
        msgspec.msgpack.encode(old_payload)
    )
    assert decoded.memory_type == "VRAM"
    remote = _agent_metadata("DRAM")
    worker = object.__new__(IluNixlWorkerMixin)
    worker.nixl_memory_type = "VRAM"
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_xfer_descs.return_value = "remote-descs"
    worker.nixl_wrapper.prep_xfer_dlist.return_value = "remote-handle"
    worker._remote_transfer_regions = lambda *_args: [(0, 64, 0)]

    worker._register_pp_remote_xfer_handle(
        "producer",
        0,
        "remote-agent",
        remote,
        [0],
        [0],
    )

    blocks_data, memory_type = worker.nixl_wrapper.get_xfer_descs.call_args.args
    assert blocks_data == [(1000, 64, 0), (1128, 64, 0)]
    assert memory_type == "DRAM"
    assert worker.nixl_memory_type == "VRAM"
