"""CPU checks of physical main-KV/scale placement and allocation budgets."""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64

from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PAYLOAD_PAGE_BYTES,
    PREFILL_WORKSPACE_BYTES,
    WORKSPACE_BYTES,
    CacheBudget,
    CompressedKVState,
    WorkspacePool,
    main_kv_format,
    plan_scale_layout,
    prefill_producer_for_config,
    retained_mapper_bytes_for_config,
    workspace_bytes_for_config,
)


def cache_config(fmt, count=3, block_outer=False):
    page = PAYLOAD_PAGE_BYTES
    spec = SimpleNamespace(
        cache_dtype_str=fmt.name,
        dtype=torch.uint8,
        state_content_size_bytes=fmt.payload_bytes(512),
        page_size_bytes=page,
        num_states=fmt.rows_per_page(512),
    )
    names = ("model.layers.2.attn", "model.layers.8.attn", "model.layers.14.attn")
    config = SimpleNamespace(
        num_blocks=count,
        kv_cache_groups=[
            SimpleNamespace(layer_names=list(names[:2]), kv_cache_spec=spec),
            SimpleNamespace(layer_names=[names[2]], kv_cache_spec=spec),
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=2 * page * count,
                layers=list(names[:2]),
                offset=0,
                layer_stride=page if block_outer else page * count,
                block_stride=2 * page if block_outer else page,
            ),
            SimpleNamespace(
                size=2 * page * count,
                layers=[names[2]],
                offset=0,
                layer_stride=page if block_outer else page * count,
                block_stride=2 * page if block_outer else page,
            ),
        ],
    )
    return config, names


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("pages", [2, 4])
def test_promoted_manager_pages_match_payload_kernel_views(fmt, pages):
    """The real allocator may promote pages while retaining codec block width."""
    from vllm.config import CacheConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
        group_kernel_blocks,
    )
    from vllm.v1.worker.utils import allocate_kv_cache

    rows = fmt.rows_per_page(512)
    spec = MLAAttentionSpec(
        block_size=rows * 2 * pages,
        storage_block_size=rows * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=fmt.payload_bytes(512),
        cache_dtype_str=fmt.name,
    )
    cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_groups=[KVCacheGroupSpec(["main"], spec)],
        kv_cache_tensors=[KVCacheTensor(
            size=2 * spec.page_size_bytes,
            layers=["main"],
            offset=0,
            layer_stride=2 * spec.page_size_bytes,
            block_stride=spec.page_size_bytes,
        )],
    )
    context = {"main": SimpleNamespace(compressed_kv_format=fmt)}
    state = CompressedKVState(
        plan_scale_layout(cfg, fmt), fmt, "cpu", context, workspace_bytes=4096
    )
    cache = CacheConfig(block_size=64, cache_dtype="bfloat16")
    cache.kv_cache_layout = "LBNHC"
    try:
        payload = allocate_kv_cache(
            cfg, torch.device("cpu"), cache.get_resolved_kv_cache_layout()
        )["main"]
        scales = state.scales["main"]
        assert payload.shape[:3] == (2 * pages, 1, rows)
        assert scales.shape == (2 * pages, rows, fmt.scale_groups(512))
        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
            MemoryRegistrationPlanner,
        )

        planner = MemoryRegistrationPlanner(cfg)
        registration = planner.describe(
            layer_name="main", tensor=payload.squeeze(1), spec=spec, num_blocks=2
        )
        scale_registration = planner.describe_scale_plane(
            layer_name="main", scales=scales, payload=registration,
            scale_page_bytes=fmt.scale_page_bytes(512),
        )
        assert registration.kernel_blocks == scale_registration.kernel_blocks == pages
        assert scale_registration.payload_segments == (
            (0, pages * fmt.scale_page_bytes(512)),
        )
        assert scale_registration.num_blocks == 2
        physical = group_kernel_blocks(scales, cfg.num_blocks)
        assert physical.shape[:2] == (2, pages)
        for block in range(2 * pages):
            scales[block].fill_(block + 1)
        assert physical[1, -1, 0, 0].item() == 2 * pages
        zeroer = SimpleNamespace(_meta=None)
        state.bind_zeroer(zeroer)
        assert zeroer._meta[1].tolist() == [physical.stride(0) // 2]
        assert zeroer._meta[2].tolist() == [physical[0].numel() // 2]
    finally:
        state.release()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_real_parent_shutdown_clears_context_before_state_unbinds(fmt, monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    cfg, names = cache_config(fmt)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    layers = list(context.values())
    state = CompressedKVState(
        plan_scale_layout(cfg, fmt), fmt, "cpu", context, workspace_bytes=4096
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=3),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    runner.aux_output_connector = None
    worker = IluWorker.__new__(IluWorker)
    worker.model_runner = runner
    worker.profiler = None
    worker.elastic_ep_executor = SimpleNamespace(shutdown=lambda: None)
    worker._compressed_kv_state = state
    worker._try_replace_model_runner_class = lambda: runner.__class__
    # Only accelerator operations/allocator inspection are isolated. The real
    # Worker and V2 runner shutdowns call real free_before_shutdown, which clears
    # static_forward_context before the plugin releases its paired State.
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        __import__(
            IluWorker.__module__, fromlist=["current_platform"]
        ).current_platform,
        "is_cuda_alike",
        lambda: False,
    )
    worker.shutdown()
    assert context == {} and runner.vllm_config.cache_config.num_gpu_blocks is None
    assert worker._compressed_kv_state is None
    assert not state.scales and state.workspace is state.scale_storage is None
    assert not state._forward_context
    assert all(layer.kv_cache_scales is None for layer in layers)
    assert all(layer.compressed_kv_workspace is None for layer in layers)
    state.release()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_partial_state_construction_keeps_bindings_when_external_context_clears(
    fmt, monkeypatch
):
    cfg, names = cache_config(fmt)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    layers = list(context.values())
    original = torch.as_strided
    calls = []
    error = RuntimeError("original scale binding failure")

    def fail_second_view(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            context.clear()
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "as_strided", fail_second_view)
    state = CompressedKVState.__new__(CompressedKVState)
    with pytest.raises(RuntimeError) as raised:
        state.__init__(
            plan_scale_layout(cfg, fmt), fmt, "cpu", context, workspace_bytes=4096
        )
    assert raised.value is error
    assert not state.scales and state.workspace is state.scale_storage is None
    assert not state._forward_context
    assert getattr(layers[0], "kv_cache_scales", None) is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("workspace_bytes", [WORKSPACE_BYTES, PREFILL_WORKSPACE_BYTES])
def test_budget_reserves_sidecars_and_one_workspace(fmt, workspace_bytes):
    page_cost = PAYLOAD_PAGE_BYTES + fmt.scale_page_bytes(512)
    budget = CacheBudget.create(
        workspace_bytes + page_cost * 7 - 1,
        fmt,
        workspace_bytes=workspace_bytes,
    )
    assert budget.workspace_bytes == workspace_bytes
    assert budget.payload_bytes == 6 * PAYLOAD_PAGE_BYTES
    budget.validate(6 * PAYLOAD_PAGE_BYTES, 6 * fmt.scale_page_bytes(512))
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        budget.validate(7 * PAYLOAD_PAGE_BYTES, 7 * fmt.scale_page_bytes(512))


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("kv_producer", 512 * 1024**2),
        ("kv_both", 512 * 1024**2),
        ("kv_consumer", 128 * 1024**2),
        (None, 128 * 1024**2),
    ],
)
def test_workspace_policy_uses_the_configured_pd_role(role, expected):
    cfg = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role=role))
    assert workspace_bytes_for_config(cfg) == expected
    assert prefill_producer_for_config(cfg) is (role in ("kv_producer", "kv_both"))
    assert workspace_bytes_for_config(SimpleNamespace()) == WORKSPACE_BYTES
    assert workspace_bytes_for_config(SimpleNamespace(kv_transfer_config=None)) == (
        WORKSPACE_BYTES
    )


def test_producer_role_does_not_follow_the_workspace_size():
    config, names = cache_config(I8G64)
    layout = plan_scale_layout(config, I8G64)
    context = {name: SimpleNamespace(compressed_kv_format=I8G64) for name in names}
    budget = CacheBudget.create(
        8 * 1024**3, I8G64, workspace_bytes=PREFILL_WORKSPACE_BYTES
    )
    assert not budget.prefill_producer
    state = CompressedKVState(
        layout,
        I8G64,
        "meta",
        context,
        workspace_bytes=budget.arena_bytes,
        prefill_producer=budget.prefill_producer,
    )
    assert not state.workspace.prefill_producer
    assert state.workspace.storage.numel() == PREFILL_WORKSPACE_BYTES
    state.release()


@pytest.mark.parametrize("workspace_bytes", [0, -1, 1.5])
def test_budget_rejects_invalid_workspace_sizes(workspace_bytes):
    with pytest.raises(ValueError, match="positive integer"):
        CacheBudget.create(8 * 1024**3, I8G64, workspace_bytes=workspace_bytes)


@pytest.mark.parametrize(
    "tokens,topk,expected", [(2048, 512, 4202496), (32, 512, 66048), (513, 257, 530432)]
)
@pytest.mark.parametrize("text_config", [False, True])
def test_retained_mapper_credit_uses_resolved_config_and_each_tensor_alignment(
    tokens, topk, expected, text_config
):
    name = "hf_text_config" if text_config else "hf_config"
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
        model_config=SimpleNamespace(**{name: SimpleNamespace(index_topk=topk)}),
    )
    assert retained_mapper_bytes_for_config(cfg) == expected


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
@pytest.mark.parametrize("tokens", [False, True])
def test_retained_mapper_credit_rejects_invalid_bounds(value, tokens):
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=value if tokens else 2048
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=512 if tokens else value)
        ),
    )
    with pytest.raises(ValueError, match="positive integers"):
        retained_mapper_bytes_for_config(cfg)


@pytest.mark.parametrize(
    "credit", [-1, 1.5, True, WORKSPACE_BYTES, WORKSPACE_BYTES + 1]
)
def test_budget_rejects_invalid_or_unaffordable_retained_credit(credit):
    with pytest.raises(ValueError, match="mapper credit"):
        CacheBudget.create(8 * 1024**3, I8G64, retained_mapper_bytes=credit)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("workspace_bytes", [WORKSPACE_BYTES, PREFILL_WORKSPACE_BYTES])
def test_retained_credit_partitions_total_without_changing_payload_budget(
    fmt, workspace_bytes
):
    baseline = CacheBudget.create(8 * 1024**3, fmt, workspace_bytes=workspace_bytes)
    credited = CacheBudget.create(
        8 * 1024**3, fmt, workspace_bytes=workspace_bytes, retained_mapper_bytes=4202496
    )
    assert credited.payload_bytes == baseline.payload_bytes
    assert credited.workspace_bytes == baseline.workspace_bytes
    assert credited.arena_bytes + credited.retained_mapper_bytes == workspace_bytes


@pytest.mark.parametrize(
    "tensor",
    [
        torch.empty(0, dtype=torch.uint8),
        torch.empty(16, dtype=torch.float32),
        torch.empty((4, 4), dtype=torch.uint8),
        torch.empty(32, dtype=torch.uint8)[::2],
    ],
)
def test_workspace_requires_an_owned_contiguous_byte_span(tensor):
    with pytest.raises(ValueError, match="contiguous 1D uint8 span"):
        WorkspacePool(tensor)


def test_metadata_uses_only_the_provided_span_and_retains_alignment():
    storage = torch.full((8192,), 0xA5, dtype=torch.uint8)
    owned = storage[2048:6144]
    pool = WorkspacePool(owned)
    metadata = pool.reserve_metadata(257)
    metadata.zero_()
    assert pool.storage is owned
    assert metadata.numel() == 257
    assert pool.tensor.numel() == 4096 - 512
    assert pool.tensor.data_ptr() - owned.data_ptr() == 512
    with pool.acquire() as lease:
        lease.fill_(7)
    assert torch.all(storage[:2048] == 0xA5)
    assert torch.all(storage[6144:] == 0xA5)
    with pytest.raises(RuntimeError, match="already reserved"):
        pool.reserve_metadata(1)


@pytest.mark.parametrize("byte_count", [-1, 1.5, 3841, 4096, 4097])
def test_metadata_rejects_out_of_span_or_invalid_counts(byte_count):
    pool = WorkspacePool(torch.empty(4096, dtype=torch.uint8))
    with pytest.raises(ValueError, match="metadata"):
        pool.reserve_metadata(byte_count)
    assert pool.tensor is pool.storage
    assert pool.reserve_metadata(256).numel() == 256


def test_zero_length_metadata_still_reserves_only_once():
    pool = WorkspacePool(torch.empty(4096, dtype=torch.uint8))
    assert pool.reserve_metadata(0).numel() == 0
    with pytest.raises(RuntimeError, match="already reserved"):
        pool.reserve_metadata(0)


def test_prefill_metadata_boundary_uses_the_larger_actual_span_on_meta():
    pool = WorkspacePool(
        torch.empty(PREFILL_WORKSPACE_BYTES, dtype=torch.uint8, device="meta")
    )
    assert pool.reserve_metadata(WORKSPACE_BYTES + 1).numel() == WORKSPACE_BYTES + 1
    assert pool.tensor.numel() == PREFILL_WORKSPACE_BYTES - WORKSPACE_BYTES - 256
    with pytest.raises(RuntimeError, match="already reserved"):
        pool.reserve_metadata(1)


@pytest.fixture
def cuda_pool_record_log(monkeypatch):
    """Exercise the real lease branches without initializing CUDA."""
    records, waits = [], []

    class Stream:
        def __init__(self, handle, device=0):
            self.handle = handle
            self.device = torch.device("cuda", device)

        def __eq__(self, other):
            # Deliberately model an equal handle on different devices, so
            # the explicit device check remains independently exercised.
            return isinstance(other, Stream) and self.handle == other.handle

        def wait_event(self, event):
            waits.append((self, event))

    first = Stream(1)
    current = SimpleNamespace(stream=first, capturing=False)
    event = SimpleNamespace(record=records.append)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current.stream)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: current.capturing
    )
    tensor = SimpleNamespace(
        dtype=torch.uint8,
        ndim=1,
        is_cuda=True,
        device=first.device,
        is_contiguous=lambda: True,
        numel=lambda: 4096,
    )
    return WorkspacePool(tensor), current, Stream, records, waits


def test_workspace_same_recorded_stream_uses_fifo_but_records_every_release(
    cuda_pool_record_log,
):
    pool, current, stream_type, records, waits = cuda_pool_record_log
    first = current.stream
    assert pool._last_recorded_stream is first and records == [first]
    for _ in range(2):
        with pool.acquire(), pool.acquire(nested=True):
            pass
    assert not waits and len(records) == 3
    equal_wrapper = stream_type(first.handle)
    assert equal_wrapper is not first
    current.stream = equal_wrapper
    with pool.acquire():
        pass
    assert not waits and pool._last_recorded_stream is equal_wrapper
    assert records[-1] is equal_wrapper
    # Equal numeric stream handles do not justify skipping a device handoff.
    current.stream = stream_type(first.handle, device=1)
    with pool.acquire():
        pass
    assert len(waits) == 1 and waits[0][0] is current.stream
    pool._last_recorded_stream = None
    with pool.acquire():
        pass
    assert len(waits) == 2


def test_workspace_stream_handoff_exception_still_records_completion(
    cuda_pool_record_log,
):
    pool, current, stream_type, records, waits = cuda_pool_record_log
    original = current.stream
    current.stream = stream_type(2)
    with pytest.raises(RuntimeError, match="lease body"), pool.acquire():
        raise RuntimeError("lease body")
    assert pool._last_recorded_stream is current.stream
    assert records[-1] is current.stream and len(waits) == 1
    assert not pool._leased and pool._stream is None and pool._thread is None
    current.stream = original
    with pool.acquire():
        pass
    assert len(waits) == 2 and records[-1] is original


def test_workspace_capture_does_not_publish_an_actual_recorded_stream(
    cuda_pool_record_log,
):
    pool, current, stream_type, records, waits = cuda_pool_record_log
    original = current.stream
    current.stream, current.capturing = stream_type(2), True
    with pool.forward_scope():
        assert pool.capturing
        with pool.read_scope():
            pass
    assert records == [original] and not waits
    assert pool._last_recorded_stream is original and not pool.capturing
    current.capturing = False
    with pool.acquire():
        pass
    assert len(waits) == 1 and records[-1] is current.stream


def test_workspace_nested_stream_guard_is_not_bypassed_by_fifo(
    cuda_pool_record_log,
):
    pool, current, stream_type, records, waits = cuda_pool_record_log
    original = current.stream
    with pool.acquire():
        current.stream = stream_type(2)
        with (
            pytest.raises(RuntimeError, match="changed CUDA streams"),
            pool.acquire(nested=True),
        ):
            pass
        current.stream = original
    assert not waits and records[-1] is original


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("block_outer", [False, True])
def test_overlay_alias_and_independent_owner_views(fmt, block_outer):
    config, names = cache_config(fmt, block_outer=block_outer)
    layout = plan_scale_layout(config, fmt)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    state = CompressedKVState(layout, fmt, "cpu", context)
    try:
        a, b, alias = (state.scales[name] for name in names)
        assert a is alias
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        a[1].fill_(3)
        b[1].fill_(7)
        assert torch.equal(alias[1], a[1])
        assert (a[1] == 3).all() and (b[1] == 7).all()
        assert layout.scale_bytes == 6 * fmt.scale_page_bytes(512)
        with state.workspace.acquire() as lease:
            assert lease.numel() == WORKSPACE_BYTES
            with (
                pytest.raises(RuntimeError, match="active lease"),
                state.workspace.acquire(),
            ):
                pass
    finally:
        state.release()
    assert context[names[0]].kv_cache_scales is None
    assert context[names[0]].compressed_kv_workspace is None


def test_invalid_page_growth_rejects_full_allocation_fallback():
    config, _ = cache_config(I8G64)
    config.kv_cache_groups[0].kv_cache_spec.page_size_bytes *= 2
    with pytest.raises(ValueError, match="page geometry"):
        plan_scale_layout(config, I8G64)


def test_none_preserves_baseline_and_invalid_format_fails(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    assert main_kv_format() is None
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "FP8")
    with pytest.raises(ValueError, match="Expected i8g64 or i4g32"):
        main_kv_format()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("workspace_bytes", [WORKSPACE_BYTES, PREFILL_WORKSPACE_BYTES])
@pytest.mark.parametrize("credit", [0, 4202496])
def test_zeroer_metadata_uses_the_existing_workspace(fmt, workspace_bytes, credit):
    config, names = cache_config(fmt)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    budget = CacheBudget.create(
        8 * 1024**3, fmt, workspace_bytes=workspace_bytes, retained_mapper_bytes=credit
    )
    state = CompressedKVState(
        plan_scale_layout(config, fmt),
        fmt,
        "cpu",
        context,
        workspace_bytes=budget.arena_bytes,
    )
    zeroer = SimpleNamespace(_meta=None)
    try:
        state.bind_zeroer(zeroer)
        addresses, strides, lengths, _, _, count = zeroer._meta
        assert count == 2
        assert addresses.tolist() == [
            state.scales[name].data_ptr() for name in names[:2]
        ]
        assert strides.tolist() == [fmt.scale_page_bytes(512) // 4] * 2
        assert lengths.tolist() == [fmt.scale_page_bytes(512) // 4] * 2
        assert (
            addresses.untyped_storage().data_ptr() == state.workspace.storage.data_ptr()
        )
        assert state.workspace.tensor.numel() + 256 + credit == workspace_bytes
    finally:
        state.release()
