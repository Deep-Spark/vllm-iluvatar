"""Actual CPU family routing with explicit native-operation substitutes."""

import sys
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from ixformer.inference.functions.deepseek.quantized_sparse_mla import (
    QuantizedSparseMlaRestoreCapacityError,
)
from vllm.forward_context import override_forward_context

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    WORKSPACE_BYTES,
    WorkspacePool,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
    CompressedFamilySelection,
    RestorePlanTracker,
    compact_owner_capacity,
    family_key,
)
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model


def key_inputs(fmt=I8G64):
    rows, width, groups = (
        fmt.rows_per_page(512),
        fmt.payload_bytes(512),
        fmt.scale_groups(512),
    )
    data = torch.empty(6 * rows * width, dtype=torch.uint8)
    scales = torch.empty(6 * rows * groups, dtype=torch.float16)
    q = torch.empty(4, 32, 512, dtype=torch.bfloat16)
    return dict(
        owner="model.layers.20.attn",
        index_source_layer_id=20,
        payload=data[: 3 * rows * width].view(3, rows, width),
        scales=scales[: 3 * rows * groups].view(3, rows, groups),
        workspace=torch.empty(8192, dtype=torch.uint8),
        query=q,
        output=torch.empty_like(q),
        indices=torch.empty(4, 5, dtype=torch.int32),
        format=fmt,
        restore_start=1,
        restore_tokens=3,
    )


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_exact_aliases_share_keys_and_offsets_and_strides_do_not(fmt):
    values = key_inputs(fmt)
    first = family_key(**values)
    assert (
        family_key(**dict(values, payload=values["payload"].view_as(values["payload"])))
        == first
    )
    for name in ("payload", "scales"):
        tensor = values[name]
        offset = tensor.as_strided(tensor.shape, tensor.stride(), tensor.numel())
        pitched = tensor.as_strided(
            tensor.shape, (2 * tensor.stride(0), *tensor.stride()[1:])
        )
        assert family_key(**dict(values, **{name: offset})) != first
        assert family_key(**dict(values, **{name: pitched})) != first


def test_owner_index_family_codec_and_workspace_span_are_not_aliased():
    values = key_inputs()
    first = family_key(**values)
    assert family_key(**dict(values, owner="pp.forwarder.5")) != first
    for family in (24, 28, 32, 36):
        assert family_key(**dict(values, index_source_layer_id=family)) != first
    assert family_key(**dict(values, format=I4G32)) != first
    yielded = family_key(**dict(values, workspace=values["workspace"][256:]))
    assert yielded != first
    assert yielded.workspace.storage_offset == 256
    assert yielded.workspace.layout.shape == (8192 - 256,)


def test_query_addresses_and_contents_do_not_control_freshness():
    values = key_inputs()
    key = family_key(**values)
    assert family_key(**dict(values, query=torch.ones_like(values["query"]))) == key
    tracker = RestorePlanTracker()
    tracker.begin_forward()
    selection = object()
    tracker.publish(tracker.begin_prepare(key), selection)
    assert tracker.lookup(key) is selection
    tracker.begin_forward()
    assert tracker.lookup(key) is None
    tracker.publish(tracker.begin_prepare(key), selection)
    tracker.main_cache_written()
    assert tracker.lookup(key) is None and tracker.main_write_epoch == 1


def test_failed_or_stale_preparation_never_exposes_old_selection():
    key = family_key(**key_inputs())
    tracker = RestorePlanTracker()
    old = tracker.begin_prepare(key)
    new = tracker.begin_prepare(replace(key, index_source_layer_id=24))
    with pytest.raises(RuntimeError, match="became stale"):
        tracker.publish(old, object())
    tracker.main_cache_written()
    with pytest.raises(RuntimeError, match="became stale"):
        tracker.publish(new, object())
    assert tracker.lookup(key) is None


def test_key_and_tracker_do_not_allocate_or_read_tensor_values(monkeypatch):
    values = key_inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("Host family key/tracker attempted tensor allocation or value read")

    for name in ("empty", "zeros", "ones", "full", "tensor"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    key = family_key(**values)
    tracker = RestorePlanTracker()
    tracker.publish(tracker.begin_prepare(key), object())
    assert tracker.lookup(key) is not None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_family_key_retains_the_strict_packed_token_contract(fmt):
    values = key_inputs(fmt)
    values["query"] = torch.empty(6, 32, 512, dtype=torch.bfloat16)
    values["output"] = torch.empty_like(values["query"])
    with pytest.raises(ValueError, match="full packed BF16"):
        family_key(**values)


class FakePlan:
    def __init__(self, workspace, indices, scratch_bytes, restore_start):
        aligned = (scratch_bytes + 255) // 256 * 256
        self.scratch = workspace[:aligned]
        self.rows = (
            workspace[aligned : aligned + 1024].view(torch.bfloat16).view(1, 512)
        )
        self.row_map = (
            workspace[aligned + 1024 : aligned + 1024 + indices.numel() * 4]
            .view(torch.int32)
            .view(indices.shape)
        )
        self.row_map.fill_(-1)
        self.restore_start = restore_start
        self.restore_tokens = indices.shape[0] * indices.shape[1] - restore_start
        self.mode = 0
        self.selected_flags = None
        self.budget_bytes = workspace.numel()
        self.query_view_calls = 0

    def for_queries(self, token_slice):
        self.query_view_calls += 1
        return SimpleNamespace(
            rows=self.rows,
            row_map=self.row_map[token_slice],
            scratch=self.scratch,
            source=self,
        )


def suffix(
    fmt,
    monkeypatch,
    *,
    old=2,
    new=4,
    factory_error=None,
    forwarder=False,
    prefill_producer=False,
):
    tokens = old + new
    pool = WorkspacePool(
        torch.empty(WORKSPACE_BYTES, dtype=torch.uint8),
        prefill_producer=prefill_producer,
    )
    pool.reserve_metadata(2048)
    owner = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(owner)
    owner.compressed_kv_format = fmt
    owner.kv_cache = torch.zeros(
        3, fmt.rows_per_page(512), fmt.payload_bytes(512), dtype=torch.uint8
    )
    owner.kv_cache_scales = torch.zeros(
        3, fmt.rows_per_page(512), fmt.scale_groups(512), dtype=torch.float16
    )
    prefix = "pp.forwarder.5" if forwarder else "owner20"
    context = {prefix: owner}
    topk = torch.zeros(tokens, 8, dtype=torch.int32)
    positions = torch.arange(tokens)
    metadata = {
        prefix: SimpleNamespace(
            block_size=fmt.rows_per_page(512),
            block_table=torch.ones(1, 1, dtype=torch.int32),
        )
    }
    calls = SimpleNamespace(maps=[], queries=[], factories=[], prepares=[], readers=[])

    def mapper(topk, reqs, table, page_rows, valid, positions, ratio, limit, walk):
        calls.maps.append(topk.shape[0])
        return topk.clone(), torch.full(
            (topk.shape[0],), topk.shape[1], dtype=torch.int32
        )

    def size(q, swa, si, data, scales, ci, *, format, output):
        calls.queries.append(q.shape[0])
        return 513 if q.shape[0] == old else 0

    def factory(workspace, indices, dim, **kwargs):
        calls.factories.append((workspace, indices, kwargs))
        if factory_error is not None:
            raise factory_error("fixture factory failure")
        return FakePlan(
            workspace, indices, kwargs["scratch_bytes"], kwargs["restore_start"]
        )

    def prepare(data, scales, indices, lengths, plan, *, format):
        calls.prepares.append(plan)
        assert plan.row_map.shape == indices.shape
        assert (plan.row_map[:old] == -1).all()

    def reader(*args, **kwargs):
        calls.readers.append((args, kwargs))
        plan = kwargs["workspace"]
        if not isinstance(plan, torch.Tensor):
            shape = (
                plan.row_map.shape if hasattr(plan, "row_map") else plan.selection_shape
            )
            assert shape == args[6].shape
        kwargs["output"].fill_(args[0].shape[0])

    module = ModuleType("ixformer.inference.functions.deepseek.quantized_sparse_mla")
    module.QuantizedSparseMlaRestoreCapacityError = (
        QuantizedSparseMlaRestoreCapacityError
    )
    module.quantized_sparse_mla_scratch_bytes = size
    module.make_quantized_sparse_mla_restore_plan = factory
    module.prepare_quantized_sparse_mla_restore_plan = prepare
    module.quantized_sparse_mla = reader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    prefill = ModuleType(
        "ixformer.inference.functions.deepseek.quantized_sparse_mla_prefill"
    )
    prefill.quantized_sparse_mla_prefill = reader
    monkeypatch.setitem(sys.modules, prefill.__name__, prefill)
    monkeypatch.setattr(torch.ops._C, "v41_global_topk_indices_and_lens", mapper)
    layers = []
    for layer_id in range(21, 40):
        layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
        torch.nn.Module.__init__(layer)
        layer.compressed_kv_format, layer.compress_ratio = fmt, 1
        layer.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), dim=-1)
        )
        layer.is_kv_source = layer.is_kv_forwarder = False
        layer.compressed_cache_prefix, layer._static_forward_context = prefix, context
        layer.compressed_kv_workspace = pool
        layer.index_source_layer_id = max(
            f for f in (20, 24, 28, 32, 36) if f <= layer_id
        )
        layer.topk_indices_buffer = topk
        swa_prefix = f"swa{layer_id}"
        layer.swa_cache_layer = SimpleNamespace(
            prefix=swa_prefix, kv_cache=torch.empty(3, 64, 512, dtype=torch.bfloat16)
        )
        layer.scale, layer.attn_sink = 512**-0.5, torch.zeros(2)
        metadata[swa_prefix] = SimpleNamespace(
            num_decodes=bool(old),
            num_prefills=bool(new),
            num_decode_tokens=old,
            num_prefill_tokens=new,
            decode_swa_indices=torch.zeros(old, 1, 3, dtype=torch.int32),
            decode_swa_lens=torch.full((old,), 3, dtype=torch.int32),
            prefill_swa_indices=torch.zeros(new, 1, 3, dtype=torch.int32),
            prefill_swa_lens=torch.full((new,), 3, dtype=torch.int32),
            is_valid_token=torch.ones(tokens, dtype=torch.bool),
            token_to_req_indices=torch.zeros(tokens, dtype=torch.int32),
        )
        layers.append(layer)
    # Match actual V2 metadata aliases; the main mapping is independent of
    # per-layer SWA storage, whose data pointer must not split this family.
    for md in (metadata[layer.swa_cache_layer.prefix] for layer in layers[1:]):
        md.is_valid_token = metadata["swa21"].is_valid_token
        md.token_to_req_indices = metadata["swa21"].token_to_req_indices
    q = torch.zeros(tokens, 2, 512, dtype=torch.bfloat16)
    outputs = [torch.empty_like(q) for _ in layers]

    def run(*, write_layer=None, capture=False):
        with (
            override_forward_context(SimpleNamespace(attn_metadata=metadata)),
            pool.forward_scope(),
        ):
            # CPU routing tests substitute the scope's capture result. Real
            # capture/replay, including changed GPU metadata, is tested below
            # in the explicitly leased GPU suite.
            if capture:
                pool.capturing = True
            for layer_id, layer, output in zip(range(21, 40), layers, outputs):
                if layer_id == write_layer:
                    pool.main_cache_written()
                layer.forward_mqa(q, torch.empty(0), positions, output)

    return pool, layers, q, positions, outputs, calls, run


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("forwarder", [False, True])
def test_pure_decode_reuses_maps_only_in_capture_without_restore(
    fmt, forwarder, monkeypatch
):
    from vllm_iluvatar.models.deepseek_v41 import attention as module

    pool, layers, q, positions, outputs, calls, run = suffix(
        fmt, monkeypatch, old=12, new=0, forwarder=forwarder
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Eager pure decode must not construct a family key")

    with monkeypatch.context() as eager:
        eager.setattr(module, "family_key", forbidden)
        run()
    assert calls.maps == [12] * 19
    assert len(calls.readers) == 19
    run(capture=True)
    assert calls.maps == [12] * 24
    run(capture=True, write_layer=22)
    assert calls.maps == [12] * 30
    assert len(calls.readers) == 57
    assert calls.queries == calls.factories == calls.prepares == []
    assert all(kwargs["workspace"] is pool.tensor for _, kwargs in calls.readers)
    assert not pool.capturing and pool.restore._selection is None
    assert pool.restore.main_write_epoch == 1


def test_capture_status_reuses_lease_and_resets_after_nested_failure(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
        RestorePlanTracker,
    )

    pool = SimpleNamespace(
        tensor=SimpleNamespace(is_cuda=True),
        capturing=False,
        _lease_capturing=True,
        _forward_depth=0,
        restore=RestorePlanTracker(),
        acquire=lambda **kwargs: nullcontext(torch.empty(0, dtype=torch.uint8)),
    )

    def forbidden():
        pytest.fail("Forward capture state must reuse the entering lease query")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
    with (
        pytest.raises(ValueError, match="nested failure"),
        WorkspacePool.forward_scope(pool),
    ):
        assert pool.capturing and pool._forward_depth == 1
        with WorkspacePool.forward_scope(pool):
            assert pool.capturing and pool._forward_depth == 2
            raise ValueError("nested failure")
    assert not pool.capturing and pool._forward_depth == 0
    assert pool.restore._selection is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("forwarder", [False, True])
def test_actual_suffix_prepares_five_families_and_retains_mixed_plan(
    fmt, forwarder, monkeypatch
):
    pool, layers, q, positions, outputs, calls, run = suffix(
        fmt, monkeypatch, forwarder=forwarder
    )
    run()
    assert calls.maps == [6] * 5
    assert calls.queries == [4, 2] * 5
    assert len(calls.factories) == len(calls.prepares) == 5
    assert [kwargs["index_source_layer_id"] for _, _, kwargs in calls.factories] == [
        20,
        24,
        28,
        32,
        36,
    ]
    assert len(calls.readers) == 38
    assert [args[0].shape[0] for args, _ in calls.readers] == [4, 2] * 19
    for workspace, _, kwargs in calls.factories:
        assert workspace is pool.tensor and workspace.numel() == WORKSPACE_BYTES - 2048
        assert kwargs["scratch_bytes"] == 513 and kwargs["restore_start"] == 2
        assert kwargs["compressed_capacity"] == 3 * fmt.rows_per_page(512)
        assert kwargs["owner"] == ("pp.forwarder.5" if forwarder else "owner20")
    for output in outputs:
        assert (output[:2] == 2).all() and (output[2:] == 4).all()
    for _, kwargs in calls.readers[1::2]:
        scratch = kwargs["workspace"]
        assert isinstance(scratch, torch.Tensor)
        assert scratch.numel() == 768 < pool.tensor.numel()
    assert sum(plan.query_view_calls for plan in calls.prepares) == 10
    for start, consumers in ((0, 3), (3, 4), (7, 4), (11, 4), (15, 4)):
        for route in (0, 1):
            views = [
                calls.readers[2 * layer + route][1]["workspace"]
                for layer in range(start, start + consumers)
            ]
            assert all(view is views[0] for view in views)
    assert pool.restore._selection is None
    run()
    assert len(calls.prepares) == 10  # Pointer/layout equality never skips a new step.
    assert calls.readers[0][1]["workspace"] is not calls.readers[38][1]["workspace"]


def test_query_view_cache_is_bounded_and_aliases_prepared_map():
    indices = torch.arange(24, dtype=torch.int32).view(4, 1, 6)
    lengths = torch.full((4,), 6, dtype=torch.int32)
    plan = FakePlan(torch.empty(8192, dtype=torch.uint8), indices, 256, 1)
    selection = CompressedFamilySelection(indices, lengths, plan, 256)
    old, new = selection.for_queries(slice(0, 1)), selection.for_queries(slice(1, 4))
    assert selection.for_queries(slice(0, 1)) is old
    assert selection.for_queries(slice(1, 4)) is new
    plan.row_map[2].fill_(7)
    assert (new[2].row_map[1] == 7).all()
    for stop in (2, 3, 4):
        selection.for_queries(slice(0, stop))
    assert len(selection._query_views) == 2
    another = CompressedFamilySelection(indices, lengths, plan, 256)
    assert another.for_queries(slice(1, 4)) is not new


@pytest.mark.parametrize("error", [QuantizedSparseMlaRestoreCapacityError, ValueError])
def test_capacity_only_fallback_keeps_all_readers_and_bad_layout_propagates(
    error, monkeypatch
):
    pool, layers, q, positions, outputs, calls, run = suffix(
        I8G64, monkeypatch, factory_error=error
    )
    if error is ValueError:
        with pytest.raises(ValueError, match="fixture factory failure"):
            run()
        assert calls.readers == [] and pool.restore._selection is None
    else:
        run()
        assert (
            len(calls.maps) == 5 and calls.prepares == [] and len(calls.readers) == 38
        )
        assert all(kwargs["workspace"] is pool.tensor for _, kwargs in calls.readers)


def test_main_write_invalidates_family_without_swa_write_invalidation(monkeypatch):
    pool, layers, q, positions, outputs, calls, run = suffix(I8G64, monkeypatch)
    run(write_layer=22)
    assert len(calls.prepares) == 6 and pool.restore.main_write_epoch == 1
    assert len(calls.readers) == 38


def test_model_and_imported_suffix_scopes_reset_reuse_under_one_outer_lease():
    pool = WorkspacePool(torch.empty(WORKSPACE_BYTES, dtype=torch.uint8))
    model = IluDeepseekV41Model.__new__(IluDeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = 0, 1
    model.layers = [SimpleNamespace(attn=SimpleNamespace(compressed_kv_workspace=pool))]
    calls = []

    def body(*args):
        calls.append((pool._leased, pool._forward_depth, pool.restore.step_epoch))
        return args[0]

    model._forward_full = body
    with pool.acquire():
        assert model(torch.empty(0), torch.empty(0), None).numel() == 0
        # Mirrors the explicit E25 scope after prefix reconstruction.
        with model.compressed_kv_forward_scope():
            assert pool.restore.step_epoch == 2 and pool._forward_depth == 1
        assert pool._leased and pool._forward_depth == 0
    assert calls == [(True, 1, 1)]
    assert not pool._leased and pool.restore._selection is None


def test_none_model_does_not_enter_a_compressed_scope():
    model = IluDeepseekV41Model.__new__(IluDeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = 0, 1
    model.layers = [SimpleNamespace(attn=SimpleNamespace(compressed_kv_workspace=None))]
    expected = object()
    model._forward_full = lambda *args: expected
    assert model(None, None, None) is expected


class CompactCapacityError(ValueError):
    pass


def compact_sdk(
    monkeypatch, *, factory_error=None, refinement_error=None, prepare_error=None
):
    """Substitute only SDK/native work; real mapper routing/tracker stays active."""
    calls = SimpleNamespace(factories=[], queries=[], prepares=[])

    class CompactPlan:
        def __init__(self, workspace, indices, scratch_bytes):
            aligned = (scratch_bytes + 255) // 256 * 256
            self.scratch = workspace[:aligned]
            end = aligned + indices.numel() * 4
            self.row_map = workspace[aligned:end].view(torch.int32).view(indices.shape)
            self.rows = workspace[end : end + 1024].view(torch.bfloat16).view(1, 512)
            self.view_calls = 0

        def view(self):
            return SimpleNamespace(
                rows=self.rows, row_map=self.row_map, scratch=self.scratch, source=self
            )

        def for_queries(self, token_slice):
            self.view_calls += 1
            return SimpleNamespace(
                rows=self.rows,
                row_map=self.row_map[token_slice],
                scratch=self.scratch,
                source=self,
            )

    def factory(workspace, indices, dim, **kwargs):
        calls.factories.append(kwargs)
        if factory_error is not None:
            raise factory_error("compact factory failure")
        if (
            refinement_error is not None
            and kwargs["scratch_bytes"] > kwargs["prepare_scratch_bytes"]
        ):
            raise refinement_error("compact scratch refinement failure")
        assert kwargs["owner_kind"] == "hash" and dim == 512
        return CompactPlan(
            workspace,
            indices,
            max(kwargs["prepare_scratch_bytes"], kwargs["scratch_bytes"]),
        )

    def size(*args, **kwargs):
        calls.queries.append(kwargs["workspace"])
        return 1297

    def prepare(data, scales, indices, lengths, plan, *, format):
        if prepare_error is not None:
            raise prepare_error("compact prepare failure")
        calls.prepares.append(plan)
        assert plan.row_map.shape == indices.shape
        assert lengths.shape[0] == indices.shape[0]

    module = ModuleType(
        "ixformer.inference.functions.deepseek.quantized_sparse_mla_compact"
    )
    module.QuantizedSparseMlaCompactCapacityError = CompactCapacityError
    module.make_quantized_sparse_mla_compact_plan = factory
    module.quantized_sparse_mla_compact_prepare_scratch_bytes = (
        lambda *args, **kwargs: (256)
    )
    module.quantized_sparse_mla_compact_scratch_bytes = size
    module.prepare_quantized_sparse_mla_compact_plan = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("forwarder", [False, True])
def test_producer_partial_prefill_prepares_and_reuses_compact_once_per_family(
    fmt, forwarder, monkeypatch
):
    pool, _, _, _, _, original, run = suffix(
        fmt, monkeypatch, old=0, forwarder=forwarder, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch)
    run()
    assert original.maps == [4] * 5
    assert not original.prepares
    assert len(compact.prepares) == len(compact.queries) == 5
    assert (
        len(compact.factories) == 10
    )  # Exact reader scratch refines aliases, not data.
    assert all(plan.scratch.numel() == 1536 for plan in compact.prepares)
    assert all(plan.view_calls == 1 for plan in compact.prepares)
    for args in compact.factories:
        assert args["compressed_capacity"] == 3 * fmt.rows_per_page(512)
        assert args["owner_capacity"] == 64
    assert len(original.readers) == 19
    assert all(
        kwargs["workspace"].source in compact.prepares for _, kwargs in original.readers
    )
    assert pool.restore._selection is None
    run()
    assert len(compact.prepares) == 10 and original.maps == [4] * 10
    run(write_layer=22)
    assert len(compact.prepares) == 16 and original.maps == [4] * 16


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize(
    "old,producer,capture", [(0, False, False), (6, True, False), (6, True, True)]
)
def test_decode_standalone_and_mixed_do_not_prepare_compact(
    fmt, old, producer, capture, monkeypatch
):
    _, _, _, _, _, original, run = suffix(
        fmt, monkeypatch, old=old, new=4, prefill_producer=producer
    )
    compact = compact_sdk(monkeypatch)
    run(capture=capture)
    assert len(original.prepares) == 5
    assert compact.factories == compact.queries == compact.prepares == []


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("error", [CompactCapacityError, ValueError])
def test_compact_capacity_only_falls_back_and_malformed_layout_propagates(
    fmt, error, monkeypatch
):
    pool, _, _, _, _, original, run = suffix(
        fmt, monkeypatch, old=0, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch, factory_error=error)
    if error is CompactCapacityError:
        run()
        assert len(original.prepares) == 5 and len(original.readers) == 19
        assert len(compact.factories) == 5
    else:
        with pytest.raises(ValueError, match="compact factory"):
            run()
        assert not original.prepares and not original.readers
    assert not compact.prepares and pool.restore._selection is None


def test_compact_prepare_error_does_not_publish_or_read_partial_metadata(monkeypatch):
    pool, _, _, _, _, original, run = suffix(
        I8G64, monkeypatch, old=0, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch, prepare_error=RuntimeError)
    with pytest.raises(RuntimeError, match="compact prepare"):
        run()
    assert len(compact.factories) == 2
    assert not original.prepares and not original.readers
    assert pool.restore._selection is pool.restore._pending is None


def test_compact_final_scratch_capacity_failure_restores_the_original_route(
    monkeypatch,
):
    _, _, _, _, _, original, run = suffix(
        I8G64, monkeypatch, old=0, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch, refinement_error=CompactCapacityError)
    run()
    assert len(compact.factories) == 10 and len(compact.queries) == 5
    assert not compact.prepares
    assert len(original.prepares) == 5 and len(original.readers) == 19


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_producer_complete_linear_keeps_the_public_affine_route(fmt, monkeypatch):
    from ixformer.inference.functions.deepseek._quantized_sparse_mla_affine import (
        LinearRestoreGeometry,
    )
    from ixformer.inference.functions.deepseek.quantized_sparse_mla import (
        QuantizedSparseMlaAffineView,
        QuantizedSparseMlaRestorePlan,
    )

    _, _, _, _, _, original, run = suffix(
        fmt, monkeypatch, old=0, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch)
    module = sys.modules["ixformer.inference.functions.deepseek.quantized_sparse_mla"]

    def factory(workspace, indices, dim, **kwargs):
        shape = tuple(indices.shape)
        map_end = (indices.numel() * 4 + 255) // 256 * 256
        row_map = workspace[: indices.numel() * 4].view(torch.int32).view(shape)
        rows = (
            workspace[map_end : map_end + indices.numel() * dim * 2]
            .view(torch.bfloat16)
            .view(indices.numel(), dim)
        )
        return QuantizedSparseMlaRestorePlan(
            rows,
            row_map,
            workspace[:0],
            None,
            0,
            shape[0],
            0,
            workspace.numel(),
            "owner20",
            20,
            _linear_geometry=LinearRestoreGeometry(
                shape, shape, (0, 0, 0), (1, 1, 1), 0, shape[0], indices.numel(), dim
            ),
        )

    module.make_quantized_sparse_mla_restore_plan = factory
    module.prepare_quantized_sparse_mla_restore_plan = (
        lambda data, scales, indices, lengths, plan, **kwargs: original.prepares.append(
            plan
        )
    )
    run()
    assert len(original.prepares) == 5
    assert compact.factories == compact.queries == compact.prepares == []
    assert all(
        isinstance(kwargs["workspace"], QuantizedSparseMlaAffineView)
        for _, kwargs in original.readers
    )


@pytest.mark.parametrize("plane", ["selected_flags", "mode"])
def test_producer_preserves_existing_owner_table_plans(plane, monkeypatch):
    _, _, _, _, _, original, run = suffix(
        I8G64, monkeypatch, old=0, prefill_producer=True
    )
    compact = compact_sdk(monkeypatch)
    module = sys.modules["ixformer.inference.functions.deepseek.quantized_sparse_mla"]
    original_factory = module.make_quantized_sparse_mla_restore_plan

    def factory(*args, **kwargs):
        plan = original_factory(*args, **kwargs)
        setattr(
            plan, plane, 1 if plane == "mode" else torch.empty(1, dtype=torch.int32)
        )
        return plan

    module.make_quantized_sparse_mla_restore_plan = factory
    run()
    assert len(original.prepares) == 5 and not compact.factories


@pytest.mark.parametrize(
    "entries,domain,expected",
    [
        (0, 1, None),
        (1, 0, None),
        (32, 100, 64),
        (100, 17, 64),
        (2048 * 512, 15251968, 2097152),
        (2048 * 512, 27961856, 2097152),
        (512, (1 << 32) + 1, 1024),
        ((1 << 31) - 1, 1 << 32, 1 << 32),
        (1 << 31, 1 << 32, None),
    ],
)
def test_compact_owner_bound_uses_full_geometry_without_reading_indices(
    entries, domain, expected
):
    assert compact_owner_capacity(entries, domain) == expected


@pytest.mark.parametrize("bad", [-1, True, 1.5, 1 << 64])
def test_compact_owner_bound_rejects_invalid_or_overflowed_geometry(bad):
    with pytest.raises(ValueError, match="nonnegative integers"):
        compact_owner_capacity(bad, 1024)
    with pytest.raises(ValueError, match="nonnegative integers"):
        compact_owner_capacity(1024, bad)


def test_selection_owns_only_one_plan_and_compact_never_enters_old_decode():
    indices = torch.empty(4, 8, dtype=torch.int32, device="meta")
    lengths = torch.empty(4, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="one prepared plan"):
        CompressedFamilySelection(indices, lengths, object(), 0, compact_plan=object())
    selection = CompressedFamilySelection(
        indices, lengths, None, 0, compact_plan=object()
    )
    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    with pytest.raises(ValueError, match="pure-prefill"):
        layer._forward_decode(None, None, None, None, None, None, selection=selection)
