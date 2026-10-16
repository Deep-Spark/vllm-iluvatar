"""Public plan types, complete-range routing and actual old/new entrypoints."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek._quantized_sparse_mla_affine import (
    LinearRestoreGeometry,
)
from ixformer.inference.functions.deepseek.quantized_sparse_mla import (
    QuantizedSparseMlaAffineView,
    QuantizedSparseMlaRestorePlan,
)

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
    CompressedFamilySelection,
    RestorePlanTracker,
)


def selection(*, old=6, new=128, topk=512, capacity=None, mode=0):
    shape = (old + new, 1, topk)
    capacity = new * topk if capacity is None else capacity
    rows = torch.empty(capacity, 512, dtype=torch.bfloat16, device="meta")
    row_map = torch.empty(shape, dtype=torch.int32, device="meta")
    plan = QuantizedSparseMlaRestorePlan(
        rows,
        row_map,
        torch.empty(768, dtype=torch.uint8, device="meta"),
        None,
        old,
        new,
        mode,
        130015232,
        "owner20",
        24,
        # Test-only geometry fixture, matching the SDK's CPU geometry tests.
        # The plugin never reads this field or derives coordinates.
        _linear_geometry=LinearRestoreGeometry(
            shape, shape, (0, 0, 0), (1, 1, 1), old, new, capacity, 512
        ),
    )
    return CompressedFamilySelection(
        row_map[:, 0],
        torch.empty(old + new, dtype=torch.int32, device="meta"),
        plan,
        768,
    )


@pytest.mark.parametrize(
    "old,new,topk", [(0, 64, 512), (6, 128, 512), (6, 129, 512), (18, 128, 257)]
)
def test_complete_range_new_view_uses_public_affine_proof(old, new, topk):
    active = selection(old=old, new=new, topk=topk)
    original = active.plan.for_queries(slice(old, old + new)).as_affine()
    view = active.for_queries(slice(old, old + new))[2]
    assert isinstance(view, QuantizedSparseMlaAffineView)
    assert view.selection_shape == original.selection_shape == (new, 1, topk)
    assert (view.origin, view.batch_rows, view.query_rows, view.key_rows) == (
        original.origin,
        original.batch_rows,
        original.query_rows,
        original.key_rows,
    )
    assert view.rows is active.plan.rows and view.scratch is active.plan.scratch
    assert view.owner == active.plan.owner


@pytest.mark.parametrize("capacity", [0, 1, 128 * 512 - 1])
def test_partial_entire_range_does_not_convert_a_complete_subview(capacity):
    active = selection(capacity=capacity)
    assert isinstance(active.for_queries(slice(6, 7))[2], QuantizedSparseMlaRestorePlan)


@pytest.mark.parametrize("mode,plane", [(0, "selected_flags"), (1, None), (2, None)])
def test_flagged_or_non_linear_plans_never_convert_with_full_capacity(mode, plane):
    active = selection(mode=mode)
    if plane:
        active = replace(
            active,
            plan=replace(active.plan, **{plane: torch.empty(1, dtype=torch.int32)}),
        )
    assert isinstance(
        active.for_queries(slice(6, 134))[2], QuantizedSparseMlaRestorePlan
    )


@pytest.mark.parametrize(
    "token_slice", [slice(0, 6), slice(0, 7), slice(6, 6), slice(6, 134, 2)]
)
def test_old_crossing_empty_and_strided_views_keep_the_original_plan(token_slice):
    active = selection()
    assert isinstance(active.for_queries(token_slice)[2], QuantizedSparseMlaRestorePlan)


def test_negative_slice_keeps_public_rejection():
    with pytest.raises(ValueError, match="positive|greater than zero"):
        selection().for_queries(slice(None, None, -1))


def test_eligible_malformed_factory_geometry_error_is_not_swallowed():
    active = selection()
    active = replace(active, plan=replace(active.plan, restore_start=5))
    with pytest.raises(ValueError, match="geometry"):
        active.for_queries(slice(6, 133))


def test_absent_plan_keeps_map_only_pure_decode():
    active = replace(selection(), plan=None)
    assert active.for_queries(slice(0, 6))[2] is None


def test_two_view_cache_is_per_preparation_not_a_freshness_shortcut(monkeypatch):
    calls = []
    original = QuantizedSparseMlaRestorePlan.as_affine

    def checked(plan):
        calls.append(plan)
        return original(plan)

    monkeypatch.setattr(QuantizedSparseMlaRestorePlan, "as_affine", checked)
    tracker, family = RestorePlanTracker(), object()
    for invalidation in (tracker.begin_forward, tracker.main_cache_written):
        invalidation()
        assert tracker.lookup(family) is None
        active = selection()
        tracker.publish(tracker.begin_prepare(family), active)
        first = active.for_queries(slice(6, 134))
        active.for_queries(slice(0, 6))
        assert active.for_queries(slice(6, 134)) is first
        assert len(active._query_views) == 2
    assert len(calls) == 2 and calls[0] is not calls[1]


@pytest.mark.parametrize("new", [128, 129])
def test_actual_prefill_and_old_decode_entrypoints_use_affine_and_scratch(new):
    active = selection(new=new)
    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = 1
    calls = []
    layer._sparse_mla = lambda *args, **kwargs: calls.append(kwargs)
    metadata = SimpleNamespace(
        num_decode_tokens=6,
        num_prefill_tokens=new,
        decode_swa_indices=None,
        decode_swa_lens=None,
        prefill_swa_indices=None,
        prefill_swa_lens=None,
    )
    q = torch.empty(6 + new, 2, 512, dtype=torch.bfloat16, device="meta")
    layer._forward_prefill_as_decode(
        q[6:], None, None, q[6:], metadata, None, selection=active
    )
    layer._forward_decode(q[:6], None, None, q[:6], metadata, None, selection=active)
    assert isinstance(calls[0]["restore_plan"], QuantizedSparseMlaAffineView)
    assert calls[1]["restore_plan"] is active.plan.scratch
    assert calls[0]["prefill"]
