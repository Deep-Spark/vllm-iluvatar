# SPDX-License-Identifier: Apache-2.0
"""Producer family reuse must preserve distinct consumers and step freshness."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
    CompressedFamilySelection,
    RestorePlanTracker,
    consumer_selection,
    prefill_validity_family,
    union_validity,
)


@pytest.fixture
def family():
    reqs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    topk = torch.zeros((4, 3), dtype=torch.int32)
    valid = [
        torch.tensor(v, dtype=torch.bool)
        for v in ([1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0])
    ]
    peers = [
        NS(
            compressed_cache_prefix="owner",
            index_source_layer_id=2,
            swa_cache_layer=NS(prefix=f"swa{i}"),
            topk_indices_buffer=topk,
        )
        for i in range(3)
    ]
    metadata = {
        f"swa{i}": NS(
            num_decodes=0,
            num_prefills=1,
            num_prefill_tokens=4,
            is_valid_token=v,
            token_to_req_indices=reqs,
        )
        for i, v in enumerate(valid)
    }
    layer = peers[0]
    layer.compressed_kv_workspace = NS(prefill_producer=True)
    layer._static_forward_context = {f"layer{i}": p for i, p in enumerate(peers)}
    return layer, peers, metadata


def test_distinct_masks_preserve_exact_consumer_lengths_and_plan(family):
    layer, _, metadata = family
    masks = prefill_validity_family(layer, metadata, metadata["swa0"], 4)
    union = union_validity(masks)
    assert union.tolist() == [True, True, False, True]
    lengths = torch.tensor([3, 2, 1, 3])
    shared = CompressedFamilySelection(
        layer.topk_indices_buffer,
        torch.where(union, lengths, 0),
        None,
        0,
        compact_plan=object(),
    )
    original = shared.lengths.clone()
    for mask in masks:
        selected = consumer_selection(shared, mask)
        assert torch.equal(selected.lengths, torch.where(mask, lengths, 0))
        assert selected.indices is shared.indices
        assert selected.compact_plan is shared.compact_plan
    assert torch.equal(shared.lengths, original)


@pytest.mark.parametrize(
    "mismatch",
    [
        "request_map",
        "topk",
        "missing",
        "short_mask",
        "no_mask",
        "mixed",
        "token_count",
        "consumer",
        "non_dict",
        "single_peer",
    ],
)
def test_incompatible_family_falls_back(family, mismatch):
    layer, peers, metadata = family
    if mismatch == "request_map":
        metadata["swa1"].token_to_req_indices = metadata[
            "swa1"
        ].token_to_req_indices.clone()
    elif mismatch == "topk":
        peers[1].topk_indices_buffer = peers[1].topk_indices_buffer.clone()
    elif mismatch == "missing":
        del metadata["swa1"]
    elif mismatch == "short_mask":
        metadata["swa1"].is_valid_token = metadata["swa1"].is_valid_token[:2]
    elif mismatch == "no_mask":
        metadata["swa1"].is_valid_token = None
    elif mismatch == "mixed":
        metadata["swa0"].num_decodes = 1
    elif mismatch == "token_count":
        metadata["swa1"].num_prefill_tokens = 3
    elif mismatch == "consumer":
        layer.compressed_kv_workspace.prefill_producer = False
    elif mismatch == "non_dict":
        assert not prefill_validity_family(layer, None, metadata["swa0"], 4)
        return
    else:
        peers[1].compressed_cache_prefix = "different_owner"
        peers[2].index_source_layer_id = 3
    assert not prefill_validity_family(layer, metadata, metadata["swa0"], 4)


def test_cached_peers_read_current_forward_masks(family):
    layer, _, metadata = family
    first = prefill_validity_family(layer, metadata, metadata["swa0"], 4)
    assert union_validity(first).any()
    for md in metadata.values():
        md.is_valid_token = torch.zeros(4, dtype=torch.bool)
    current = prefill_validity_family(layer, metadata, metadata["swa0"], 4)
    assert not union_validity(current).any()
    assert current[0] is not first[0]


def test_union_selection_keeps_forward_and_write_invalidation(family):
    layer, _, metadata = family
    masks = prefill_validity_family(layer, metadata, metadata["swa0"], 4)
    selection = CompressedFamilySelection(
        layer.topk_indices_buffer, union_validity(masks).int(), None, 0
    )
    tracker = RestorePlanTracker()
    tracker.begin_forward()
    tracker.publish(tracker.begin_prepare(("family",)), selection)
    assert tracker.lookup(("family",)) is selection
    tracker.main_cache_written()
    assert tracker.lookup(("family",)) is None
    tracker.publish(tracker.begin_prepare(("family",)), selection)
    tracker.begin_forward()
    assert tracker.lookup(("family",)) is None
