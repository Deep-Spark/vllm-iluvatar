"""Full-history, lookup-only scoping and unsupported-model regressions."""

from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
    override_forward_context,
)

from vllm_iluvatar.models.deepseek_v41 import engram


class Lookup(torch.nn.Module):
    def __init__(self, world=4):
        super().__init__()
        self._dp_group = SimpleNamespace(world_size=world) if world else None
        self.calls = []

    def forward(self, ids):
        slot = (
            engram.engram_dp_token_slot(self._dp_group)
            if self._dp_group is not None
            else len(ids)
        )
        assert len(ids) <= slot
        self.calls.append((len(ids), slot))
        return ids[..., None].float() + torch.arange(3)


def make_layer(world=4):
    layer = engram.IluEngram.__new__(engram.IluEngram)
    torch.nn.Module.__init__(layer)
    layer.embed = Lookup(world)
    layer._prefetch_stream = None
    layer._prefetch_done = None
    layer._rows = None
    layer._token_mask = None
    layer.layer_hash_index = 0
    return layer


class Model:
    def __init__(self, rows, *, suffix=False, supported=True):
        self.first, self.second = make_layer(), make_layer()
        self.first.supports_prefix_lookup = supported
        self.layers = [
            SimpleNamespace(engram=self.first),
            SimpleNamespace(engram=None),
            SimpleNamespace(engram=self.second),
            SimpleNamespace(engram=make_layer() if suffix else None),
        ]
        self.end_layer = len(self.layers)
        self.history = torch.full((rows + 2,), -1, dtype=torch.int64)
        self.calls = 0

    def _prepare_engram(self, ids, positions, lookback):
        self.calls += 1
        self.history[: len(ids)].copy_(ids)
        hashes = torch.stack((ids + positions, ids * 3 + positions), dim=1)
        keep = ids != 255
        for layer in self.layers:
            if layer.engram is not None:
                layer.engram.prepare_embeddings(hashes)
                layer.engram._token_mask = keep


@pytest.mark.parametrize("healthy", [0, 6, 12, 24])
@pytest.mark.parametrize("tail", [128, 129])
def test_lookup_subset_preserves_complete_history_and_context(healthy, tail):
    count = healthy + tail
    ids = torch.arange(count) % 256
    ids[-1] = 255
    positions = torch.arange(count) + 1_000_000 - count
    full, compact = Model(count), Model(count)
    slot = max(12, healthy)
    marker = object()
    original = SimpleNamespace(
        dp_metadata=DPMetadata(torch.full((4,), count, dtype=torch.int32)),
        additional_kwargs={"other": marker},
    )
    with override_forward_context(original):
        full._prepare_engram(ids, positions, None)
        assert engram.prepare_prefix_engram(
            compact, ids, positions, None, healthy, slot, 3
        )
        assert get_forward_context() is original
        assert original.additional_kwargs == {"other": marker}
    assert full.calls == compact.calls == 1
    assert torch.equal(full.history[:count], ids)
    assert torch.equal(compact.history[:count], ids)
    for layer in (compact.first, compact.second):
        assert layer.embed.calls == [(healthy, slot)]
        mask = layer._token_mask
        assert mask is not None and mask.numel() == healthy


def test_suffix_or_unsupported_falls_back():
    count = 8
    ids = torch.arange(count)
    positions = torch.arange(count)
    original = SimpleNamespace(
        dp_metadata=DPMetadata(torch.full((4,), count, dtype=torch.int32)),
        additional_kwargs={},
    )
    with override_forward_context(original):
        suffix = Model(count, suffix=True)
        assert not engram.prepare_prefix_engram(suffix, ids, positions, None, 2, 4, 3)
        unsupported = Model(count, supported=False)
        assert not engram.prepare_prefix_engram(
            unsupported, ids, positions, None, 2, 4, 3
        )
    assert suffix.first.embed.calls[0][0] == count
    assert unsupported.first.embed.calls[0][0] == count
