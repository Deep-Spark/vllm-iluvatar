"""Full-history, lookup-only scoping and unsupported-model regressions."""

from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import DPMetadata, get_forward_context, override_forward_context

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
    return layer


class Model:
    def __init__(self, rows, *, suffix=False, supported=True):
        self.first, self.second = make_layer(), make_layer()
        self.first.supports_prefix_lookup = supported
        self.layers = [
            SimpleNamespace(engram=self.first), SimpleNamespace(engram=None),
            SimpleNamespace(engram=self.second),
            SimpleNamespace(engram=make_layer() if suffix else None),
        ]
        self.end_layer = len(self.layers)
        self.history = torch.full((rows + 2,), -1, dtype=torch.int64)
        self.calls = 0

    def _prepare_engram(self, ids, positions, lookback):
        self.calls += 1
        self.history[:len(ids)].copy_(ids)
        hashes = torch.stack((ids + positions, ids * 3 + positions), dim=1)
        keep = ids != 255
        for layer in self.layers:
            if layer.engram is not None:
                layer.engram.prepare_embeddings(hashes, keep)


@pytest.mark.parametrize("healthy", [0, 6, 12, 24])
@pytest.mark.parametrize("tail", [128, 129])
def test_lookup_subset_preserves_complete_history_and_context(healthy, tail):
    count = healthy + tail
    ids = torch.arange(count) % 256
    ids[-1] = 255  # Keep-mask handling remains correct for image dead tokens.
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
    assert torch.equal(full.history, compact.history)
    assert torch.all(compact.history[-2:] == -1)
    for first, second in ((full.first, compact.first), (full.second, compact.second)):
        assert torch.equal(second._rows, first._rows[:healthy])
        assert torch.equal(second._hash_ids, first._hash_ids[:healthy])
        assert torch.equal(second._token_mask, first._token_mask[:healthy])
        assert second.embed.calls == [(healthy, slot)]
    assert torch.equal(original.dp_metadata.num_tokens_across_dp_cpu, torch.full((4,), count))


@pytest.mark.parametrize("suffix,supported", [(True, True), (False, False)])
def test_unsupported_geometry_keeps_full_lookup(suffix, supported):
    model = Model(135, suffix=suffix, supported=supported)
    original = SimpleNamespace(
        dp_metadata=DPMetadata(torch.full((4,), 135, dtype=torch.int32)),
        additional_kwargs={},
    )
    with override_forward_context(original):
        assert not engram.prepare_prefix_engram(
            model, torch.arange(135), torch.arange(135), None, 6, 12, 3
        )
    assert model.first.embed.calls == [(135, 135)]


def test_no_healthy_rows_still_writes_history_without_lookup():
    model = Model(129)
    context = SimpleNamespace(additional_kwargs={})
    with override_forward_context(context):
        assert engram.prepare_prefix_engram(
            model, torch.arange(129), torch.arange(129), None, 0, 0, 3
        )
    assert model.calls == 1
    assert torch.equal(model.history[:129], torch.arange(129))
    assert model.first._rows is None
    assert not model.first.embed.calls


@pytest.mark.parametrize("local,slot", [(-1, 12), (13, 12), (0, -1)])
def test_invalid_lookup_plan_is_rejected(local, slot):
    with pytest.raises(ValueError):
        engram.PrefixEngramLookup(local, slot)


def test_regular_preparation_does_not_require_forward_context():
    layer = make_layer(world=0)
    ids = torch.tensor([[1, 2], [3, 4]])
    with override_forward_context(None):
        layer.prepare_embeddings(ids)
    assert layer.embed.calls == [(2, 2)]
