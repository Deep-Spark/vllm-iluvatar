"""Check sharded lookup results, padding, and TP-before-DP reduction order."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import engram


def _table_values(ids, dim=3):
    valid = (ids >= 0) & (ids < 97)
    columns = torch.arange(dim, dtype=torch.float32)
    values = (ids.clamp(0, 96).float()[..., None] * dim + columns) / 32
    return values.to(torch.bfloat16) * valid[..., None]


@pytest.mark.parametrize("counts", [(0, 0), (1, 6), (0, 3), (12, 1, 6, 0), (129, 0, 3, 1)])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_lookup_reduces_only_consumed_rows(monkeypatch, counts, tp_size):
    world = len(counts)
    slot = max(counts)
    requests = []
    for rank, count in enumerate(counts):
        ids = (torch.arange(count * 2).reshape(count, 2) + rank * 7) % 101
        if count:
            ids[0, 0] = -1
        requests.append(ids)
    gathered = torch.cat([
        torch.cat([ids, torch.full((slot - len(ids), 2), -1)])
        for ids in requests
    ])
    expected_global = _table_values(gathered)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: tp_size)
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
    monkeypatch.setattr(engram, "get_forward_context", lambda: SimpleNamespace(dp_metadata=metadata))
    # Use contiguous global table shards, including a partially filled last shard.
    part_size = (97 + world * tp_size - 1) // (world * tp_size)
    shard = gathered.clamp_min(0) // part_size
    for rank, ids in enumerate(requests):
        calls = []
        partial = expected_global * (shard == rank)[..., None]

        def all_gather(padded, dim):
            calls.append("gather")
            assert dim == 0
            assert torch.equal(padded[:len(ids)], ids)
            assert torch.all(padded[len(ids):] == -1)
            return gathered.clone()

        def tp_reduce(values):
            calls.append("TP")
            assert torch.equal(values, partial)
            return expected_global * ((shard % world) == rank)[..., None]

        def reduce_scatter(values, dim):
            calls.append("DP")
            assert dim == 0
            mask = (shard % world) == rank if tp_size > 1 else shard == rank
            assert torch.equal(values, expected_global * mask[..., None])
            return expected_global.chunk(world, dim=0)[rank].clone()

        monkeypatch.setattr(engram, "tensor_model_parallel_all_reduce", tp_reduce)
        layer = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
        torch.nn.Module.__init__(layer)
        layer.weight = layer.scale = torch.empty(0)
        layer._dp_group = SimpleNamespace(world_size=world, rank_in_group=rank,
                                         all_gather=all_gather, reduce_scatter=reduce_scatter)
        layer._lookup = lambda received: partial.clone()
        saved = ids.clone()
        result = layer(ids)
        assert torch.equal(result, _table_values(ids))
        assert torch.equal(ids, saved)
        assert calls == (["gather", "TP", "DP"] if tp_size > 1 else ["gather", "DP"])


def test_dp_one_keeps_direct_lookup(monkeypatch):
    layer = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.weight = layer.scale = torch.empty(0)
    layer._dp_group = None
    layer._lookup = _table_values
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 1)
    ids = torch.tensor([[5, -1], [0, 100]])
    assert torch.equal(layer(ids), _table_values(ids))


def test_local_rows_cannot_exceed_coordinated_slot(monkeypatch):
    layer = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.weight = layer.scale = torch.empty(0)
    layer._dp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(engram, "get_forward_context", lambda: SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([2, 3]))))
    with pytest.raises(ValueError, match="exceeds the DP token slot"):
        layer(torch.zeros(4, 2, dtype=torch.int64))
