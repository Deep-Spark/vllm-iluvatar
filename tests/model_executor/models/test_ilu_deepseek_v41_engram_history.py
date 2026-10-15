# SPDX-License-Identifier: Apache-2.0
"""Engram page geometry and runner-provided cross-chunk history."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import model
from vllm_iluvatar.models.deepseek_v41.engram import NgramHashState


@pytest.mark.parametrize("spec_size,bound_size", [(16, 64), (16, 16), (64, 128)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_history_cache_uses_bound_page_geometry(spec_size, bound_size, wrapped):
    kv = torch.empty(4, bound_size, 1, 8, dtype=torch.bfloat16)
    cache = SimpleNamespace(block_size=spec_size, kv_cache=[kv] if wrapped else kv)
    state = NgramHashState(
        SimpleNamespace(max_ngram_size=4, primes=(((101,), (103,), (107,)),)),
        cache,
        list(range(256)),
        torch.tensor([[13, 17, 19, 23]]),
        2,
        "cpu",
    )
    assert state.ensure_cache()
    assert state.block_size == bound_size
    assert state._nslots == 4 * bound_size
    assert state._cache.numel() == 4 * bound_size + 1
    # Rebinding larger physical pages must resize the history allocation too.
    cache.kv_cache = torch.empty(4, bound_size * 2, 1, 8)
    assert state.ensure_cache()
    assert state._nslots == 8 * bound_size
    assert state._cache.numel() == 8 * bound_size + 1


def test_prepare_engram_receives_original_history(monkeypatch):
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(2),
        query_start_loc=torch.tensor([0, 2]),
        block_table=torch.tensor([[0]]),
        token_to_req_indices=torch.tensor([0, 0]),
    )
    monkeypatch.setattr(model, "_window_metadata", lambda: metadata)
    monkeypatch.setattr(model, "_num_reqs", lambda _: 1)
    hasher = MagicMock(return_value=torch.tensor([[[7]], [[9]]]))
    hasher.ensure_cache.return_value = True
    hasher.lookback_depth = 3
    engram = SimpleNamespace(layer_hash_index=0, prepare_embeddings=MagicMock())
    instance = SimpleNamespace(
        engram_hash=hasher,
        _engram_layers=lambda: [engram],
        # Text-only checkpoint: no image span to mask out.
        image_dead_token_id=None,
    )
    history = torch.tensor([[11, 13, 17]], dtype=torch.int32)
    model.IluDeepseekV41Model._prepare_engram(
        instance, torch.tensor([19, 23]), torch.tensor([3, 4]), history
    )
    assert hasher.call_args.args[4] is history
    torch.testing.assert_close(
        engram.prepare_embeddings.call_args.args[0], torch.tensor([[7], [9]])
    )
    with pytest.raises(ValueError, match="runner-provided"):
        model.IluDeepseekV41Model._prepare_engram(
            instance, torch.tensor([19, 23]), torch.tensor([3, 4])
        )


@pytest.mark.parametrize("layers,expected", [([], 0), ([1], 3)])
def test_model_requests_the_engram_lookback_depth(layers, expected):
    instance = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=layers, engram_max_ngram_size=4)
    )
    assert (
        model.IluDeepseekV41ForCausalLM.token_lookback_depth.fget(instance) == expected
    )
