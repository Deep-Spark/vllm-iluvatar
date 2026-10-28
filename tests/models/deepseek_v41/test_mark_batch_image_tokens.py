from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model


def _stub() -> IluDeepseekV41Model:
    model = IluDeepseekV41Model.__new__(IluDeepseekV41Model)
    model.image_dead_token_id = 99
    return model


def test_mark_image_tokens_skips_item_sync_during_capture(monkeypatch) -> None:
    model = _stub()
    ctx = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_iluvatar.models.deepseek_v41.model.get_forward_context",
        lambda: ctx,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ids = MagicMock()
    ids.numel.return_value = 4
    ids.__eq__.side_effect = AssertionError(".item() path must not run in capture")

    model._mark_batch_image_tokens(ids)

    assert ctx.ilu_v41_has_image_tokens is False


def test_mark_image_tokens_eager_sets_flag(monkeypatch) -> None:
    model = _stub()
    ctx = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_iluvatar.models.deepseek_v41.model.get_forward_context",
        lambda: ctx,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ids = torch.tensor([1, 99, 2])

    model._mark_batch_image_tokens(ids)

    assert ctx.ilu_v41_has_image_tokens is True


def test_mark_image_tokens_without_forward_context_does_not_raise(
    monkeypatch,
) -> None:
    model = _stub()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    model._mark_batch_image_tokens(torch.tensor([1, 99, 2]))


def test_prepare_engram_filters_image_tokens_during_capture(monkeypatch) -> None:
    """Capture skips the .item() flag; engram keep still drops image ids."""
    model = _stub()
    ctx = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_iluvatar.models.deepseek_v41.model.get_forward_context",
        lambda: ctx,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    hashes = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    hash_state = MagicMock()
    hash_state.ensure_cache.return_value = True
    hash_state.return_value = hashes
    model.engram_hash = hash_state

    md = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2]),
        is_valid_token=torch.tensor([True, True, True]),
    )
    monkeypatch.setattr(
        "vllm_iluvatar.models.deepseek_v41.model._window_metadata",
        lambda: md,
    )
    engram = SimpleNamespace(
        layer_hash_index=0,
        prepare_embeddings=MagicMock(),
        _token_mask=None,
    )
    model._engram_layers = lambda: [engram]

    ids = torch.tensor([1, 99, 2])
    positions = torch.arange(3)
    model._prepare_engram(ids, positions)

    assert ctx.ilu_v41_has_image_tokens is False
    assert engram._token_mask.tolist() == [True, False, True]
    engram.prepare_embeddings.assert_called_once()
