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
