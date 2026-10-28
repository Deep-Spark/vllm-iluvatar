"""PP hops must carry token ids so later stages can route MoE bias_vl."""

from types import SimpleNamespace

import torch

from vllm_iluvatar.models.deepseek_v41 import model as model_mod
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model


def _model() -> IluDeepseekV41Model:
    model = object.__new__(IluDeepseekV41Model)
    model.hc_mult = 4
    model.config = SimpleNamespace(hidden_size=8)
    model.pp_plan = SimpleNamespace(
        recvs_kv=False,
        recvs_topk=False,
        recvs_candidates=False,
    )
    return model


def test_first_rank_does_not_reserve_input_ids(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    tensors = _model().make_empty_intermediate_tensors(
        5, torch.bfloat16, torch.device("cpu")
    )
    assert "input_ids" not in tensors.tensors


def test_later_rank_reserves_input_ids(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    tensors = _model().make_empty_intermediate_tensors(
        5, torch.bfloat16, torch.device("cpu")
    )
    assert tensors["input_ids"].shape == (5,)
    assert tensors["input_ids"].dtype == torch.int32
