# SPDX-License-Identifier: Apache-2.0
"""DSpark PP aux contract, draft embedding remap, and vocab-shard argmax."""

import torch
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model
from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import vocab_parallel_argmax


class _PP:
    def __init__(self, world_size: int, rank: int, start_layer: int):
        self.world_size = world_size
        self.rank = rank
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1
        self.start_layer = start_layer


def _install_pp(monkeypatch, group: _PP) -> None:
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(ps, "get_pp_group", lambda: group)


def test_v41_model_opts_into_aux_relay():
    assert IluDeepseekV41Model.supports_aux_hidden_states_over_pp is True


def test_aux_slots_follow_idx_plus_one(monkeypatch):
    """Layer id ``idx + 1`` is what the V4.1 forward captures.

    Rank 0 owns layers [0, 4) and therefore ids 1..4. Rank 1 starts at 4, so
    its upstream slot count is every configured id <= 4.
    """
    group = _PP(world_size=2, rank=1, start_layer=4)
    _install_pp(monkeypatch, group)

    model = EagleModelMixin()
    model.start_layer = 4
    model._set_aux_hidden_state_layers((1, 3, 4, 6))

    assert model._aux_slot_base_cached == 3
    assert model._aux_upstream_total_cached == 3

    local = [torch.ones(2, 4), torch.ones(2, 4) * 2]
    packed = model.pack_local_aux_hidden_states(local)
    assert list(packed) == ["aux_hidden_states_3", "aux_hidden_states_4"]

    incoming = IntermediateTensors(
        {
            "aux_hidden_states_0": torch.zeros(2, 4),
            "aux_hidden_states_1": torch.zeros(2, 4),
            "aux_hidden_states_2": torch.zeros(2, 4),
        }
    )
    remote = model.collect_remote_aux_hidden_states(incoming)
    assert len(remote) == 3


def test_vocab_parallel_argmax_tp1(monkeypatch):
    import vllm_iluvatar.v1.worker.gpu.spec_decode.dspark as dspark_mod

    monkeypatch.setattr(dspark_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.1, 0.2]])
    ids = vocab_parallel_argmax(logits.clone(), vocab_start=0, num_vocab_padding=0)
    assert ids.tolist() == [1, 0]


def test_remap_dspark_loads_embed_under_pp():
    from vllm_iluvatar.models.deepseek_v41.dspark import remap_dspark_name

    assert remap_dspark_name("embed.weight", load_own_embed=True) == (
        "model.embed_tokens.weight"
    )
    assert remap_dspark_name("embed.weight", load_own_embed=False) is None
    assert remap_dspark_name("mtp.0.norm.weight") == "model.norm.weight"
    assert remap_dspark_name("mtp.1.attn.wq_a.weight") == (
        "model.layers.1.attn.wq_a.weight"
    )
