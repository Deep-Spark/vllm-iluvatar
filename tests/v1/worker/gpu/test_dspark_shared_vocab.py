"""Shared vocabulary weights must not be materialized before target aliasing."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.deepseek_v41.dspark import _make_vocab_layer
from vllm.v1.worker.gpu.spec_decode.eagle.utils import maybe_share_target_embed


class Vocabulary(nn.Module):
    def __init__(self, vocab, hidden, prefix):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, hidden), requires_grad=False)
        self.num_embeddings_per_partition = vocab
        self.prefix = prefix


@pytest.mark.parametrize("owns_weight", [False, True])
def test_shared_placeholder_does_not_materialize_or_change_default_device(owns_weight):
    with torch.device("cpu"):
        layer = _make_vocab_layer(Vocabulary, 8, 4, prefix="draft", owns_weight=owns_weight)
        following = torch.empty(1)
    assert layer.weight.shape == (8, 4)
    assert layer.num_embeddings_per_partition == 8
    assert layer.weight.device.type == ("cpu" if owns_weight else "meta")
    assert following.device.type == "cpu"


def test_community_sharing_replaces_meta_embedding_with_target_owner(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
    from vllm_iluvatar.v1.worker.gpu.spec_decode import dspark_pp

    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(dspark_pp, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    placeholder = _make_vocab_layer(Vocabulary, 8, 4, prefix="draft", owns_weight=False)
    target = Vocabulary(8, 4, prefix="target")
    draft_inner = nn.Module()
    target_inner = nn.Module()
    draft_inner.embed_tokens = placeholder
    target_inner.embed_tokens = target
    draft = SimpleNamespace(has_own_embed_tokens=False)
    maybe_share_target_embed(draft, draft_inner, target_inner)
    assert draft_inner.embed_tokens is target
    assert draft_inner.embed_tokens.weight is target.weight
