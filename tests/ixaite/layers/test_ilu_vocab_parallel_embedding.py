# SPDX-License-Identifier: Apache-2.0
"""Check direct ixaite gather binding, TP masking and eager/graph numerics."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

_MODULE_NAME = "vllm_iluvatar.backends.ixaite.layers.embedding"
_COMMUNITY_NAME = "vllm.model_executor.layers.vocab_parallel_embedding"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="an Iluvatar/CUDA device is required for this test",
)


# Import the plugin module against a throwaway community stand-in so its OOT
# registration side effect never touches the real registry (a second real
# registration in one process raises "Duplicate layer name").
@pytest.fixture()
def embed_mod(monkeypatch):
    class FakeUnquantizedEmbeddingMethod:
        pass

    class FakeVocabParallelEmbedding:
        @classmethod
        def register_oot(cls):
            return lambda oot_cls: oot_cls

        def forward(self, input_):
            return ("community", input_)

    fake_community = ModuleType(_COMMUNITY_NAME)
    fake_community.UnquantizedEmbeddingMethod = FakeUnquantizedEmbeddingMethod
    fake_community.VocabParallelEmbedding = FakeVocabParallelEmbedding
    fake_community.get_masked_input_and_mask = Mock()
    monkeypatch.setitem(sys.modules, _COMMUNITY_NAME, fake_community)

    sys.modules.pop(_MODULE_NAME, None)
    module = importlib.import_module(_MODULE_NAME)
    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", lambda t: t)
    yield module
    # Leave no fake-bound module behind for later real importers.
    sys.modules.pop(_MODULE_NAME, None)


def _make_layer(module, *, tp_size=1, quant_method=None, weight=None):
    layer = object.__new__(module.IluVocabParallelEmbedding)
    layer.tp_size = tp_size
    layer.quant_method = quant_method
    layer.weight = weight
    return layer


# An unbound layer cannot execute before startup completes.
def test_uninitialized_gather_is_an_error(embed_mod) -> None:
    assert embed_mod._IXAITE_EMBEDDING_GATHER is None

    layer = _make_layer(embed_mod)
    with pytest.raises(RuntimeError, match="not initialized"):
        layer.forward("ids")


# Startup binds the resolved ixaite callable directly.
def test_provider_binds_ixaite_gather(embed_mod, monkeypatch) -> None:
    gather = Mock()
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        embed_mod,
        "load_ixaite_python_op",
        Mock(return_value=(gather, "supported on ivcore11")),
    )

    embed_mod.configure_embedding_gather_provider()

    assert embed_mod._IXAITE_EMBEDDING_GATHER is gather


# Missing gather support fails during startup.
def test_provider_rejects_unavailable_ixaite(embed_mod, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        embed_mod,
        "load_ixaite_python_op",
        Mock(return_value=(None, "not validated on this arch")),
    )

    with pytest.raises(RuntimeError, match="embedding_gather"):
        embed_mod.configure_embedding_gather_provider()


# TP=1: the bound gather receives the long ids and the layer weight directly.
def test_forward_uses_bound_gather_tp1(embed_mod, monkeypatch) -> None:
    gather = Mock(return_value="gathered")
    monkeypatch.setattr(embed_mod, "_IXAITE_EMBEDDING_GATHER", gather)
    quant_method = embed_mod.UnquantizedEmbeddingMethod()
    weight = Mock(name="weight")
    layer = _make_layer(embed_mod, tp_size=1, quant_method=quant_method, weight=weight)
    ids = torch.tensor([3, 1, 4], dtype=torch.int32)

    assert layer.forward(ids) == "gathered"
    gather.assert_called_once()
    call_ids, call_weight = gather.call_args.args
    assert call_ids.dtype == torch.int64
    assert torch.equal(call_ids, ids.long())
    assert call_weight is weight


# Quantized tables fail before the unquantized gather can execute.
def test_forward_rejects_quantized_method(embed_mod, monkeypatch) -> None:
    gather = Mock()
    monkeypatch.setattr(embed_mod, "_IXAITE_EMBEDDING_GATHER", gather)
    layer = _make_layer(embed_mod, tp_size=1, quant_method=object())

    with pytest.raises(ValueError, match="unquantized"):
        layer.forward("ids")
    gather.assert_not_called()


# TP>1: masking precedes the gather and the all_reduce follows it, exactly
# like the community forward; the gather itself only sees the local shard.
def test_forward_tp2_masks_gathers_and_reduces(embed_mod, monkeypatch) -> None:
    weight = torch.randn(8, 4)
    gathered = torch.ones(3, 4)
    gather = Mock(return_value=gathered.clone())
    masked_ids = torch.tensor([0, 5, 0])
    input_mask = torch.tensor([True, False, True])
    embed_mod.get_masked_input_and_mask = Mock(return_value=(masked_ids, input_mask))
    monkeypatch.setattr(embed_mod, "_IXAITE_EMBEDDING_GATHER", gather)

    shard_indices = Mock(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    layer = _make_layer(
        embed_mod,
        tp_size=2,
        quant_method=embed_mod.UnquantizedEmbeddingMethod(),
        weight=weight,
    )
    layer.shard_indices = shard_indices

    out = layer.forward(torch.tensor([0, 5, 6]))

    gather.assert_called_once()
    call_ids, call_weight = gather.call_args.args
    assert call_ids.dtype == torch.int64
    assert torch.equal(call_ids, masked_ids.long())
    assert call_weight is weight
    # Rows masked out of this shard are zeroed before the reduce.
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(out[2]) == 0
    assert torch.equal(out[1], gathered[1])


# Real-GPU smoke: the OOT layer's forward with the real ixaite gather bound.
@requires_cuda
def test_forward_smoke_with_real_ixaite_op(monkeypatch) -> None:
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    # The real import registers the OOT class in the real registry; restore it.
    saved = op_registry_oot.get("VocabParallelEmbedding")
    sys.modules.pop(_MODULE_NAME, None)
    try:
        module = importlib.import_module(_MODULE_NAME)

        import ixaite

        ixaite.initialize("ivcore11")
        from ixaite.ops.embedding_gather import embedding_gather

        monkeypatch.setattr(module, "_IXAITE_EMBEDDING_GATHER", embedding_gather)
        monkeypatch.setattr(
            module, "tensor_model_parallel_all_reduce", lambda tensor: tensor
        )

        torch.manual_seed(37)
        weight = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
        layer = _make_layer(
            module,
            tp_size=1,
            quant_method=UnquantizedEmbeddingMethod(),
            weight=weight,
        )
        ids = torch.randint(0, 1024, (17,), device="cuda", dtype=torch.int64)

        actual = layer.forward(ids)

        torch.testing.assert_close(actual, F.embedding(ids, weight), rtol=0, atol=0)
    finally:
        sys.modules.pop(_MODULE_NAME, None)
        if saved is None:
            op_registry_oot.pop("VocabParallelEmbedding", None)
        else:
            op_registry_oot["VocabParallelEmbedding"] = saved
