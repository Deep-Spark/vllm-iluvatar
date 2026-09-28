# SPDX-License-Identifier: Apache-2.0
"""DSpark draft and target MoE must not share one NVSM EP communicator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep_mod
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)


@pytest.fixture(autouse=True)
def _reset_ll_cache():
    IluFusedMoEEP._dp_ep_ll_all2all_manager = None
    IluFusedMoEEP._dp_ep_ll_handles = {}
    yield
    IluFusedMoEEP._dp_ep_ll_all2all_manager = None
    IluFusedMoEEP._dp_ep_ll_handles = {}


def _make_ep(
    *,
    num_experts: int,
    experts_per_token: int,
    hidden_dim: int = 5120,
    max_num_tokens: int = 96,
) -> IluFusedMoEEP:
    ep = IluFusedMoEEP()
    ep.prefix = "layers.0.ffn"
    ep.experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            hidden_dim=hidden_dim,
            max_num_tokens=max_num_tokens,
        ),
        _quant_method=SimpleNamespace(moe_quant_config=None),
    )
    return ep


def test_ll_handles_are_keyed_by_experts_and_topk(monkeypatch):
    monkeypatch.setattr(ep_mod, "VLLM_ILU_LL_FUSED_AG_AR", False)
    monkeypatch.setattr(ep_mod, "VLLM_ILU_DEEPEP_USE_SCATTER", False)
    monkeypatch.setattr(ep_mod, "_get_max_num_seqs", lambda: 12)
    monkeypatch.setattr(ep_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    ep_comm = SimpleNamespace(cpu_group="cpu", tcp_store_group=None)
    monkeypatch.setattr(
        ep_mod,
        "get_ep_group",
        lambda: SimpleNamespace(device_communicator=ep_comm),
    )

    created_handles: list[SimpleNamespace] = []

    class FakeManager:
        world_size = 16

        def get_handle(self, args):
            handle = SimpleNamespace(buffer=None, group="ep", args=dict(args))
            created_handles.append(handle)
            return handle

    monkeypatch.setattr(
        "vllm_iluvatar.distributed.device_communicators.all2all.DeepEPLLAll2AllManager",
        lambda *a, **k: FakeManager(),
    )

    inits: list[dict] = []

    def fake_nvsm_init(**kwargs):
        inits.append(dict(kwargs))
        return f"comm-{kwargs['num_experts']}-{kwargs['topk']}"

    fake_nvsm = SimpleNamespace(nvsm_init_ep_comm=fake_nvsm_init)
    monkeypatch.setitem(
        __import__("sys").modules,
        "ixformer.distributed.nvsm_ep",
        fake_nvsm,
    )
    # Also patch the import site used inside the method.
    import ixformer.distributed as ixfd

    monkeypatch.setattr(ixfd, "nvsm_ep", fake_nvsm, raising=False)

    target = _make_ep(num_experts=384, experts_per_token=6)
    draft = _make_ep(num_experts=128, experts_per_token=3)

    # Force the lazy import path to see our fake module.
    import ixformer.distributed.nvsm_ep as nvsm_ep_mod

    monkeypatch.setattr(nvsm_ep_mod, "nvsm_init_ep_comm", fake_nvsm_init)

    _, target_handle = target._get_forward_dp_ep_ll_all2all()
    _, draft_handle = draft._get_forward_dp_ep_ll_all2all()

    assert target_handle is not draft_handle
    assert len(IluFusedMoEEP._dp_ep_ll_handles) == 2
    assert [c["topk"] for c in inits] == [6, 3]
    assert [c["num_experts"] for c in inits] == [384, 128]
    assert target_handle.buffer == "comm-384-6"
    assert draft_handle.buffer == "comm-128-3"

    # Re-fetch must reuse cached communicators.
    _, target_again = target._get_forward_dp_ep_ll_all2all()
    _, draft_again = draft._get_forward_dp_ep_ll_all2all()
    assert target_again is target_handle
    assert draft_again is draft_handle
    assert len(inits) == 2


def test_handle_key_includes_topk():
    ep = _make_ep(num_experts=384, experts_per_token=6)
    args = dict(
        max_num_tokens_per_dp_rank=96,
        token_hidden_size=5120,
        num_ep_ranks=16,
        num_global_experts=384,
        num_local_experts=24,
    )
    quant = {"use_int8": True, "use_fp8": False}
    k6 = ep._dp_ep_ll_handle_key(args, 12, 6, quant)
    k3 = ep._dp_ep_ll_handle_key(args, 12, 3, quant)
    assert k6 != k3


def test_get_max_num_seqs_uses_init_cache_without_vllm_config(monkeypatch):
    monkeypatch.setattr(ep_mod, "get_current_vllm_config_or_none", lambda: None)
    ep_mod._MAX_NUM_SEQS = None
    with pytest.raises(RuntimeError, match="max_num_seqs is unavailable"):
        ep_mod._get_max_num_seqs()

    # Speculative decode budget: max_num_seqs=16 * (1+5) = 96.
    ep_mod.set_max_num_seqs(96)
    assert ep_mod._get_max_num_seqs() == 96
    ep_mod._MAX_NUM_SEQS = None


def test_nvsm_ll_max_tokens_applies_tp_scatter(monkeypatch):
    monkeypatch.setattr(ep_mod, "_get_max_num_seqs", lambda: 96)
    monkeypatch.setattr(ep_mod, "VLLM_ILU_LL_FUSED_AG_AR", True)
    monkeypatch.setattr(ep_mod, "VLLM_ILU_DEEPEP_USE_SCATTER", False)
    monkeypatch.setattr(ep_mod, "get_tensor_model_parallel_world_size", lambda: 8)
    assert ep_mod._nvsm_ll_max_tokens_per_rank() == 12


def test_prepare_finalize_uses_nvsm_token_budget(monkeypatch):
    """PrepareFinalize max_tokens must match nvsm_init, not moe_config.max_num_tokens."""
    monkeypatch.setattr(ep_mod, "VLLM_ILU_LL_FUSED_AG_AR", False)
    monkeypatch.setattr(ep_mod, "VLLM_ILU_DEEPEP_USE_SCATTER", False)
    monkeypatch.setattr(ep_mod, "_get_max_num_seqs", lambda: 12)
    monkeypatch.setattr(ep_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    ep_comm = SimpleNamespace(cpu_group="cpu", tcp_store_group=None)
    monkeypatch.setattr(
        ep_mod,
        "get_ep_group",
        lambda: SimpleNamespace(device_communicator=ep_comm),
    )

    class FakeManager:
        world_size = 16

        def get_handle(self, args):
            return SimpleNamespace(buffer="comm", group="ep", args=dict(args))

    monkeypatch.setattr(
        "vllm_iluvatar.distributed.device_communicators.all2all.DeepEPLLAll2AllManager",
        lambda *a, **k: FakeManager(),
    )

    captured: dict = {}

    class FakePF:
        def __init__(self, handle, max_tokens_per_rank, **kwargs):
            captured["max_tokens_per_rank"] = max_tokens_per_rank
            captured["handle"] = handle

    monkeypatch.setattr(
        "vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize."
        "ilu_deepep_ll.DeepEPLLPrepareAndFinalize",
        FakePF,
    )

    class FakeGemm:
        pass

    ep = _make_ep(num_experts=384, experts_per_token=6, max_num_tokens=2048)
    quant_method = SimpleNamespace(
        moe_quant_config=None,
        select_gemm_impl=lambda *a, **k: FakeGemm(),
    )
    ep.experts._quant_method = quant_method
    ep.experts.routed_experts = SimpleNamespace()
    ep.experts._expert_routing_tables = lambda: None
    ep.experts.moe_config.moe_parallel_config = SimpleNamespace(
        all2all_backend="allgather_reducescatter"
    )

    # Avoid importing real FusedMoEKernel; stub the constructor.
    class FakeKernel:
        def __init__(self, pf, gemm):
            self.prepare_finalize = pf
            self.fused_experts = gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel",
        FakeKernel,
    )
    monkeypatch.setattr(
        ep,
        "_nvsm_dispatch_quant_kwargs",
        lambda: {"use_int8": True, "use_fp8": False},
    )

    kernel = ep._get_forward_dp_ep_ll_moe_kernel()
    assert captured["max_tokens_per_rank"] == 12
    assert kernel.prepare_finalize is not None
