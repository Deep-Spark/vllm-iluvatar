from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import ParallelConfig

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep
from vllm_iluvatar.models.glm_moe_dsa.model import IluGlmMoeDsaMoE
from vllm_iluvatar.v1.worker.dp_utils import _post_process_all_ranks_decode_only


def _make_stub_moe(backend="allgather_reducescatter") -> IluGlmMoeDsaMoE:
    moe = IluGlmMoeDsaMoE.__new__(IluGlmMoeDsaMoE)
    moe.experts = MagicMock()
    moe.experts.is_internal_router = True
    moe.experts.moe_config.moe_parallel_config = SimpleNamespace(
        dp_size=2,
        use_ep=True,
        all2all_backend=backend,
    )
    moe.experts._quant_method.supports_internal_mk = False
    moe.experts._quant_method.is_monolithic = False
    moe.experts._ilu_nvsm_init_patched = False
    moe.gate = MagicMock()
    return moe


@pytest.mark.parametrize(
    "backend,override",
    [
        (backend, override)
        for backend in (
            "allgather_reducescatter",
            "deepep_low_latency",
            "deepep_high_throughput",
        )
        for override in (None, "naive", "deepep_low_latency")
        if override != "naive" or backend == "allgather_reducescatter"
    ],
)
@pytest.mark.parametrize(
    "rank_decode", [(False, False), (True, False), (True, True), None]
)
@pytest.mark.parametrize("prequantized", [False, True])
def test_glm_phase_dispatch(monkeypatch, backend, override, rank_decode, prequantized):
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", override)
    metadata = None
    if rank_decode is not None:
        # Use the coordinator's real reduction for prefill, mixed, and decode.
        coordinated = torch.zeros((6, 2), dtype=torch.int32)
        coordinated[5] = torch.tensor(rank_decode)
        metadata = SimpleNamespace(
            all_ranks_decode_only=_post_process_all_ranks_decode_only(coordinated)
        )
    monkeypatch.setattr(
        ep, "get_forward_context", lambda: SimpleNamespace(dp_metadata=metadata)
    )
    moe = _make_stub_moe(backend)
    x = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    q = torch.zeros_like(x, dtype=torch.int8)
    scales = torch.ones((2, 3), dtype=torch.float32)
    flat = x.reshape(-1, 4)
    moe.experts.return_value = flat + 1
    moe.forward_dp_ep_naive = MagicMock(return_value=flat + 2)
    moe.forward_dp_ep_deep_ep = MagicMock(return_value=flat + 3)

    output = moe.forward((q, scales, x) if prequantized else x)

    selected = override if rank_decode == (True, True) else None
    targets = {
        None: moe.experts,
        "naive": moe.forward_dp_ep_naive,
        "deepep_low_latency": moe.forward_dp_ep_deep_ep,
    }
    for name, target in targets.items():
        assert target.call_count == int(name == selected)
    torch.testing.assert_close(
        output, x + {None: 1, "naive": 2, "deepep_low_latency": 3}[selected]
    )
    call = targets[selected].call_args
    if selected is None:
        actual = call.kwargs["hidden_states"]
        if prequantized:
            torch.testing.assert_close(actual[0], q.reshape(-1, 4))
            torch.testing.assert_close(actual[1], scales)
            actual = actual[2]
        torch.testing.assert_close(actual, flat)
    else:
        torch.testing.assert_close(call.args[0], flat)
        shared = call.kwargs["shared_gate_up_input"]
        if prequantized:
            torch.testing.assert_close(shared[0], q.reshape(-1, 4))
            torch.testing.assert_close(shared[1], scales)
            assert shared[2] == x.dtype
        else:
            assert shared is None


@pytest.mark.parametrize("backend", ["allgather_reducescatter", "ag-rs", "ag_rs"])
@pytest.mark.parametrize("override", [None, "naive", "deepep_low_latency"])
def test_decode_override_initialization(monkeypatch, backend, override):
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", override)
    monkeypatch.setattr(ep.IluFusedMoEEP, "_dp_ep_ll_preinit_done", False)
    monkeypatch.setattr(ep.IluFusedMoEEP, "_dp_ep_ll_preinit_shapes", set())
    moe = _make_stub_moe(backend)
    original_init = moe.experts.routed_experts._ensure_moe_quant_config_init
    moe._get_forward_dp_ep_ll_all2all = MagicMock()
    moe.init_ep()
    assert ep.IluFusedMoEEP._dp_ep_ll_preinit_done == (override == "deepep_low_latency")
    # LL resource setup is deferred until weight processing has finished.
    moe._get_forward_dp_ep_ll_all2all.assert_not_called()
    moe.experts.routed_experts._ensure_moe_quant_config_init()
    original_init.assert_called_once()
    assert moe._get_forward_dp_ep_ll_all2all.call_count == int(
        override == "deepep_low_latency"
    )


@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_high_throughput"])
def test_naive_override_rejects_incompatible_backend_at_init(monkeypatch, backend):
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", "naive")
    with pytest.raises(ValueError, match="requires an AG-RS"):
        _make_stub_moe(backend).init_ep()


@pytest.mark.parametrize("override", ["naive", "deepep_low_latency"])
@pytest.mark.parametrize("unsupported", ["no_dp", "no_ep", "monolithic"])
def test_override_rejects_unsupported_layer_at_init(monkeypatch, override, unsupported):
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", override)
    moe = _make_stub_moe()
    config = moe.experts.moe_config.moe_parallel_config
    if unsupported == "no_dp":
        config.dp_size = 1
    elif unsupported == "no_ep":
        config.use_ep = False
    else:
        moe.experts._quant_method.is_monolithic = True
    with pytest.raises(ValueError, match="requires a supported DP\\+EP"):
        moe.init_ep()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),
        ("  ", None),
        ("naive", "naive"),
        (" deepep_low_latency ", "deepep_low_latency"),
    ],
)
def test_decode_override_env_values(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_ILU_DECODE_EP_BACKEND", value)
    assert ilu_envs._decode_ep_backend() == expected


def test_invalid_decode_override_rejected_when_read(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_DECODE_EP_BACKEND", "typo")
    with pytest.raises(ValueError, match="Invalid VLLM_ILU_DECODE_EP_BACKEND"):
        ilu_envs._decode_ep_backend()


@pytest.mark.parametrize(
    "backend",
    ["allgather_reducescatter", "deepep_low_latency", "deepep_high_throughput"],
)
@pytest.mark.parametrize("decode", [False, True])
def test_existing_dp_ep_dispatch_without_override(monkeypatch, backend, decode):
    """Call DSv4's actual model entry; downstream kernels remain test doubles."""
    from vllm_iluvatar.models.deepseek_v4.model import IluDeepseekV4MoE

    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", None)
    monkeypatch.setattr(
        ep,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(all_ranks_decode_only=decode)
        ),
    )
    moe = IluDeepseekV4MoE.__new__(IluDeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    stub = _make_stub_moe(backend)
    moe.experts = stub.experts
    moe.gate = MagicMock()
    moe.gate.tid2eid = torch.zeros(1)  # Exercise hash-routing input propagation.
    moe.hash_indices_dtype = torch.int64
    x = torch.zeros(2, 4)
    ids = torch.tensor([1, 2], dtype=torch.int32)
    q = torch.zeros_like(x, dtype=torch.int8)
    shared = (q, torch.ones(2, 1), x.dtype)
    moe.forward_dp_ep_naive = MagicMock(return_value=x + 1)
    moe.forward_dp_ep_deep_ep = MagicMock(return_value=x + 2)
    result = moe.forward(x, ids, shared_gate_up_input=shared)
    selected = (
        moe.forward_dp_ep_naive
        if backend == "allgather_reducescatter"
        else moe.forward_dp_ep_deep_ep
    )
    selected.assert_called_once()
    call = selected.call_args
    assert call.args[0] is x
    assert call.args[1].dtype == torch.int64
    torch.testing.assert_close(call.args[1], ids.long())
    assert call.kwargs["shared_gate_up_input"] is shared
    torch.testing.assert_close(
        result, x + (1 if backend == "allgather_reducescatter" else 2)
    )
    moe.experts.assert_not_called()
    if backend == "deepep_high_throughput":
        assert call.kwargs["enable_fused_ag_ar"] is False
        assert call.kwargs["moe_kernel"] is moe.experts._quant_method.moe_kernel


def test_configured_ll_does_not_preinit_duplicate_handle(monkeypatch):
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", "deepep_low_latency")
    monkeypatch.setattr(ep.IluFusedMoEEP, "_dp_ep_ll_preinit_done", False)
    moe = _make_stub_moe("deepep_low_latency")
    original_init = moe.experts.routed_experts._ensure_moe_quant_config_init
    moe.init_ep()
    assert moe.experts.routed_experts._ensure_moe_quant_config_init is original_init
    assert not ep.IluFusedMoEEP._dp_ep_ll_preinit_done


@pytest.mark.parametrize("override", [None, "naive", "deepep_low_latency"])
def test_fp8_ll_capacity_tracks_override(monkeypatch, override):
    from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp4.fp8_moe import (
        _shrink_ll_max_tokens_to_decode,
    )

    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", override)
    monkeypatch.setattr(ep, "_decode_ll_max_tokens_per_rank", lambda: 16)
    prepare = SimpleNamespace(max_tokens_per_rank=1024)
    _shrink_ll_max_tokens_to_decode(prepare)
    assert prepare.max_tokens_per_rank == (
        16 if override == "deepep_low_latency" else 1024
    )


@pytest.mark.parametrize("cli_backend", ["naive", "allgather_reducescatter"])
def test_naive_override_accepts_normalized_parallel_config(monkeypatch, cli_backend):
    config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend=cli_backend,
        distributed_executor_backend="mp",
    )
    assert config.all2all_backend == "allgather_reducescatter"
    monkeypatch.setattr(ep, "_DECODE_EP_BACKEND", "naive")
    moe = _make_stub_moe(config.all2all_backend)
    moe.init_ep()
    monkeypatch.setattr(
        ep,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(all_ranks_decode_only=True)
        ),
    )
    x = torch.zeros(2, 4)
    moe.forward_dp_ep_naive = MagicMock(return_value=x)
    torch.testing.assert_close(moe.forward(x), x)
    moe.forward_dp_ep_naive.assert_called_once()
    moe.experts.assert_not_called()
