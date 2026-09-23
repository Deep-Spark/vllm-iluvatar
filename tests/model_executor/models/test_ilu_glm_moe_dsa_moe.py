from unittest.mock import MagicMock

import torch

from vllm_iluvatar.models.glm_moe_dsa.model import IluGlmMoeDsaMoE


def _make_stub_moe() -> IluGlmMoeDsaMoE:
    moe = IluGlmMoeDsaMoE.__new__(IluGlmMoeDsaMoE)
    moe.experts = MagicMock()
    moe.experts.is_internal_router = True
    moe.gate = MagicMock()
    return moe


def test_tuple_input_uses_dp_ep_with_prequantized_shared_input():
    moe = _make_stub_moe()
    q = torch.zeros((2, 3, 4), dtype=torch.int8)
    scales = torch.ones((2, 3), dtype=torch.float32)
    normed = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    moe._should_use_forward_dp_ep = MagicMock(return_value=True)
    moe.forward_dp_ep = MagicMock(return_value=normed.reshape(-1, 4) + 1)

    output = moe.forward((q, scales, normed))

    moe._should_use_forward_dp_ep.assert_called_once_with()
    moe.experts.assert_not_called()
    args, kwargs = moe.forward_dp_ep.call_args
    torch.testing.assert_close(args[0], normed.reshape(-1, 4))
    shared_q, shared_scales, shared_dtype = kwargs["shared_gate_up_input"]
    torch.testing.assert_close(shared_q, q.reshape(-1, 4))
    torch.testing.assert_close(shared_scales, scales)
    assert shared_dtype == torch.bfloat16
    torch.testing.assert_close(output, normed + 1)


def test_tensor_input_uses_dp_ep_without_prequantized_shared_input():
    moe = _make_stub_moe()
    hidden_states = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    moe._should_use_forward_dp_ep = MagicMock(return_value=True)
    moe.forward_dp_ep = MagicMock(return_value=hidden_states.reshape(-1, 4) + 1)

    output = moe.forward(hidden_states)

    moe.forward_dp_ep.assert_called_once()
    args, kwargs = moe.forward_dp_ep.call_args
    torch.testing.assert_close(args[0], hidden_states.reshape(-1, 4))
    assert kwargs["shared_gate_up_input"] is None
    moe.experts.assert_not_called()
    torch.testing.assert_close(output, hidden_states + 1)
