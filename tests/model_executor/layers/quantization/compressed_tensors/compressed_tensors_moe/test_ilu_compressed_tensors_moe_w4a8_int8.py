# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing tests for Iluvatar W4A8 compressed-tensors MoE."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe_w4a8_int8 as w4a8_moe,
)


def _layer_without_ep_rank():
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=2),
        expert_map=torch.tensor([0]),
        local_num_experts=4,
        global_num_experts=10,
    )


def _install_zero_expert_ep(monkeypatch):
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index_ep.return_value = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        0,
    )
    monkeypatch.setattr(w4a8_moe, "ixfops", fake_ixfops)
    return fake_ixfops


def _non_ep_layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=2),
        expert_map=None,
        top_k=1,
        local_num_experts=4,
        global_num_experts=4,
        routed_scaling_factor=9.0,
        _ilu_output_scale=2.5,
        activation=SimpleNamespace(value="silu"),
        w13_weight=torch.empty((4, 2, 5)),
        w2_weight=torch.empty((4, 5, 5)),
        w13_weight_scale=torch.ones(4),
        w2_weight_scale=torch.ones(4),
        w13_i8_weight_scale=torch.ones(4),
        w2_i8_weight_scale=torch.ones(4),
        w13_i8_weight_zero=torch.zeros(4),
        w2_i8_weight_zero=torch.zeros(4),
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
    )


class _GemmInfo:
    format = 0

    @staticmethod
    def get_n():
        return 5

    @staticmethod
    def get_pad_n():
        return 5


def test_w4a8_ep_uses_moe_config_ep_rank(monkeypatch):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    ixfops = _install_zero_expert_ep(monkeypatch)
    layer = _layer_without_ep_rank()
    x = torch.ones((3, 5), dtype=torch.float32)
    topk_ids = torch.zeros((3, 1), dtype=torch.int32)

    result = method._apply_ep(
        layer=layer,
        x=x,
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=topk_ids,
        shared_experts_input=None,
    )

    torch.testing.assert_close(result, torch.zeros_like(x))
    ixfops.moe_compute_token_index_ep.assert_called_once()
    assert ixfops.moe_compute_token_index_ep.call_args.kwargs["start_expert_id"] == 8
    assert ixfops.moe_compute_token_index_ep.call_args.kwargs["end_expert_id"] == 10


def test_w4a8_non_ep_uses_regular_token_index_and_group_gemm(monkeypatch):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.version = 2
    method.gemm1_info = _GemmInfo()
    method.gemm2_info = _GemmInfo()
    method._expand_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    method._activation_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    expected = torch.full((3, 5), 7.0)
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake_ixfops.moe_w4a8_group_gemm.side_effect = [
        torch.ones((3, 10)),
        torch.ones((3, 5)),
    ]
    fake_ixfops.moe_output_reduce_sum.return_value = expected
    monkeypatch.setattr(w4a8_moe, "ixfops", fake_ixfops)
    monkeypatch.setattr(w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor)
    layer = _non_ep_layer()

    result = method._apply_general(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
    )

    assert result is expected
    assert fake_ixfops.moe_output_reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    fake_ixfops.moe_compute_token_index.assert_called_once()
    assert fake_ixfops.moe_w4a8_group_gemm.call_count == 2
    fake_ixfops.moe_w4a8_group_gemv.assert_not_called()
    fake_ixfops.moe_compute_token_index_ep.assert_not_called()


def test_w4a8_decode_uses_decode_token_index_and_group_gemv(monkeypatch):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.gemm1_info = _GemmInfo()
    method.gemm2_info = _GemmInfo()
    method._expand_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    method._activation_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    expected = torch.full((3, 5), 11.0)
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index_for_decode.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake_ixfops.moe_w4a8_group_gemv.side_effect = [
        torch.ones((3, 10)),
        torch.ones((3, 5)),
    ]
    fake_ixfops.moe_output_reduce_sum.return_value = expected
    monkeypatch.setattr(w4a8_moe, "ixfops", fake_ixfops)
    monkeypatch.setattr(w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor)
    layer = _non_ep_layer()

    result = method._apply_decode(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
    )

    assert result is expected
    assert fake_ixfops.moe_output_reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    fake_ixfops.moe_compute_token_index_for_decode.assert_called_once()
    assert fake_ixfops.moe_w4a8_group_gemv.call_count == 2
    fake_ixfops.moe_w4a8_group_gemm.assert_not_called()
    fake_ixfops.moe_compute_token_index.assert_not_called()
    fake_ixfops.moe_compute_token_index_ep.assert_not_called()
