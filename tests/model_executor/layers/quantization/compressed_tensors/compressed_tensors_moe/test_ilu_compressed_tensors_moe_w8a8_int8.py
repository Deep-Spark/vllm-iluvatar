# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing tests for Iluvatar W8A8 compressed-tensors MoE."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe_w8a8_int8 as w8a8_moe,
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
    monkeypatch.setattr(w8a8_moe, "ixfops", fake_ixfops)
    return fake_ixfops


def _non_ep_layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=2),
        expert_map=None,
        top_k=1,
        local_num_experts=4,
        global_num_experts=4,
        routed_scaling_factor=1.0,
        activation=SimpleNamespace(value="silu"),
        w13_weight=torch.empty((4, 2, 5)),
        w2_weight=torch.empty((4, 5, 5)),
        w13_weight_scale=torch.ones(4),
        w2_weight_scale=torch.ones(4),
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
    )


class IluFlashAttentionMetadata:
    num_decodes = 1
    num_prefills = 0


def _decode_metadata():
    return IluFlashAttentionMetadata()


def test_w8a8_ep_uses_moe_config_ep_rank(monkeypatch):
    method = w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod.__new__(
        w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod
    )
    ixfops = _install_zero_expert_ep(monkeypatch)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    layer = _layer_without_ep_rank()
    x = torch.ones((3, 5), dtype=torch.float32)
    topk_ids = torch.zeros((3, 1), dtype=torch.int32)

    result = method.apply(
        layer=layer,
        x=x,
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    torch.testing.assert_close(result, torch.zeros_like(x))
    ixfops.moe_compute_token_index_ep.assert_called_once()
    assert ixfops.moe_compute_token_index_ep.call_args.kwargs["start_expert_id"] == 8
    assert ixfops.moe_compute_token_index_ep.call_args.kwargs["end_expert_id"] == 10


def test_w8a8_non_ep_uses_regular_token_index_and_group_gemm(monkeypatch):
    method = w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod.__new__(
        w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod
    )
    method.gemm_format = "TN"
    method.padding_dim = -1
    method.group_size = -1
    expected = torch.full((3, 5), 9.0)
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake_ixfops.moe_expand_input_dynamic_scaled_int8.return_value = (
        torch.ones((3, 5), dtype=torch.int8),
        torch.ones(1),
    )
    fake_ixfops.moe_w8a8_group_gemm.side_effect = [
        torch.ones((3, 10)),
        torch.ones((3, 5)),
    ]
    fake_ixfops.activation_dynamic_scaled_int8.return_value = (
        torch.ones((3, 5), dtype=torch.int8),
        torch.ones(1),
    )
    fake_ixfops.moe_output_reduce_sum.return_value = expected
    monkeypatch.setattr(w8a8_moe, "ixfops", fake_ixfops)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    layer = _non_ep_layer()

    result = method.apply(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is expected
    fake_ixfops.moe_compute_token_index.assert_called_once()
    assert fake_ixfops.moe_w8a8_group_gemm.call_count == 2
    fake_ixfops.moe_w8a8_group_gemv.assert_not_called()
    fake_ixfops.moe_compute_token_index_ep.assert_not_called()


def test_w8a8_decode_uses_group_gemv_path(monkeypatch):
    method = w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod.__new__(
        w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod
    )
    method.gemm_format = "NN"
    method.padding_dim = -2
    method.group_size = -1
    expected = torch.full((3, 5), 13.0)
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake_ixfops.moe_expand_input_dynamic_scaled_int8.return_value = (
        torch.ones((3, 5), dtype=torch.int8),
        torch.ones(1),
    )
    fake_ixfops.moe_w8a8_group_gemv.side_effect = [
        torch.ones((3, 10)),
        torch.ones((3, 5)),
    ]
    fake_ixfops.activation_dynamic_scaled_int8.return_value = (
        torch.ones((3, 5), dtype=torch.int8),
        torch.ones(1),
    )
    fake_ixfops.moe_output_reduce_sum.return_value = expected
    monkeypatch.setattr(w8a8_moe, "ixfops", fake_ixfops)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"decode": _decode_metadata()}),
    )
    layer = _non_ep_layer()

    result = method.apply(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is expected
    fake_ixfops.moe_compute_token_index.assert_called_once()
    assert fake_ixfops.moe_w8a8_group_gemv.call_count == 2
    fake_ixfops.moe_w8a8_group_gemm.assert_not_called()
    fake_ixfops.moe_compute_token_index_ep.assert_not_called()
