# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar W8A8 compressed-tensors MoE.

Covers W8A8GroupGemmInfo (NN-only), create_weights (GemmInfo-driven shapes +
NN shard_dim), process_weights_after_loading (deferred padding), and the
apply() EP / non-EP group-gemm / decode group-gemv paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe_w8a8_int8 as w8a8_moe,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W8A8GroupGemmInfo,
)

_HIDDEN = 64
_INTER = 64
_NUM_EXPERTS = 4
_2INTER = 2 * _INTER


# --- W8A8GroupGemmInfo -------------------------------------------------------
def test_w8a8_group_gemm_info_nn_aligned():
    info = W8A8GroupGemmInfo("NN", k=64, n=128)
    assert (info.pad_k, info.pad_n) == (64, 128)
    assert info.get_weight_shape() == (64, 128)
    assert info.get_pad_weight_shape() == (64, 128)


def test_w8a8_group_gemm_info_nn_pads_non_aligned():
    info = W8A8GroupGemmInfo("NN", k=48, n=96)
    assert (info.pad_k, info.pad_n) == (64, 128)
    assert info.get_weight_shape() == (48, 96)
    assert info.get_pad_weight_shape() == (64, 128)


def test_w8a8_group_gemm_info_rejects_tn():
    with pytest.raises(AssertionError, match="only supports NN"):
        W8A8GroupGemmInfo("TN", k=64, n=128)


# --- create_weights / process_weights_after_loading --------------------------
def _make_method(*, has_bias: bool = False):
    method = w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod.__new__(
        w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod
    )
    method.moe = SimpleNamespace(is_act_and_mul=True)
    method.gemm_format = "NN"
    method.has_bias = has_bias
    method.weight_quant = SimpleNamespace(strategy=QuantizationStrategy.CHANNEL)
    method.static_input_scales = False
    return method


def test_create_weights_unpadded_shapes_and_shard_dim():
    method = _make_method()
    layer = torch.nn.Module()
    method.create_weights(layer, _NUM_EXPERTS, _HIDDEN, _INTER, torch.int8)
    # NN: w13 [E, k=hidden, n=2*inter]; w2 [E, k=inter, n=hidden].
    assert tuple(layer.w13_weight.shape) == (_NUM_EXPERTS, _HIDDEN, _2INTER)
    assert tuple(layer.w2_weight.shape) == (_NUM_EXPERTS, _INTER, _HIDDEN)
    # NN shard_dim override: w13 -> 1 (inter on dim1), w2 -> 0 (inter on dim0).
    assert layer.w13_weight.shard_dim == 1
    assert layer.w2_weight.shard_dim == 0
    # Per-channel weight scales on the n axis: [E, n, 1].
    assert tuple(layer.w13_weight_scale.shape) == (_NUM_EXPERTS, _2INTER, 1)
    assert tuple(layer.w2_weight_scale.shape) == (_NUM_EXPERTS, _HIDDEN, 1)
    assert layer.w13_bias is None and layer.w2_bias is None
    assert layer.w13_input_scale is None and layer.w2_input_scale is None
    assert method.gemm1_info.pad_k == _HIDDEN and method.gemm1_info.pad_n == _2INTER
    assert method.gemm2_info.pad_k == _INTER and method.gemm2_info.pad_n == _HIDDEN


def test_process_weights_after_loading_pads_non_aligned():
    method = _make_method()
    layer = torch.nn.Module()
    method.create_weights(layer, _NUM_EXPERTS, 48, 48, torch.int8)
    assert tuple(layer.w13_weight.shape) == (_NUM_EXPERTS, 48, 96)
    method.process_weights_after_loading(layer)
    # gemm1: k=48->64, n=96->128; gemm2: k=48->64, n=48->64.
    assert tuple(layer.w13_weight.shape) == (_NUM_EXPERTS, 64, 128)
    assert tuple(layer.w2_weight.shape) == (_NUM_EXPERTS, 64, 64)
    assert tuple(layer.w13_weight_scale.shape) == (_NUM_EXPERTS, 128, 1)
    assert tuple(layer.w2_weight_scale.shape) == (_NUM_EXPERTS, 64, 1)
    assert layer.w13_weight.shard_dim == 1 and layer.w2_weight.shard_dim == 0


def test_process_weights_after_loading_aligned_is_noop():
    method = _make_method()
    layer = torch.nn.Module()
    method.create_weights(layer, _NUM_EXPERTS, _HIDDEN, _INTER, torch.int8)
    method.process_weights_after_loading(layer)
    assert tuple(layer.w13_weight.shape) == (_NUM_EXPERTS, _HIDDEN, _2INTER)
    assert tuple(layer.w2_weight.shape) == (_NUM_EXPERTS, _INTER, _HIDDEN)


# --- apply() helpers --------------------------------------------------------
def _non_ep_layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=2),
        expert_map=None,
        top_k=1,
        local_num_experts=_NUM_EXPERTS,
        global_num_experts=_NUM_EXPERTS,
        routed_scaling_factor=9.0,
        _ilu_output_scale=2.5,
        activation=SimpleNamespace(value="silu"),
        w13_weight=torch.empty((_NUM_EXPERTS, _HIDDEN, _2INTER)),
        w2_weight=torch.empty((_NUM_EXPERTS, _INTER, _HIDDEN)),
        w13_weight_scale=torch.ones(_NUM_EXPERTS),
        w2_weight_scale=torch.ones(_NUM_EXPERTS),
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
    )


def _ep_layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=2),
        expert_map=torch.tensor([0]),
        local_num_experts=4,
        global_num_experts=10,
    )


def _apply_method():
    method = w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod.__new__(
        w8a8_moe.IluCompressedTensorsW8A8Int8MoEMethod
    )
    method.gemm_format = "NN"
    method.padding_dim = -2
    method.group_size = -1
    return method


def _fake_ixfops_non_decode(expected):
    fake = MagicMock()
    fake.moe_compute_token_index.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake.moe_expand_input_dynamic_scaled_int8.return_value = (
        torch.ones((3, _HIDDEN), dtype=torch.int8),
        torch.ones(1),
    )
    fake.moe_w8a8_group_gemm.side_effect = [
        torch.ones((3, _2INTER)),
        torch.ones((3, _HIDDEN)),
    ]
    fake.activation_dynamic_scaled_int8.return_value = (
        torch.ones((3, _INTER), dtype=torch.int8),
        torch.ones(1),
    )
    fake.moe_output_reduce_sum.return_value = expected
    return fake


class _DecodeMeta:
    num_prefills = 0
    num_decodes = 1


def _fake_ixfops_decode(expected):
    fake = MagicMock()
    fake.moe_compute_token_index.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake.moe_expand_input_dynamic_scaled_int8.return_value = (
        torch.ones((3, _HIDDEN), dtype=torch.int8),
        torch.ones(1),
    )
    fake.moe_w8a8_group_gemv.side_effect = [
        torch.ones((3, _2INTER)),
        torch.ones((3, _HIDDEN)),
    ]
    fake.activation_dynamic_scaled_int8.return_value = (
        torch.ones((3, _INTER), dtype=torch.int8),
        torch.ones(1),
    )
    fake.moe_output_reduce_sum.return_value = expected
    return fake


# --- apply() tests ----------------------------------------------------------
def test_w8a8_ep_uses_moe_config_ep_rank(monkeypatch):
    method = _apply_method()
    fake = MagicMock()
    fake.moe_compute_token_index_ep.return_value = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        0,
    )
    monkeypatch.setattr(w8a8_moe, "ixfops", fake)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    layer = _ep_layer()
    x = torch.ones((3, _HIDDEN), dtype=torch.float32)

    result = method.apply(
        layer=layer,
        x=x,
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    torch.testing.assert_close(result, torch.zeros_like(x))
    fake.moe_compute_token_index_ep.assert_called_once()
    assert fake.moe_compute_token_index_ep.call_args.kwargs["start_expert_id"] == 8
    assert fake.moe_compute_token_index_ep.call_args.kwargs["end_expert_id"] == 10


def test_w8a8_non_ep_uses_group_gemm(monkeypatch):
    method = _apply_method()
    expected = torch.full((3, _HIDDEN), 9.0)
    fake = _fake_ixfops_non_decode(expected)
    monkeypatch.setattr(w8a8_moe, "ixfops", fake)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    result = method.apply(
        layer=_non_ep_layer(),
        x=torch.ones((3, _HIDDEN), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is expected
    assert fake.moe_output_reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    fake.moe_compute_token_index.assert_called_once()
    assert fake.moe_w8a8_group_gemm.call_count == 2
    fake.moe_w8a8_group_gemv.assert_not_called()
    fake.moe_compute_token_index_ep.assert_not_called()


def test_w8a8_decode_uses_group_gemv(monkeypatch):
    method = _apply_method()
    expected = torch.full((3, _HIDDEN), 13.0)
    fake = _fake_ixfops_decode(expected)
    monkeypatch.setattr(w8a8_moe, "ixfops", fake)
    monkeypatch.setattr(
        w8a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"decode": _DecodeMeta()}),
    )

    result = method.apply(
        layer=_non_ep_layer(),
        x=torch.ones((3, _HIDDEN), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is expected
    assert fake.moe_output_reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    fake.moe_compute_token_index.assert_called_once()
    assert fake.moe_w8a8_group_gemv.call_count == 2
    fake.moe_w8a8_group_gemm.assert_not_called()
    fake.moe_compute_token_index_ep.assert_not_called()
