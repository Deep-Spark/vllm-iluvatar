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
    method.version = 2
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
    method._try_fused_w13_swiglu_quant = MagicMock(return_value=None)
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


class _PadGemm1:
    pad_k = 256

    @staticmethod
    def get_n():
        return 384


class _PadGemm2:
    pad_k = 256


def _method_with_pad_gemm():
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.gemm1_info = _PadGemm1()
    method.gemm2_info = _PadGemm2()
    return method


def test_expand_and_quant_fuses_k_padding(monkeypatch):
    method = _method_with_pad_gemm()
    captured: dict[str, object] = {}

    def fake_expand(**kwargs):
        captured.update(kwargs)
        hidden = kwargs["hidden_states"].shape[-1]
        padding = kwargs["output_padding"]
        dst_tokens = kwargs["dst_tokens"]
        return (
            torch.zeros((dst_tokens, hidden + padding), dtype=torch.int8),
            torch.ones(dst_tokens),
        )

    monkeypatch.setattr(
        w4a8_moe,
        "ixfops",
        SimpleNamespace(moe_expand_input_dynamic_scaled_int8=fake_expand),
    )
    monkeypatch.setattr(
        w4a8_moe,
        "pad_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pad_tensor should not run after fused output_padding")
        ),
    )

    hidden = torch.ones((2, 192), dtype=torch.float32)
    output, _scale = method._expand_and_quant(
        layer=SimpleNamespace(top_k=8, w13_input_scale=None),
        input=hidden,
        expand_tokens=16,
        sorted_token_ids=torch.arange(16, dtype=torch.int32),
        src_to_dst=torch.arange(16, dtype=torch.int32),
        output_format=1,
        check_valid=True,
    )

    assert captured["output_padding"] == 64
    assert captured["output_format"] == 1
    assert captured["check_valid"] is True
    assert tuple(output.shape) == (16, 256)


def test_activation_and_quant_fuses_k_padding(monkeypatch):
    method = _method_with_pad_gemm()
    captured: dict[str, object] = {}

    def fake_act(**kwargs):
        captured.update(kwargs)
        rows = kwargs["input"].shape[0]
        valid = kwargs["input"].shape[-1] // 2
        padding = kwargs["output_padding"]
        return (
            torch.zeros((rows, valid + padding), dtype=torch.int8),
            torch.ones(rows),
        )

    monkeypatch.setattr(
        w4a8_moe,
        "ixfops",
        SimpleNamespace(activation_dynamic_scaled_int8=fake_act),
    )
    monkeypatch.setattr(
        w4a8_moe,
        "pad_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pad_tensor should not run after fused output_padding")
        ),
    )

    act_in = torch.ones((4, 384), dtype=torch.float32)
    output, _scale = method._activation_and_quant(
        layer=SimpleNamespace(
            activation=SimpleNamespace(value="silu"),
            w2_input_scale=None,
        ),
        input=act_in,
        sorted_token_ids=torch.arange(4, dtype=torch.int32),
        output_format=1,
        check_valid=False,
    )

    assert captured["output_padding"] == 64
    assert captured["output_format"] == 1
    assert captured["act_type"] == "swiglu"
    assert tuple(output.shape) == (4, 256)


def test_standard_experts_expand_and_act_fuse_k_padding(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        ilu_moe_standard_experts_w4a8 as std_w4a8,
    )

    experts = std_w4a8.IluMoEStandardExpertsW4A8.__new__(
        std_w4a8.IluMoEStandardExpertsW4A8
    )
    experts.gemm1_info = _PadGemm1()
    experts.gemm2_info = _PadGemm2()
    experts.w13_input_scale = None
    experts.w2_input_scale = None
    experts.swiglu_limit = None

    expand_kw: dict[str, object] = {}
    act_kw: dict[str, object] = {}

    def fake_expand(**kwargs):
        expand_kw.update(kwargs)
        hidden = kwargs["hidden_states"].shape[-1]
        padding = kwargs["output_padding"]
        dst_tokens = kwargs["dst_tokens"]
        return (
            torch.zeros((dst_tokens, hidden + padding), dtype=torch.int8),
            torch.ones(dst_tokens),
        )

    def fake_act(**kwargs):
        act_kw.update(kwargs)
        rows = kwargs["input"].shape[0]
        valid = kwargs["input"].shape[-1] // 2
        padding = kwargs["output_padding"]
        return (
            torch.zeros((rows, valid + padding), dtype=torch.int8),
            torch.ones(rows),
        )

    monkeypatch.setattr(
        std_w4a8,
        "ixff",
        SimpleNamespace(
            moe_expand_input_dynamic_scaled_int8=fake_expand,
            activation_dynamic_scaled_int8=fake_act,
        ),
    )

    expand_out, _ = experts._expand_and_quant(
        input=torch.ones((2, 192), dtype=torch.float32),
        expand_tokens=16,
        sorted_token_ids=torch.arange(16, dtype=torch.int32),
        src_to_dst=torch.arange(16, dtype=torch.int32),
        output_format=0,
        topk=8,
    )
    act_out, _ = experts._activation_and_quant(
        input=torch.ones((4, 384), dtype=torch.float32),
        sorted_token_ids=torch.arange(4, dtype=torch.int32),
        output_format=1,
        activation=SimpleNamespace(value="swiglu"),
    )

    assert expand_kw["output_padding"] == 64
    assert tuple(expand_out.shape) == (16, 256)
    assert act_kw["output_padding"] == 64
    assert tuple(act_out.shape) == (4, 256)
