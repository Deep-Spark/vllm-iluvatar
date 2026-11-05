# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing tests for Iluvatar W4A8 compressed-tensors MoE."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_iluvatar.backends.ixaite.moe import w4a8 as w4a8_moe


def _install_zero_expert_ep(monkeypatch):
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index_ep.return_value = (
        torch.full((3,), -1, dtype=torch.int64),
        torch.full((3,), -1, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int32),
        None,
        torch.tensor(0),
        torch.zeros(9, dtype=torch.int32),
    )
    # Route tests stub the bound wrappers so provider caches cannot bypass mocks.
    for name in (
        "moe_compute_token_index",
        "moe_compute_token_index_for_decode",
        "moe_compute_token_index_ep",
        "moe_output_reduce_sum",
    ):
        monkeypatch.setattr(w4a8_moe, "_" + name, getattr(fake_ixfops, name))
    fake_ixfops.moe_w4a8_group_gemm.return_value = torch.zeros(3, 10)
    monkeypatch.setattr(
        w4a8_moe, "_moe_w4a8_group_gemm", fake_ixfops.moe_w4a8_group_gemm
    )
    fake_ixfops.moe_output_reduce_sum.return_value = torch.zeros(3, 5)
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
    layer = _non_ep_layer()
    layer.moe_config.ep_rank = 2
    layer.global_num_experts = 10
    method.gemm1_info = method.gemm2_info = _GemmInfo()
    method.group_size = -1
    method.version = 2
    method._expand_and_quant = method._activation_and_quant = lambda **kwargs: (
        torch.zeros(3, 5),
        torch.ones(3),
    )
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
    assert ixfops.moe_compute_token_index_ep.call_args.kwargs["graph_safe"] is True
    assert ixfops.moe_output_reduce_sum.call_args.kwargs["mask"].all()


def test_w4a8_non_ep_uses_regular_token_index_and_group_gemm(monkeypatch):
    method, expert_sizes_gpu, expert_sizes_cpu = _stub_general_method()
    expected = torch.full((3, 5), 7.0)
    token_index = MagicMock()
    group_gemm = MagicMock(
        side_effect=[
            torch.ones((3, 10)),
            torch.ones((3, 5)),
        ]
    )
    group_gemv = MagicMock()
    token_index_ep = MagicMock()
    reduce_sum = MagicMock(return_value=expected)
    monkeypatch.setattr(w4a8_moe, "_moe_compute_token_index", token_index)
    monkeypatch.setattr(w4a8_moe, "_moe_compute_token_index_ep", token_index_ep)
    monkeypatch.setattr(w4a8_moe, "_moe_w4a8_group_gemm", group_gemm)
    monkeypatch.setattr(w4a8_moe, "_moe_w4a8_group_gemv", group_gemv)
    monkeypatch.setattr(w4a8_moe, "_moe_output_reduce_sum", reduce_sum)
    monkeypatch.setattr(
        w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor
    )
    layer = _non_ep_layer()

    result = method._apply_general(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
    )

    assert result is expected
    assert reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    token_index.assert_called_once()
    assert group_gemm.call_count == 2
    group_gemv.assert_not_called()
    token_index_ep.assert_not_called()
    # ixformer-style GEMM consumes the CPU histogram: D2H + host sync happen.
    method._copy_hist_async.assert_not_called()
    method._wait_hist_d2h.assert_not_called()
    for call in group_gemm.call_args_list:
        assert call.kwargs["tokens_per_experts"] is expert_sizes_gpu
        assert "tokens_per_experts_host" not in call.kwargs
        assert call.kwargs["tokens_per_experts_gpu"] is expert_sizes_gpu


def test_w4a8_decode_uses_decode_token_index_and_group_gemv(monkeypatch):
    # This case verifies the split GEMVs; fusion has separate upstream coverage.
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.gemm1_info = _GemmInfo()
    method.gemm2_info = _GemmInfo()
    method._decode_ws = {}
    method._expand_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    method._activation_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(1))
    )
    expected = torch.full((3, 5), 11.0)
    token_index_decode = MagicMock(
        return_value=(
            torch.arange(3),
            torch.arange(3),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )
    )
    token_index = MagicMock()
    token_index_ep = MagicMock()
    group_gemm = MagicMock()
    group_gemv = MagicMock(
        side_effect=[
            torch.ones((3, 10)),
            torch.ones((3, 5)),
        ]
    )
    reduce_sum = MagicMock(return_value=expected)
    monkeypatch.setattr(
        w4a8_moe, "_moe_compute_token_index_for_decode", token_index_decode
    )
    monkeypatch.setattr(w4a8_moe, "_moe_compute_token_index", token_index)
    monkeypatch.setattr(w4a8_moe, "_moe_compute_token_index_ep", token_index_ep)
    monkeypatch.setattr(w4a8_moe, "_moe_w4a8_group_gemm", group_gemm)
    monkeypatch.setattr(w4a8_moe, "_moe_w4a8_group_gemv", group_gemv)
    monkeypatch.setattr(w4a8_moe, "_moe_output_reduce_sum", reduce_sum)
    monkeypatch.setattr(
        w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor
    )
    layer = _non_ep_layer()

    result = method._apply_decode(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
    )

    assert result is expected
    assert reduce_sum.call_args.kwargs["scaling_factor"] == 2.5
    token_index_decode.assert_called_once()
    assert group_gemv.call_count == 2
    expert_sizes_gpu = token_index_decode.return_value[2]
    for call in group_gemv.call_args_list:
        assert call.kwargs["tokens_per_experts"] is expert_sizes_gpu
    group_gemm.assert_not_called()
    token_index.assert_not_called()
    token_index_ep.assert_not_called()


# Preserve upstream fusion in baseline/auto while forced ixaite uses split GEMV.
@pytest.mark.parametrize("provider", ["baseline"])
def test_w13_fusion_respects_forced_ixaite(monkeypatch, provider):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_compressed_tensors_moe_w4a8_int8 as vendor_w4a8,
    )

    method = vendor_w4a8.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        vendor_w4a8.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.version = 2
    method.group_size = -1
    method.gemm1_info = SimpleNamespace(format_str="TN")
    method.gemm2_info = SimpleNamespace(pad_k=2048)
    layer = SimpleNamespace(
        # Meta tensors exercise a supported fusion layout without weight storage.
        w13_weight=torch.empty((32, 4096, 3072), device="meta"),
        w13_weight_scale=torch.empty((32, 4096), device="meta"),
        activation=SimpleNamespace(value="silu"),
    )
    expected = (torch.zeros((1, 2048), dtype=torch.int8), torch.ones(1))
    fused = MagicMock(return_value=expected)
    monkeypatch.setattr(vendor_w4a8.ixfops, "moe_w4a8_group_gemv_swiglu_quant", fused)
    monkeypatch.setattr(vendor_w4a8.envs, "VLLM_ILU_FUSE_MOE_W13", True)
    monkeypatch.setattr(vendor_w4a8.envs, "VLLM_ILU_KERNEL_PROVIDER", provider)

    actual = method._try_fused_w13_swiglu_quant(
        layer=layer,
        act=torch.zeros((1, 6144), dtype=torch.int8),
        a_scale=torch.ones(1),
        dense_expert_sizes_gpu=torch.zeros(65, dtype=torch.int32),
        aggregate_rows=1,
        sorted_token_ids=torch.zeros(1, dtype=torch.int32),
    )

    if provider == "ixaite":
        assert actual is None
        fused.assert_not_called()
    else:
        assert actual[0] is expected[0]
        assert actual[1] is expected[1]
        fused.assert_called_once()


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

    monkeypatch.setattr(w4a8_moe, "_moe_expand_input_dynamic_scaled_int8", fake_expand)

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

    monkeypatch.setattr(w4a8_moe, "_activation_dynamic_scaled_int8", fake_act)

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
    from vllm_iluvatar.backends.ixaite.moe import standard_w4a8 as std_w4a8

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

    monkeypatch.setattr(w4a8_moe, "_moe_expand_input_dynamic_scaled_int8", fake_expand)
    monkeypatch.setattr(w4a8_moe, "_activation_dynamic_scaled_int8", fake_act)

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


def _stub_general_method():
    """Instance state for _apply_general without running __init__."""
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
    src_to_dst = torch.arange(3)
    sorted_token_ids = torch.arange(3)
    expert_sizes_gpu = torch.ones(4, dtype=torch.int32)
    expert_sizes_cpu = torch.ones(4, dtype=torch.int32)
    method._prefill_token_index_workspace = MagicMock(
        return_value=(src_to_dst, sorted_token_ids, expert_sizes_gpu)
    )
    method._copy_hist_async = MagicMock()
    method._wait_hist_d2h = MagicMock()
    return method, expert_sizes_gpu, expert_sizes_cpu


def test_w4a8_prefill_skips_cpu_hist_when_gemm_plans_on_gpu(monkeypatch):
    method, expert_sizes_gpu, expert_sizes_cpu = _stub_general_method()
    expected = torch.full((3, 5), 7.0)
    group_gemm = MagicMock(
        side_effect=[
            torch.ones((3, 10)),
            torch.ones((3, 5)),
        ]
    )
    reduce_sum = MagicMock(return_value=expected)
    monkeypatch.setattr(w4a8_moe, "_moe_compute_token_index", MagicMock())
    monkeypatch.setattr(w4a8_moe, "_moe_w4a8_group_gemm", group_gemm)
    monkeypatch.setattr(w4a8_moe, "_moe_output_reduce_sum", reduce_sum)
    monkeypatch.setattr(
        w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor
    )
    layer = _non_ep_layer()

    result = method._apply_general(
        layer=layer,
        x=torch.ones((3, 5), dtype=torch.float32),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
    )

    assert result is expected
    assert group_gemm.call_count == 2
    # GPU-planned GEMM: no histogram D2H, no host sync, no CPU mirror passed.
    method._copy_hist_async.assert_not_called()
    method._wait_hist_d2h.assert_not_called()
    for call in group_gemm.call_args_list:
        assert call.kwargs["tokens_per_experts"] is expert_sizes_gpu
        assert "tokens_per_experts_host" not in call.kwargs
        assert call.kwargs["tokens_per_experts_gpu"] is expert_sizes_gpu
