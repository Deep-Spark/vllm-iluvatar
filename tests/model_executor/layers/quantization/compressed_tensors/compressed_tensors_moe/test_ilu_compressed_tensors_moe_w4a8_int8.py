# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing tests for Iluvatar W4A8 compressed-tensors MoE."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe_w4a8_int8 as w4a8_moe,
)


def _make_w4a8_method(gemm_format: str, pack_version: int):
    return w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod(
        weight_quant=SimpleNamespace(
            group_size=None,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
        ),
        input_quant=SimpleNamespace(
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
        ),
        moe=SimpleNamespace(has_bias=False),
        ilu_params={"gemm_format": gemm_format, "pack_version": pack_version},
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
    monkeypatch.setattr(w4a8_moe, "w4a8_group_gemv", fake_ixfops.moe_w4a8_group_gemv)
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
    format_str = "TN"

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
    monkeypatch.setattr(w4a8_moe, "w4a8_group_gemv", fake_ixfops.moe_w4a8_group_gemv)
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
    monkeypatch.setattr(w4a8_moe, "w4a8_group_gemv", fake_ixfops.moe_w4a8_group_gemv)
    monkeypatch.setattr(
        w4a8_moe, "unpad_tensor", lambda tensor, *_args, **_kwargs: tensor
    )
    method._try_fused_w13_swiglu_quant = MagicMock(return_value=None)
    method.decode_group_gemm = False
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


def test_w4a8_decode_group_gemm_uses_graph_gemm(monkeypatch):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.version = 1
    method.decode_group_gemm = True
    method.gemm1_info = _GemmInfo()
    method.gemm2_info = _GemmInfo()
    method._expand_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(3))
    )
    method._activation_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(3))
    )
    expected = torch.full((3, 5), 13.0)
    fake_ixfops = MagicMock()
    fake_ixfops.moe_compute_token_index_for_decode.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    fake_ixfops.moe_output_reduce_sum.return_value = expected
    monkeypatch.setattr(w4a8_moe, "ixfops", fake_ixfops)
    monkeypatch.setattr(w4a8_moe, "w4a8_group_gemv", fake_ixfops.moe_w4a8_group_gemv)
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
    assert fake_ixfops.moe_w4a8_group_gemm.call_count == 2
    fake_ixfops.moe_w4a8_group_gemv.assert_not_called()
    for call in fake_ixfops.moe_w4a8_group_gemm.call_args_list:
        assert call.kwargs["enable_cuda_graph"] is True
        assert call.kwargs["version"] == 1
    method._expand_and_quant.assert_called_once()
    assert method._expand_and_quant.call_args.kwargs["output_format"] == 0
    method._activation_and_quant.assert_called_once()
    assert method._activation_and_quant.call_args.kwargs["output_format"] == 0


def test_native_nn_v1_enables_decode_group_gemm():
    method = _make_w4a8_method("NN", 1)
    assert method.version == 1
    assert method.gemm_format == "NN"
    assert method.decode_group_gemm is True


def test_tn_v2_stays_on_decode_group_gemv():
    method = _make_w4a8_method("TN", 2)
    assert method.decode_group_gemm is False


def test_nn_v2_and_tn_v1_do_not_enable_decode_group_gemm():
    nn_v2 = _make_w4a8_method("NN", 2)
    tn_v1 = _make_w4a8_method("TN", 1)
    assert nn_v2.decode_group_gemm is False
    assert tn_v1.decode_group_gemm is False


def test_unsupported_pack_version_still_rejected():
    with pytest.raises(AssertionError, match="pack_version 1 or 2"):
        _make_w4a8_method("NN", 3)


def test_pack_unpack_int4_roundtrip():
    torch.manual_seed(0)
    values = torch.randint(-8, 8, (3, 64), dtype=torch.int8)
    packed = w4a8_moe._pack_int4(values)
    assert packed.shape[-1] == values.shape[-1] // 2
    torch.testing.assert_close(w4a8_moe._unpack_int4(packed), values)


def test_relayout_int4_tn_to_nn_recovers_values():
    torch.manual_seed(0)
    num_experts, n, k = 2, 64, 64
    values = torch.randint(-8, 8, (num_experts, n, k), dtype=torch.int8)
    w_tn = torch.stack([w4a8_moe._pack_int4(values[e]) for e in range(num_experts)])
    w_nn = w4a8_moe._relayout_int4_tn_to_nn(w_tn)
    assert w_nn.shape == (num_experts, k, n // 2)

    recovered = []
    for e in range(num_experts):
        swizzled = w4a8_moe._unpack_int4(w_nn[e])
        swizzled = swizzled.reshape(k // 32, 2, 16, n // 32, 2, 16)
        raw = swizzled.permute(0, 4, 2, 3, 1, 5).contiguous().reshape(k, n)
        recovered.append(raw.transpose(0, 1).contiguous())
    torch.testing.assert_close(torch.stack(recovered), values)


@pytest.mark.parametrize("decode_gemm", [False, True])
def test_ep_decode_quantizes_activation_once_with_matching_layout(
    monkeypatch, decode_gemm
):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.version = 1 if decode_gemm else 2
    method.decode_group_gemm = decode_gemm
    method._try_fused_w13_swiglu_quant = MagicMock(return_value=None)
    method.gemm1_info = _GemmInfo()
    method.gemm2_info = _GemmInfo()
    method._expand_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(3))
    )
    method._activation_and_quant = MagicMock(
        return_value=(torch.ones((3, 5), dtype=torch.int8), torch.ones(3))
    )
    ops = MagicMock()
    ops.moe_compute_token_index_ep.return_value = (
        torch.arange(3),
        torch.arange(3),
        torch.ones(4, dtype=torch.int32),
        torch.ones(4, dtype=torch.int32),
        3,
        torch.ones(4, dtype=torch.int32),
    )
    ops.moe_w4a8_group_gemv.return_value = torch.ones((3, 5))
    monkeypatch.setattr(w4a8_moe, "ixfops", ops)
    monkeypatch.setattr(w4a8_moe, "w4a8_group_gemv", ops.moe_w4a8_group_gemv)
    monkeypatch.setattr(w4a8_moe, "unpad_tensor", lambda tensor, *_a, **_kw: tensor)
    layer = _non_ep_layer()
    layer.moe_config.ep_rank = 0
    method._apply_ep(
        layer=layer,
        x=torch.ones((3, 5)),
        topk_weights=torch.ones((3, 1)),
        topk_ids=torch.zeros((3, 1), dtype=torch.int32),
        shared_experts_input=None,
        run_decode=True,
    )
    method._activation_and_quant.assert_called_once()
    assert method._activation_and_quant.call_args.kwargs["output_format"] == (
        0 if decode_gemm else 1
    )
    assert method._expand_and_quant.call_args.kwargs["output_format"] == (
        0 if decode_gemm else 1
    )
    assert ops.moe_w4a8_group_gemm.call_count == (2 if decode_gemm else 0)
    assert ops.moe_w4a8_group_gemv.call_count == (0 if decode_gemm else 2)


@pytest.mark.parametrize(
    "use_ep,local_prefills,global_decode,expected_decode",
    [
        (True, 0, False, False),  # Local decode, remote prefill after EP gather.
        (True, 1, False, False),
        (True, 0, True, True),  # All DP ranks decode, including speculative decode.
        (True, 0, None, True),  # TP/EP without DP uses local semantics.
        (True, 1, None, False),
        (False, 0, False, True),  # Non-EP input is local despite remote prefill.
        (False, 1, True, False),
    ],
)
def test_apply_dispatch_uses_gathered_batch_semantics(
    monkeypatch, use_ep, local_prefills, global_decode, expected_decode
):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    expected = object()
    method._apply_ep = MagicMock(return_value=expected)
    method._apply_decode = MagicMock(return_value=expected)
    method._apply_general = MagicMock(return_value=expected)
    method._valid_extra_residual = MagicMock(return_value=None)
    metadata = SimpleNamespace(num_prefills=local_prefills, num_decodes=4)
    dp = (
        None
        if global_decode is None
        else SimpleNamespace(all_ranks_decode_only=global_decode)
    )
    monkeypatch.setattr(
        w4a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"attention": metadata}, dp_metadata=dp),
    )
    guard = MagicMock()
    monkeypatch.setattr(w4a8_moe, "_ensure_cuda_graph_safe_dispatch", guard)
    layer = SimpleNamespace(expert_map=torch.tensor([0]) if use_ep else None)
    result = method.apply(
        layer,
        torch.ones((4, 5)),
        torch.ones((4, 1)),
        torch.zeros((4, 1), dtype=torch.int32),
        None,
        None,
    )
    assert result is expected
    guard.assert_called_once_with(expected_decode, {"attention": metadata})
    if use_ep:
        assert method._apply_ep.call_args.kwargs["run_decode"] is expected_decode
        method._apply_decode.assert_not_called()
        method._apply_general.assert_not_called()
    elif expected_decode:
        method._apply_decode.assert_called_once()
        method._apply_general.assert_not_called()
    else:
        method._apply_general.assert_called_once()
        method._apply_decode.assert_not_called()


def test_apply_dispatch_uses_dummy_decode_metadata_during_capture(monkeypatch):
    method = w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )
    expected = object()
    method._apply_ep = MagicMock(return_value=expected)
    method._apply_decode = MagicMock()
    method._apply_general = MagicMock()
    method._valid_extra_residual = MagicMock(return_value=None)
    metadata = SimpleNamespace(num_prefills=0, num_decodes=16)
    monkeypatch.setattr(
        w4a8_moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"attention": metadata},
            # Startup capture has no real DP batch synchronization yet.
            dp_metadata=SimpleNamespace(all_ranks_decode_only=False),
        ),
    )
    monkeypatch.setattr(w4a8_moe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        w4a8_moe.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    guard = MagicMock()
    monkeypatch.setattr(w4a8_moe, "_ensure_cuda_graph_safe_dispatch", guard)
    layer = SimpleNamespace(expert_map=torch.tensor([0]))

    result = method.apply(
        layer,
        torch.ones((16, 5)),
        torch.ones((16, 1)),
        torch.zeros((16, 1), dtype=torch.int32),
        None,
        None,
    )

    assert result is expected
    assert method._apply_ep.call_args.kwargs["run_decode"] is True
    guard.assert_called_once_with(True, {"attention": metadata})


@pytest.mark.parametrize(
    "setting,expected", [(None, "TN"), ("nn", "NN"), (" TN ", "TN")]
)
def test_runtime_layout_is_frozen_at_construction(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("VLLM_ILU_MOE_RUNTIME_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_ILU_MOE_RUNTIME_LAYOUT", setting)
    method = _make_w4a8_method("TN", 2)
    assert method.runtime_gemm_format == expected
    monkeypatch.setenv("VLLM_ILU_MOE_RUNTIME_LAYOUT", "invalid-after-construction")
    assert method.runtime_gemm_format == expected


@pytest.mark.parametrize("setting", ["", "NT", "invalid"])
def test_runtime_layout_rejects_invalid_configuration(monkeypatch, setting):
    monkeypatch.setenv("VLLM_ILU_MOE_RUNTIME_LAYOUT", setting)
    with pytest.raises(ValueError, match="must be NN or TN"):
        _make_w4a8_method("TN", 2)


@pytest.mark.parametrize(
    "available,supported,expected",
    [
        (["ixinfer", "ixformer_perf"], True, "ixformer_perf"),
        (["ixinfer", "ixformer_perf"], False, "ixinfer"),
        (["ixinfer"], True, "ixinfer"),
        (["ixformer_perf"], False, None),
    ],
)
def test_nn_backend_selected_before_execution(
    monkeypatch, available, supported, expected
):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    monkeypatch.setenv("VLLM_ILU_MOE_W4A8_GEMV_PERF", "1")

    monkeypatch.setattr(gemv.registry, "available_backends", lambda op, arch: available)
    seen = []
    monkeypatch.setattr(
        gemv.ixf, "moe_w4a8_group_gemv_perf_supported", lambda **kw: supported
    )
    monkeypatch.setattr(
        gemv.ixf, "moe_w4a8_group_gemv", lambda **kw: seen.append(kw) or "output"
    )
    remap = object()
    if expected is None:
        with pytest.raises(RuntimeError, match="No available"):
            gemv.w4a8_group_gemv(format=0, dst_to_src=remap)
        assert not seen
    else:
        assert gemv.w4a8_group_gemv(format=0, dst_to_src=remap) == "output"
        assert seen == [dict(format=0, dst_to_src=remap, backend=expected)]


def test_g1_g2_backend_decisions_are_independent_and_tn_uses_registry(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    monkeypatch.setenv("VLLM_ILU_MOE_W4A8_GEMV_PERF", "1")

    monkeypatch.setattr(
        gemv.registry,
        "available_backends",
        lambda op, arch: ["ixinfer", "ixformer_perf"],
    )
    monkeypatch.setattr(
        gemv.ixf,
        "moe_w4a8_group_gemv_perf_supported",
        lambda **kw: kw["dst_to_src"] is None,
    )
    seen = []
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", lambda **kw: seen.append(kw))
    remap = object()
    gemv.w4a8_group_gemv(format=0, dst_to_src=None)
    gemv.w4a8_group_gemv(format=0, dst_to_src=remap)
    gemv.w4a8_group_gemv(format=2, dst_to_src=remap)
    assert [call["backend"] for call in seen] == ["ixformer_perf", "ixinfer", None]
    assert seen[1]["dst_to_src"] is remap and seen[2]["dst_to_src"] is remap


def test_nn_vendor_rejection_is_not_retried(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    monkeypatch.setattr(
        gemv.registry, "available_backends", lambda op, arch: ["ixinfer"]
    )
    calls = []

    def reject(**kw):
        calls.append(kw)
        raise RuntimeError("vendor unsupported shape")

    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", reject)
    with pytest.raises(RuntimeError, match="vendor unsupported shape"):
        gemv.w4a8_group_gemv(format=0, dst_to_src="preserved")
    assert calls == [dict(format=0, dst_to_src="preserved", backend="ixinfer")]


def test_batched_g1_g2_use_actual_separate_calls(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_moe_batch_experts_w4a8 as batch,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    expert = batch.IluMoEBatchedExpertsW4A8.__new__(batch.IluMoEBatchedExpertsW4A8)
    expert.quant_mode = "w4a8"
    expert.version, expert.group_size, expert.format, expert.out_dtype = (
        2,
        -1,
        "NN",
        torch.bfloat16,
    )
    expert.quant_config = SimpleNamespace(
        w1_scale=torch.ones(2, 128),
        w2_scale=torch.ones(2, 64),
        g1_alphas=None,
        g2_alphas=None,
        w1_zp=None,
        w2_zp=None,
        w1_bias=None,
        w2_bias=None,
    )
    expert.swiglu_limit = None
    x = torch.ones(4, 64, dtype=torch.int8)
    w1 = torch.zeros(2, 64, 64, dtype=torch.int8)
    w2 = torch.zeros(2, 64, 32, dtype=torch.int8)
    output = torch.full((4, 64), 7, dtype=torch.bfloat16)
    dense = torch.zeros(5, dtype=torch.uint32)
    monkeypatch.setenv("VLLM_ILU_MOE_W4A8_GEMV_PERF", "1")

    monkeypatch.setattr(
        batch.ixff,
        "moe_group_gemv_reformat_input_fused_dense",
        lambda **kw: (kw["input"], kw["input_scales"], dense),
    )
    monkeypatch.setattr(
        batch.ixff,
        "moe_group_gemv_swiglu_quant",
        lambda **kw: (torch.zeros(4, 64, dtype=torch.int8), torch.ones(4)),
    )
    monkeypatch.setattr(
        gemv.registry,
        "available_backends",
        lambda op, arch: ["ixformer_perf", "ixinfer"],
    )
    monkeypatch.setattr(
        gemv.ixf, "moe_w4a8_group_gemv_perf_supported", lambda **kw: kw["weight"] is w1
    )
    calls = []

    def run(**kw):
        calls.append(kw)
        out = kw.get("output", torch.zeros(4, 128, dtype=torch.bfloat16))
        out.fill_(3)
        return out

    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", run)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    result = expert.apply_group_gemv(
        output,
        x,
        w1,
        w2,
        torch.ones(1, 1),
        ids,
        None,
        2,
        None,
        torch.ones(4),
        None,
        None,
        None,
    )
    # The existing batched method reshapes output and returns that view.
    assert result.data_ptr() == output.data_ptr()
    assert torch.equal(output, torch.full_like(output, 3))
    assert [kw["backend"] for kw in calls] == ["ixformer_perf", "ixinfer"]
    assert calls[0]["weight"] is w1 and calls[1]["weight"] is w2
    assert calls[0]["output_dtype"] == calls[1]["output_dtype"] == torch.bfloat16
    assert (
        calls[0]["dense_tokens_per_experts"]
        is calls[1]["dense_tokens_per_experts"]
        is dense
    )
    assert calls[1]["output"] is result


@pytest.mark.parametrize("operand", ["weight", "w_scales", "output"])
def test_nn_cross_device_inputs_are_not_sent_to_perf(monkeypatch, operand):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    mode = FakeTensorMode()

    def fake(shape, dtype, device="cuda:0"):
        return FakeTensor(
            mode, torch.empty(shape, dtype=dtype, device="meta"), torch.device(device)
        )

    kwargs = dict(
        input=fake((128, 5120), torch.int8),
        weight=fake((8, 5120, 2304), torch.int8),
        i_scales=fake((128,), torch.float32),
        w_scales=fake((8, 4608), torch.float32),
        output_dtype=torch.bfloat16,
        tokens_per_experts=fake((8,), torch.int32),
        dense_tokens_per_experts=fake((17,), torch.uint32),
        output=fake((128, 4608), torch.bfloat16),
        format=0,
    )
    assert gemv.ixf.moe_w4a8_group_gemv_perf_supported(**kwargs)
    original = kwargs[operand]
    kwargs[operand] = fake(original.shape, original.dtype, "cuda:1")
    assert not gemv.ixf.moe_w4a8_group_gemv_perf_supported(**kwargs)
    monkeypatch.setattr(
        gemv.registry,
        "available_backends",
        lambda op, arch: ["ixformer_perf", "ixinfer"],
    )
    execute = MagicMock()
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", execute)
    monkeypatch.setenv("VLLM_ILU_MOE_W4A8_GEMV_PERF", "1")
    gemv.w4a8_group_gemv(**kwargs)
    execute.assert_called_once_with(**kwargs, backend="ixinfer")


@pytest.fixture(autouse=True)
def _clear_nn_perf_policy_cache():
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    gemv._nn_backends.cache_clear()
    gemv._nn_perf_enabled.cache_clear()
    yield
    gemv._nn_backends.cache_clear()
    gemv._nn_perf_enabled.cache_clear()


@pytest.mark.parametrize(
    "setting,enabled",
    [(None, False), ("0", False), ("false", False), ("1", True), (" true ", True)],
)
def test_nn_perf_opt_in_defaults_off_and_preserves_arguments(
    monkeypatch, setting, enabled
):
    from vllm_iluvatar import envs as ilu_envs
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    key = "VLLM_ILU_MOE_W4A8_GEMV_PERF"
    if setting is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, setting)
    assert ilu_envs._ILU_CANONICAL_ONLY_ENV_VARS[key]() is enabled
    monkeypatch.setattr(
        gemv.registry,
        "available_backends",
        lambda op, arch: ["ixformer_perf", "ixinfer"],
    )
    query = MagicMock(return_value=True)
    execute = MagicMock(return_value="output")
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv_perf_supported", query)
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", execute)
    weight, metadata = object(), object()
    assert (
        gemv.w4a8_group_gemv(format=0, weight=weight, dense_tokens_per_experts=metadata)
        == "output"
    )
    assert execute.call_args.kwargs == dict(
        format=0,
        weight=weight,
        dense_tokens_per_experts=metadata,
        backend="ixformer_perf" if enabled else "ixinfer",
    )
    assert query.call_count == int(enabled)


def test_disabled_perf_is_not_an_implicit_fallback_when_vendor_missing(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    monkeypatch.delenv("VLLM_ILU_MOE_W4A8_GEMV_PERF", raising=False)
    monkeypatch.setattr(
        gemv.registry, "available_backends", lambda op, arch: ["ixformer_perf"]
    )
    query, execute = MagicMock(), MagicMock()
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv_perf_supported", query)
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", execute)
    with pytest.raises(RuntimeError, match="perf opt-in"):
        gemv.w4a8_group_gemv(format=0)
    query.assert_not_called()
    execute.assert_not_called()


@pytest.mark.parametrize("initial", ["0", "1"])
def test_nn_perf_opt_in_is_frozen_after_first_call(monkeypatch, initial):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_w4a8_gemv as gemv,
    )

    key = "VLLM_ILU_MOE_W4A8_GEMV_PERF"
    monkeypatch.setenv(key, initial)
    available = MagicMock(return_value=["ixformer_perf", "ixinfer"])
    monkeypatch.setattr(gemv.registry, "available_backends", available)
    monkeypatch.setattr(
        gemv.ixf, "moe_w4a8_group_gemv_perf_supported", lambda **kw: True
    )
    execute = MagicMock()
    monkeypatch.setattr(gemv.ixf, "moe_w4a8_group_gemv", execute)
    gemv.w4a8_group_gemv(format=0)
    monkeypatch.setenv(key, "1" if initial == "0" else "0")
    gemv.w4a8_group_gemv(format=0)
    expected = "ixformer_perf" if initial == "1" else "ixinfer"
    assert [call.kwargs["backend"] for call in execute.call_args_list] == [
        expected,
        expected,
    ]
    available.assert_called_once()
