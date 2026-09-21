# SPDX-License-Identifier: Apache-2.0
"""CPU-only byte-exact tests for TN checkpoint to NN runtime repacking."""

import pytest
import torch

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_w4a8_layout import (  # noqa: E501
    repack_w4a8_tn_to_nn,
)


def _pack_reference(raw: torch.Tensor, layout: str) -> torch.Tensor:
    experts, n, k = raw.shape
    if layout == "NN":
        raw = (
            raw.transpose(-2, -1)
            .contiguous()
            .view(experts, k // 32, 2, 16, n // 32, 2, 16)
            .permute(0, 1, 5, 3, 4, 2, 6)
            .contiguous()
            .view(experts, k, n)
        )
    rows, columns = raw.shape[-2:]
    pairs = raw.view(experts, rows, columns // 32, 32)
    return (
        (pairs[..., :16] & 0x0F) | (pairs[..., 16:] << 4)
    ).reshape(experts, rows, columns // 2).contiguous()


@pytest.mark.parametrize("shape", [(2, 128, 256), (3, 256, 128), (1, 144, 192), (1, 129, 257)])
def test_tn_to_nn_matches_reference_with_padding(shape):
    experts, n, k = shape
    torch.manual_seed(19 + n + k)
    raw = torch.randint(-8, 8, shape, dtype=torch.int8)
    tn_n = (n + 15) // 16 * 16
    tn_k = max(256, (k + 127) // 128 * 128)
    nn_n = (n + 127) // 128 * 128
    nn_k = max(256, (k + 63) // 64 * 64)
    tn_plain = torch.zeros((experts, tn_n, tn_k), dtype=torch.int8)
    tn_plain[:, :n, :k] = raw
    nn_plain = torch.zeros((experts, nn_n, nn_k), dtype=torch.int8)
    nn_plain[:, :n, :k] = raw

    actual = repack_w4a8_tn_to_nn(
        _pack_reference(tn_plain, "TN"),
        logical_n=n,
        logical_k=k,
        nn_n=nn_n,
        nn_k=nn_k,
    )
    expected = _pack_reference(nn_plain, "NN")
    assert actual.shape == (experts, nn_k, nn_n // 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tn_to_nn_rejects_wrong_source_shape():
    with pytest.raises(ValueError, match="packed_tn"):
        repack_w4a8_tn_to_nn(
            torch.zeros((1, 128, 128), dtype=torch.float32),
            logical_n=128,
            logical_k=256,
            nn_n=128,
            nn_k=256,
        )


def test_tn_to_nn_matches_ixformer_v2_packer():
    from ixformer.inference.functions.moe import quant_repack_int4

    torch.manual_seed(42)
    raw = torch.randint(-8, 8, (2, 128, 256), dtype=torch.int8).float() * 18
    tn, _, _ = quant_repack_int4(raw, group_size=-1, version=2, format="TN")
    expected, _, _ = quant_repack_int4(
        raw, group_size=-1, version=2, format="NN"
    )
    actual = repack_w4a8_tn_to_nn(
        tn,
        logical_n=128,
        logical_k=256,
        nn_n=128,
        nn_k=256,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loaded_w4a8_weights_repack_without_requantization(monkeypatch):
    from ixformer.inference.functions.moe import quant_repack_int4

    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
        W4A8GroupGemmInfo,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8 import (  # noqa: E501
        IluCompressedTensorsW4A8Int8MoEMethod,
    )

    monkeypatch.setenv("VLLM_ILU_MOE_EP_LL_MOE_ALGO", "GEMV")
    method = IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.gemm_format = "TN"
    method.runtime_gemm_format = "NN"
    method.version = 2
    method.group_size = -1
    method.weight_symmetric = True
    method.has_bias = False
    method.static_input_scales = False
    method.gemm1_info = W4A8GroupGemmInfo("TN", k=256, n=128)
    method.gemm2_info = W4A8GroupGemmInfo("TN", k=256, n=256)

    layer = torch.nn.Module()
    expected = {}
    for name, n in (("w13", 128), ("w2", 256)):
        raw = torch.randint(-8, 8, (2, n, 256), dtype=torch.int8).float() * 18
        tn, _, _ = quant_repack_int4(raw, group_size=-1, version=2, format="TN")
        nn, _, _ = quant_repack_int4(raw, group_size=-1, version=2, format="NN")
        layer.register_parameter(
            name + "_weight", torch.nn.Parameter(tn, requires_grad=False)
        )
        layer.register_parameter(
            name + "_weight_scale",
            torch.nn.Parameter(torch.ones((2, 1, n)), requires_grad=False),
        )
        expected[name] = nn

    method.process_weights_after_loading(layer)
    assert method.gemm_format == "NN"
    assert method.gemm1_info.format == method.gemm2_info.format == 0
    for name, shard_dim in (("w13", 1), ("w2", 0)):
        actual = getattr(layer, name + "_weight")
        assert actual.shard_dim == shard_dim
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)


def test_nn_experts_use_logical_output_width():
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
        W4A8GroupGemmInfo,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_batch_experts_w4a8 import (  # noqa: E501
        IluMoEBatchedExpertsW4A8,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_standard_experts_w4a8 import (  # noqa: E501
        IluMoEStandardExpertsW4A8,
    )

    for cls in (IluMoEBatchedExpertsW4A8, IluMoEStandardExpertsW4A8):
        experts = cls.__new__(cls)
        experts.gemm1_info = W4A8GroupGemmInfo("NN", k=4096, n=2048)
        dimensions = experts.moe_problem_size(
            torch.empty((8, 4096)),
            torch.empty((16, 4096, 1024)),
            torch.empty((16, 2048, 2048)),
            torch.empty((8, 6), dtype=torch.int32),
        )
        assert dimensions == (16, 8, 2048, 4096, 6)
