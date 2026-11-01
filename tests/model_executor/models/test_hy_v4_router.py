# SPDX-License-Identifier: Apache-2.0
"""HY4 router loading, backend wiring and FP32 output on real devices."""

from unittest.mock import patch

import ixformer.inference.functions as ixf_ops
import pytest
import torch
from ixformer.inference.backend_registry import registry
from vllm import distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_iluvatar.models.hy_v4 import moe


@pytest.fixture(scope="module", autouse=True)
def distributed_gate(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("requires an Iluvatar GPU")
    with set_current_vllm_config(VllmConfig()):
        torch.cuda.set_device(0)
        dist.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method="file://"
            + str(tmp_path_factory.mktemp("router") / "init"),
        )
        dist.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        try:
            yield
        finally:
            dist.destroy_model_parallel()
            dist.destroy_distributed_environment()


def make_gate():
    gate = moe.HYV4RouterGate(6144, 256, "test.gate").cuda()
    generator = torch.Generator(device="cuda").manual_seed(42)
    weight = (
        torch.randn((256, 6144), generator=generator, device="cuda") * 0.02
    ).bfloat16()
    gate.weight_loader(gate.weight, weight)
    gate.quant_method.process_weights_after_loading(gate)
    return gate, weight


@pytest.mark.parametrize(
    "arch,backend", [("ivcore11", "ixinfer"), ("ivcore30", "ixmm")]
)
def test_arch_dispatch_keeps_bf16_operands_and_fp32_logits(
    default_vllm_config, arch, backend
):
    # This checks wiring for both branches, not CoreX11 device numerics.
    with (
        patch.object(moe, "cuda_arch", return_value=arch),
        patch.object(registry, "available_backends", return_value=[backend]),
    ):
        gate, _ = make_gate()
    x = torch.randn((2, 6144), dtype=torch.bfloat16, device="cuda")

    def capture(a, w, **kwargs):
        assert a is x and a.dtype == w.dtype == torch.bfloat16
        assert kwargs == {"format": "TN", "backend": backend}
        return torch.nn.functional.linear(a.float(), w.float())

    with patch.object(ixf_ops, "gemm_bf16_bf16_fp32", side_effect=capture) as call:
        out, bias = gate(x)
    assert call.call_count == 1 and out.dtype == torch.float32 and bias is None


@pytest.mark.parametrize("m", [2048, 725, 635])
@torch.inference_mode()
def test_real_backend_matches_ixinfer_at_prefill_sizes(default_vllm_config, m):
    gate, weight = make_gate()
    x = torch.randn((m, 6144), dtype=torch.bfloat16, device="cuda")
    out, bias = gate(x)
    reference = ixf_ops.gemm_bf16_bf16_fp32(x, weight, backend="ixinfer")
    assert out.dtype == torch.float32 and bias is None
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-4)
    assert gate.weight.dtype == torch.float32 and gate.weight.is_inference()


def test_load_invalidates_cache_without_serializing_it(default_vllm_config):
    gate, weight = make_gate()
    assert "_router_weight_bf16" not in gate.state_dict()
    source = gate.weight
    gate.weight_loader(gate.weight, weight * 2)
    assert gate._router_weight_bf16 is None and gate.weight is source
    gate.quant_method.process_weights_after_loading(gate)
    torch.testing.assert_close(gate._router_weight_bf16, weight * 2, atol=0, rtol=0)
    with torch.no_grad():
        gate.weight.add_(1)
    x = torch.randn((2, 6144), dtype=torch.bfloat16, device="cuda")
    with patch.object(
        ixf_ops, "gemm_bf16_bf16_fp32", side_effect=AssertionError("stale cache")
    ):
        out, _ = gate(x)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x.float(), gate.weight), atol=0, rtol=0
    )


def test_true_fp32_weight_keeps_original_compute(default_vllm_config):
    gate, weight = make_gate()
    precise = weight.float()
    precise[0, 0] += 1e-6
    gate.weight_loader(gate.weight, precise)
    gate.quant_method.process_weights_after_loading(gate)
    assert gate._router_weight_bf16 is None
    x = torch.randn((2, 6144), dtype=torch.bfloat16, device="cuda")
    with patch.object(
        ixf_ops, "gemm_bf16_bf16_fp32", side_effect=AssertionError("rounded FP32 W")
    ):
        out, _ = gate(x)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x.float(), precise), atol=0, rtol=0
    )


def test_required_backend_missing_is_an_error(default_vllm_config):
    gate, _ = make_gate()
    with (
        patch.object(registry, "available_backends", return_value=["torch"]),
        pytest.raises(RuntimeError, match="requires backend"),
    ):
        gate.prepare_router_weight()
