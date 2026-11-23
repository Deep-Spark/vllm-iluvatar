# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed experts converted to the IxInfer W4A8 contract on ivcore11.

The load-bearing claim is that NVFP4 restated as block-FP8 on a ``[1, 16]``
grid is the *same tensor*, not an approximation: e2m1 magnitudes are exact in
float8_e4m3fn and the two scale levels multiply into one. These tests pin that
restatement, the conversion against the converter's own reference, and the
properties of the shared method that the W4A8 target depends on.
"""

import pytest
import torch

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.nvfp4_to_w4a8 import (
    _E2M1_MAGNITUDES,
    NVFP4_BLOCK_SHAPE,
    convert_nvfp4_to_w4a8,
    nvfp4_source_as_block_fp8,
)

NN_SPEC = MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18)


def _nvfp4_source(experts, n, k, device="cpu"):
    """Random packed NVFP4 weights with their block and global scales."""
    torch.manual_seed(7)
    weight = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device
    )
    block = (torch.rand(experts, n, k // 16, device=device) * 2 + 0.25).to(
        torch.float8_e4m3fn
    )
    global_scale = torch.rand(experts, device=device) * 0.05 + 0.01
    return weight, block, global_scale


def test_e2m1_magnitudes_are_exact_in_e4m3():
    """The restatement is only lossless because this holds."""
    values = torch.tensor(
        list(_E2M1_MAGNITUDES) + [-v for v in _E2M1_MAGNITUDES], dtype=torch.float32
    )
    assert torch.equal(values, values.to(torch.float8_e4m3fn).float())


def test_restatement_reproduces_the_scaled_source_exactly():
    experts, n, k = 2, 64, 128
    weight, block, _ = _nvfp4_source(experts, n, k)
    values, scale = nvfp4_source_as_block_fp8(weight, block)
    assert values.dtype is torch.float8_e4m3fn
    assert values.shape == (experts, n, k)
    assert scale.shape == (experts, n, k // 16)

    # Independent unpack: low nibble first, sign bit 0x08, magnitude bits 0x07.
    table = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float64)
    nibbles = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2)
    expected = table[(nibbles & 0x07).long()]
    expected = torch.where(nibbles & 0x08 != 0, -expected, expected)
    expected = expected * block.float().double().repeat_interleave(16, -1)
    restated = values.float().double() * scale.double().repeat_interleave(16, -1)
    assert torch.equal(restated, expected)


@pytest.mark.parametrize("experts,n,k", [(2, 64, 128), (1, 256, 512)])
def test_conversion_matches_the_converter_reference(experts, n, k):
    from ixformer.inference.backend_registry import registry

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    if cuda_arch() != "ivcore11":
        pytest.skip("NVFP4 experts convert to IxInfer W4A8 on ivcore11")
    weight, block, global_scale = _nvfp4_source(experts, n, k, device="cuda")
    weight_shape, scale_shape = NN_SPEC.shapes(experts, n, k)
    out_weight = torch.empty(weight_shape, dtype=torch.int8, device="cuda")
    out_scale = torch.empty(scale_shape, dtype=torch.float32, device="cuda")
    convert_nvfp4_to_w4a8(
        weight, block, global_scale, out_weight, out_scale, weight_format="ixinfer_nn"
    )

    values, scale = nvfp4_source_as_block_fp8(weight, block)
    ref = registry.get_reference_impl("fp8_block_to_w4a8")
    expected = torch.empty_like(out_weight)
    expected_scale = torch.empty_like(out_scale)
    for e in range(experts):
        ref(
            values[e],
            scale[e],
            expected[e],
            expected_scale[e],
            block_shape=NVFP4_BLOCK_SHAPE,
            weight_format="ixinfer_nn",
        )
    expected_scale = expected_scale * global_scale.reshape(-1, 1, 1)
    assert torch.equal(out_weight, expected)
    torch.testing.assert_close(out_scale, expected_scale, rtol=1e-6, atol=0)


def test_int4_codes_do_not_depend_on_the_global_scale():
    """Why the global is applied to the channel scale instead of folded in.

    It is a uniform factor, so it cancels in the per-channel ``amax / 127``
    normalization; keeping it out of the block scale means no FP32 rounding
    can reach the codes.
    """
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    if cuda_arch() != "ivcore11":
        pytest.skip("requires the ivcore11 native converter")
    experts, n, k = 2, 64, 128
    weight, block, global_scale = _nvfp4_source(experts, n, k, device="cuda")
    weight_shape, scale_shape = NN_SPEC.shapes(experts, n, k)
    codes = []
    scales = []
    for factor in (1.0, 1e-4):
        out_weight = torch.empty(weight_shape, dtype=torch.int8, device="cuda")
        out_scale = torch.empty(scale_shape, dtype=torch.float32, device="cuda")
        convert_nvfp4_to_w4a8(
            weight,
            block,
            global_scale * factor,
            out_weight,
            out_scale,
            weight_format="ixinfer_nn",
        )
        codes.append(out_weight)
        scales.append(out_scale)
    assert torch.equal(codes[0], codes[1])
    torch.testing.assert_close(scales[0] * 1e-4, scales[1], rtol=1e-6, atol=0)


def test_whole_w13_conversion_equals_converting_the_halves():
    """The NN packing tiles N by 32, so a gate/up split cannot straddle a tile.

    The method converts ``[2I, K]`` in one call rather than per shard; this is
    what makes that safe.
    """
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    if cuda_arch() != "ivcore11":
        pytest.skip("requires the ivcore11 native converter")
    experts, half, k = 1, 96, 128
    n = 2 * half
    weight, block, global_scale = _nvfp4_source(experts, n, k, device="cuda")
    weight_shape, scale_shape = NN_SPEC.shapes(experts, n, k)
    whole = torch.empty(weight_shape, dtype=torch.int8, device="cuda")
    whole_scale = torch.empty(scale_shape, dtype=torch.float32, device="cuda")
    convert_nvfp4_to_w4a8(
        weight, block, global_scale, whole, whole_scale, weight_format="ixinfer_nn"
    )

    split = torch.empty_like(whole)
    split_scale = torch.empty_like(whole_scale)
    for offset in (0, half):
        part_shape, part_scale_shape = NN_SPEC.shapes(experts, half, k)
        part = torch.empty(part_shape, dtype=torch.int8, device="cuda")
        part_scale = torch.empty(part_scale_shape, dtype=torch.float32, device="cuda")
        convert_nvfp4_to_w4a8(
            weight[:, offset : offset + half],
            block[:, offset : offset + half],
            global_scale,
            part,
            part_scale,
            weight_format="ixinfer_nn",
        )
        split[..., offset // 2 : offset // 2 + half // 2] = part
        split_scale[..., offset : offset + half] = part_scale
    assert torch.equal(whole, split)
    torch.testing.assert_close(whole_scale, split_scale, rtol=0, atol=0)


def test_class_name_still_selects_the_modelopt_loading_protocol():
    """``ilu_routed_experts.weight_loader`` dispatches NVFP4 on the class name.

    It checks ``"ModelOpt" in self.quant_method.__class__.__name__`` to pick
    the NVFP4 scale protocol. Renaming this class without that substring sends
    the block ``w13_weight_scale`` into the generic per-scale branch, which
    raises "quant method must be one of ..." because community's
    ``create_weights`` attaches no ``quant_method`` attribute to it.
    """
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
        IluModelOptNvFp4MoEMethod,
    )

    assert "ModelOpt" in IluModelOptNvFp4MoEMethod.__name__


def test_k_alignment_follows_the_conversion_target(default_vllm_config):
    """The requirement is the converter's, so it moves with the selection."""
    from vllm_iluvatar.ilu_util.arch import cuda_arch
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
        IluModelOptNvFp4MoEMethod,
    )

    method = object.__new__(IluModelOptNvFp4MoEMethod)
    method.selection = None
    assert method.k_alignment == (256 if cuda_arch() == "ivcore30" else 32)


def test_the_post_conversion_lifecycle_comes_from_the_mixin():
    """Nothing after the conversion may stay NVFP4-specific."""
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
        IluModelOptNvFp4MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
        MoEComputeBindingMixin,
    )

    for name in (
        "apply",
        "get_fused_moe_quant_config",
        "maybe_make_prepare_finalize",
        "select_gemm_impl",
    ):
        assert getattr(IluModelOptNvFp4MoEMethod, name) is getattr(
            MoEComputeBindingMixin, name
        )
