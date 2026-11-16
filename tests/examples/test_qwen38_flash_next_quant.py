# SPDX-License-Identifier: Apache-2.0
"""Flash-Next export contract: real expert dimensions and all TN/v2 scales."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples/iluvatar-compressed-tensors"
_PROFILE = "qwen/qwen38_flash_next_bi150_w4a8_w8a8.json"


@pytest.fixture
def quantizer():
    sys.path.insert(0, str(_EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(
            "flash_next_quant", _EXAMPLES / "quant.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(_EXAMPLES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
def test_real_expert_dimensions_export_all_v2_scales(tmp_path, quantizer):
    from types import SimpleNamespace

    from compressed_tensors.quantization import QuantizationStrategy

    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8 import (
        IluCompressedTensorsW4A8Int8MoEMethod,
    )

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    prefix = "model.language_model.layers.2."
    tensors = {
        prefix + "mlp.experts.gate_up_proj": torch.randn(2, 1280, 2560).bfloat16(),
        prefix + "mlp.experts.down_proj": torch.randn(2, 2560, 640).bfloat16(),
        prefix + "mlp.shared_expert.gate_proj.weight": torch.randn(
            640, 2560
        ).bfloat16(),
        prefix + "ple.ple_embedding.ngram_embedding.shard_0.weight": torch.randn(
            9, 160
        ).bfloat16(),
        prefix + "ple.key_proj.weight": torch.randn(16, 32).bfloat16(),
        prefix + "linear_attn.in_proj_a.weight": torch.randn(48, 32).bfloat16(),
    }
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text("{}")
    quantizer.convert(
        str(source),
        str(output),
        str(_EXAMPLES / "configs" / _PROFILE),
        str(_EXAMPLES / "rules" / _PROFILE),
    )
    exported = load_file(output / "model.safetensors")
    method = IluCompressedTensorsW4A8Int8MoEMethod(
        weight_quant=SimpleNamespace(
            group_size=None, strategy=QuantizationStrategy.CHANNEL, symmetric=True
        ),
        input_quant=SimpleNamespace(strategy=QuantizationStrategy.TOKEN, dynamic=True),
        moe=SimpleNamespace(has_bias=False),
        ilu_params={"gemm_format": "TN", "pack_version": 2},
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_routed_experts import (
        IluRoutedExperts,
    )

    layer = IluRoutedExperts.__new__(IluRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = prefix + "mlp.experts"
    layer.quant_method = method
    layer.quant_config = None
    layer.moe_config = SimpleNamespace(
        hidden_dim_unpadded=2560,
        hidden_dim=2560,
        intermediate_size=640,
        tp_rank=0,
        is_act_and_mul=True,
        tp_shard_with_padding=False,
        moe_parallel_config=SimpleNamespace(tp_size=1),
    )
    layer._map_global_expert_id_to_local_expert_id = lambda index: index
    mapping = layer.build_expert_params_mapping(
        "gate_proj",
        "down_proj",
        "up_proj",
        2,
        routed_experts_prefix="",
        include_fused=True,
    )
    layer.get_expert_mapping = lambda include_fused: mapping
    method.create_weights(
        layer, 2, 2560, 640, torch.bfloat16, weight_loader=layer.weight_loader
    )
    for parameter in layer.parameters():
        parameter.data.zero_()
    loaded = set(
        layer.load_weights(
            (name.removeprefix(layer.layer_name + "."), value)
            for name, value in exported.items()
            if name.startswith(layer.layer_name + ".")
        )
    )
    assert loaded == {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"}
    for checkpoint, runtime in (("gate_up_proj", "w13"), ("down_proj", "w2")):
        for source_suffix, parameter_suffix in (
            ("", "weight"),
            ("_scale", "weight_scale"),
            ("_i8_weight_scale", "i8_weight_scale"),
            ("_i8_weight_zero", "i8_weight_zero"),
        ):
            parameter = getattr(layer, runtime + "_" + parameter_suffix)
            key = prefix + "mlp.experts." + checkpoint + source_suffix
            if parameter is None:
                assert key not in exported
                continue
            value = exported[key]
            assert value.shape == parameter.shape
            assert value.dtype == parameter.dtype
            assert torch.isfinite(value.float()).all()
            torch.testing.assert_close(parameter, value, atol=0, rtol=0)
    table = prefix + "ple.ple_embedding.ngram_embedding.shard_0"
    assert exported[table + ".weight"].dtype == torch.int8
    assert exported[table + ".weight_scale"].shape == (9, 1)
    restored = exported[table + ".weight"].float() * exported[table + ".weight_scale"]
    torch.testing.assert_close(
        restored, tensors[table + ".weight"].float(), atol=0.025, rtol=0
    )
    assert exported[prefix + "mlp.shared_expert.gate_proj.weight"].dtype == torch.int8
    for name in (
        prefix + "ple.key_proj.weight",
        prefix + "linear_attn.in_proj_a.weight",
    ):
        assert torch.equal(exported[name], tensors[name])
    assert not any(str(t.dtype).startswith("torch.float8") for t in exported.values())
    config = json.loads((output / "config.json").read_text())
    assert (
        config["quantization_config"]["config_groups"]["routed_experts"]["iluvatar"][
            "pack_version"
        ]
        == 2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
@pytest.mark.parametrize(
    "projection,n,k", [("gate_up", 1280, 2560), ("down", 2560, 640)]
)
@pytest.mark.parametrize("tokens_per_expert", [1, 17])
@torch.inference_mode()
def test_real_dimensions_w4a8_gemm_and_gemv(
    quantizer, projection, n, k, tokens_per_expert
):
    import ixformer.inference.functions as ops
    from ixformer.inference.backend_registry import registry

    torch.manual_seed(23)
    experts = 2
    weight = torch.randn(experts, n, k, device="cuda") * 0.02
    options = {
        "group_size": -1,
        "symmetric": True,
        "gemm_format": "TN",
        "pack_version": 2,
        "scale_layout": "row",
        "packed": projection == "gate_up",
        "chunk_dim": -2,
    }
    packed = quantizer.QUANT_METHODS["pack_int4_moe"](weight, options)
    inputs = torch.randn(
        experts * tokens_per_expert, k, device="cuda", dtype=torch.bfloat16
    )
    quant_input, input_scale = ops.scaled_int8_quant(inputs)
    sizes = torch.full((experts,), tokens_per_expert, dtype=torch.int32)
    args = dict(
        input=quant_input,
        weight=packed.weight,
        i_scales=input_scale.flatten(),
        w_scales=packed.weight_scale.reshape(experts, n),
        output_dtype=torch.bfloat16,
        tokens_per_experts=sizes,
        format=2,
        version=2,
        group_size=-1,
    )
    ref = registry.get_reference_impl("moe_w4a8_group_gemm")(**args)
    actual = ops.moe_w4a8_group_gemm(**args, backend="ixinfer")
    error = (actual.float() - ref.float()).norm() / ref.float().norm()
    assert error < 0.01
    dense = torch.cat(
        [
            inputs[i * tokens_per_expert : (i + 1) * tokens_per_expert].float()
            @ weight[i].T
            for i in range(experts)
        ]
    )
    quant_error = (ref.float() - dense).norm() / dense.norm()
    assert quant_error < 0.3

    gpu_sizes = sizes.cuda()
    gemv_input, gemv_scale, dense_sizes = ops.moe_group_gemv_reformat_input_fused_dense(
        quant_input,
        input_scale.flatten(),
        gpu_sizes,
        tokens_per_expert,
    )
    gemv_args = {
        **args,
        "input": gemv_input,
        "i_scales": gemv_scale,
        "tokens_per_experts": gpu_sizes,
        "dense_tokens_per_experts": dense_sizes,
    }
    # The reference consumes logical row-major activations; the native GEMV
    # consumes the reordered buffer returned above.
    ref_gemv = registry.get_reference_impl("moe_w4a8_group_gemv")(**args)
    actual_gemv = ops.moe_w4a8_group_gemv(**gemv_args, backend="ixinfer")
    gemv_error = (
        actual_gemv.float() - ref_gemv.float()
    ).norm() / ref_gemv.float().norm()
    assert gemv_error < 0.01
    print(
        f"{projection} tokens/expert={tokens_per_expert}: "
        f"GEMM_NRMSE={error.item():.6g}, GEMV_NRMSE={gemv_error.item():.6g}, "
        f"quantization_NRMSE={quant_error.item():.6g}"
    )
