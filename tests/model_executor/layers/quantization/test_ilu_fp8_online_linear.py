# SPDX-License-Identifier: Apache-2.0
"""Online FP8 dense-linear loading, fallback, and dispatch tests."""

from __future__ import annotations

import logging

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_iluvatar.model_executor.layers.quantization import ilu_fp8


@pytest.fixture(autouse=True)
def _single_rank_parameter_state(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _method() -> ilu_fp8.IluFp8PerChannelOnlineLinearMethod:
    # __init__ reads the active VllmConfig. The loading mechanics under test do
    # not depend on it, so keep these unit tests isolated from engine config.
    method = object.__new__(ilu_fp8.IluFp8PerChannelOnlineLinearMethod)
    method.gemm_format = "TN"
    method._unquantized_method = UnquantizedLinearMethod()
    return method


def _copy_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)


def _layer(prefix: str = "model.layers.0.proj") -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.prefix = prefix
    return layer


def _materialized_weight(meta_weight, device, dtype):
    weight = ModelWeightParameter(
        data=torch.empty(tuple(meta_weight.shape), device=device, dtype=dtype),
        input_dim=meta_weight.input_dim,
        output_dim=meta_weight.output_dim,
        weight_loader=meta_weight.weight_loader,
    )
    weight.tp_rank = meta_weight.tp_rank
    weight.tp_size = meta_weight.tp_size
    return weight


def test_shape_fallback_preserves_original_dtype_and_warns(monkeypatch, caplog):
    method = _method()
    layer = _layer("model.layers.0.mlp.shared_expert.down_proj")
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)

    with caplog.at_level(logging.WARNING):
        method.create_weights(
            layer,
            input_size_per_partition=160,
            output_partition_sizes=[2560],
            input_size=640,
            output_size=2560,
            params_dtype=torch.bfloat16,
            weight_loader=_copy_loader,
        )

    assert layer.weight.is_meta
    assert layer.weight.dtype == torch.bfloat16
    assert "K=160 is not divisible by 64" in layer._ilu_fp8_online_fallback_reason
    assert "fallback to torch.bfloat16" in caplog.text


def test_gemv_n_alignment_uses_bf16_fallback(monkeypatch):
    method = _method()
    layer = _layer("model.layers.0.linear_attn.in_proj_ba")
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)

    method.create_weights(
        layer,
        input_size_per_partition=2560,
        output_partition_sizes=[12, 12],
        input_size=2560,
        output_size=96,
        params_dtype=torch.bfloat16,
        weight_loader=_copy_loader,
    )

    assert layer.weight.dtype == torch.bfloat16
    assert layer._ilu_fp8_online_fallback_reason == (
        "N=24 is not divisible by 32 for GEMV"
    )


def test_meta_counting_does_not_allocate_staging(monkeypatch):
    method = _method()
    layer = _layer()
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=_copy_loader,
    )

    layer.weight.weight_loader(
        layer.weight, torch.empty((32, 64), dtype=torch.bfloat16)
    )

    assert layer.weight.is_meta
    assert "_ilu_fp8_staging" not in layer.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_loader_quantizes_into_stable_fp8_buffer_bit_exact(monkeypatch):
    method = _method()
    layer = _layer()
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=_copy_loader,
    )

    meta_weight = layer.weight
    layer.weight = _materialized_weight(meta_weight, "cuda", torch.float8_e4m3fn)
    source = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, source)
    expected_weight, expected_scale = ilu_fp8.fp8_per_token_dynamic_scaled(source)
    weight_ptr = layer.weight.data_ptr()

    method.process_weights_after_loading(layer)

    assert layer.weight.data_ptr() == weight_ptr
    assert torch.equal(layer.weight.float(), expected_weight.float())
    assert torch.equal(layer.weight_scale, expected_scale)
    assert "_ilu_fp8_staging" not in layer.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("loader_kind", ["qkv", "merged", "row_parallel"])
def test_fragmented_and_tp_loaders_quantize_only_after_assembly(
    monkeypatch, loader_kind
):
    method = _method()
    layer = _layer(f"model.layers.0.{loader_kind}")
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)

    if loader_kind == "row_parallel":

        def loader(param, loaded_weight):
            width = param.shape[1]
            local = loaded_weight.narrow(1, param.tp_rank * width, width)
            param.data.copy_(local)

    else:

        def loader(param, loaded_weight, shard_offset):
            target = param.data.narrow(0, shard_offset, loaded_weight.shape[0])
            target.copy_(loaded_weight)

    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=128 if loader_kind == "row_parallel" else 64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    meta_weight = layer.weight
    if loader_kind == "row_parallel":
        meta_weight.tp_rank = 1
        meta_weight.tp_size = 2
    layer.weight = _materialized_weight(meta_weight, "cuda", torch.float8_e4m3fn)

    if loader_kind == "qkv":
        shards = [
            torch.randn((16, 64), device="cuda", dtype=torch.bfloat16),
            torch.randn((8, 64), device="cuda", dtype=torch.bfloat16),
            torch.randn((8, 64), device="cuda", dtype=torch.bfloat16),
        ]
        offsets = [0, 16, 24]
        for shard, offset in zip(shards, offsets):
            layer.weight.weight_loader(layer.weight, shard, shard_offset=offset)
        assembled = torch.cat(shards)
    elif loader_kind == "merged":
        shards = [
            torch.randn((16, 64), device="cuda", dtype=torch.bfloat16),
            torch.randn((16, 64), device="cuda", dtype=torch.bfloat16),
        ]
        for offset, shard in zip((0, 16), shards):
            layer.weight.weight_loader(layer.weight, shard, shard_offset=offset)
        assembled = torch.cat(shards)
    else:
        full_weight = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, full_weight)
        assembled = full_weight[:, 64:]

    expected_weight, expected_scale = ilu_fp8.fp8_per_token_dynamic_scaled(assembled)
    method.process_weights_after_loading(layer)

    assert torch.equal(layer.weight.float(), expected_weight.float())
    assert torch.equal(layer.weight_scale, expected_scale)


def test_missing_staging_fails_instead_of_silent_fp8_cast():
    method = _method()
    layer = _layer()
    layer.orig_dtype = torch.bfloat16
    layer.output_size_per_partition = 32
    layer._ilu_fp8_online_fallback_reason = None
    layer.weight = torch.nn.Parameter(
        torch.empty((32, 64), dtype=torch.float8_e4m3fn), requires_grad=False
    )

    with pytest.raises(RuntimeError, match="staging weight is missing"):
        method.process_weights_after_loading(layer)


def test_config_dispatches_online_and_preserves_ignored_layers(monkeypatch):
    config = ilu_fp8.IluFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.layers.0.keep"],
            "none_moe_is_checkpoint_fp8_serialized": False,
            "none_moe_use_weight_block_size": False,
        }
    )
    online = object()
    monkeypatch.setattr(
        ilu_fp8, "IluFp8PerChannelOnlineLinearMethod", lambda config: online
    )
    layer = object.__new__(LinearBase)

    assert config.get_quant_method(layer, "model.layers.0.proj") is online
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.keep"),
        UnquantizedLinearMethod,
    )
