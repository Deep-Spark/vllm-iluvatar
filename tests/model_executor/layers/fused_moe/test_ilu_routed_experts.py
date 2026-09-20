# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar RoutedExperts OOT replacement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_routed_experts import (
    _ROUTED_EXPERTS_LOAD_WEIGHTS_BASELINE,
    _ROUTED_EXPERTS_WEIGHT_LOADER_BASELINE,
    IluRoutedExperts,
)

_MAIN_6E448D0_WEIGHT_LOADER_SHA = (
    "7a6d280797aa1f803cfd0df14202356cbd1607bf1c1d33e2dc2ef20a56870d9c"
)
_MAIN_6E448D0_LOAD_WEIGHTS_SHA = (
    "f16604b8d023796a0d876c5141a4bb43b7a33b8b545c66fe9f678932da87fe9c"
)


class IluAutoGPTQMoEMethod:
    pass


class OtherMoEMethod:
    pass


class IluCompressedTensorsW4A8Int8MoEMethod:
    def __init__(self, gemm_format: str):
        self.gemm_format = gemm_format


class IluCompressedTensorsW8A8Int8MoEMethod:
    def __init__(self, gemm_format: str):
        self.gemm_format = gemm_format


def _uninitialized_routed_experts() -> IluRoutedExperts:
    obj = IluRoutedExperts.__new__(IluRoutedExperts)
    obj.__dict__["_parameters"] = {}
    obj.__dict__["_buffers"] = {}
    obj.__dict__["_modules"] = {}
    obj.__dict__["_non_persistent_buffers_set"] = set()
    obj.__dict__["training"] = False
    obj.__dict__["moe_config"] = SimpleNamespace(tp_rank=0)
    obj.__dict__["quant_config"] = None
    obj.__dict__["quant_method"] = OtherMoEMethod()
    obj.__dict__["opt_level"] = 0
    obj.__dict__["layer_name"] = "layer"
    obj.__dict__["_map_global_expert_id_to_local_expert_id"] = lambda _: 0
    obj.__dict__["_load_model_weight_or_group_weight_scale"] = MagicMock()
    return obj


def test_routed_experts_oot_registration_uses_upstream_class_name():
    assert op_registry_oot["RoutedExperts"] is IluRoutedExperts


def test_weight_loader_fingerprint_baseline_matches_declared_main_6e448d0_source():
    assert p.source_sha(_ROUTED_EXPERTS_WEIGHT_LOADER_BASELINE) == (
        _MAIN_6E448D0_WEIGHT_LOADER_SHA
    )
    assert p.source_sha(_ROUTED_EXPERTS_LOAD_WEIGHTS_BASELINE) == (
        _MAIN_6E448D0_LOAD_WEIGHTS_SHA
    )


def test_ilu_auto_gptq_requests_full_intermediate_size():
    routed = _uninitialized_routed_experts()

    assert routed._needs_intermediate_size_param(IluAutoGPTQMoEMethod()) is True


def test_weight_loader_uses_parameter_shard_dim_override():
    routed = _uninitialized_routed_experts()
    param = torch.nn.Parameter(torch.empty((1, 2, 2)))
    param.shard_dim = 1

    loaded_weight = torch.ones((2, 2))

    assert (
        routed.weight_loader(
            param=param,
            loaded_weight=loaded_weight,
            weight_name="w13_weight",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )
        is True
    )
    routed._load_model_weight_or_group_weight_scale.assert_called_once()
    assert (
        routed._load_model_weight_or_group_weight_scale.call_args.kwargs["shard_dim"]
        == 1
    )


def _recording_param(
    calls: list[tuple[str, torch.Size]], label: str = ""
) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def weight_loader(**kwargs):
        calls.append((label or kwargs["weight_name"], kwargs["loaded_weight"].shape))
        return True

    param.weight_loader = weight_loader
    return param


@pytest.mark.parametrize(
    ("gemm_format", "scale_shape"),
    [
        ("TN", (128, 3072, 2)),
        ("NN", (128, 1536, 2)),
    ],
)
def test_load_weights_trims_fused_w13_weight_scale_to_half_intermediate(
    gemm_format, scale_shape
):
    routed = _uninitialized_routed_experts()
    routed.moe_config.hidden_dim_unpadded = 2048
    routed.moe_config.intermediate_size = 1536
    routed.__dict__["quant_method"] = IluCompressedTensorsW4A8Int8MoEMethod(gemm_format)
    routed.get_expert_mapping = lambda include_fused=True: [
        ("experts.w13_weight_scale", "experts.w13.weight_scale", 0, "w1"),
        ("experts.w13_weight_scale", "experts.w13.weight_scale", 1, "w3"),
        ("experts.w13_weight", "experts.w13", 0, "w1"),
        ("experts.w13_weight", "experts.w13", 1, "w3"),
    ]
    calls: list[tuple[str, torch.Size]] = []
    routed.__dict__["experts.w13_weight"] = _recording_param(calls)
    routed.__dict__["experts.w13_weight_scale"] = _recording_param(
        calls, "w13_weight_scale"
    )
    routed.__dict__["experts.w13_weight.weight_scale"] = _recording_param(calls)

    list(
        routed.load_weights(
            [
                ("experts.w13", torch.empty((128, 3072, 2048))),
                ("experts.w13.weight_scale", torch.empty(scale_shape)),
            ]
        )
    )

    scale_shapes = [shape for name, shape in calls if name == "w13_weight_scale"]
    assert scale_shapes
    assert set(scale_shapes) == {torch.Size([1, 1536])}


def test_load_weights_keeps_fused_w13_weight_scale_when_dim_differs():
    routed = _uninitialized_routed_experts()
    routed.moe_config.hidden_dim_unpadded = 2048
    routed.get_expert_mapping = lambda include_fused=True: [
        ("experts.w13_weight_scale", "experts.w13.weight_scale", 0, "w1"),
        ("experts.w13_weight", "experts.w13", 0, "w1"),
    ]
    calls: list[tuple[str, torch.Size]] = []
    routed.__dict__["experts.w13_weight"] = _recording_param(calls)
    routed.__dict__["experts.w13_weight_scale"] = _recording_param(
        calls, "w13_weight_scale"
    )
    routed.__dict__["experts.w13_weight.weight_scale"] = _recording_param(calls)

    list(
        routed.load_weights(
            [
                ("experts.w13", torch.empty((128, 3072, 2048))),
                ("experts.w13.weight_scale", torch.empty((128, 4096, 2))),
            ]
        )
    )

    scale_shapes = [shape for name, shape in calls if name == "w13_weight_scale"]
    assert scale_shapes
    assert set(scale_shapes) == {torch.Size([1, 4096])}


@pytest.mark.parametrize(
    ("gemm_format", "scale_shape", "expected_shape"),
    [
        ("TN", (128, 2048, 1), torch.Size([1, 2048])),
        ("NN", (128, 1024, 1), torch.Size([1, 1024])),
    ],
)
def test_load_weights_transposes_fused_w2_weight_scale_layout(
    gemm_format, scale_shape, expected_shape
):
    routed = _uninitialized_routed_experts()
    routed.moe_config.hidden_dim_unpadded = 2048
    routed.moe_config.intermediate_size = 1536
    routed.__dict__["quant_method"] = IluCompressedTensorsW4A8Int8MoEMethod(gemm_format)
    routed.get_expert_mapping = lambda include_fused=True: [
        ("experts.w2_weight_scale", "experts.w2.weight_scale", 0, "w2"),
        ("experts.w2_weight", "experts.w2", 0, "w2"),
    ]
    calls: list[tuple[str, torch.Size]] = []
    routed.__dict__["experts.w2_weight"] = _recording_param(calls)
    routed.__dict__["experts.w2_weight_scale"] = _recording_param(
        calls, "w2_weight_scale"
    )
    routed.__dict__["experts.w2_weight.weight_scale"] = _recording_param(calls)

    list(
        routed.load_weights(
            [
                ("experts.w2", torch.empty((128, 3072, 2048))),
                ("experts.w2.weight_scale", torch.empty(scale_shape)),
            ]
        )
    )

    scale_shapes = [shape for name, shape in calls if name == "w2_weight_scale"]
    assert scale_shapes
    assert set(scale_shapes) == {expected_shape}


def test_weight_loader_keeps_moe_loading_marker():
    assert IluRoutedExperts.weight_loader.supports_moe_loading is True


def test_load_weights_w8a8_nn_chunks_fused_w13_on_last_dim():
    routed = _uninitialized_routed_experts()
    routed.moe_config.hidden_dim_unpadded = 2048
    routed.__dict__["quant_method"] = IluCompressedTensorsW8A8Int8MoEMethod("NN")
    routed.get_expert_mapping = lambda include_fused=True: [
        ("experts.w13_weight", "experts.w13", 0, "w1"),
        ("experts.w13_weight", "experts.w13", 1, "w3"),
    ]
    calls: list[tuple[str, torch.Size]] = []
    routed.__dict__["experts.w13_weight"] = _recording_param(calls)

    list(routed.load_weights([("experts.w13", torch.empty((4, 2048, 3072)))]))

    assert calls
    assert set(shape for _, shape in calls) == {torch.Size([2048, 1536])}


def test_load_weights_w8a8_nn_keeps_fused_w2_as_inter_hidden():
    routed = _uninitialized_routed_experts()
    routed.moe_config.hidden_dim_unpadded = 2048
    routed.__dict__["quant_method"] = IluCompressedTensorsW8A8Int8MoEMethod("NN")
    routed.get_expert_mapping = lambda include_fused=True: [
        ("experts.w2_weight", "experts.w2", 0, "w2"),
    ]
    calls: list[tuple[str, torch.Size]] = []
    routed.__dict__["experts.w2_weight"] = _recording_param(calls)

    list(routed.load_weights([("experts.w2", torch.empty((4, 1536, 2048)))]))

    assert calls
    assert set(shape for _, shape in calls) == {torch.Size([1536, 2048])}
