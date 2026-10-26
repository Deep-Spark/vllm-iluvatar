# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar W16A16 unquantized MoE wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_iluvatar.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
    IluUnquantizedFusedMoEMethod,
)


class _Method(IluUnquantizedFusedMoEMethod):
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.moe_quant_config = object()

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.output


def _layer(*, routed_scaling_factor: float = 7.0):
    layer = MagicMock()
    layer.routed_scaling_factor = routed_scaling_factor
    layer.w13_weight = torch.ones((1, 2, 2))
    layer.w2_weight = torch.ones((1, 2, 2))
    layer.activation = "swiglustep"
    layer.swiglu_limit = 3.0
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = 1
    layer.expert_map = None
    return layer


def test_unquantized_method_uses_custom_op_oot_registration():
    assert issubclass(IluUnquantizedFusedMoEMethod, UnquantizedFusedMoEMethod)
    assert op_registry_oot["UnquantizedFusedMoEMethod"] is IluUnquantizedFusedMoEMethod


def test_initialization_preserves_oot_state(default_vllm_config):
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )

    method = IluUnquantizedFusedMoEMethod(SimpleNamespace())
    assert method.unquantized_backend == UnquantizedMoeBackend.OOT
    assert method.experts_cls is None
    assert method.moe_kernel is None
    assert method.supports_pre_processed_weights is True


def test_apply_adds_shared_residual_without_runner_scale():
    """Runner, not this method, owns routed_scaling_factor application."""
    method = _Method(torch.full((2,), 3.0))
    shared = torch.full((2,), 5.0)

    result = method.apply(
        layer=_layer(routed_scaling_factor=11.0),
        x=torch.ones((2, 2)),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=shared,
    )

    torch.testing.assert_close(result, torch.full((2,), 8.0))


def test_forward_native_calls_plugin_w16a16_helper():
    method = _Method(torch.ones((2, 2)))
    layer = _layer()
    expected = torch.full((2, 2), 9.0)

    with patch(
        "vllm_iluvatar.model_executor.layers.fused_moe.fused_moe."
        "ilu_w16a16_fused_experts",
        return_value=expected,
    ) as helper:
        result = method.forward_native(
            layer=layer,
            x=torch.ones((2, 2)),
            topk_weights=torch.ones((2, 1)),
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )

    assert result is expected
    helper.assert_called_once()
    assert helper.call_args.kwargs["w1"] is layer.w13_weight
    assert helper.call_args.kwargs["swiglu_limit"] == 3.0


def test_w16_helper_rejects_wrong_weight_dtype():
    from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
        validate_nn_weights,
    )

    with pytest.raises(ValueError, match="BF16"):
        validate_nn_weights(torch.ones(1, 64, 64), torch.ones(1, 32, 64))


def test_modular_wrapper_preserves_layer_activation_limit():
    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep import (
        ilu_modular_apply,
    )

    layer = _layer()
    layer._ilu_step5_native_activation = True
    wrapper = SimpleNamespace(_ilu_is_monolithic=True, moe_quant_config=object())
    with patch(
        "vllm_iluvatar.model_executor.layers.fused_moe.fused_moe.ilu_w16a16_fused_experts"
    ) as helper:
        ilu_modular_apply(
            wrapper,
            layer,
            torch.ones(1, 2),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
        )
    assert helper.call_args.kwargs["swiglu_limit"] == 3.0
    assert helper.call_args.kwargs["step5_native_activation"] is True
    assert helper.call_args.kwargs["global_num_experts"] == layer.global_num_experts
    assert helper.call_args.kwargs["quant_config"] is wrapper.moe_quant_config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_corex11_loading_preserves_tn_weights_and_loader(dtype):
    method = _loading_method()
    layer = torch.nn.Module()
    layer.activation = "gelu"
    layer.apply_router_weight_on_input = False
    loaders = {}

    def allocate(*args, **kwargs):
        for name, shape in (("w13_weight", (2, 64, 96)), ("w2_weight", (2, 96, 32))):
            weight = torch.nn.Parameter(torch.ones(shape, dtype=dtype))
            loaders[name] = weight.weight_loader = MagicMock(return_value="loaded")
            layer.register_parameter(name, weight)

    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=True),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=False),
        patch.object(UnquantizedFusedMoEMethod, "create_weights", side_effect=allocate),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
        patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.fused_moe.normalize_nn_weights"
        ) as normalize,
    ):
        method.create_weights(layer, 2, 96, 32, dtype)
        original = (layer.w13_weight, layer.w2_weight)
        method.process_weights_after_loading(layer)
        for name in loaders:
            assert getattr(layer, name).weight_loader is loaders[name]
            assert getattr(layer, name).weight_loader(None) == "loaded"
    parent.assert_called_once_with(layer)
    normalize.assert_not_called()
    assert layer.w13_weight is original[0] and layer.w2_weight is original[1]
    assert layer._ilu_w16a16_weight_format == "TN"
    assert "_ilu_w16a16_input_formats" not in layer.__dict__


def test_corex11_rejects_nn_weights_before_upstream_processing():
    method = _loading_method()
    layer = torch.nn.Module()
    layer._ilu_w16a16_input_formats = {"w13_weight": "NN", "w2_weight": "TN"}
    layer.activation = "silu"
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=True),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=False),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
        pytest.raises(ValueError, match="TN weights"),
    ):
        method.process_weights_after_loading(layer)
    parent.assert_not_called()


@pytest.mark.parametrize(
    "corex11,weight_format", [(True, "TN"), (False, "TN"), (False, "NN")]
)
def test_architecture_dispatch_keeps_distinct_execution_paths(corex11, weight_format):
    from vllm_iluvatar.model_executor.layers.fused_moe import (
        fused_moe as moe,
    )

    layer = _layer()
    quant_config = object()
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=corex11),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=not corex11),
        patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.fused_moe._fused_experts_tn"
        ) as tn,
        patch.object(moe, "_fused_experts_nn") as nn,
    ):
        result = moe.ilu_w16a16_fused_experts(
            torch.ones(1, 2),
            layer.w13_weight,
            layer.w2_weight,
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            "gelu",
            expert_map=layer.expert_map,
            global_num_experts=5,
            quant_config=quant_config,
            swiglu_limit=3.0,
            weight_format=weight_format,
        )
    selected, other = (tn, nn) if weight_format == "TN" else (nn, tn)
    selected.assert_called_once()
    other.assert_not_called()
    assert result is selected.return_value
    if weight_format == "TN":
        assert tn.call_args.kwargs["global_num_experts"] == 5
        assert tn.call_args.kwargs["quant_config"] is quant_config
        assert tn.call_args.kwargs["swiglu_limit"] == 3.0


def _loading_method():
    method = object.__new__(IluUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False, all2all_backend="allgather_reducescatter"
        ),
        has_bias=False,
        is_lora_enabled=False,
    )
    return method


def _loading_layer(dtype=torch.bfloat16, hidden=96):
    layer = torch.nn.Module()
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 64, hidden, dtype=dtype))
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, hidden, 32, dtype=dtype))
    return layer


@pytest.mark.parametrize(
    "reason",
    [
        "fp16",
        "gelu",
        "swigluoai",
        "bias",
        "lora",
        "input_weight",
        "eplb",
        "deepep",
        "preprocessed",
        "alignment",
        "backend",
    ],
)
def test_corex30_keeps_tn_when_nn_is_ineligible(reason):
    method = _loading_method()
    layer = _loading_layer(
        dtype=torch.float16 if reason == "fp16" else torch.bfloat16,
        hidden=48 if reason == "alignment" else 96,
    )
    if reason in ("gelu", "swigluoai"):
        layer.activation = reason
    method.moe.has_bias = reason == "bias"
    method.moe.is_lora_enabled = reason == "lora"
    layer.apply_router_weight_on_input = reason == "input_weight"
    method.moe.moe_parallel_config.enable_eplb = reason == "eplb"
    if reason == "deepep":
        method.moe.moe_parallel_config.all2all_backend = "deepep_low_latency"
    original = (layer.w13_weight, layer.w2_weight)
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=False),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=True),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch(
            "vllm.model_executor.utils.is_weights_pre_processed",
            return_value=reason == "preprocessed",
        ),
        patch(
            "ixformer.inference.backend_registry.registry.available_backends",
            return_value=[] if reason == "backend" else ["ixmm"],
        ),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
    ):
        method.process_weights_after_loading(layer)
    if reason == "preprocessed":
        parent.assert_not_called()
    else:
        parent.assert_called_once_with(layer)
    assert layer._ilu_w16a16_weight_format == "TN"
    assert layer.w13_weight is original[0] and layer.w2_weight is original[1]


def test_eligible_corex30_converts_once_and_keeps_nn_storage():
    method = _loading_method()
    layer = _loading_layer()
    expected = [
        weight.transpose(-1, -2).contiguous()
        for weight in (layer.w13_weight, layer.w2_weight)
    ]
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=False),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=True),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch("vllm.model_executor.utils.is_weights_pre_processed", return_value=False),
        patch(
            "ixformer.inference.backend_registry.registry.available_backends",
            return_value=["ixmm"],
        ),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
    ):
        method.process_weights_after_loading(layer)
        pointers = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
        method.process_weights_after_loading(layer)
    parent.assert_not_called()
    assert layer._ilu_w16a16_weight_format == "NN"
    assert method.supports_pre_processed_weights is False
    assert pointers == (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    torch.testing.assert_close(layer.w13_weight, expected[0])
    torch.testing.assert_close(layer.w2_weight, expected[1])


def test_tn_selection_stays_fixed_across_weight_updates():
    method = _loading_method()
    layer = _loading_layer()
    layer._ilu_w16a16_weight_format = "TN"
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=False),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=True),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
        patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.fused_moe.normalize_nn_weights"
        ) as normalize,
    ):
        method.process_weights_after_loading(layer)
    parent.assert_called_once_with(layer)
    normalize.assert_not_called()


def test_ineligible_nn_input_never_reaches_tn():
    method = _loading_method()
    method.moe.has_bias = True
    layer = _loading_layer()
    layer._ilu_w16a16_input_formats = {"w13_weight": "NN", "w2_weight": "TN"}
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=True),
        patch.object(
            UnquantizedFusedMoEMethod, "process_weights_after_loading"
        ) as parent,
        pytest.raises(ValueError, match="TN execution requires TN weights"),
    ):
        method.process_weights_after_loading(layer)
    parent.assert_not_called()


@pytest.mark.parametrize("corex11", [True, False])
def test_preprocessed_tn_keeps_storage_without_community_kernel(corex11):
    method = _loading_method()
    layer = _loading_layer()
    before = (layer.w13_weight, layer.w2_weight)
    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore11", return_value=corex11),
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=not corex11),
        patch("vllm.model_executor.utils.is_weights_pre_processed", return_value=True),
        patch.object(UnquantizedFusedMoEMethod, "_init_moe_kernel") as community_kernel,
    ):
        method.process_weights_after_loading(layer)
    community_kernel.assert_not_called()
    assert layer._ilu_w16a16_weight_format == "TN"
    assert layer.w13_weight is before[0] and layer.w2_weight is before[1]
