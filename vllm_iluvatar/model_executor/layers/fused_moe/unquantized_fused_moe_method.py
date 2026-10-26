# SPDX-License-Identifier: Apache-2.0
"""Iluvatar W16A16 unquantized fused-MoE method wrappers.

Corresponds to: vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py

Target   : vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.
           UnquantizedFusedMoEMethod
Baseline : v0.31.0 (db9527a468) + fork ILUVATAR_W16A16 delta
Why      : preserve the fork-only ILUVATAR_W16A16 ``ilu_apply`` dispatch without
           editing community vLLM source.
How      : ``@UnquantizedFusedMoEMethod.register_oot()`` installs an official
           CustomOp OOT subclass. ``forward_native`` calls
           ``ilu_w16a16_fused_experts`` directly, replacing the fork's temporary
           community ``fused_moe.py`` edit.
Scope    : imported during ``apply_oot_registrations()``.
Safety   : this implementation is intentionally only for ILUVATAR_W16A16. Do
           not add generic community fallback logic here; use the upstream method
           when running a non-Iluvatar backend.
Remove   : when community vLLM exposes an official unquantized MoE vendor
           backend hook for this ``ilu_apply`` path.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


@UnquantizedFusedMoEMethod.register_oot()
class IluUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """ILUVATAR_W16A16 OOT implementation of UnquantizedFusedMoEMethod."""

    def __init__(self, moe):
        # IluPlatform inherits CudaPlatform, so the community oracle would
        # search CUDA backends. Both TN and NN execution belong to this OOT.
        FusedMoEMethodBase.__init__(self, moe)
        self.unquantized_backend = UnquantizedMoeBackend.OOT
        self.experts_cls = None
        self.supports_pre_processed_weights = True

    @property
    def is_monolithic(self) -> bool:
        return False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        if not is_ivcore30():
            return
        layer._ilu_w16a16_input_formats = {"w13_weight": "TN", "w2_weight": "TN"}
        for name in ("w13_weight", "w2_weight"):
            parameter = getattr(layer, name)
            loader = parameter.weight_loader

            def guard_loader(original):
                @wraps(original)
                def guarded(*loader_args, **loader_kwargs):
                    if layer.__dict__.get("_ilu_w16a16_weight_format") == "NN":
                        raise RuntimeError(
                            "NN W16 weights do not support TN hot reload; restart the model"
                        )
                    return original(*loader_args, **loader_kwargs)

                return guarded

            parameter.weight_loader = guard_loader(loader)

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import is_weights_pre_processed

        from vllm_iluvatar.ilu_util.arch import is_ivcore11, is_ivcore30
        from vllm_iluvatar.model_executor.layers.fused_moe.activation import (
            swiglustep_limit,
        )
        from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
            normalize_nn_weights,
        )

        if not (is_ivcore11() or is_ivcore30()):
            raise ValueError("W16A16 MoE requires CoreX11 or CoreX30")
        selected = layer.__dict__.get("_ilu_w16a16_weight_format")
        formats = layer.__dict__.get(
            "_ilu_w16a16_input_formats", {"w13_weight": "TN", "w2_weight": "TN"}
        )
        if selected == "NN":
            formats = {"w13_weight": "NN", "w2_weight": "NN"}
        if set(formats) != {"w13_weight", "w2_weight"} or any(
            fmt not in ("TN", "NN") for fmt in formats.values()
        ):
            raise ValueError("Both W16 parameter input layouts must be explicit TN/NN")
        original_nn = "NN" in formats.values()
        activation = getattr(layer.activation, "value", layer.activation)
        if activation == "swiglustep":
            swiglustep_limit(getattr(layer, "swiglu_limit", None))
        parallel = self.moe.moe_parallel_config
        use_nn = (
            is_ivcore30()
            and selected != "TN"
            and not is_weights_pre_processed()
            and not parallel.enable_eplb
            and parallel.all2all_backend
            not in ("deepep_low_latency", "deepep_high_throughput", "deepep_v2")
            and not self.moe.has_bias
            and not self.moe.is_lora_enabled
            and not layer.apply_router_weight_on_input
            and activation in ("silu", "swiglustep")
            and layer.w13_weight.dtype == torch.bfloat16
            and layer.w2_weight.dtype == torch.bfloat16
            and layer.w13_weight.is_cuda
            and layer.w13_weight.device == layer.w2_weight.device
        )
        if use_nn:
            from ixformer.inference.backend_registry import registry

            use_nn = "ixmm" in registry.available_backends("moe_w16a16_group_gemm_ixmm")
        if use_nn:
            try:
                # Validation happens before either parameter is replaced.
                normalize_nn_weights(layer)
            except ValueError:
                if original_nn:
                    raise
            else:
                self.supports_pre_processed_weights = False
                return
        if original_nn:
            raise ValueError(
                "NN W16 weights require a supported CoreX30 NN configuration; TN execution requires TN weights"
            )
        # Retain the original implementation and its support range. Once chosen,
        # TN remains TN across later weight updates, including dtype changes.
        if not is_weights_pre_processed():
            super().process_weights_after_loading(layer)
        # Preprocessed TN weights are already in runtime format. This OOT
        # dispatches directly and must not build a community modular kernel.
        layer._ilu_w16a16_weight_format = "TN"

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        output = self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
        if shared_experts_input is not None:
            return output + shared_experts_input
        return output

    def forward_native(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
            ilu_w16a16_fused_experts,
        )

        return ilu_w16a16_fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            weight_format=layer.__dict__.get("_ilu_w16a16_weight_format", "TN"),
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            quant_config=self.moe_quant_config,
            expert_map=layer.expert_map,
            step5_native_activation=layer.__dict__.get(
                "_ilu_step5_native_activation", False
            ),
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )
