# SPDX-License-Identifier: Apache-2.0
"""Iluvatar W16A16 unquantized fused-MoE method wrappers.

Corresponds to:
    vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py

Target   : vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.
           UnquantizedFusedMoEMethod
Baseline : main_6e448d0 + fork ILUVATAR_W16A16 delta
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

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


@UnquantizedFusedMoEMethod.register_oot()
class IluUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """ILUVATAR_W16A16 OOT implementation of UnquantizedFusedMoEMethod."""

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # ILUVATAR_W16A16 handles only routed experts in ``ilu_apply``. The
        # shared-expert residual was fused into ``shared_experts_input`` by
        # IluMoERunner, so this wrapper adds it back after routed experts run.
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
        # Fork delta formerly lived in community fused_moe.py. Keep the
        # ixformer W16A16 dispatch local to the plugin.
        from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe import (
            ilu_w16a16_fused_experts,
        )

        return ilu_w16a16_fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
