# SPDX-License-Identifier: Apache-2.0
"""Wiring for fused MoE runner/shared-expert rebinds.

Corresponds to: vllm/model_executor/layers/fused_moe/layer.py
                   vllm/model_executor/layers/fused_moe/runner/moe_runner.py
                   vllm/model_executor/layers/fused_moe/runner/shared_experts.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

This patch is a pure module-global rebind with no method-body copy.

Target   : vllm.model_executor.layers.fused_moe.layer.MoERunner
           vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner
           vllm.model_executor.layers.fused_moe.runner.shared_experts.SharedExperts
Baseline : main_6e448d0
Why      : MoERunner is not a CustomOp (no register_oot); Iluvatar needs
           elastic-EP, input_ids dispatch, FP16 scaling, fused norm-quant tuple
           support, and the fork's RoutedExperts scaling placement. SharedExperts
           needs to always allow multi-stream overlap (community disables it
           under eplb/flashinfer conditions).
How      : rebind module-global ``MoERunner`` / ``SharedExperts`` names to
           ``IluMoERunner`` / ``IluSharedExperts`` (inheritance-based, zero
           method-body copy). Uses ``rebind_module_func`` to propagate to
           already ``from-import``-ed references.
Scope    : all MoE model runs on Iluvatar.
Remove   : when upstream MoERunner gains an official OOT extension point
           (e.g. runner_cls factory hook or CustomOp-style register_oot).
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register IluMoERunner and IluSharedExperts overrides."""
    p.register_once(
        __name__,
        _apply,
        detail="IluMoERunner (inherit MoERunner), IluSharedExperts",
    )


def _apply() -> None:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_mod
    import vllm.model_executor.layers.fused_moe.runner.shared_experts as se_mod

    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner import (  # noqa: E501
        IluMoERunner,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_shared_experts import (  # noqa: E501
        IluSharedExperts,
    )

    # [ILU override] MoERunner -> IluMoERunner — delta: rebind at definition
    # module; rebind_module_func propagates to layer.py and all other
    # from-import consumers automatically.
    p.rebind_module_func(moe_runner_mod, "MoERunner", IluMoERunner)

    # [ILU override] SharedExperts -> IluSharedExperts — delta: rebind at
    # definition module; propagates to MoERunner.__init__ consumer.
    p.rebind_module_func(se_mod, "SharedExperts", IluSharedExperts)
