# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Qwen2MoeMLP shared-expert gate override.

Corresponds to: vllm/model_executor/models/qwen2_moe.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.model_executor.models.qwen2_moe.Qwen2MoeMLP.forward
Baseline : main_61ac3680210
Why      : the community forward gates the shared expert with
           ``F.sigmoid(self.expert_gate(x)[0]) * out`` — two eager CUDA
           kernels per MoE layer. Iluvatar's ixaite provider fuses them into
           one rowwise FlyDSL kernel to cut host dispatch on the prefill
           critical path. ``Qwen3NextMLP`` is a same-object alias of
           ``Qwen2MoeMLP`` (qwen3_next.py), so a class-method swap covers
           every consumer, including aliases a module-global rebind would
           miss.
How      : ``setattr`` the override forward onto the class; instances and
           aliases resolve through the class at call time. The gate callable
           itself is provider-bound by
           ``configure_shared_expert_gate_provider``.
Scope    : all models using Qwen2MoeMLP (qwen2_moe / qwen3_moe /
           qwen3_next / qwen3_5_moe shared experts). Layers without
           ``expert_gate`` skip the branch exactly like upstream.
Remove   : when upstream fuses the shared-expert sigmoid gate itself.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Swap ``Qwen2MoeMLP.forward`` for the provider-selected gate path."""
    p.register_once(
        __name__,
        _apply,
        detail="Qwen2MoeMLP.forward (rowwise sigmoid gate provider hook)",
    )


def _apply() -> None:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from vllm_iluvatar.backends.ixaite.models.qwen2_moe import (
        ilu_qwen2_moe_mlp_forward,
    )

    # [ILU override] Qwen2MoeMLP.forward — delta: the shared-expert gate
    # ``F.sigmoid(gate) * out`` goes through the provider-bound callable
    # (ixaite rowwise_sigmoid_gate when enabled; the community Torch
    # expression otherwise). Method swap on the class covers the
    # ``Qwen3NextMLP`` alias and all already-imported references.
    Qwen2MoeMLP.forward = ilu_qwen2_moe_mlp_forward
