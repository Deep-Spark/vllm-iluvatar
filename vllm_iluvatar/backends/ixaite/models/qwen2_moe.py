# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override of the Qwen2MoeMLP shared-expert gate.

Baseline : community ``model_executor/models/qwen2_moe.py`` @ main_61ac3680210
Delta    : ``Qwen2MoeMLP.forward`` applies the shared-expert gate as
           ``F.sigmoid(self.expert_gate(x)[0]) * out`` — two eager CUDA kernels
           (a [tokens, 1] sigmoid and a broadcast multiply) per MoE layer.
           Under the ixaite provider the pair is replaced by one FlyDSL
           rowwise kernel (``ixaite.ops.attn_output_sigmoid_gate.
           rowwise_sigmoid_gate``), halving launch/host dispatch cost on the
           prefill critical path. Baseline/auto-without-ixaite keep the exact
           community Torch expression.
"""

from __future__ import annotations

import torch

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

# Bound once by configure_shared_expert_gate_provider(); forward never
# reselects. None until the provider is configured (patch applied but
# configure not yet run) — the Torch fallback keeps that window correct.
_SHARED_EXPERT_GATE = None


# Bind the single rowwise gate before shared experts can run.
def configure_shared_expert_gate_provider():
    global _SHARED_EXPERT_GATE
    _SHARED_EXPERT_GATE, reason = load_ixaite_python_op(
        "attn_output_sigmoid_gate", "rowwise_sigmoid_gate"
    )
    if _SHARED_EXPERT_GATE is None:
        raise RuntimeError(f"ixaite shared expert gate is unavailable: {reason}")


def ilu_qwen2_moe_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``Qwen2MoeMLP.forward`` with a provider-selected shared-expert gate."""
    gate_up, _ = self.gate_up_proj(x)
    out = self.act_fn(gate_up)
    out, _ = self.down_proj(out)

    if self.expert_gate is not None:
        gate, _ = self.expert_gate(x)
        out = _SHARED_EXPERT_GATE(out, gate)

    return out
