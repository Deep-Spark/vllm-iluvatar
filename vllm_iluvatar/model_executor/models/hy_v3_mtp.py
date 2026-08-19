# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Qwen3-MoE model override.

Corresponds to:
    vllm/model_executor/models/hy_v3_mtp.py

Target:
    vllm.model_executor.models.hy_v3_mtp.HYV3MTP
    (arch "HYV3MTPModel").
Baseline:
    main_6e448d0.
Why:
    The MoE router optionally runs bf16 to match the
    ixformer grouped-topk kernel (see
    ``hy_v3_moe_bf16_router_overrides``).
How:
    ModelRegistry.register_model points the arch at the subclass below;
    the subclass inherits the community model unchanged and, after
    ``super().__init__()``, process hy_v3_moe_bf16_router_overrides.
Scope:
    HunYuan-V3-MTP on the Iluvatar platform.
Remove:
    when backend grouped_tokp support float32 inputs.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.hy_v3 import HYV3MoEFused
from vllm.model_executor.models.hy_v3_mtp import HYV3MTP as _HYV3MTP


def hy_v3_moe_bf16_router_overrides(model) -> None:
    """Run the MoE router (gate logits + expert_bias) in bf16.

    The community default is float32.  PR#26 switched the router to bf16 to match
    the Iluvatar ixformer grouped-topk kernel, whose current ``moe_grouped_topk``
    path does not accept float32 gating_output (see the TODO in
    ``ilu_grouped_topk_router``).  Keep this consistent with the router kernel;
    drop this call and inherit the community float32 once ixformer supports
    float32 gating_output.

    ``gate.out_dtype`` is documented as mutable-after-init; ``expert_bias`` is the
    same ``Parameter`` object held by ``FusedMoE`` as ``e_score_correction_bias``,
    so converting it in place keeps both views in sync.
    """
    for _, layer in model.layers.items():
        mlp = layer.mtp_block.mlp
        assert isinstance(mlp, HYV3MoEFused)
        mlp.gate.out_dtype = torch.bfloat16
        if getattr(mlp, "expert_bias", None) is not None:
            mlp.expert_bias.data = mlp.expert_bias.data.to(torch.bfloat16)


class HYV3MTP(_HYV3MTP):
    """HunYuan-V3-MTP for Iluvatar."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hy_v3_moe_bf16_router_overrides(self.model)


def register_model() -> None:
    """Model registration hook: point the HYV3-MTP arch at the Iluvatar
    class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "HYV3MTPModel",
        "vllm_iluvatar.model_executor.models.hy_v3_mtp:HYV3MTP",
    )
