# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Qwen3-MoE model override.

Corresponds to:
    vllm/model_executor/models/hy_v3.py

Target:
    vllm.model_executor.models.hy_v3.HYV3ForCausalLM
    (arch "HYV3ForCausalLM").
Baseline:
    main_6e448d0.
Why:
    the only Iluvatar delta vs community is fusing the two separate
    q_norm/k_norm passes into one fused q/k RMSNorm (ixformer
    rms_norm_qk).  The MoE router optionally runs bf16 to match the
    ixformer grouped-topk kernel (see
    ``hy_v3_moe_bf16_router_overrides``).
How:
    ModelRegistry.register_model points the arch at the subclass below;
    the subclass inherits the community model unchanged and, after
    ``super().__init__()``, swaps each layer's ``self_attn.__class__`` to
    a fused attention subclass (weights preserved) and attaches a
    ``qk_norm`` op.  IluPlatform does not use the HPC_ATTN backend, so the
    community HpcRopeNorm fast path is never built (``rope_support`` is
    False); the fused fallback forward is the only active path.  vLLM has
    no finer-grained (per-method) model hook, so whole-class
    re-registration is the only official extension point.
Scope:
    HunYuan-V3 causal-LM on the Iluvatar platform.
Remove:
    when community HYV3 attention fuses q/k norm natively.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.hy_v3 import HYV3Attention as _HYV3Attention
from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM as _HYV3ForCausalLM
from vllm.model_executor.models.hy_v3 import HYV3MoEFused

from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK


class _HYV3AttentionFusedMixin:
    """Provides the fused q/k norm forward for HunYuan-V3 attention.

    Requires a ``qk_norm`` (:class:`RMSNormQK`) attribute on the instance,
    attached by :func:`hy_v3_attention_fusion` after the community
    ``__init__`` has built ``q_norm`` / ``k_norm``.  Only installed when
    ``use_qk_norm`` is True, so ``q_norm`` / ``k_norm`` always exist here.

    Mirrors the community non-HPC fallback in
    ``vllm.model_executor.models.hy_v3.HYV3Attention.forward``, with the two
    separate ``q_norm`` / ``k_norm`` passes replaced by one fused op.

    TODO:
    The original logic is as follows:
    q/k bf16                          ->
    q_norm/k_norm weight bf16 -> f32  -> norm (calculated in f32) -> rope with cos_sin float32 -> bf16 write

    Our logic is as follows:
    q/k bf16                          ->
    q_norm/k_norm bf16                -> norm (calculated in f32) -> q/k norm bf16 write -> rope (calculated in f32) -> bf16 write
    Thus, a pair of redundant reciprocal operations are executed:
    __float2bfloat16(q/k) in the norm kernel -> q/k bf16 (norm results) -> __bfloat162float(q/k) in the rotary kernel -> q/k bf16 (final results)

    For more details, please refer to the following code:
    https://github.com/Tencent/hpc-ops/blob/main/src/rope/rope.cu#L80-L344
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        out_q, out_k = self.qk_norm(
            q_by_head,
            k_by_head,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
        )
        q = out_q.view(q.shape)
        k = out_k.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class HYV3Attention(_HYV3AttentionFusedMixin, _HYV3Attention):
    """HYV3Attention with fused q/k RMSNorm."""


def hy_v3_attention_fusion(model) -> None:
    """Swap each decoder layer's ``self_attn`` to the fused subclass.

    ``__class__`` reassignment preserves all weights / state and only changes the
    resolved ``forward``.  A ``qk_norm`` op is attached using the head dim and
    eps already carried by the community ``q_norm``.  Layers that do not use q/k
    norm (``use_qk_norm`` False) are left on the community class.
    """
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not getattr(attn, "use_qk_norm", False):
            continue
        attn.__class__ = HYV3Attention
        attn.qk_norm = RMSNormQK(
            attn.head_dim,
            attn.head_dim,
            eps=attn.q_norm.variance_epsilon,
        )


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
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, HYV3MoEFused):
            continue
        mlp.gate.out_dtype = torch.bfloat16
        if mlp.expert_bias is not None:
            mlp.expert_bias.data = mlp.expert_bias.data.to(torch.bfloat16)


class HYV3ForCausalLM(_HYV3ForCausalLM):
    """HunYuan-V3 causal-LM wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hy_v3_attention_fusion(self.model)
        hy_v3_moe_bf16_router_overrides(self.model)


def register_model() -> None:
    """Model registration hook: point the HYV3 causal-LM arch at the Iluvatar
    class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "HYV3ForCausalLM",
        "vllm_iluvatar.model_executor.models.hy_v3:HYV3ForCausalLM",
    )
