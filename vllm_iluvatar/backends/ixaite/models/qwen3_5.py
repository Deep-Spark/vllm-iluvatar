# Source: vllm_iluvatar/model_executor/models/qwen3_5.py at 47fb804; independent text and vision bindings.
# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/models/qwen3_5.py

Iluvatar OOT Qwen3.5 models: optimize
``Qwen3NextAttention._project_qkv_gate`` with fused kernels, route the vision
tower's LayerNorm and target GELUs off the eager Torch path, and replace
ViT attention einops with packed-QKV / output layouts.

Target   : vllm.model_executor.models.qwen3_5.
           Qwen3_5ForConditionalGeneration and
           Qwen3_5MoeForConditionalGeneration (arches with the same names)
Baseline : main_6e448d0
Why      : the fork replaces the Torch implementation of q/k RMSNorm, rotary
           embedding, and output-gate projection with fused Iluvatar kernels.
           The ViT also runs stock nn.LayerNorm at every block's norm1 /
           norm2 and at the merger entry, which no CustomOp registry covers.
How      : ModelRegistry.register_model points both arches at the subclasses
           below; each subclass inherits the community model unchanged and,
           after ``super().__init__()``, swaps every layer's
           ``self_attn.__class__`` to an optimized ``Qwen3NextAttention``
           subclass (weights preserved), overriding Q/K projection and
           attention-output gating, then swaps vision attention and every
           vision-tower ``nn.LayerNorm.__class__`` the same way. The dense
           subclass also replaces the Qwen3.5-9B Vision MLP and merger
           GELUs with routed modules.
Scope    : Attention, vision packed-QKV/RoPE, and Vision LayerNorm cover
           dense and MoE; Vision GELU covers dense Qwen3.5-9B only.
Remove   : when community Qwen3NextAttention uses equivalent fused kernels
           and the vision tower routes LayerNorm/GELU through CustomOps.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention as _Qwen2_5_VisionAttention,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGeneration as _Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention as _Qwen3NextAttention,
)

import vllm_iluvatar._C  # noqa: F401
from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite.attention.vit import (
    vit_attn_output_layout,
    vit_fused_qkv_preproc,
)
from vllm_iluvatar.backends.ixaite.runtime import (
    load_ixaite_python_op,
    load_ixaite_torch_op,
)

_FUSED_QK_RMSNORM_ROPE_GATE = None
_FUSED_QK_RMSNORM_GATE = None
_VIDEO_EVS = None
_VISION_GELU = None
_ATTN_OUTPUT_SIGMOID_GATE = None


# Required fixed model functions bind before constructor fusion changes classes.
def _required_torch_op(name):
    implementation, reason = load_ixaite_torch_op(name)
    if implementation is None:
        raise RuntimeError(f"ixaite {name} is unavailable: {reason}")
    return implementation


# Python ABI functions carry model metadata that the Torch schema cannot express.
def _required_python_op(name, function):
    implementation, reason = load_ixaite_python_op(name, function)
    if implementation is None:
        raise RuntimeError(f"ixaite {name}.{function} is unavailable: {reason}")
    return implementation


# Both fused projection forms share the same model construction contract.
def configure_fused_qk_provider():
    global _FUSED_QK_RMSNORM_ROPE_GATE, _FUSED_QK_RMSNORM_GATE
    _FUSED_QK_RMSNORM_ROPE_GATE = _required_torch_op("fused_qk_rmsnorm_rope_gate")
    _FUSED_QK_RMSNORM_GATE = _required_torch_op("fused_qk_rmsnorm_gate")


# Attention output gating is independent of the projection's weight layout.
def configure_attn_output_sigmoid_gate_provider():
    global _ATTN_OUTPUT_SIGMOID_GATE
    _ATTN_OUTPUT_SIGMOID_GATE = _required_python_op(
        "attn_output_sigmoid_gate", "attn_output_sigmoid_gate"
    )


# Vision GELU binds once before replacing the parameter-free modules.
def configure_vision_gelu_provider():
    global _VISION_GELU
    _VISION_GELU = _required_torch_op("gelu")


# Preserve video retention semantics through the dedicated ixaite entry.
def configure_video_evs_provider():
    global _VIDEO_EVS
    _VIDEO_EVS = _required_python_op("video_evs", "compute_retention_mask")


class _VisionGELU(torch.nn.Module):
    """Parameter-free GELU module using the startup-selected provider."""

    def __init__(self, approximate: str) -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _VISION_GELU(input, self.approximate)


def _replace_vision_gelu(module, approximate: str, location: str) -> None:
    activation = getattr(module, "act_fn", None)
    if (
        not isinstance(activation, torch.nn.GELU)
        or activation.approximate != approximate
    ):
        raise RuntimeError(
            f"{location} must be nn.GELU(approximate={approximate!r}), "
            f"got {activation!r}"
        )
    module.act_fn = _VisionGELU(approximate)


def qwen35_vision_gelu(visual) -> None:
    """Route the Qwen3.5-9B Vision MLP and merger GELUs."""
    for index, block in enumerate(visual.blocks):
        _replace_vision_gelu(block.mlp, "tanh", f"visual.blocks.{index}.mlp.act_fn")
    _replace_vision_gelu(visual.merger, "none", "visual.merger.act_fn")


class Qwen3_5VisionAttention(_Qwen2_5_VisionAttention):
    """Qwen3.5 vision attention with packed-QKV and output layouts."""

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        q, k, v = vit_fused_qkv_preproc(
            qkv_packed=x,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=self.num_attention_heads_per_partition,
            head_dim=self.hidden_size_per_attention_head,
            padded_head_dim=self.attn.padded_head_dim,
        )
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output, _ = self.proj(vit_attn_output_layout(context_layer))
        return output


def qwen35_vision_attention_fusion(visual) -> None:
    """Replace community vision attention with packed-QKV split and rotary."""
    if visual is None:
        return
    for block in visual.blocks:
        attention = getattr(block, "attn", None)
        if type(attention) is _Qwen2_5_VisionAttention:
            attention.__class__ = Qwen3_5VisionAttention


class Qwen3NextAttention(_Qwen3NextAttention):
    """Qwen3NextAttention with routed Q/K fusion and output gating."""

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            pos = positions[0] if positions.ndim == 2 else positions

            q, k, gate = _FUSED_QK_RMSNORM_ROPE_GATE(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )

            q, k, gate = _FUSED_QK_RMSNORM_GATE(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.num_kv_heads * self.head_dim
            )

        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = _ATTN_OUTPUT_SIGMOID_GATE(
                attn_output,
                gate,
                attn_output,
            )
        output, _ = self.o_proj(attn_output)
        return output


class IluVisionLayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` with its forward routed to the selected provider.

    Installed by ``__class__`` reassignment, so weight / bias / eps and every
    other module attribute survive and only the resolved ``forward`` changes.
    A parameter dtype that does not match the activation stays off the
    routed kernels, which reject a widen. Community ``F.layer_norm`` on
    CoreX is equally strict, so the input is aligned to the parameter
    dtype and the output is cast back.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.dtype is not x.dtype:
            return super().forward(x.to(self.weight.dtype)).to(x.dtype)
        return ops.layer_norm(x, self.weight, self.bias, self.eps)


def qwen35_vision_layernorm_fusion(visual) -> None:
    """Route every vision-tower LayerNorm through the selected provider.

    Covers each vision block's norm1 / norm2 plus the patch-merger entry norm
    -- 55 call sites per forward on the 9B tower, and the last eager norm left
    in the ViT. The exact-type match leaves LayerNorm subclasses (community
    fp32-weight helper, this one) alone, and a multi-axis normalized_shape is
    skipped because the kernels normalize the trailing dim only.
    """
    if visual is None:
        return
    for module in visual.modules():
        if type(module) is nn.LayerNorm and len(module.normalized_shape) == 1:
            module.__class__ = IluVisionLayerNorm


def qwen35_attention_fusion(model) -> None:
    """Swap each decoder layer's ``self_attn`` to the fused subclass.

    ``__class__`` reassignment preserves all weights / state and only changes the
    resolved ``forward``.  See Qwen3NextAttention._project_qkv_gate for details.
    """
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if attn.__class__.__name__ == "Qwen3NextAttention":
            attn.__class__ = Qwen3NextAttention


def qwen35_vision_mlp_padding(model) -> None:
    from ixaite.ops.linear_qwen35_shapes import padded_mr_nk

    for block in model.visual.blocks:
        fc1 = block.mlp.linear_fc1
        fc2 = block.mlp.linear_fc2
        target_fc1 = padded_mr_nk(*map(int, fc1.weight.shape))
        target_fc2 = padded_mr_nk(*map(int, fc2.weight.shape))
        if target_fc1 is None or target_fc2 is None:
            continue
        if target_fc1[0] != target_fc2[1]:
            raise ValueError(
                "Qwen3.5 vision MLP padding must preserve the shared width"
            )

        fc1_weight = fc1.weight.data.new_zeros(target_fc1)
        fc1_weight[: fc1.weight.shape[0], : fc1.weight.shape[1]].copy_(fc1.weight.data)
        fc1.weight.data = fc1_weight
        fc2_weight = fc2.weight.data.new_zeros(target_fc2)
        fc2_weight[: fc2.weight.shape[0], : fc2.weight.shape[1]].copy_(fc2.weight.data)
        fc2.weight.data = fc2_weight

        if fc1.bias is not None:
            fc1_bias = fc1.bias.data.new_zeros(target_fc1[0])
            fc1_bias[: fc1.bias.shape[0]].copy_(fc1.bias.data)
            fc1.bias.data = fc1_bias

        fc1.output_size_per_partition = target_fc1[0]
        fc1.output_partition_sizes = [target_fc1[0]]
        fc1.output_size = target_fc1[0] * fc1.tp_size
        fc2.input_size_per_partition = target_fc2[1]
        fc2.input_size = target_fc2[1] * fc2.tp_size


class _IluVideoEvsMixin:
    def _postprocess_video_embeds_evs(self, video_embeds_split, video_input):
        if not self.is_multimodal_pruning_enabled or self.video_pruning_method != "evs":
            return super()._postprocess_video_embeds_evs(
                video_embeds_split, video_input
            )

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        merge_size = self.visual.spatial_merge_size
        video_embeds_out = []
        for video_idx, (emb, size) in enumerate(
            zip(video_embeds_split, grid_thw.tolist())
        ):
            timestamps = video_input.timestamps[video_idx]
            retention_mask = _VIDEO_EVS(
                emb,
                size,
                spatial_merge_size=merge_size,
                q=self.video_pruning_rate,
            )
            emb = emb[retention_mask]
            t, h, w = size
            num_tokens_per_frame = (
                retention_mask.reshape(
                    t,
                    h // merge_size,
                    w // merge_size,
                )
                .sum(dim=(1, 2))
                .long()
                .tolist()
            )
            video_embeds_out.append(
                self._create_final_video_embeddings(
                    video_embeddings=emb,
                    num_tokens_per_frame=num_tokens_per_frame,
                    timestamps=timestamps,
                    video_grid_thw=size,
                    retention_mask=retention_mask,
                )
            )
        return tuple(video_embeds_out)


class Qwen3_5ForConditionalGeneration(
    _IluVideoEvsMixin, _Qwen3_5ForConditionalGeneration
):
    """Qwen3_5 dense model wired for Iluvatar fused q/k norm attention."""

    supports_mamba_prefix_caching = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            from .qwen3_5_apc import initialize_apc_model

            initialize_apc_model(self, vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Inherited video postprocessing requires the configured pruning method.
        self.video_pruning_method = self.multimodal_config.video_pruning_method
        qwen35_vision_gelu(self.visual)
        qwen35_vision_attention_fusion(self.visual)
        qwen35_attention_fusion(self.language_model.model)
        qwen35_vision_layernorm_fusion(getattr(self, "visual", None))

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        qwen35_vision_mlp_padding(self)
        return loaded


class Qwen3_5MoeForConditionalGeneration(
    _IluVideoEvsMixin, _Qwen3_5MoeForConditionalGeneration
):
    """Qwen3_5 moe model wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Inherited video postprocessing requires the configured pruning method.
        self.video_pruning_method = self.multimodal_config.video_pruning_method
        qwen35_vision_attention_fusion(getattr(self, "visual", None))
        qwen35_attention_fusion(self.language_model.model)
        qwen35_vision_layernorm_fusion(getattr(self, "visual", None))

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        qwen35_vision_mlp_padding(self)
        return loaded


def register_model() -> None:
    """Model registration hook: point the Qwen3 dense arch at the Iluvatar class."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .qwen3_5_apc import IluQwen3_5Config

    MODELS_CONFIG_MAP["Qwen3_5ForConditionalGeneration"] = IluQwen3_5Config

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_iluvatar.backends.ixaite.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_iluvatar.backends.ixaite.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    )
