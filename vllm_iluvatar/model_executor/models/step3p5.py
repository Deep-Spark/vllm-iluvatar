# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Step3.5 model overrides.

Corresponds to: vllm/model_executor/models/step3p5.py
Target   : arch ``Step3p5ForCausalLM``
Baseline : main_6e448d0
Why      : Iluvatar W8A8 uses fused dynamic-INT8 norm outputs, fused Q/K
           RMSNorm and head-wise gating kernels, and must transport the residual
           stream across pipeline-parallel ranks. DP/EP MoE must use the same
           flatten path as DeepSeek-V4 so shared experts overlap dispatch.
How      : Small model-owned subclasses are promoted after community
           construction and selected with ``ModelRegistry.register_model``.
           MoE blocks mixin ``IluFusedMoEEP`` after promotion.
Scope    : Step3.5 on Iluvatar, including W8A8, PP, and DP/EP execution.
Remove   : when upstream supports these Iluvatar kernel and PP contracts.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parameter import Parameter
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.step3p5 import (
    FusedMoEBlock as _FusedMoEBlock,
)
from vllm.model_executor.models.step3p5 import (
    Step3p5Attention as _Step3p5Attention,
)
from vllm.model_executor.models.step3p5 import (
    Step3p5DecoderLayer as _Step3p5DecoderLayer,
)
from vllm.model_executor.models.step3p5 import (
    Step3p5ForCausalLM as _Step3p5ForCausalLM,
)
from vllm.model_executor.models.step3p5 import Step3p5MLP as _Step3p5MLP
from vllm.model_executor.models.step3p5 import Step3p5Model as _Step3p5Model
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
)
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
    set_max_num_seqs,
)


def _uses_ilu_w8a8_int8(proj: Any) -> bool:
    """Whether ``proj`` accepts Iluvatar's fused quantized-input tuple."""
    scheme = getattr(proj, "scheme", None)
    return (
        scheme is not None
        and scheme.__class__.__name__ == "IluCompressedTensorsW8A8Int8"
    )


class IluStep3p5MLP(_Step3p5MLP):
    """Community MLP with W8A8 fused-input capability detection."""

    def check_fused_quant_for_input(self) -> bool:
        return _uses_ilu_w8a8_int8(self.gate_up_proj)


class IluStep3p5Attention(_Step3p5Attention):
    """Community attention with Iluvatar fused Q/K norm and gate kernels."""

    def check_fused_quant_for_input(self) -> bool:
        if not _uses_ilu_w8a8_int8(self.qkv_proj):
            return False
        return not self.use_head_wise_attn_gate or _uses_ilu_w8a8_int8(self.g_proj)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.dtype],
    ) -> torch.Tensor:
        import ixformer.inference.functions as ixf_ops

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        q_by_head, k_by_head = ixf_ops.rms_norm_qk(
            input_q=q_by_head,
            input_k=k_by_head,
            weight_q=self.q_norm.weight.data,
            weight_k=self.k_norm.weight.data,
            eps=self.q_norm.variance_epsilon,
            norm_type="gemma",
        )
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            attn_output = ixf_ops.step3_5_mul_sigmoid(
                attn_output.view(
                    *attn_output.shape[:-1], self.num_heads, self.head_dim
                ),
                extra_dims.unsqueeze(-1),
            ).view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class IluReplicatedLinear(ReplicatedLinear):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # assert self.weight.dtype != torch.float32
        output, output_bias = super().forward(x)
        return output.to(torch.float32), output_bias


class IluFusedMoEBlock(_FusedMoEBlock, IluFusedMoEEP):
    """Community MoE block that preserves the fused W8A8 input tuple."""

    @staticmethod
    def promote(layer: _FusedMoEBlock) -> None:
        gate_weight = layer.gate.weight
        if gate_weight.dtype == torch.float32:
            gate_weight = gate_weight.to(torch.get_default_dtype())
            from vllm.model_executor.utils import replace_parameter

            replace_parameter(layer.gate, "weight", gate_weight)
            layer.gate.__class__ = IluReplicatedLinear

    def check_fused_quant_for_input(self) -> bool:
        return _uses_ilu_w8a8_int8(self.share_expert.gate_up_proj)

    def forward(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        origin = (
            hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states
        )
        num_tokens, hidden_dim = origin.shape
        # W8A8 fused MoE input is (int8, scale, origin). forward_dp_ep expects
        # a BF16 tensor plus optional (int8, scale, dtype), so keep the tuple
        # on the community runner path until that packing is wired.
        if self._should_use_forward_dp_ep() and isinstance(hidden_states, torch.Tensor):
            final_hidden_states = self.forward_dp_ep(hidden_states)
        elif self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return final_hidden_states.view(num_tokens, hidden_dim)


class IluStep3p5DecoderLayer(_Step3p5DecoderLayer):
    """Step3.5 block retaining its residual stream across PP ranks."""

    @staticmethod
    def promote(layer: _Step3p5DecoderLayer) -> None:
        """Promote community submodules without rebuilding loaded parameters."""
        layer.__class__ = IluStep3p5DecoderLayer
        layer.self_attn.__class__ = IluStep3p5Attention
        if layer.use_moe:
            layer.moe.__class__ = IluFusedMoEBlock
            IluFusedMoEBlock.promote(layer.moe)
            layer.moe.prefix = f"{layer.prefix}.moe"
            layer.moe.init_ep()
        else:
            layer.mlp.__class__ = IluStep3p5MLP

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_norm_quant = self.self_attn.check_fused_quant_for_input()
        if fused_norm_quant:
            if residual is None:
                residual = hidden_states.clone()
                packed = self.input_layernorm(hidden_states, fused_int8_quant=True)
            else:
                packed = self.input_layernorm(
                    hidden_states, residual, fused_int8_quant=True
                )
                residual = packed[-1]
            hidden_states = (*packed[:2], hidden_states.dtype)
        elif residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        fused_norm_quant = (
            self.moe.check_fused_quant_for_input()
            if self.use_moe
            else self.mlp.check_fused_quant_for_input()
        )
        if fused_norm_quant:
            packed = self.post_attention_layernorm(
                hidden_states,
                residual,
                fused_int8_quant=True,
                return_origin_output=self.use_moe,
            )
            residual = packed[-1]
            hidden_states = (
                packed[:3] if self.use_moe else (*packed[:2], hidden_states.dtype)
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        ffn_output = (
            self.moe(hidden_states) if self.use_moe else self.mlp(hidden_states)
        )
        return ffn_output, residual


class IluStep3p5Model(_Step3p5Model):
    """Community model with promoted blocks and PP residual transport."""

    @staticmethod
    def promote(model: _Step3p5Model) -> None:
        """Promote a constructed community model in place."""
        model.__class__ = IluStep3p5Model
        for layer in model.layers:
            if not isinstance(layer, PPMissingLayer):
                IluStep3p5DecoderLayer.promote(layer)
        model.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], model.config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for index in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[index](
                positions, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        return hidden_states + residual


class IluStep3p5ForCausalLM(_Step3p5ForCausalLM):
    """OOT Step3.5 causal LM with promoted Iluvatar model components."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        set_max_num_seqs(vllm_config.scheduler_config.max_num_seqs)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        IluStep3p5Model.promote(self.model)
        # The community CausalLM caches this factory before ``self.model`` is
        # promoted. Keep the PP transport contract aligned with the new model.
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


def register_model() -> None:
    """Register the Iluvatar Step3.5 causal-LM implementation."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Step3p5ForCausalLM",
        "vllm_iluvatar.model_executor.models.step3p5:IluStep3p5ForCausalLM",
    )
