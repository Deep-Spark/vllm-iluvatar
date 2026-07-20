# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT GDN overrides.

Corresponds to:
    vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py

Why:
    Iluvatar routes GDN chunk/prefill/decode work through ixformer/CoreX kernels
    and needs plugin-owned GDN metadata for speculative decode output ordering.
How:
    Official OOT hooks keep community source untouched:
    ``ChunkGatedDeltaRule.register_oot`` owns the chunk op and
    ``QwenGatedDeltaNetAttention.register_oot`` narrows the attention delta to
    CoreX kernel preparation/dispatch.
Scope:
    Qwen3-Next GDN layers on the Iluvatar platform.
"""

from __future__ import annotations

from typing import Any

import ixformer._C as ixf_c_ops
import torch
import vllm.envs as envs
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,  # pyright: ignore[reportMissingImports]
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton

from vllm_iluvatar.model_executor.layers.mamba.ops.ilu_causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm_iluvatar.third_party.flash_linear_attention.ops.ilu_l2norm import (
    l2norm_fwd_inplace as l2norm_fwd,
)
from vllm_iluvatar.v1.attention.backends.ilu_gdn_attn import (
    IluGDNAttentionMetadata,
)


def _register_qwen_gdn_attention(cls):
    """Register the metadata-dependent attention override only with its backend.

    Without ``VLLM_ILU_GDN_ATTN_BACKEND``, vLLM uses the community GDN backend
    and produces community ``GDNAttentionMetadata``. Registering this OOT layer
    in that mode would mix plugin layer code with community metadata.
    """
    if envs.VLLM_ILU_GDN_ATTN_BACKEND:
        return QwenGatedDeltaNetAttention.register_oot(cls)
    return cls


@ChunkGatedDeltaRule.register_oot()
class IluChunkGatedDeltaRule(ChunkGatedDeltaRule):
    """Use the Iluvatar FLA chunk path for GDN Triton prefill."""

    def __init__(self) -> None:
        super().__init__()
        if self.gdn_prefill_backend == "flashinfer":
            # Community selects FlashInfer on SM90. Iluvatar keeps the same
            # public resolver but routes the chunk op to the plugin FLA path.
            self.gdn_prefill_backend = "triton"
        if self.gdn_prefill_backend == "triton":
            self._forward_method = self.forward_native

    def forward_native(
        self,
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        initial_state: Any,
        output_final_state: bool,
        cu_seqlens: Any = None,
        chunk_indices: Any = None,
        chunk_offsets: Any = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: Any = None,
    ):
        from vllm_iluvatar.third_party.flash_linear_attention.ops.chunk import (
            chunk_gated_delta_rule,
        )

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    output_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    """
    batch, num_heads = a.shape
    token_stride = a.stride(0)
    assert b.stride(0) == token_stride, (
        "a and b must all contiguous or split from a same tensor"
    )
    grid = (batch, triton.cdiv(num_heads, 8))

    g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        output_index,
        token_stride,
        num_heads,
        beta,
        threshold,
        8,
        USE_OUTPUT_INDEX=output_index is not None,
        num_warps=1,
    )
    return g, beta_output


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    output_index,
    TOKEN_STRIDE,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    USE_OUTPUT_INDEX: tl.constexpr,
):
    i_b, i_d = tl.program_id(0), tl.program_id(1)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off_in = i_b * TOKEN_STRIDE + head_off
    if USE_OUTPUT_INDEX:
        off_output = tl.load(output_index + i_b).to(tl.int32) * NUM_HEADS + head_off
    else:
        off_output = i_b * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off_in, mask=mask)
    blk_b = tl.load(b + off_in, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off_output, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off_output,
        blk_beta_output.to(beta_output.dtype.element_ty),
        mask=mask,
    )


@_register_qwen_gdn_attention
class IluQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None

        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        query, key = map(
            lambda x: rearrange(x, "l (h d) -> l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> l h d", d=self.head_v_dim)

        return query, key, value

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, IluGDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        non_spec_query_start_loc_cpu = attn_metadata.non_spec_query_start_loc_cpu
        output_index = attn_metadata.output_index

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_spec = None
        mixed_qkv_non_spec = None

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
                input_index=spec_token_indx,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            mixed_qkv_T = mixed_qkv.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
                input_index=non_spec_token_indx,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_non_spec, beta_non_spec = None, None
                g_spec, beta_spec = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            else:
                non_spec_nums = non_spec_token_indx.numel()
                g, beta = fused_gdn_gating(
                    self.A_log, a, b, self.dt_bias, output_index=output_index
                )
                g_non_spec, beta_non_spec = g[:non_spec_nums], beta[:non_spec_nums]
                g_spec, beta_spec = g[non_spec_nums:], beta[non_spec_nums:]
        else:
            g_non_spec, beta_non_spec = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            g_spec, beta_spec = None, None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            ixf_c_ops.infer.fused_recurrent_gated_delta_rule(
                query_spec,
                key_spec,
                value_spec,
                core_attn_out,
                g_spec,
                beta_spec,
                ssm_state,
                spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                spec_state_indices_tensor,
                key_spec.shape[-1] ** -0.5,
                True,
                num_accepted_tokens,
                spec_token_indx
                if attn_metadata.num_prefills > 0
                else None,  # none for cuda graph, else use spec_token_indx to inplace update
            )

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            l2norm_fwd(query_non_spec, key_non_spec)
            ixf_c_ops.infer.chunk_gated_delta_rule(
                query_non_spec,
                key_non_spec,
                value_non_spec,
                core_attn_out,
                g_non_spec,
                beta_non_spec,
                ssm_state,
                has_initial_state,
                non_spec_query_start_loc,
                non_spec_query_start_loc_cpu,
                non_spec_state_indices_tensor,
                key_non_spec.shape[-1] ** -0.5,
                non_spec_token_indx,  # not None if spec_sequence_masks is not None to inplace write output, else None
            )
        elif attn_metadata.num_decodes > 0:
            ixf_c_ops.infer.fused_recurrent_gated_delta_rule(
                query_non_spec,
                key_non_spec,
                value_non_spec,
                core_attn_out,
                g_non_spec,
                beta_non_spec,
                ssm_state,
                non_spec_query_start_loc[: attn_metadata.num_actual_tokens + 1],
                non_spec_state_indices_tensor.view(-1, 1),
                key_non_spec.shape[-1] ** -0.5,
                True,  # inplace kernel norm
                None,  # num_accepted_tokens is not needed for none spec decoode
                None,  # for none spec, do no use output index whether cuda graph or not
            )
