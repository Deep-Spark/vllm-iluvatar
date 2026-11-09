# SPDX-License-Identifier: Apache-2.0
# Source: ilu_qwen_gdn_linear_attn.py at 47fb804; independent GDN lifecycle.
"""Ixaite GDN with matched convolution, metadata and recurrent-state contracts."""

from collections.abc import Callable
from functools import cache
from typing import Any

import torch
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import _encode_layer_name

from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite.attention.gdn import IluGDNAttentionMetadata
from vllm_iluvatar.backends.ixaite.layers import causal_conv1d as conv
from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
from vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn import (
    fused_gdn_gating,
)
from vllm_iluvatar.third_party.flash_linear_attention.ops.ilu_l2norm import (
    l2norm_fwd_inplace as l2norm_fwd,
)

_IXAITE_POST_CONV_PREP = None
_IXAITE_POST_CONV_PREP_GEOMETRY = None
_GDN_CHUNK = None
_IXAITE_APC_CHUNK = None
_IXAITE_PACKED_DECODE = None
_IXAITE_PACKED_DECODE_GEOMETRY = None
_IXAITE_SPEC_DELTA_UPDATE = None
_IXAITE_SPEC_DELTA_UPDATE_GEOMETRY = None
_STRIDED_Z_NORM = None


# Required lifecycle entries bind before any GDN weights or cache state exist.
def _required(op_name, func_name):
    implementation, reason = load_ixaite_python_op(op_name, func_name)
    if implementation is None:
        raise RuntimeError(f"ixaite {op_name}.{func_name} is unavailable: {reason}")
    return implementation


# Fixed geometry is available after TP sharding, before mutable state is used.
def _require_geometry(name, checker, geometry):
    supported, reason = checker(**geometry)
    if not supported:
        raise RuntimeError(f"ixaite {name} geometry is not supported: {reason}")


# Bind preparation and its geometry validator as a complete contract.
def configure_gdn_post_conv_prep_provider():
    global _IXAITE_POST_CONV_PREP, _IXAITE_POST_CONV_PREP_GEOMETRY
    _IXAITE_POST_CONV_PREP = _required("gdn_post_conv_prep", "fused_post_conv_prep")
    _IXAITE_POST_CONV_PREP_GEOMETRY = _required(
        "gdn_post_conv_prep", "is_geometry_supported"
    )
    _fused_prep_supports_geometry.cache_clear()


# The existing ABI adapter retains state-pool writeback and output reordering.
def configure_gdn_chunk_provider():
    global _GDN_CHUNK, _IXAITE_APC_CHUNK
    _IXAITE_APC_CHUNK = _required("gdn_chunk", "chunk_gated_delta_rule")
    _GDN_CHUNK = _adapt_ixaite_gdn_chunk(_IXAITE_APC_CHUNK)


# Bind packed recurrent decode before the layer can mutate convolution state.
def configure_gdn_packed_decode_provider():
    global _IXAITE_PACKED_DECODE, _IXAITE_PACKED_DECODE_GEOMETRY
    _IXAITE_PACKED_DECODE = _required(
        "gdn_packed_decode", "fused_recurrent_gated_delta_rule_packed_decode"
    )
    _IXAITE_PACKED_DECODE_GEOMETRY = _required(
        "gdn_packed_decode", "is_geometry_supported"
    )
    _packed_decode_supports_geometry.cache_clear()


# Speculative and mixed-batch recurrence share the same ixaite state ABI.
def configure_gdn_spec_delta_update_provider():
    global _IXAITE_SPEC_DELTA_UPDATE, _IXAITE_SPEC_DELTA_UPDATE_GEOMETRY
    _IXAITE_SPEC_DELTA_UPDATE = _required(
        "gdn_spec_delta_update", "fused_recurrent_gated_delta_rule"
    )
    _IXAITE_SPEC_DELTA_UPDATE_GEOMETRY = _required(
        "gdn_spec_delta_update", "is_geometry_supported"
    )
    _spec_delta_update_supports_geometry.cache_clear()


# The independent facade already binds the ABI that accepts strided z.
def configure_gdn_strided_z_norm_provider():
    global _STRIDED_Z_NORM
    _STRIDED_Z_NORM = ops.rms_norm_gated


# A module-global binding is traceable without reflection during forward.
def _ixaite_strided_z_norm():
    return _STRIDED_Z_NORM


# No alternate backend is reachable after a support check fails.
def _fail_closed_if_forced_ixaite(op_name, reason):
    raise RuntimeError(f"ixaite {op_name} is not supported: {reason()}")


# Cache immutable head geometry so repeated forwards do not revalidate it.
@cache
def _fused_prep_supports_geometry(num_k_heads, head_k_dim, num_v_heads, head_v_dim):
    return (
        _IXAITE_POST_CONV_PREP_GEOMETRY is not None
        and _IXAITE_POST_CONV_PREP_GEOMETRY(
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
        )[0]
    )


# Packed decode uses the same final TP-local geometry as preparation.
@cache
def _packed_decode_supports_geometry(num_k_heads, head_k_dim, num_v_heads, head_v_dim):
    return (
        _IXAITE_PACKED_DECODE_GEOMETRY is not None
        and _IXAITE_PACKED_DECODE_GEOMETRY(
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
        )[0]
    )


# Speculative state updates must map every head before any state is written.
@cache
def _spec_delta_update_supports_geometry(
    num_k_heads, head_k_dim, num_v_heads, head_v_dim
):
    return (
        _IXAITE_SPEC_DELTA_UPDATE_GEOMETRY is not None
        and _IXAITE_SPEC_DELTA_UPDATE_GEOMETRY(
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
        )[0]
    )


# Selection has no vendor fallback, including mixed speculative batches.
def _select_spec_delta_update(num_k_heads, head_k_dim, num_v_heads, head_v_dim):
    if _IXAITE_SPEC_DELTA_UPDATE is None or not _spec_delta_update_supports_geometry(
        num_k_heads, head_k_dim, num_v_heads, head_v_dim
    ):
        raise RuntimeError("ixaite gdn_spec_delta_update geometry is not supported")
    return _IXAITE_SPEC_DELTA_UPDATE


def _adapt_ixaite_gdn_chunk(implementation: Callable[..., object]):
    def gdn_chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float,
        output_indices: torch.Tensor | None,
    ) -> None:
        # The kernel owns initialization and fp32 pool writeback.
        direct_out = out if output_indices is None else None
        chunk_out, final_state = implementation(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=state,
            initial_state_indices=state_indices,
            has_initial_state=has_initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=64,
            use_qk_l2norm_in_kernel=False,
            out=direct_out,
        )
        if output_indices is not None:
            out.index_copy_(0, output_indices.to(torch.long), chunk_out)
        assert final_state is not None

    return gdn_chunk


def _ixaite_geometry_reason(
    checker: Callable[..., tuple[bool, str | None]] | None,
    **kwargs: object,
) -> str:
    if checker is None:
        return "it is not bound"
    supported, reason = checker(**kwargs)
    if supported:
        return "supported"
    return reason or "the TP-local geometry is not supported"


class IluChunkGatedDeltaRule(ChunkGatedDeltaRule):
    """Use the Iluvatar FLA chunk path for GDN Triton prefill."""

    def __init__(self) -> None:
        super().__init__()
        # Parent backend preferences must not replace the selected implementation.
        self.gdn_prefill_backend = "triton"
        self._ixaite_chunk = _required("gdn_chunk", "chunk_gated_delta_rule")
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
        if use_qk_l2norm_in_kernel:
            # The generic prefill ABI must not mutate its input projections.
            q, k = q.clone(), k.clone()
            l2norm_fwd(q, k)
        return self._ixaite_chunk(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            out=core_attn_out,
        )


# Independent executable layer; metadata comes from the paired backend.
class IluQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(
        self,
        config,
        vllm_config,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(
            config,
            vllm_config,
            prefix,
            gqa_interleaved_layout,
            reduce_results,
        )
        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        if output_gate_type not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported {output_gate_type=}")
        self.output_gate_type = output_gate_type
        # Validate every entry used by this layer before convolution or SSM state exists.
        geometry = dict(
            num_k_heads=self.key_dim // self.tp_size // self.head_k_dim,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        for name, checker in (
            ("gdn_post_conv_prep", _IXAITE_POST_CONV_PREP_GEOMETRY),
            ("gdn_packed_decode", _IXAITE_PACKED_DECODE_GEOMETRY),
            ("gdn_spec_delta_update", _IXAITE_SPEC_DELTA_UPDATE_GEOMETRY),
        ):
            if checker is None:
                raise RuntimeError(f"ixaite {name} is not initialized")
            _require_geometry(name, checker, geometry)
        _require_geometry(
            "gdn_chunk", _required("gdn_chunk", "is_geometry_supported"), geometry
        )
        if (
            _GDN_CHUNK is None
            or _STRIDED_Z_NORM is None
            or conv.causal_conv1d_fn is None
            or conv.causal_conv1d_update is None
        ):
            raise RuntimeError("ixaite GDN lifecycle is not initialized")
        self._gate_act_code = {"sigmoid": 0, "silu": 1}[output_gate_type]

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use provider-safe output buffers and retain parent initialization elsewhere."""
        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        core_attn_out = torch.empty(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Profile run: custom op returns without writing. Do not feed
        # uninitialized rows into RMSNormGated.
        if get_forward_context().attn_metadata is None:
            core_attn_out.zero_()

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )
        return self._output_projection(core_attn_out, z)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Project attention output using the selected provider's gated norm."""
        norm = self.norm
        rms_norm_gated = _ixaite_strided_z_norm()
        if (
            rms_norm_gated is not None
            and norm.bias is None
            and norm.group_size is None
            and norm.activation in ("swish", "silu", "sigmoid")
            and core_attn_out.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and norm.weight.dtype == core_attn_out.dtype
            and z.dtype == core_attn_out.dtype
            and norm.weight.is_contiguous()
            and core_attn_out.is_contiguous()
            and z.dim() == 3
            and z.size(0) == core_attn_out.size(0)
            and z.size(1) == core_attn_out.size(1)
            and z.size(2) == core_attn_out.size(2)
            and z.stride(-1) == 1
            and z.stride(1) == z.size(2)
        ):
            num_tokens, num_heads, head_dim = z.shape
            x2d = core_attn_out.view(num_tokens * num_heads, head_dim)
            out2d = torch.empty_like(x2d)
            rms_norm_gated(
                out2d,
                x2d,
                z.view(num_tokens, num_heads * head_dim),
                norm.weight.data,
                float(norm.eps),
                bool(norm.norm_before_gate),
                norm.activation,
                num_heads,
            )
            output, _ = self.out_proj(out2d.view(num_tokens, num_heads * head_dim))
            return output
        raise RuntimeError("ixaite GDN gated-norm layout is not supported")

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

    def packed_qkv_views(self, packed: torch.Tensor):
        """G4: [T,H,D] views of packed conv_out [T, C=q|k|v], no copy."""
        t, c = packed.size(0), packed.size(1)
        d = self.head_k_dim
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size
        hq = q_dim // d
        hv = v_dim // self.head_v_dim
        query = packed.as_strided((t, hq, d), (c, d, 1), storage_offset=0)
        key = packed.as_strided((t, hq, d), (c, d, 1), storage_offset=q_dim)
        value = packed.as_strided(
            (t, hv, self.head_v_dim),
            (c, self.head_v_dim, 1),
            storage_offset=q_dim + k_dim,
        )
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
        if attn_metadata.state_block_table is not None:
            return self._forward_core_apc(mixed_qkv, b, a, core_attn_out, attn_metadata)

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
        spec_conv_state_indices_tensor = attn_metadata.spec_conv_state_indices_tensor
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
            assert spec_conv_state_indices_tensor is not None
            mixed_qkv_spec = conv.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_conv_state_indices_tensor[
                    : attn_metadata.num_spec_decodes
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
            mixed_qkv_non_spec = conv.causal_conv1d_fn(
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
            mixed_qkv_non_spec = conv.causal_conv1d_update(
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

        # ixaite fused prep: rearrange_mixed_qkv + fused_gdn_gating + optional
        # l2norm. Covers pure non-spec and pure spec-decode batches. Mixed
        # spec+prefill/decode still uses Triton gating: a/b stay in packed
        # token order while conv outputs are split, so output_index scatter is
        # not a single conv_out + a/b geometry the fused kernel can own.
        #
        # A pure non-spec decode batch returns from _forward_core_decode_non_spec
        # above whenever VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE is set (the
        # default), so it never reaches here; it only does with that env off.
        #
        # apply_l2norm is exclusive with the downstream in-kernel L2Norm:
        #   apply_l2norm=True  (prefill) -> chunk MUST NOT re-normalize q/k
        #   apply_l2norm=False (decode / spec) -> recurrent keeps in-kernel L2Norm
        # Double normalization corrupts results silently.
        num_k_heads_local = self.key_dim // self.tp_size // self.head_k_dim
        fused_geometry_ok = _fused_prep_supports_geometry(
            num_k_heads_local,
            self.head_k_dim,
            a.shape[-1],
            self.head_v_dim,
        )
        fused_available = _IXAITE_POST_CONV_PREP is not None and fused_geometry_ok
        fused_prep_l2norm_applied = False
        spec_only = (
            spec_sequence_masks is not None
            and mixed_qkv_spec is not None
            and mixed_qkv_non_spec is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        # Mixed spec+prefill/decode still uses Triton gating: a/b stay in
        # packed token order while conv outputs are split.
        mixed_spec = spec_sequence_masks is not None and not spec_only
        if not fused_available and not mixed_spec:
            _fail_closed_if_forced_ixaite(
                "gdn_post_conv_prep",
                lambda: _ixaite_geometry_reason(
                    _IXAITE_POST_CONV_PREP_GEOMETRY,
                    num_k_heads=num_k_heads_local,
                    head_k_dim=self.head_k_dim,
                    num_v_heads=a.shape[-1],
                    head_v_dim=self.head_v_dim,
                ),
            )
        if fused_available and spec_only:
            (
                query_spec,
                key_spec,
                value_spec,
                g_spec,
                beta_spec,
            ) = _IXAITE_POST_CONV_PREP(
                mixed_qkv_spec,
                a,
                b,
                self.A_log,
                self.dt_bias,
                num_k_heads_local,
                self.head_k_dim,
                self.head_v_dim,
                apply_l2norm=False,
            )
            query_non_spec = key_non_spec = value_non_spec = None
            g_non_spec, beta_non_spec = None, None
        elif (
            fused_available
            and spec_sequence_masks is None
            and mixed_qkv_non_spec is not None
        ):
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
            fused_prep_l2norm_applied = attn_metadata.num_prefills > 0
            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = _IXAITE_POST_CONV_PREP(
                mixed_qkv_non_spec,
                a,
                b,
                self.A_log,
                self.dt_bias,
                num_k_heads_local,
                self.head_k_dim,
                self.head_v_dim,
                apply_l2norm=fused_prep_l2norm_applied,
            )
            g_spec, beta_spec = None, None
        else:
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
                g_non_spec, beta_non_spec = fused_gdn_gating(
                    self.A_log, a, b, self.dt_bias
                )
                g_spec, beta_spec = None, None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            spec_delta_update = _select_spec_delta_update(
                query_spec.shape[-2],
                key_spec.shape[-1],
                value_spec.shape[-2],
                value_spec.shape[-1],
            )
            spec_delta_update(
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
            if not fused_prep_l2norm_applied:
                # Fused prep already normalized q/k (apply_l2norm=True);
                # normalizing again here would silently corrupt prefill.
                l2norm_fwd(query_non_spec, key_non_spec)
            assert _GDN_CHUNK is not None
            _GDN_CHUNK(
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
            # Mixed output ordering uses the same ixaite recurrent ABI.
            _IXAITE_SPEC_DELTA_UPDATE(
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

    def _forward_core_apc(self, mixed_qkv, b, a, core_attn_out, metadata):
        """Read the source block once and persist all scheduled destination states."""
        if metadata.spec_sequence_masks is not None:
            raise NotImplementedError(
                "GDN all prefix caching does not support speculative decoding"
            )
        count = metadata.num_actual_tokens
        mixed_qkv, a, b = mixed_qkv[:count], a[:count], b[:count]
        conv_state, ssm_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        if metadata.num_prefills == 0:
            # Decode keeps the packed fast path; only source/destination slots differ.
            mixed = conv.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=metadata.state_block_table,
                initial_state_idx=metadata.initial_state_idx,
                block_idx_last_scheduled_token=metadata.block_idx_last_scheduled_token,
                pad_slot_id=0,
            )
            _IXAITE_PACKED_DECODE(
                mixed,
                a,
                b,
                self.A_log,
                self.dt_bias,
                self.head_k_dim**-0.5,
                ssm_state,
                core_attn_out[:count].unsqueeze(1),
                metadata.non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
                output_state_indices=metadata.output_state_indices,
            )
            return
        mixed = conv.causal_conv1d_fn(
            mixed_qkv.T,
            weights,
            self.conv1d.bias,
            conv_state,
            metadata.non_spec_query_start_loc,
            cache_indices=metadata.state_block_table,
            has_initial_state=metadata.has_initial_state,
            activation=self.activation,
            pad_slot_id=0,
            block_idx_first_scheduled_token=metadata.block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=metadata.block_idx_last_scheduled_token,
            initial_state_idx=metadata.initial_state_idx,
            num_computed_tokens=metadata.num_computed_tokens,
            block_size_to_align=metadata.cache_block_size,
            metadata=metadata,
        ).T
        q, k, v, g, beta = _IXAITE_POST_CONV_PREP(
            mixed,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.key_dim // self.tp_size // self.head_k_dim,
            self.head_k_dim,
            self.head_v_dim,
            apply_l2norm=True,
        )
        _IXAITE_APC_CHUNK(
            q,
            k,
            v,
            g,
            beta,
            initial_state=ssm_state,
            initial_state_indices=metadata.non_spec_state_indices_tensor,
            has_initial_state=metadata.has_initial_state,
            cu_seqlens=metadata.non_spec_query_start_loc,
            cu_seqlens_cpu=metadata.non_spec_query_start_loc_cpu,
            state_block_table=metadata.state_block_table,
            num_computed_tokens=metadata.num_computed_tokens,
            num_computed_tokens_cpu=metadata.num_computed_tokens_cpu,
            cache_block_size=metadata.cache_block_size,
            out=core_attn_out[:count],
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: IluGDNAttentionMetadata,
    ):
        # Reject unsupported projections before the convolution updates its state.
        if _IXAITE_PACKED_DECODE is None or not _packed_decode_supports_geometry(
            self.key_dim // self.tp_size // self.head_k_dim,
            self.head_k_dim,
            a.shape[-1],
            self.head_v_dim,
        ):
            raise RuntimeError("ixaite gdn_packed_decode geometry is not supported")
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

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = conv.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        _IXAITE_PACKED_DECODE(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )


# APC requires complete bindings before constructing its alternative module tree.
def require_ixaite_apc_providers():
    if any(
        fn is None
        for fn in (
            _IXAITE_APC_CHUNK,
            _IXAITE_PACKED_DECODE,
            _IXAITE_POST_CONV_PREP,
            conv.causal_conv1d_fn,
            conv.causal_conv1d_update,
        )
    ):
        raise RuntimeError(
            "ixaite GDN prefix caching requires the complete convolution and recurrent lifecycle"
        )
