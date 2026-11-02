# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT GDN overrides.

Corresponds to: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py

Why      : Iluvatar routes GDN chunk/prefill/decode work through ixformer/CoreX kernels
           and needs plugin-owned GDN metadata for speculative decode output ordering.
How      : Official OOT hooks keep community source untouched:
           ``ChunkGatedDeltaRule.register_oot`` owns the chunk op and
           ``QwenGatedDeltaNetAttention.register_oot`` narrows the attention delta to
           CoreX kernel preparation/dispatch.
Scope    : Shared Iluvatar GDN metadata/OOT for Qwen3-Next / Qwen3.5 / Qwen3.8.
           On CoreX30, ``_qwen38_kernel()`` returns ``ixformer`` and
           ``_forward_core`` dispatches to ``_forward_core_ixformer`` (the
           community ``Falling back to the Triton GDN decode path`` log is
           parent ``__init__`` only and is not the hot path).
"""

from __future__ import annotations

import os
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
from vllm.utils.torch_utils import _encode_layer_name

from vllm_iluvatar.ilu_util.arch import is_ivcore30
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

_shared_gdn_workspaces: dict[tuple[int, str, int | None], torch.Tensor] = {}
_GDN_GATE_ACT = {"sigmoid": 0, "silu": 1}
_ixformer_gdn_activation_abi_ready = False


def _run_ixformer_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    activation: int,
) -> None:
    """Call the activation-aware ABI and report stale ixformer clearly."""
    global _ixformer_gdn_activation_abi_ready
    if _ixformer_gdn_activation_abi_ready:
        ixf_c_ops.infer.qwen_gdn_rmsnorm_gated(x, z, weight, out, eps, activation)
        return
    try:
        ixf_c_ops.infer.qwen_gdn_rmsnorm_gated(x, z, weight, out, eps, activation)
    except TypeError as exc:
        raise RuntimeError(
            "The installed ixformer is too old for activation-aware qwen38 "
            "GDN kernels; rebuild and reinstall ixformer."
        ) from exc
    _ixformer_gdn_activation_abi_ready = True


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
        from vllm.third_party.flash_linear_attention.ops.chunk import (
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
        self._gate_act_code = _GDN_GATE_ACT[output_gate_type]
        self._gdn_workspace: torch.Tensor | None = None
        self._conv_out: torch.Tensor | None = None
        self._g_buf: torch.Tensor | None = None
        self._beta_buf: torch.Tensor | None = None
        self._core_attn_out: torch.Tensor | None = None
        self._conv_w: torch.Tensor | None = None
        self._has_init_u8: torch.Tensor | None = None
        self._has_init_i32: torch.Tensor | None = None
        self._gdn_workspace_group = id(vllm_config)
        # Cache sizes at construct time: CustomOp forward has no
        # get_current_vllm_config() context.
        sched = vllm_config.scheduler_config
        self._gdn_b_max = int(sched.max_num_seqs)
        self._gdn_t_max = int(sched.max_num_batched_tokens)
        if self._qwen38_kernel() == "ixformer":
            self._check_ixformer_supported(vllm_config)

    def _qwen38_kernel(self) -> str:
        """CoreX30 uses ixformer CUDA; other arches keep plugin FLA."""
        return "ixformer" if is_ivcore30() else "plugin_fla"

    def _check_ixformer_supported(self, vllm_config) -> None:
        """Fail at model load when the CoreX30 kernel cannot support config."""
        fallback = " Set VLLM_ILU_GDN_ATTN_BACKEND=0 to use the community path."
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            raise ValueError(
                "Iluvatar GDN ixformer kernels require head_k_dim == "
                f"head_v_dim == 128, got {self.head_k_dim}/{self.head_v_dim}."
                + fallback
            )
        if self.conv_kernel_size != 4:
            raise ValueError(
                "Iluvatar GDN ixformer kernels require conv_kernel_size == 4, "
                f"got {self.conv_kernel_size}." + fallback
            )
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "Iluvatar GDN ixformer kernels require num_v_heads % "
                f"num_k_heads == 0, got {self.num_v_heads}/{self.num_k_heads}."
                + fallback
            )

    @staticmethod
    def _ixformer_spec_enabled() -> bool:
        return os.getenv("VLLM_ILU_GDN_IXFORMER_SPEC", "1").lower() not in {
            "0",
            "false",
            "off",
            "no",
        }

    def _ensure_ixformer_buffers(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> None:
        if self._gdn_workspace is not None:
            return
        hv = self.num_v_heads // self.tp_size
        n_chunk_capacity = (self._gdn_t_max + 15) // 16 + self._gdn_b_max
        nbytes = n_chunk_capacity * hv * 13856
        conv_c = self.conv_dim // self.tp_size
        out_dtype = dtype if dtype is not None else torch.bfloat16
        workspace_key = (
            self._gdn_workspace_group,
            device.type,
            device.index,
        )
        shared = _shared_gdn_workspaces.get(workspace_key)
        if shared is None or shared.numel() < nbytes:
            shared = torch.empty(nbytes, dtype=torch.uint8, device=device)
            _shared_gdn_workspaces[workspace_key] = shared
        self._gdn_workspace = shared
        self._conv_out = torch.empty(
            self._gdn_t_max, conv_c, dtype=torch.bfloat16, device=device
        )
        self._g_buf = torch.empty(
            self._gdn_t_max, hv, dtype=torch.float32, device=device
        )
        self._beta_buf = torch.empty(
            self._gdn_t_max, hv, dtype=torch.bfloat16, device=device
        )
        self._core_attn_out = torch.empty(
            self._gdn_t_max, hv, self.head_v_dim, dtype=out_dtype, device=device
        )
        conv_w = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self._conv_w = conv_w if conv_w.is_contiguous() else conv_w.contiguous()
        self._has_init_u8 = torch.empty(
            self._gdn_b_max, dtype=torch.uint8, device=device
        )
        self._has_init_i32 = torch.empty(
            self._gdn_b_max, dtype=torch.int32, device=device
        )

    def _ixformer_has_init(
        self,
        has_init: torch.Tensor | None,
        batch: int,
        *,
        as_uint8: bool,
    ) -> torch.Tensor | None:
        """Reuse preallocated [max_num_seqs] flags. None stays None."""
        if has_init is None:
            if not as_uint8:
                assert self._has_init_i32 is not None
                out = self._has_init_i32[:batch]
                out.fill_(1)
                return out
            return None
        src = has_init[:batch]
        if as_uint8:
            assert self._has_init_u8 is not None
            if (
                src.dtype == torch.uint8
                and src.is_contiguous()
                and src.numel() == batch
            ):
                return src
            out = self._has_init_u8[:batch]
            out.copy_(src)
            return out
        if src.dtype == torch.int32 and src.is_contiguous() and src.numel() == batch:
            return src
        assert self._has_init_i32 is not None
        out = self._has_init_i32[:batch]
        out.copy_(src)
        return out

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """G5/G9 only on the ixformer core. Other kernels keep parent zeros+copy.

        ``IluQwenGatedDeltaNetAttention`` is the OOT class for every
        ``QwenGatedDeltaNetAttention`` once ``VLLM_ILU_GDN_ATTN_BACKEND=1``.
        CoreX11 takes the parent FLA / cuinfer path:
        - ``a/b.contiguous()``: community ``fused_gdn_gating`` assumes packed
          stride; packed-decode Triton is stride-safe, but keep the parent
          copies so we do not change a fallback we are not measuring.
        - ``torch.zeros`` for ``core_attn_out``: FLA / warmup may not write
          every row (vLLM PR 28182). ixformer writes the live ``T`` rows, so
          we reuse a preallocated buffer and only zero on profile warmup.
        """
        if self._qwen38_kernel() != "ixformer":
            return super().forward_cuda(hidden_states)

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

        self._ensure_ixformer_buffers(hidden_states.device, hidden_states.dtype)
        assert self._core_attn_out is not None
        core_attn_out = self._core_attn_out[:num_tokens]
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
        """RMSNormGated + out_proj.

        Parent flattens strided ``z`` from ``mixed_qkvz.split`` and copies.
        The ixformer kernel eats ``[T, Hv, 128]`` with token stride >= Hv*128.
        """
        if self._qwen38_kernel() != "ixformer":
            return super()._output_projection(core_attn_out, z)
        if z.dim() == 2:
            z = z.view(z.size(0), -1, self.head_v_dim)
        x = core_attn_out
        if x.dim() == 2:
            x = x.view(x.size(0), -1, self.head_v_dim)
        _run_ixformer_rmsnorm_gated(
            x,
            z,
            self.norm.weight,
            x,
            float(self.norm.eps),
            self._gate_act_code,
        )
        output, _ = self.out_proj(x.flatten(-2))
        return output

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

    def _forward_core_ixformer(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: IluGDNAttentionMetadata,
    ) -> None:
        self._ensure_ixformer_buffers(mixed_qkv.device, mixed_qkv.dtype)
        assert self._gdn_workspace is not None
        assert self._conv_out is not None
        assert self._conv_w is not None

        # split() of in_proj_qkvz is a non-contiguous [T,C] view; conv reads
        # x[t * x_stride + c] and does not need a packed copy.
        if mixed_qkv.dim() != 2:
            mixed_qkv = mixed_qkv.view(mixed_qkv.size(0), -1)

        conv_cache = self.kv_cache[0]
        # The ixformer kernel consumes SD [N, 3, C] directly, including a
        # padded block stride from the mamba page. Never materialize the full
        # cache here: that would copy every block on every layer and step.
        if is_conv_state_dim_first():
            raise RuntimeError(
                "Qwen GDN ixformer kernel requires SD conv-state layout; "
                "dim-first layout would require a full-cache copy"
            )
        conv_state = conv_cache
        ssm_cache = self.kv_cache[1]
        if ssm_cache.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise RuntimeError(
                "Qwen GDN ixformer kernel needs a float32/bfloat16/float16 SSM "
                f"cache (got {ssm_cache.dtype}). Set --mamba-ssm-cache-dtype."
            )
        # Packed pages are strided on dim0; the CUDA op accepts that stride.
        ssm_state = ssm_cache

        tokens = mixed_qkv.size(0)
        conv_out = self._conv_out[:tokens]
        has_init = attn_metadata.has_initial_state
        spec = attn_metadata.spec_sequence_masks is not None
        has_non_spec = attn_metadata.num_prefills > 0 or attn_metadata.num_decodes > 0
        if spec:
            spec_cu = attn_metadata.spec_query_start_loc
            spec_conv_idx = attn_metadata.spec_conv_state_indices
            accepted = attn_metadata.num_accepted_tokens
            assert spec_cu is not None
            assert spec_conv_idx is not None
            assert accepted is not None
            try:
                ixf_c_ops.infer.qwen_causal_conv1d_spec_update(
                    mixed_qkv,
                    self._conv_w,
                    conv_out,
                    conv_state,
                    spec_cu,
                    spec_conv_idx,
                    accepted,
                    attn_metadata.spec_token_starts,
                )
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "The installed ixformer is too old for Qwen GDN MTP; "
                    "rebuild and reinstall ixformer."
                ) from exc
        if has_non_spec:
            cu = attn_metadata.non_spec_query_start_loc
            idx = attn_metadata.non_spec_state_indices_tensor
            assert cu is not None and idx is not None
            has_init_u8 = None
            if attn_metadata.num_prefills > 0 and has_init is not None:
                has_init_u8 = self._ixformer_has_init(
                    has_init, int(has_init.numel()), as_uint8=True
                )
            ixf_c_ops.infer.qwen_causal_conv1d(
                mixed_qkv,
                self._conv_w,
                conv_out,
                conv_state,
                cu,
                idx,
                has_init_u8,
                attn_metadata.non_spec_token_starts,
            )

        query, key, value = self.packed_qkv_views(conv_out)
        assert self._g_buf is not None and self._beta_buf is not None
        g = self._g_buf[:tokens]
        beta = self._beta_buf[:tokens]
        ixf_c_ops.infer.qwen_gdn_gating(
            self.A_log, a, b, self.dt_bias, g, beta, 1.0, 20.0
        )
        scale = self.head_k_dim**-0.5
        if spec:
            spec_cu = attn_metadata.spec_query_start_loc
            spec_idx = attn_metadata.spec_state_indices_tensor
            accepted = attn_metadata.num_accepted_tokens
            assert spec_cu is not None and spec_idx is not None
            assert accepted is not None
            try:
                ixf_c_ops.infer.qwen_gdn_recurrent_fwd(
                    query,
                    key,
                    value,
                    core_attn_out[:tokens],
                    g,
                    beta,
                    ssm_state,
                    spec_cu,
                    spec_idx,
                    scale,
                    True,
                    accepted,
                    attn_metadata.spec_token_starts,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "The installed ixformer is too old for Qwen GDN MTP; "
                    "rebuild and reinstall ixformer."
                ) from exc
        if attn_metadata.num_prefills > 0:
            batch = int(cu.numel()) - 1
            has_init_i32 = self._ixformer_has_init(has_init, batch, as_uint8=False)
            assert has_init_i32 is not None
            chunk_offsets = attn_metadata.chunk_offsets
            assert chunk_offsets is not None
            n_max = max(1, int(attn_metadata.n_chunk_max))
            ixf_c_ops.infer.qwen_gdn_chunk_fwd(
                query,
                key,
                value,
                core_attn_out[:tokens],
                g,
                beta,
                ssm_state,
                has_init_i32,
                cu,
                idx,
                chunk_offsets,
                self._gdn_workspace,
                n_max,
                int(attn_metadata.n_chunk_total),
                scale,
                True,
                attn_metadata.non_spec_token_starts,
            )
        elif attn_metadata.num_decodes > 0:
            ixf_c_ops.infer.qwen_gdn_recurrent_fwd(
                query,
                key,
                value,
                core_attn_out[:tokens],
                g,
                beta,
                ssm_state,
                cu,
                idx,
                scale,
                True,
                None,
                attn_metadata.non_spec_token_starts,
            )

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
            self._qwen38_kernel() != "ixformer"
            and self.enable_packed_recurrent_decode
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

        use_ixformer_spec = (
            attn_metadata.spec_sequence_masks is not None
            and self._ixformer_spec_enabled()
        )
        if self._qwen38_kernel() == "ixformer" and (
            attn_metadata.spec_sequence_masks is None or use_ixformer_spec
        ):
            return self._forward_core_ixformer(
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
                (
                    spec_token_indx if attn_metadata.num_prefills > 0 else None
                ),  # none for cuda graph, else use spec_token_indx to inplace update
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
