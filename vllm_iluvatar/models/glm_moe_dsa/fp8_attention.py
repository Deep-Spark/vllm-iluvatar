# SPDX-License-Identifier: Apache-2.0
"""FP8 variants of the GLM-5.2 attention layers (official ZhipuAI FP8 weights).

Thin subclasses of the W4A8 chain in :mod:`.attention`; only the pieces that
differ are overridden. Everything else -- the DSA indexer, RoPE, KV insert,
sparse MLA kernels, FP8 KV packing -- is shared verbatim.

What differs from W4A8, and why:

* ``kv_b_proj``. The W4A8 checkpoint routes it to ``IluGlmKVBProjW8A16Method``
  (triggered by ``mla_bmm_w8a16`` in the checkpoint's ``quantization_config``),
  which splits it into Marlin-repacked ``W_UK`` / ``W_UV``. The official FP8
  checkpoint has no such field, so ``kv_b_proj`` falls to the ordinary
  ``IluFp8LinearMethod`` and ``W_UK`` / ``W_UV`` never exist -- the absorbed
  up-proj has nothing to read. This module builds them.

* The RMSNorm+RoPE+FP8-KV-insert fusion is pinned off, see ``__init__``.

* Everything the base class already handles by feature detection is left
  alone. Notably the INT8 norm+quant fusion disarms itself: its guard requires
  ``q_b_proj.scheme`` to be ``IluCompressedTensorsW8A8Int8``, which FP8 layers
  do not have, so ``forward`` falls back to the split RMSNorm path and the
  tuple-unpacking branch is never taken.

Absorbed up-proj in FP8
-----------------------
Both halves run as ``fp8_grouped_per_token_quant`` (activation) followed by
``fp8_per_channel_einsum`` (grouped fp8 Tensor-Core GEMM), with one group per
head (``G = num_heads``, ``heads_per_group = 1``)::

    k-up:  q_nope [T,H,192] -> a[H,T,192] x W_UK[H,512,192] -> [T,H,512]
    v-up:  x      [T,H,512] -> a[H,T,512] x W_UV[H,256,512] -> [T,H,256]

``fp8_per_channel_einsum`` wants the weight scale on the *output* axis, which
the two halves reach differently (see :meth:`_build_fp8_absorb`):

* v-up is free. ``kv_b_proj``'s per-channel scale is one entry per output row
  of ``[H*(nope+v), lora]``, and v-up contracts over ``lora`` -- so the v rows
  are already ``[H, v, lora]`` fp8 with a per-output-channel ``[H, v]`` scale.
  The checkpoint bytes are reused verbatim; no requantisation, no extra error.

* k-up is not. It contracts over ``nope``, so the checkpoint's per-``nope``-row
  scale lands on the *contraction* axis of the transposed weight. There is no
  contraction-axis-scaled fp8 GEMM, so the weight is dequantised once at load
  time and requantised per output channel (``lora``). A scale-direction probe
  on the real weights (``llm-adapt-notes/kernels/GLM-5.2/kv-b-proj-fp8/
  scale_direction_probe.py``) measured the two directions as equally accurate,
  which is what makes the requant an acceptable trade for keeping the existing
  GEMM.

Weight cost is ~3.7 MiB per layer at TP4, half the BF16 fallback's ~7.3 MiB and
a quarter of the ~14 MiB ``kv_b_proj.weight`` it replaces. Set
``VLLM_ILU_GLM_DSA_FP8_ABSORB_UP=0`` to fall back to the BF16 grouped einsums
for an A/B.
"""

import ixformer.inference.functions as ixf_ops
import torch
import vllm.envs as envs
from vllm.logger import init_logger

from vllm_iluvatar.models.glm_moe_dsa.attention import (
    IluGlmMoeDsaMLAAttention,
    IluGlmMoeDsaSparseMLA,
)

logger = init_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


class IluGlmMoeDsaFp8SparseMLA(IluGlmMoeDsaSparseMLA):
    """Sparse MLA whose absorbed up-proj is built from the FP8 ``kv_b_proj``.

    The build has to run *after* the FP8 linear method has requantised
    ``kv_b_proj`` to per-channel, which happens in the loader's
    ``process_weights_after_loading`` pass. Rather than guess at that timing we
    chain onto the tail of ``kv_b_proj``'s own hook (:meth:`_install_absorb_hook`),
    so the ordering holds however the loader drives the pass -- including the
    ``device_loading_context`` that may have moved the weight to GPU first.
    """

    # FP8 path: W_UK [H, lora, nope] + W_UK_S [H, lora] (read-safe),
    #           W_UV [H, v, lora]    + W_UV_S [H, v]    (read-safe).
    # BF16 fallback: W_UK [H, nope, lora], W_UV [H, v, lora], scales None.
    W_UK: torch.Tensor | None = None
    W_UV: torch.Tensor | None = None
    W_UK_S: torch.Tensor | None = None
    W_UV_S: torch.Tensor | None = None
    _absorb_logged: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The fused RMSNorm+RoPE+FP8-KV-insert kernel gives wrong numerics, and
        # ``VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV`` defaults to on -- the base class
        # arms the fusion the moment an offline scale table freezes the scale in
        # ``__init__``, so online calibration is the only reason FP8 has not hit
        # it yet. Pin it off here instead of relying on the env switch.
        self.use_rms_rope_fp8_kv_fusion = False
        self.use_fp8_absorb_up = envs.VLLM_ILU_GLM_DSA_FP8_ABSORB_UP
        self._install_absorb_hook()

    def _install_absorb_hook(self) -> None:
        layer = self.kv_b_proj
        quant_method = getattr(layer, "quant_method", None)
        if quant_method is None:
            raise RuntimeError(
                f"{self.prefix}: kv_b_proj has no quant_method, so the FP8 "
                "absorbed up-proj cannot be scheduled. The FP8 model expects "
                "kv_b_proj to be handled by IluFp8LinearMethod."
            )
        if getattr(layer, "_ilu_fp8_absorb_hooked", False):
            return
        layer._ilu_fp8_absorb_hooked = True

        original = quant_method.process_weights_after_loading

        def hooked(processed_layer, *args, **kwargs):
            original(processed_layer, *args, **kwargs)
            # ``quant_method`` is per-layer today, but guard in case a future
            # config hands the same instance to several layers.
            if processed_layer is layer:
                self._build_absorb()

        quant_method.process_weights_after_loading = hooked

    def _read_kv_b_weight(self) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """``(weight[H, nope+v, lora] e4m3, scale[H, nope+v] fp32, geometry)``.

        Runs after ``IluFp8BlockScaledMMKernel.process_weights_after_loading``
        has requantised the weight in place to per-channel FP8, so the source
        here is ``weight[N, K]`` e4m3 plus ``scale[N]`` -- not the original
        128x128 block scale. That costs a little accuracy versus dequantising
        straight from the block scale, but it is the same per-channel precision
        every other FP8 linear in the model runs at.
        """
        layer = self.kv_b_proj
        geom = layer.bmm_geom
        heads, nope, v, lora = (
            geom.num_heads,
            geom.qk_nope_head_dim,
            geom.v_head_dim,
            geom.kv_lora_rank,
        )

        weight = getattr(layer, "weight", None)
        if weight is None:
            raise RuntimeError(
                f"{self.prefix}: kv_b_proj has no 'weight' to build the absorbed "
                "up-proj from. Was process_weights_after_loading already run "
                "with a method that frees it (e.g. the W4A8 "
                "IluGlmKVBProjW8A16Method)?"
            )
        weight = weight.data
        if tuple(weight.shape) != (heads * (nope + v), lora):
            raise ValueError(
                f"{self.prefix}: kv_b_proj weight shape {tuple(weight.shape)} "
                f"!= ({heads} * ({nope} + {v}), {lora}) implied by bmm_geom"
            )

        # Per-channel scale along N. The FP8 linear kernel leaves it under
        # either name depending on how the requant path resolved.
        scale = getattr(layer, "weight_scale", None)
        if scale is None:
            scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            raise RuntimeError(
                f"{self.prefix}: kv_b_proj has neither weight_scale nor "
                "weight_scale_inv; cannot dequantise."
            )
        scale = scale.data.reshape(-1)
        if scale.numel() != weight.shape[0]:
            raise ValueError(
                f"{self.prefix}: expected a per-channel kv_b_proj scale of "
                f"{weight.shape[0]} elements, got {scale.numel()}. A 2-D block "
                "scale means process_weights_after_loading has not run yet."
            )

        return (
            weight.view(heads, nope + v, lora),
            scale.float().view(heads, nope + v),
            (heads, nope, v, lora),
        )

    def _build_absorb(self) -> None:
        """Split ``kv_b_proj`` into W_UK / W_UV and free the packed weight.

        ``kv_b_proj.weight`` is dropped afterwards: nothing else reads it (the
        absorbed form is the only consumer) and it is ~14 MiB per layer.
        """
        if self.use_fp8_absorb_up:
            self._build_fp8_absorb()
        else:
            self._build_bf16_absorb()
        delattr(self.kv_b_proj, "weight")
        # One line per layer is too noisy at 78 layers; log the first only.
        # It is the only way to tell the two paths apart in a serving log.
        if not IluGlmMoeDsaFp8SparseMLA._absorb_logged:
            IluGlmMoeDsaFp8SparseMLA._absorb_logged = True
            bytes_per_layer = (
                self.W_UK.numel() * self.W_UK.element_size()
                + self.W_UV.numel() * self.W_UV.element_size()
            )
            logger.info(
                "GLM DSA absorbed up-proj: %s (W_UK %s %s, W_UV %s %s, %.2f MiB/layer)",
                "fp8" if self.use_fp8_absorb_up else "bf16",
                tuple(self.W_UK.shape),
                self.W_UK.dtype,
                tuple(self.W_UV.shape),
                self.W_UV.dtype,
                bytes_per_layer / 2**20,
            )

    def _build_fp8_absorb(self) -> None:
        """FP8 W_UK / W_UV laid out for :func:`fp8_per_channel_einsum`."""
        w, s, (heads, nope, v, lora) = self._read_kv_b_weight()

        # --- v-up: checkpoint bytes reused as-is -----------------------------
        # einsum wants b[G, D, R] with Bs[G, D]; v-up has G=H, D=v, R=lora,
        # which is exactly the v rows of kv_b_proj and their per-row scales.
        self.W_UV = w[:, nope:, :].contiguous()
        self.W_UV_S = ixf_ops.alloc_readsafe_scale(heads, v, device=w.device)
        self.W_UV_S.copy_(s[:, nope:])

        # --- k-up: transpose + requantise onto the output axis ---------------
        # b[G, D, R] with G=H, D=lora, R=nope. The checkpoint scale is per nope
        # row, i.e. per R after the transpose, which the einsum cannot consume;
        # dequantise and rescale per lora channel instead.
        wk = w[:, :nope, :].float() * s[:, :nope].unsqueeze(-1)  # [H, nope, lora]
        wk = wk.transpose(1, 2).contiguous()  # [H, lora, nope]
        ks = (wk.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)  # [H, lora]
        self.W_UK = (
            (wk / ks.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        ).contiguous()
        self.W_UK_S = ixf_ops.alloc_readsafe_scale(heads, lora, device=w.device)
        self.W_UK_S.copy_(ks)

    def _build_bf16_absorb(self) -> None:
        """A/B baseline: BF16 grouped GEMMs off a dequantised ``kv_b_proj``."""
        w, s, (heads, nope, v, lora) = self._read_kv_b_weight()
        wd = (w.float() * s.unsqueeze(-1)).to(torch.bfloat16)
        # k-up consumes [H, nope, lora]; v-up consumes [H, v, lora]. Both are
        # used as ``einsum`` operands below, so no Marlin-style repack.
        self.W_UK = wd[:, :nope, :].contiguous()
        self.W_UV = wd[:, nope:, :].contiguous()
        self.W_UK_S = None
        self.W_UV_S = None

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        # (T, H, nope) -> (T, H, lora)
        if self.W_UK_S is None:
            return torch.einsum("thp,hpl->thl", q_nope, self.W_UK)
        # q_nope is a split view of q, so its head stride is qk_head_dim, not
        # nope; the quant kernel reads it strided rather than forcing a copy.
        a, a_s = ixf_ops.fp8_grouped_per_token_quant(q_nope, q_nope.shape[1])
        return ixf_ops.fp8_per_channel_einsum(a, self.W_UK, a_s, self.W_UK_S)

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        # (T, H, lora) -> (T, H, v)
        if self.W_UV_S is None:
            return torch.einsum("thl,hvl->thv", x, self.W_UV)
        a, a_s = ixf_ops.fp8_grouped_per_token_quant(x, x.shape[1])
        return ixf_ops.fp8_per_channel_einsum(a, self.W_UV, a_s, self.W_UV_S)


class IluGlmMoeDsaFp8MLAAttention(IluGlmMoeDsaMLAAttention):
    sparse_mla_cls = IluGlmMoeDsaFp8SparseMLA
