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
  up-proj has nothing to read.

  k-up is also the one GEMM with no ready FP8 kernel: its per-channel scale
  sits on the *contraction* axis, so ``fp8_per_channel_einsum`` (scale on the
  output axis) does not apply. Until the dedicated kernel lands, both k-up and
  v-up run as BF16 grouped GEMMs off a dequantised copy of ``kv_b_proj``
  (``_build_bf16_absorb``). Cost is ~7.3 MiB per layer at TP4.

* The RMSNorm+RoPE+FP8-KV-insert fusion is pinned off, see ``__init__``.

* Everything the base class already handles by feature detection is left
  alone. Notably the INT8 norm+quant fusion disarms itself: its guard requires
  ``q_b_proj.scheme`` to be ``IluCompressedTensorsW8A8Int8``, which FP8 layers
  do not have, so ``forward`` falls back to the split RMSNorm path and the
  tuple-unpacking branch is never taken.
"""

import torch

from vllm_iluvatar.models.glm_moe_dsa.attention import (
    IluGlmMoeDsaMLAAttention,
    IluGlmMoeDsaSparseMLA,
)


class IluGlmMoeDsaFp8SparseMLA(IluGlmMoeDsaSparseMLA):
    """Sparse MLA whose absorbed up-proj is a BF16 grouped GEMM.

    ``_build_bf16_absorb`` has to run *after* the FP8 linear method has
    requantised ``kv_b_proj`` to per-channel, which happens in the loader's
    ``process_weights_after_loading`` pass. Rather than guess at that timing we
    chain onto the tail of ``kv_b_proj``'s own hook (:meth:`_install_absorb_hook`),
    so the ordering holds however the loader drives the pass -- including the
    ``device_loading_context`` that may have moved the weight to GPU first.
    """

    W_UK: torch.Tensor | None = None
    W_UV: torch.Tensor | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The fused RMSNorm+RoPE+FP8-KV-insert kernel gives wrong numerics, and
        # ``VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV`` defaults to on -- the base class
        # arms the fusion the moment an offline scale table freezes the scale in
        # ``__init__``, so online calibration is the only reason FP8 has not hit
        # it yet. Pin it off here instead of relying on the env switch.
        self.use_rms_rope_fp8_kv_fusion = False
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
                self._build_bf16_absorb()

        quant_method.process_weights_after_loading = hooked

    def _build_bf16_absorb(self) -> None:
        """Dequantise ``kv_b_proj`` to BF16 and split into W_UK / W_UV.

        Runs after ``IluFp8BlockScaledMMKernel.process_weights_after_loading``
        has requantised the weight in place to per-channel FP8, so the source
        here is ``weight[N, K]`` e4m3 plus ``scale[N]`` -- not the original
        128x128 block scale. That costs a little accuracy versus dequantising
        straight from the block scale, but it is the same per-channel precision
        every other FP8 linear in the model runs at, and this path is a
        stepping stone to the real kernel anyway.

        ``kv_b_proj.weight`` is freed afterwards: nothing else reads it (the
        absorbed form is the only consumer) and it is ~14 MiB per layer.
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
                f"{self.prefix}: kv_b_proj has no 'weight' to build the BF16 "
                "absorbed up-proj from. Was process_weights_after_loading "
                "already run with a method that frees it (e.g. the W4A8 "
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

        w = (weight.to(torch.float32) * scale.unsqueeze(1)).to(torch.bfloat16)
        w = w.view(heads, nope + v, lora)
        # k-up consumes [H, nope, lora]; v-up consumes [H, v, lora]. Both are
        # used as ``einsum`` operands below, so no Marlin-style repack.
        self.W_UK = w[:, :nope, :].contiguous()
        self.W_UV = w[:, nope:, :].contiguous()

        delattr(layer, "weight")

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        # (T, H, nope) @ [H, nope, lora] -> (T, H, lora)
        return torch.einsum("thp,hpl->thl", q_nope, self.W_UK)

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        # (T, H, lora) @ [H, v, lora]^T -> (T, H, v)
        return torch.einsum("thl,hvl->thv", x, self.W_UV)


class IluGlmMoeDsaFp8MLAAttention(IluGlmMoeDsaMLAAttention):
    sparse_mla_cls = IluGlmMoeDsaFp8SparseMLA
