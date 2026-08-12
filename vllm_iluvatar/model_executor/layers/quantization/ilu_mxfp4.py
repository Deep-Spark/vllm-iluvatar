# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP4 MoE quantization method.

Corresponds to: vllm/model_executor/layers/quantization/mxfp4.py

Target   : vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod
Baseline : main_9c12a81ac
Why      : community ``Mxfp4MoEMethod`` picks a backend through
           ``select_deepseek_v4_mxfp4_moe_backend``, and off ROCm that function
           can only return NVIDIA-specific backends (TRTLLM / DeepGEMM / Marlin /
           FlashInfer) or raise ``NotImplementedError``. Iluvatar needs the
           community weight layout with an Iluvatar compute.
How      : subclass ``Mxfp4MoEMethod`` and keep ``create_weights`` verbatim (so
           the four routed ``uint8`` parameters stay byte-identical to the
           checkpoint, which is also what Phase 2's zero-copy
           ``group_gemm_fp4`` needs); override only the five members listed below
           and route ``apply`` to ``ilu_mxfp4_fused_experts()``.
Scope    : routed MXFP4 experts on Iluvatar; selected by
           ``IluDeepseekV4FP8Config.get_quant_method``.
Remove   : when community exposes a vendor MXFP4 MoE backend hook.

Named after the **quantization format**, not the model: everything
DeepSeek-V4-specific (``swiglu_limit``, hash routing) already lives in
``moe_config`` / the router, so a GPT-OSS-MXFP4 checkpoint could reuse this class
as-is. The checkpoint-shaped decisions live one level up, in
``ilu_deepseek_v4_fp8.py``.

Overrides (five, each marked ``[ILU override]`` inline)
------------------------------------------------------
``__init__``                      do not call ``select_deepseek_v4_mxfp4_moe_backend``;
                                  pin ``mxfp4_backend`` to ``NONE`` and set no
                                  ``experts_cls`` attribute at all
``is_monolithic``                 hard ``False`` (we are a modular method)
``supports_eplb``                 hard ``False`` (Phase 1 records no expert load)
``process_weights_after_loading`` fallback: zero-copy dtype re-label;
                                  default: chunked
                                  ``preprocess_weight_for_ixmm`` on ivcore30
``get_fused_moe_quant_config``    ``None`` -> no community MoE kernel is built
``maybe_make_prepare_finalize``   ``None`` -> never wrapped into
                                  ``FusedMoEModularMethod``
``apply``                         wiring only; the compute lives in
                                  ``ilu_mxfp4_fused_moe.py``

Why ``experts_cls`` must be *absent* rather than ``None``
--------------------------------------------------------
Community ``FusedMoEMethodBase.is_monolithic``
(``fused_moe_method_base.py:163-170``) reads::

    if self.moe_kernel is None:
        if hasattr(self, "experts_cls"):
            return self.experts_cls.is_monolithic()
        return False

``Mxfp4MoEMethod.__init__`` (``mxfp4.py:481``) assigns ``self.experts_cls =
None``, so the ``hasattr`` guard passes and the attribute access raises
``AttributeError: 'NoneType' object has no attribute 'is_monolithic'``. That is
exactly the EP-only crash the retired patch P3
(``patch_dsv4_moe_modular_kernel``) worked around. Not setting the attribute
makes ``hasattr`` false; the explicit ``is_monolithic`` override below makes the
answer independent of that subtlety altogether. ``__init__`` asserts both.
"""

from __future__ import annotations

import torch

# Same import surface community ``quantization/mxfp4.py:8-15`` uses -- the
# package root, not the defining modules, which is what keeps this free of the
# fused_moe <-> quantization import cycle.
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)

_IXMM_EXPERT_CHUNK = 4


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    """The runner-side output scale, ``1.0`` unless a kernel opted in.

    Same helper the other Iluvatar MoE methods carry (``ilu_auto_awq.py:67``,
    ``ilu_auto_gptq.py:64``, ``ilu_compressed_tensors_moe_w4a8_int8.py:52``).
    ``IluMoERunner.__init__`` (``runner/ilu_moe_runner.py:67``) writes the
    attribute from the runner's ``routed_scaling_factor``, which the ``FusedMoE``
    factory pins to ``1.0`` whenever ``apply_routed_scale_to_output`` is False
    (``layer.py:414-416``) -- the default, and our case.

    Do **not** substitute ``layer.routed_scaling_factor``: that one is
    router-aligned (``1.5`` for DeepSeek-V4, ``layer.py:385-387``) and the router
    has already multiplied it into ``topk_weights``.
    """
    return getattr(layer, "_ilu_output_scale", 1.0)


class IluMxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 routed-expert method backed by ``ilu_mxfp4_fused_experts()``."""

    # [ILU override] Mxfp4MoEMethod.__init__ — delta: skip the parent body (it
    # calls select_deepseek_v4_mxfp4_moe_backend, which off ROCm either returns
    # an NVIDIA-only backend or raises) and mirror its remaining field
    # initialisation, minus ``experts_cls``.
    #
    # Mirrors community ``mxfp4.py:478-489``. If that constructor grows a field
    # the asserts below will not catch it, but every consumer of a missing field
    # raises ``AttributeError`` loudly rather than computing a wrong number.
    def __init__(self, moe: FusedMoEConfig):
        # Skip Mxfp4MoEMethod.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "mxfp4"
        # NONE is still required (not merely convenient): community
        # ``maybe_roundup_sizes`` (mxfp4.py:507), ``skip_forward_padding``
        # (:494) and ``has_unpadded_output`` (:499) all read this attribute, and
        # ``maybe_roundup_sizes`` runs during FusedMoE construction. Under NONE
        # the round-up degrades to ``round_up(intermediate, 64)``
        # (``oracle/mxfp4.py:671-672``), a no-op for DeepSeek-V4's 4096 / 2048.
        self.mxfp4_backend = Mxfp4MoeBackend.NONE
        self.max_capture_size = moe.max_capture_size
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel = None
        self.w13_precision_config = None
        self.w2_precision_config = None

        assert not hasattr(self, "experts_cls"), (
            "IluMxfp4MoEMethod must not define experts_cls; community "
            "is_monolithic() would dereference it (see module docstring)"
        )
        assert self.is_monolithic is False

    # [ILU override] FusedMoEMethodBase.is_monolithic — delta: constant False.
    # We always take the modular path (router first, then
    # ``RoutedExperts.forward_modular`` -> ``apply``), so the answer never
    # depends on ``moe_kernel`` / ``experts_cls`` introspection.
    @property
    def is_monolithic(self) -> bool:
        return False

    # [ILU override] Mxfp4MoEMethod.supports_eplb — delta: community returns
    # True. Phase 1's torch expert loop records no expert load, so claiming EPLB
    # support would silently disable rebalancing. Returning False makes
    # ``RoutedExperts.__init__`` (``routed_experts.py:138-150``) raise
    # ``NotImplementedError`` when EPLB is requested, which is the same
    # behaviour as the W4A8 method.
    @property
    def supports_eplb(self) -> bool:
        return False

    # [ILU override] Mxfp4MoEMethod.process_weights_after_loading — delta:
    # Re-labels or converts MXFP4 tensors to the target layout at load time.
    # The selected backend always sees the layout matching its dtype contract.
    #
    #   w*_weight       uint8 [E, n, k/2] -> int8
    #       Reason §2.2: ``group_gemm_fp4`` dispatches on ``weight.dtype``; the
    #       original MXFP4 layout MUST be ``int8`` or it gets misread as the
    #       ixmm preprocessed ``uint8 [E, k/2, n]`` format (silently wrong on
    #       symmetric shapes, e.g. some TP splits of w2).
    #   w*_weight_scale uint8 [E, n, k/32] -> float8_e8m0fnu
    #       Reason §2.1: the facade's ``.float()`` converts a ``uint8`` to
    #       0..255 integers, not ``2**(v-127)``; the e8m0 decode only fires for
    #       an fp8 dtype view. Passing raw uint8 scales silently scales the
    #       weights by ~1e38.
    #
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        import vllm.envs as envs

        backend = envs.VLLM_ILU_MXFP4_MOE_BACKEND
        if backend not in ("auto", "torch", "ixformer_native", "ixmm"):
            raise ValueError(
                "VLLM_ILU_MXFP4_MOE_BACKEND must be one of auto/torch/"
                f"ixformer_native/ixmm, got {backend!r}"
            )

        use_ixmm = backend == "ixmm" or (
            backend == "auto" and _ixmm_backend_available()
        )
        if backend == "ixmm" and not _ixmm_backend_available():
            raise RuntimeError(
                "VLLM_ILU_MXFP4_MOE_BACKEND=ixmm, but ixmm is unavailable. "
                "Install the ixmm wheel and rebuild ixformer with HAVE_IXMM "
                "for ivcore30."
            )

        if use_ixmm:
            # Lazy import keeps CPU/native-only imports independent of ixformer.
            import ixformer.inference.functions as ixfops  # type: ignore

        for w_name, s_name in (
            ("w13_weight", "w13_weight_scale"),
            ("w2_weight", "w2_weight_scale"),
        ):
            w = getattr(layer, w_name)
            s = getattr(layer, s_name)
            assert w.dtype is torch.uint8 and w.is_contiguous(), (
                f"{w_name} expected uint8 contiguous, got {w.dtype} "
                f"contiguous={w.is_contiguous()}"
            )
            assert s.dtype is torch.uint8 and s.is_contiguous(), (
                f"{s_name} expected uint8 contiguous, got {s.dtype} "
                f"contiguous={s.is_contiguous()}"
            )

            w_int8 = w.data.view(torch.int8)
            s_e8m0 = s.data.view(torch.float8_e8m0fnu)
            if not use_ixmm:
                replace_parameter(layer, w_name, w_int8)
                replace_parameter(layer, s_name, s_e8m0)
                continue

            # Chunk by expert to cap preprocessing temporaries. For DSv4 w13,
            # processing all 32 local experts at once would require several GB.
            weight_parts: list[torch.Tensor] = []
            scale_parts: list[torch.Tensor] = []
            for begin in range(0, w_int8.shape[0], _IXMM_EXPERT_CHUNK):
                end = min(begin + _IXMM_EXPERT_CHUNK, w_int8.shape[0])
                ixmm_weight, ixmm_scale = ixfops.preprocess_weight_for_ixmm(
                    w_int8[begin:end],
                    s_e8m0[begin:end],
                )
                weight_parts.append(ixmm_weight)
                scale_parts.append(ixmm_scale)

            ixmm_weight = torch.cat(weight_parts, dim=0).contiguous()
            ixmm_scale = torch.cat(scale_parts, dim=0).contiguous()
            expected_weight_shape = (w.shape[0], w.shape[2], w.shape[1])
            expected_scale_shape = (
                w.shape[0],
                (w.shape[2] * 2) // 256,
                w.shape[1],
            )
            if ixmm_weight.dtype is not torch.uint8:
                raise TypeError(
                    f"preprocess_weight_for_ixmm returned {ixmm_weight.dtype} "
                    f"for {w_name}, expected torch.uint8"
                )
            if ixmm_scale.dtype is not torch.float32:
                raise TypeError(
                    f"preprocess_weight_for_ixmm returned {ixmm_scale.dtype} "
                    f"for {s_name}, expected torch.float32"
                )
            if tuple(ixmm_weight.shape) != expected_weight_shape:
                raise ValueError(
                    f"preprocess_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_weight.shape)} for {w_name}, expected "
                    f"{expected_weight_shape}"
                )
            if tuple(ixmm_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"preprocess_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_scale.shape)} for {s_name}, expected "
                    f"{expected_scale_shape}"
                )
            replace_parameter(layer, w_name, ixmm_weight)
            replace_parameter(layer, s_name, ixmm_scale)

        layer._ilu_mxfp4_ixmm = use_ixmm

    # [ILU override] Mxfp4MoEMethod.get_fused_moe_quant_config — delta: None.
    # A ``FusedMoEQuantConfig`` only exists to configure a community modular
    # kernel; we build none. Returning None also keeps ``topk_indices_dtype``
    # at ``None`` (``fused_moe_method_base.py:135-140``), so the router emits
    # int32 ids -- the dtype ``ilu_mxfp4_fused_experts`` expects. Same choice as
    # the W4A8 method (``ilu_compressed_tensors_moe_w4a8_int8.py:491-494``).
    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    # [ILU override] FusedMoEMethodBase.maybe_make_prepare_finalize — delta:
    # None. Community would build a prepare/finalize pair once
    # ``use_all2all_kernels`` is true (EP together with DP / PCP / SP), and
    # ``MoERunner.maybe_init_modular_kernel`` would then wrap us into a
    # ``FusedMoEModularMethod`` whose ``apply`` expects a real kernel. Iluvatar
    # dispatch/combine is the naive EP-group path instead
    # (``IluMoERunner.do_naive_dispatch_combine``), so we opt out explicitly
    # rather than relying on the deployment happening to have ``dp_size == 1``.
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        return None

    # [ILU override] Mxfp4MoEMethod.apply — delta: wiring only. Community
    # delegates to ``self.moe_kernel.apply``; we call the plugin's compute
    # function with explicit tensors.
    #
    # ``shared_experts`` is accepted for signature compatibility and
    # deliberately unused: the community modular kernel only calls it under
    # ``SharedExpertsOrder.MK_INTERNAL_OVERLAPPED``, which requires
    # ``mk_can_overlap_shared_experts`` (``fused_moe_method_base.py:45-50``,
    # ``moe_kernel is not None``) and is therefore unreachable here. The shared
    # output arrives through ``shared_experts_input`` instead -- see
    # ``ilu_mxfp4_fused_experts``' ``extra_residual``.
    #
    # ``run_decode`` is not passed: ``group_gemm_fp4`` auto-selects gemv/gemm via
    # ``mode=None``, and computing it cost a ``torch.cuda.is_available()`` plus a
    # scan of the whole ``attn_metadata`` dict for every MoE layer on every step.
    # ``backend=None`` lets ``VLLM_ILU_MXFP4_MOE_BACKEND`` decide (default ixmm).
    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return ilu_mxfp4_fused_experts(
            hidden_states=x,
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            expert_map=layer.expert_map,
            output_scale=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
            w13_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            backend=None,
        )
