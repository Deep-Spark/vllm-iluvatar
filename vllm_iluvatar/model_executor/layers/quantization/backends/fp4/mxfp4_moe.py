# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP4 MoE quantization method.

Corresponds to: vllm/model_executor/layers/quantization/mxfp4.py

Target   : vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod
Baseline : v0.31.0 (db9527a468)
Why      : community ``Mxfp4MoEMethod`` picks a backend through
           ``select_deepseek_v4_mxfp4_moe_backend``, and off ROCm that function
           can only return NVIDIA-specific backends (TRTLLM / DeepGEMM / Marlin /
           FlashInfer) or raise ``NotImplementedError``. Iluvatar needs the
           community weight layout with an Iluvatar compute.
How      : subclass the community Method. Selected compute backends declare
           final weight contracts; MXFP4 loaders convert paired checkpoint
           shards directly into those parameters. Post-load only finalizes and
           binds compute. Unselected legacy torch/native and runtime-quant
           entry points retain their existing implementation.
Scope    : routed MXFP4 experts on Iluvatar; selected by
           ``IluDeepseekV4FP8Config.get_quant_method``.
Remove   : when community exposes a vendor MXFP4 MoE backend hook.

The Method is named after the source quantization format. Model routing and
activation parameters come from ``moe_config`` / the router. Reuse by another
model still requires compatible checkpoint shard, activation and communication
contracts; the shared MXFP4 name alone does not establish compatibility.

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
from vllm.config import get_current_vllm_config_or_none

# Same import surface community ``quantization/mxfp4.py:8-15`` uses -- the
# package root, not the defining modules, which is what keeps this free of the
# fused_moe <-> quantization import cycle.
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
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
    """Load MXFP4 checkpoints and compose the selected runtime expert backend.

    Explicit torch/native modes retain the original checkpoint execution path.
    """

    # [ILU override] Mxfp4MoEMethod.__init__ — delta: skip the parent body (it
    # calls select_deepseek_v4_mxfp4_moe_backend, which off ROCm either returns
    # an NVIDIA-only backend or raises) and mirror its remaining field
    # initialisation, minus ``experts_cls``.
    #
    # Mirrors community ``mxfp4.py:478-489``. If that constructor grows a field
    # the asserts below will not catch it, but every consumer of a missing field
    # raises ``AttributeError`` loudly rather than computing a wrong number.
    def __init__(self, moe: FusedMoEConfig, selection=None):
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

        self.selection = selection
        self._compute = None
        self._runtime_weights = None
        self._checkpoint_loader = None
        if selection is not None:
            self.ilu_compute_backend = selection.name
            self.ilu_weight_format = selection.layout
            from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
                get_moe_backend,
            )

            self.ilu_prefer_hybrid_ep = get_moe_backend(selection).prefer_hybrid_ep

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
    def _legacy_process_weights_after_loading(self, layer: RoutedExperts) -> None:
        import vllm.envs as envs

        backend = (
            getattr(self, "compute_backend", None) or envs.VLLM_ILU_MXFP4_MOE_BACKEND
        )
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

    # [ILU override] Mxfp4MoEMethod.get_fused_moe_quant_config — delta: describe
    # the ix30 tensors ``process_weights_after_loading`` left on the layer (FP8
    # activations over MXFP4 weights) instead of returning None.
    #
    # ``apply`` still reads the weights straight off the layer, and no community
    # MoE kernel is built either way -- ``moe_kernel`` stays ``None``, so
    # ``topk_indices_dtype`` stays None
    # (``fused_moe_method_base.py:109-112``) and the router still emits the
    # int32 ids ``ilu_mxfp4_fused_experts`` expects. The config exists for the
    # batched DP+EP path: ``select_gemm_impl`` hands ``w1_scale`` / ``w2_scale``
    # to ``IluMoEBatchedExpertsFp8``, and ``nvsm_dispatch_quant_kwargs``
    # (``prepare_finalize/ilu_deepep_ll.py:42-68``) reads the ``"mxfp4"`` weight
    # dtype to put the LL dispatch into FP8 mode. Without it that helper falls
    # back to ``use_int8=VLLM_ILU_DEEPEP_USE_INT8, use_fp8=False`` and the
    # dispatched activations no longer match what the experts consume.
    #
    # ``gemm1_clamp_limit`` carries DeepSeek-V4's ``swiglu_limit`` through to the
    # batched activation; see ``select_gemm_impl`` for why a missing limit is a
    # hard error rather than "no clamp".
    def _legacy_get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    # [ILU override] — delta: return None.
    #
    # vLLM v0.31.0 (db9527a468) has no
    # ``FusedMoEMethodBase.maybe_make_prepare_finalize``. The factory is the
    # free function ``all2all_utils.maybe_make_prepare_finalize``, and nothing
    # on this pin calls it through the quant method. With
    # ``allow_new_interface`` left false it returns None for naive /
    # allgather_reducescatter (``all2all_utils.py:189``; the AG-RS branch at
    # ``:392`` also requires the flag). Returning None here keeps
    # ``moe_kernel`` unset, so ``supports_internal_mk`` stays False and
    # ``IluFusedMoEEP`` keeps driving dispatch/combine.
    #
    # The low-latency kernel is built in
    # ``IluFusedMoEEP._get_forward_dp_ep_ll_moe_kernel`` via
    # ``select_gemm_impl``, and only when the configured backend is not
    # already ``deepep_low_latency``. That function returns ``moe_kernel``
    # for a configured LL backend; this method leaves it None, so
    # ``forward_dp_ep_deep_ep`` raises ``NotImplementedError``. The supported
    # MXFP4 entry is ``VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`` on top
    # of a non-LL configured backend.
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        return None

    # [ILU override] FusedMoEMethodBase.select_gemm_impl — delta: build the
    # batched FP4-weight / FP8-activation experts instead of raising.
    #
    # Reached from ``IluFusedMoEEP._get_forward_dp_ep_ll_moe_kernel`` when
    # ``VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`` and the configured
    # backend is not itself ``deepep_low_latency``. Under ``ixmm`` our
    # ``process_weights_after_loading`` leaves exactly the ix30 pair
    # (``uint8 [E, k/2, n]`` weight + ``float32 [E, k/256, n]`` running-acc
    # scale) that ``IluMoEBatchedExpertsFp8`` was written against -- the same
    # tensors ``IluFp8MoEMethod`` produces by dequantising block-FP8 -- so the
    # experts are reused verbatim rather than reimplemented here.
    def _legacy_select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> FusedMoEExpertsModular:
        # Private import on purpose: the LL decode sizing rule belongs to
        # whichever method owns the NVSHMEM buffer, and both FP8 and MXFP4
        # routed experts share one.
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.batch_experts import (  # noqa: E501
            IluMoEBatchedExpertsFp8,
        )
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
            _shrink_ll_max_tokens_to_decode,
        )

        if (
            prepare_finalize.activation_format
            != FusedMoEActivationFormat.BatchedExperts
        ):
            raise NotImplementedError(
                "IluMoEBatchedExpertsFp8 requires the batched activation format, "
                f"got {prepare_finalize.activation_format}"
            )

        # The fallback layouts keep the checkpoint's ``int8 [E, n, k/2]``
        # weights and ``float8_e8m0fnu`` scales, which the batched experts
        # reject by dtype. Fail here instead of at the first decode step.
        if not getattr(layer, "_ilu_mxfp4_ixmm", False):
            raise NotImplementedError(
                "batched MXFP4 experts require the ixmm weight layout; "
                "VLLM_ILU_MXFP4_MOE_BACKEND resolved to a torch / "
                "ixformer_native layout. Use the configured all2all backend "
                "(no decode override) or set VLLM_ILU_MXFP4_MOE_BACKEND=ixmm."
            )

        assert self.moe_quant_config is not None
        activation = getattr(layer.activation, "value", layer.activation)
        swiglu_limit = self.moe_quant_config.gemm1_clamp_limit
        if activation in ("silu", "swiglu"):
            # ``IluMoEBatchedExpertsFp8`` maps silu/swiglu onto
            # ``moe_group_gemv_swiglu_quant_fp8(act_type="swiglu")``, whose
            # clamp is DeepSeek-V4's ``SiluAndMulWithClamp`` only while a limit
            # is supplied -- a ``None`` silently degrades to plain SwiGLU
            # instead of raising the way the legacy ``dsv4_clamp`` variant does.
            if swiglu_limit is None or swiglu_limit <= 0:
                raise ValueError(
                    "batched MXFP4 experts need a positive swiglu_limit for "
                    f"activation {activation!r} (clamp-before-silu), got "
                    f"{swiglu_limit!r}"
                )
        elif activation != "swiglustep":
            raise NotImplementedError(
                "batched MXFP4 experts support silu/swiglu (clamped) and "
                f"swiglustep activations, got {activation!r}"
            )

        _shrink_ll_max_tokens_to_decode(prepare_finalize)
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        if max_num_tokens is None:
            raise ValueError("batched MXFP4 experts require max_num_tokens")
        return IluMoEBatchedExpertsFp8(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            swiglu_limit=swiglu_limit,
            # ``apply`` folds this into ``ilu_mxfp4_fused_experts``'
            # ``output_scale``; the batched combine reads it back off the
            # experts (``ilu_fused_moe_ep.py:889-896``), so pass it through
            # rather than letting the two paths disagree if it ever leaves 1.0.
            runner_scaling_factor=_get_ilu_output_scale(layer),
        )

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
    def _legacy_apply(
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

    def _compute_config(self, layer):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEComputeConfig,
        )

        return MoEComputeConfig(
            hidden=layer.moe_config.hidden_dim,
            intermediate=layer.moe_config.intermediate_size_per_partition,
            local_num_experts=layer.local_num_experts,
            global_num_experts=layer.global_num_experts,
            ep_rank=layer.moe_config.ep_rank,
            activation=getattr(layer.activation, "value", layer.activation),
            limit=getattr(layer, "swiglu_limit", None),
            output_scale=float(_get_ilu_output_scale(layer)),
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        if self.selection is None:
            return super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **attrs,
            )
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            make_moe_compute,
        )
        from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
            Mxfp4LoadSpec,
            make_mxfp4_loader,
        )

        if self._checkpoint_loader is not None:
            raise RuntimeError("MXFP4 Method cannot create weights twice")
        if (
            self.moe.has_bias
            or not self.moe.is_act_and_mul
            or self.moe.hidden_dim_unpadded != hidden_size
            or self.moe.intermediate_size_per_partition_unpadded
            not in (None, intermediate_size_per_partition)
        ):
            raise ValueError(
                "MXFP4 streaming requires unpadded bias-free gated experts"
            )
        config = get_current_vllm_config_or_none()
        if config is not None:
            fmt = config.load_config.load_format
            if str(getattr(fmt, "value", fmt)) not in (
                "auto",
                "safetensors",
                "pt",
                "npcache",
            ):
                raise ValueError(
                    "MXFP4 streaming requires a sequential checkpoint loader"
                )
        backend = make_moe_compute(self._compute_config(layer), self.selection)
        if backend.requires_contiguous_experts and layer.expert_map is not None:
            start = layer.moe_config.ep_rank * layer.local_num_experts
            expected = torch.full_like(layer.expert_map, -1)
            expected[start : start + layer.local_num_experts] = torch.arange(
                layer.local_num_experts, device=expected.device, dtype=expected.dtype
            )
            if not torch.equal(layer.expert_map, expected):
                raise ValueError("W4A8 compute requires contiguous expert placement")
        loader = make_mxfp4_loader(
            backend.weight_spec,
            Mxfp4LoadSpec(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                self.moe.tp_size,
                self.moe.tp_rank,
            ),
        )
        layer.params_dtype = params_dtype
        self.num_experts = layer.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        loader.create_weights(layer, attrs)
        self._checkpoint_loader = loader
        self.supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer):
        if getattr(self, "selection", None) is None:
            return self._legacy_process_weights_after_loading(layer)
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            make_moe_compute,
        )

        if self._compute is not None or self._checkpoint_loader is None:
            raise RuntimeError("MXFP4 weights must be created once before finalization")
        if self._checkpoint_loader._layer is not layer:
            raise ValueError("MXFP4 Method can only finalize its own layer")
        weights = self._checkpoint_loader.finalize()
        # Runner-owned output scaling is installed after create_weights. Freeze
        # it now, before any Experts adapter or graph captures the backend.
        compute = make_moe_compute(self._compute_config(layer), self.selection)
        compute.validate_weights(weights)
        self._runtime_weights, self._compute = weights, compute
        layer._ilu_mxfp4_ixmm = self.selection.name == "ixmm_fp4fp8"

    def get_fused_moe_quant_config(self, layer):
        if getattr(self, "selection", None) is None:
            return self._legacy_get_fused_moe_quant_config(layer)
        if self.selection.name == "ixmm_fp4fp8":
            return mxfp4_w4a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4",
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        if getattr(self, "selection", None) is None:
            return self._legacy_select_gemm_impl(prepare_finalize, layer)
        if self._compute is None:
            raise RuntimeError(
                "MXFP4 weights must be finalized before binding communication"
            )
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
                _shrink_ll_max_tokens_to_decode,
            )

            _shrink_ll_max_tokens_to_decode(prepare_finalize)
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
            bind_compute_experts,
        )

        return bind_compute_experts(
            self._compute,
            self._runtime_weights,
            self.moe,
            self.moe_quant_config,
            prepare_finalize,
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if getattr(self, "selection", None) is None:
            return self._legacy_apply(
                layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
            )
        if self._compute is None:
            raise RuntimeError("MXFP4 weights have not been finalized")
        residual = shared_experts_input
        if residual is not None and residual.shape[0] != x.shape[0]:
            residual = None
        workspace = self._compute.workspace_spec(topk_ids.numel(), x.dtype).allocate(
            x.device
        )
        return self._compute.run_standard(
            x,
            self._runtime_weights,
            topk_weights,
            topk_ids,
            layer.expert_map,
            workspace=workspace,
            residual=residual,
        )


# Public MXFP4 execution modes. Backend/layout mechanics remain internal; source
# compatibility and the supported platform matrix belong to the Method policy.
_MXFP4_MODES = {
    "ivcore11": {
        "ixinfer_w4a8_nn_v2": ("ixinfer_w4a8", "NN"),
        "ixinfer_w4a8_tn_v2": ("ixinfer_w4a8", "TN"),
    },
    "ivcore30": {
        "ixmm_fp4fp8": ("ixmm_fp4fp8", "fp4fp8"),
        "ixmm_w4a8_nn": ("ixmm_w4a8", "NN"),
    },
}
_MXFP4_DEFAULTS = {"ivcore11": "ixinfer_w4a8_nn_v2", "ivcore30": "ixmm_fp4fp8"}


def select_mxfp4_compute(arch, mode="auto"):
    """Resolve one public mode before loading; reject unsupported overrides."""
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        select_moe_backend,
    )

    modes = _MXFP4_MODES.get(arch, {})
    if mode == "auto":
        mode = _MXFP4_DEFAULTS.get(arch)
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError(
            f"Unsupported MXFP4 mode {mode!r} on {arch}; supported: {tuple(modes)}"
        )
    name, layout = modes[mode]
    return select_moe_backend(arch, name=name, layout=layout)


def select_mxfp4_method(layer, original, runtime=None):
    """Quantization-owned policy: source compatibility and legacy entry points."""
    from vllm.logger import init_logger

    from vllm_iluvatar import envs
    from vllm_iluvatar.config.ilu_compute import compute_options
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    options = compute_options("moe")
    if runtime is not None:
        if any(value != "auto" for value in options.values()):
            raise ValueError(
                "Explicit ilu_quant MoE conversion cannot be combined with ilu_compute.moe overrides yet"
            )
        return runtime
    arch = cuda_arch()
    if (
        not options
        and arch == "ivcore30"
        and envs.VLLM_ILU_MXFP4_MOE_BACKEND in ("torch", "ixformer_native")
    ):
        return original
    mode = options.get("mode", "auto")
    if not options and arch == "ivcore11":
        # Preserve the existing default/legacy layout override when no explicit
        # MoE configuration is supplied. An explicit mode always takes precedence.
        layout = envs.VLLM_ILU_MOE_RUNTIME_LAYOUT
        if layout:
            mode = {"NN": "ixinfer_w4a8_nn_v2", "TN": "ixinfer_w4a8_tn_v2"}.get(
                layout, layout
            )
    selection = select_mxfp4_compute(arch, mode)
    init_logger(__name__).info_once(
        "MXFP4 compute: %s / %s on %s (streaming weights)",
        selection.name,
        selection.layout,
        arch,
    )
    return IluMxfp4MoEMethod(layer.moe_config, selection)
