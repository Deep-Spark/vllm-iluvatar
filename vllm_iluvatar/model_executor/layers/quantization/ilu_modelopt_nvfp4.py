# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ModelOpt NVFP4 MoE quantization method.

Corresponds to: vllm/model_executor/layers/quantization/modelopt.py

Target   : vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4Config
           (registered under the name ``modelopt_fp4``) and its
           ``ModelOptNvFp4FusedMoE`` routed-expert method.
Baseline : main_9c12a81ac
Why      : community ``ModelOptNvFp4FusedMoE.__init__`` calls
           ``select_nvfp4_moe_backend`` which, off NVIDIA/ROCm, only returns
           NVIDIA-specific backends and lands Step-3.7 on MARLIN ->
           ``gptq_marlin_repack`` (absent from the Iluvatar ``_C`` extension,
           so it crashes). Iluvatar needs the community NVFP4 checkpoint layout
           driven by an Iluvatar compute.
How      : ``@register_quantization_config("modelopt_fp4")`` on a subclass that
           overrides only the routed-expert branch of ``get_quant_method`` to
           return ``IluModelOptNvFp4MoEMethod``; that method keeps community
           ``create_weights`` verbatim (so the checkpoint parameters load
           unchanged), reprocesses the NVFP4 weights into the ixmm ix30 layout
           at load time, and routes ``apply`` to ``ilu_mxfp4_fused_experts()``.
Scope    : routed NVFP4 experts on Iluvatar (ivcore30 + ixmm); selected by
           ``IluModelOptNvFp4Config.get_quant_method``.
Remove   : when community exposes a vendor NVFP4 MoE backend hook (and/or
           ixformer ships a native NVFP4 -> ix30 preprocessing kernel, Phase 2).

Zero-copy reuse of the MXFP4 compute
------------------------------------
``ilu_mxfp4_fused_experts(backend="ixmm")`` consumes the ix30 packed format
(``uint8 [E, k/2, n]`` weight + ``float32 [E, k/256, n]`` running-acc scale).
That format is produced by dequantising the checkpoint and re-quantising per
256-element group -- a step that is agnostic to the *source* scale protocol.
MXFP4 (per-32 e8m0) and NVFP4 (per-16 e4m3 + per-expert global) therefore yield
byte-identical ix30 tensors, so the whole GEMM / SwiGLU pipeline is reused with
no change; the only NVFP4-specific work is the dequant in
``preprocess_nvfp4_weight_for_ixmm`` below.

Why the community name (not a new ``ilu_*``)
--------------------------------------------
The NVFP4 weights are the official ModelOpt release: their config resolves to
``modelopt_fp4`` through community ``override_quantization_method``
(``modelopt.py:1064-1071``). This is the "community format, Iluvatar
implementation" case, mirroring ``IluDeepseekV4FP8Config`` covering
``deepseek_v4_fp8``. ``register_quantization_config`` permits shadowing a
built-in name; ``get_quantization_config`` resolves the customized map last, so
our class wins.
"""

from __future__ import annotations

import torch
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
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)

# Process this many experts per dequant/requant temporary. DSv4-scale w13 across
# all local experts at once would allocate several GB of fp32; mirror the MXFP4
# method's chunking (``ilu_mxfp4.py:93``).
_IXMM_EXPERT_CHUNK = 4
_NVFP4_GROUP_SIZE = 16
_IXMM_GROUP_SIZE = 256
_FP4_MAX_MAG = 6.0

# Step-3.5 routed SwiGLU clamp bound. Community ``FusedMoEBlock`` stores the
# per-layer activation as a string on ``RoutedExperts`` (``step3p5.py:350-368``):
# ``"swiglustep"`` (clamp at 7) or ``"silu"`` (no clamp). ``swiglustep_dynamic_
# scaled_fp8`` clamps after silu and reads a numeric limit with ``limit <= 0`` =
# no clamp, so we derive it here directly from ``activation``.
_SWIGLUSTEP_CLAMP = 7.0


def preprocess_nvfp4_weight_for_ixmm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one expert-chunk of NVFP4 weights to the ixmm ix30 format.

    This is a NVFP4 counterpart of ixformer's
    ``_preprocess_weight_for_ixmm_torch`` (``fp4/gemm/gemm.py:183-220``): the
    only difference is step 2's dequant (per-16 e4m3 block scale times a
    per-expert global scale instead of per-32 e8m0). Steps 3-7 reuse ixformer's
    leaf helpers verbatim, which is what guarantees the output is byte-identical
    to the MXFP4 ix30 tensors the compute path already accepts.

    Args:
        weight:       ``[E, n, k/2]`` uint8, two e2m1 fp4 packed along ``k``.
        weight_scale: ``[E, n, k/16]`` float8_e4m3fn per-16 block scale.
        global_scale: ``[E]`` float32 per-expert global scale (``weight_scale_2``).

    Returns:
        ixmm_weight: ``[E, k/2, n]`` uint8 in ix30 packed layout.
        ixmm_scale:  ``[E, k/256, n]`` float32 running-acc ratio scale.
    """
    # Lazy import keeps this module importable without ixformer (CPU-only unit
    # tests), matching the MXFP4 method's lazy-import discipline.
    from ixformer.inference.functions.fp4.gemm import gemm as ixmm_gemm

    expert, n_dim, packed_k = weight.shape
    k = packed_k * 2
    assert k % _IXMM_GROUP_SIZE == 0, f"k={k} must be a multiple of {_IXMM_GROUP_SIZE}"

    # 1. Unpack uint8 -> nibbles [E, n, k].
    nibbles = ixmm_gemm._unpack_int8_to_nibbles(weight)

    # 2. NVFP4 dequant: fp4 magnitude * per-16 block scale * per-expert global.
    block_scale = weight_scale.float()  # [E, n, k/16]
    weight_float = ixmm_gemm._fp4_to_float(nibbles)  # [E, n, k]
    weight_float = weight_float * block_scale.repeat_interleave(
        _NVFP4_GROUP_SIZE, dim=-1
    )
    weight_float = weight_float * global_scale.to(torch.float32).reshape(expert, 1, 1)

    # 3. Transpose [E, n, k] -> [E, k, n].
    weight_float = weight_float.transpose(1, 2).contiguous()

    # 4. Per-256 absolute scale: amax / FP4_MAX_MAG.
    groups_256 = k // _IXMM_GROUP_SIZE
    grouped = weight_float.reshape(expert, groups_256, _IXMM_GROUP_SIZE, n_dim)
    amax = grouped.abs().amax(dim=2)  # [E, groups, n]
    ixmm_scale = (amax / _FP4_MAX_MAG).clamp(min=1e-12)

    # 5. Re-quantise to fp4 nibbles [E, k, n].
    normalized = grouped / ixmm_scale.unsqueeze(2)
    new_nibbles = ixmm_gemm._quantize_fp4(normalized.reshape(expert, k, n_dim))

    # 6. Pack into ix30 layout [E, k/2, n].
    ixmm_weight = ixmm_gemm._pack_nibbles_ix30(new_nibbles)

    # 7. Absolute scale -> running-acc ratio for the ixmm kernel.
    ixmm_scale = ixmm_gemm.preprocess_group_scale_for_running_acc(ixmm_scale)

    return ixmm_weight, ixmm_scale


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    """The runner-side output scale, ``1.0`` unless a kernel opted in.

    Same helper the other Iluvatar MoE methods carry (see
    ``ilu_mxfp4.py:96-110`` for the full rationale): ``IluMoERunner`` writes
    ``_ilu_output_scale`` from the runner's ``routed_scaling_factor`` (pinned to
    ``1.0`` unless ``apply_routed_scale_to_output`` is set). Do **not** use
    ``layer.routed_scaling_factor``; the router already folded that into
    ``topk_weights``.
    """
    return getattr(layer, "_ilu_output_scale", 1.0)


class IluModelOptNvFp4MoEMethod(ModelOptNvFp4FusedMoE):
    """NVFP4 routed-expert method backed by ``ilu_mxfp4_fused_experts()``."""

    # [ILU override] ModelOptNvFp4FusedMoE.__init__ — delta: skip the parent
    # body (it calls select_nvfp4_moe_backend, which off NVIDIA lands on MARLIN)
    # and mirror only the field initialisation ``create_weights`` needs.
    #
    # ``experts_cls`` is deliberately left unset (not None) so community
    # ``FusedMoEMethodBase.is_monolithic`` never dereferences it; the explicit
    # ``is_monolithic`` override below makes the answer independent of that
    # subtlety regardless. See ``ilu_mxfp4.py`` module docstring.
    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        # Skip ModelOptNvFp4FusedMoE.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.use_a16 = quant_config.quant_method == "W4A16_NVFP4"
        # No community backend is selected: the ix30 path is our own compute.
        self.nvfp4_backend = None
        self.moe_kernel = None
        # ``create_weights`` reads this to size ``w*_input_scale``. False = the
        # per-local-expert layout MARLIN (the backend this checkpoint would
        # otherwise use) expects; the scales are dropped after loading anyway.
        self.use_global_sf = False

        assert not hasattr(self, "experts_cls"), (
            "IluModelOptNvFp4MoEMethod must not define experts_cls; community "
            "is_monolithic() would dereference it"
        )
        assert self.is_monolithic is False

    # [ILU override] FusedMoEMethodBase.is_monolithic — delta: constant False.
    # We always take the modular path (router first, then
    # ``RoutedExperts.forward_modular`` -> ``apply``).
    @property
    def is_monolithic(self) -> bool:
        return False

    # [ILU override] ModelOptNvFp4FusedMoE.supports_eplb — delta: community
    # returns True. Phase 1's ixmm expert loop records no expert load, so
    # claiming EPLB support would silently disable rebalancing; matches the
    # MXFP4 method.
    @property
    def supports_eplb(self) -> bool:
        return False

    # [ILU override] ModelOptNvFp4FusedMoE.maybe_make_prepare_finalize — delta:
    # None. Community *raises* here (it assumes the modular-kernel init path);
    # Iluvatar dispatch/combine is the naive EP-group path, so opt out.
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        return None

    # [ILU override] ModelOptNvFp4FusedMoE.get_fused_moe_quant_config — delta:
    # None. A ``FusedMoEQuantConfig`` only configures a community modular
    # kernel, which we do not build. Returning None also keeps
    # ``topk_indices_dtype`` at None so the router emits int32 ids -- the dtype
    # ``ilu_mxfp4_fused_experts`` expects.
    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    # [ILU override] ModelOptNvFp4FusedMoE.process_weights_after_loading —
    # delta: replace community's ``convert_to_nvfp4_moe_kernel_format`` +
    # kernel build with a NVFP4 -> ix30 conversion, then drop the now-unused
    # global / input scales so they do not stay resident (see plan §5).
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if not _ixmm_backend_available():
            raise RuntimeError(
                "IluModelOptNvFp4MoEMethod requires the ixmm backend (ix30 "
                "group_gemm_fp4 only accepts the preprocessed uint8 layout). "
                "Install the ixmm wheel and rebuild ixformer with HAVE_IXMM on "
                "ivcore30."
            )

        # A single global scale per expert for w13 (w1 / w3 must match; community
        # warns otherwise at ``modelopt.py:1560-1566``).
        w13_global = layer.w13_weight_scale_2[:, 0].contiguous()
        w2_global = layer.w2_weight_scale_2.contiguous()

        for w_name, s_name, g in (
            ("w13_weight", "w13_weight_scale", w13_global),
            ("w2_weight", "w2_weight_scale", w2_global),
        ):
            w = getattr(layer, w_name)
            s = getattr(layer, s_name)
            assert w.dtype is torch.uint8 and w.is_contiguous(), (
                f"{w_name} expected uint8 contiguous, got {w.dtype} "
                f"contiguous={w.is_contiguous()}"
            )
            assert s.dtype is torch.float8_e4m3fn, (
                f"{s_name} expected float8_e4m3fn, got {s.dtype}"
            )

            weight_parts: list[torch.Tensor] = []
            scale_parts: list[torch.Tensor] = []
            for begin in range(0, w.shape[0], _IXMM_EXPERT_CHUNK):
                end = min(begin + _IXMM_EXPERT_CHUNK, w.shape[0])
                ixmm_weight, ixmm_scale = preprocess_nvfp4_weight_for_ixmm(
                    w[begin:end],
                    s[begin:end],
                    g[begin:end],
                )
                weight_parts.append(ixmm_weight)
                scale_parts.append(ixmm_scale)

            ixmm_weight = torch.cat(weight_parts, dim=0).contiguous()
            ixmm_scale = torch.cat(scale_parts, dim=0).contiguous()
            expected_weight_shape = (w.shape[0], w.shape[2], w.shape[1])
            expected_scale_shape = (
                w.shape[0],
                (w.shape[2] * 2) // _IXMM_GROUP_SIZE,
                w.shape[1],
            )
            if tuple(ixmm_weight.shape) != expected_weight_shape:
                raise ValueError(
                    f"preprocess_nvfp4_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_weight.shape)} for {w_name}, expected "
                    f"{expected_weight_shape}"
                )
            if tuple(ixmm_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"preprocess_nvfp4_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_scale.shape)} for {s_name}, expected "
                    f"{expected_scale_shape}"
                )
            replace_parameter(layer, w_name, ixmm_weight)
            replace_parameter(layer, s_name, ixmm_scale)

        # Free the checkpoint scales the ix30 path no longer references. Phase 1
        # uses dynamic per-token W4A8 activation, so ``*_input_scale`` (W4A4) is
        # dropped as well; the global scales were folded into the ix30 tensors.
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)

        layer._ilu_mxfp4_ixmm = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [ILU override] ModelOptNvFp4FusedMoE.apply — delta: wiring only. Community
    # delegates to ``self.moe_kernel.apply``; we call the plugin's ixmm compute
    # with explicit tensors, selecting the Step-3.5 ``step3_swiglustep`` variant
    # and deriving its numeric clamp from ``layer.activation``. NVFP4 experts
    # carry no bias, and the shared output arrives through
    # ``shared_experts_input`` (``extra_residual``), so ``shared_experts`` is
    # accepted for signature compatibility but unused.
    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # Step-3.5 routed SwiGLU: read the community activation string off the
        # experts and map it to the numeric clamp the ``step3_swiglustep`` op
        # expects (``"swiglustep"`` -> 7.0, ``"silu"`` -> 0.0 = no clamp). This
        # keeps all SwiGLU semantics in the method + ixformer op, so the model
        # file carries no swiglu wiring.
        activation = getattr(layer, "activation", "silu")
        swiglu_limit = _SWIGLUSTEP_CLAMP if activation == "swiglustep" else 0.0
        return ilu_mxfp4_fused_experts(
            hidden_states=x,
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            swiglu_limit=swiglu_limit,
            expert_map=layer.expert_map,
            output_scale=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
            w13_bias=None,
            w2_bias=None,
            backend="ixmm",
            swiglu_variant="step3_swiglustep",
        )


@register_quantization_config("modelopt_fp4")
class IluModelOptNvFp4Config(ModelOptNvFp4Config):
    """ModelOpt NVFP4 config whose routed experts land on Iluvatar kernels.

    Inherits ``__init__`` / ``get_name`` / ``override_quantization_method`` /
    ``_from_config`` unchanged, so checkpoint detection and every linear /
    attention branch stay community behaviour.
    """

    # [ILU override] ModelOptQuantConfigBase.get_quant_method — delta: routed
    # experts (not excluded) return IluModelOptNvFp4MoEMethod instead of
    # community ModelOptNvFp4FusedMoE. Everything else (linear, attention, KV,
    # excluded layers) is left to super().
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts) and not self.is_layer_excluded(prefix):
            return IluModelOptNvFp4MoEMethod(
                quant_config=self, moe_config=layer.moe_config
            )
        return super().get_quant_method(layer, prefix)
