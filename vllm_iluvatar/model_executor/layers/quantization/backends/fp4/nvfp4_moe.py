# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ivcore30 backend for ModelOpt NVFP4 routed experts.

Owns the NVFP4-to-ix30 conversion, layout helpers, and routed-expert method.
The quantization config and checkpoint selection remain in
``configs.modelopt.nvfp4``.
"""

from __future__ import annotations

from contextlib import nullcontext

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
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _STEP3_SWIGLUSTEP_VARIANT,
    _SWIGLUOAI_VARIANT,
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.common.weight_utils import (
    publish_derived_weight,
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
    from ixformer.inference.functions.ops.gemm.fp4.gemm import gemm as ixmm_gemm

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
        *,
        stage_on_cpu: bool = False,
        validate_checkpoint: bool = False,
        expert_chunk_size: int = _IXMM_EXPERT_CHUNK,
    ) -> None:
        # Skip ModelOptNvFp4FusedMoE.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        if quant_config.group_size != _NVFP4_GROUP_SIZE:
            raise ValueError("NVFP4 requires group_size=16")
        if expert_chunk_size < 1:
            raise ValueError("expert_chunk_size must be positive")
        self.stage_on_cpu = stage_on_cpu
        self.validate_checkpoint = validate_checkpoint
        self.expert_chunk_size = expert_chunk_size
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

    # [ILU override] ModelOptNvFp4FusedMoE.create_weights — delta: optional
    # host staging and per-local-expert load coverage; reuse community layouts.
    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if not self.moe.is_act_and_mul:
            raise NotImplementedError("NVFP4 ixmm experts require a gated activation")
        if (
            hidden_size % _IXMM_GROUP_SIZE
            or intermediate_size_per_partition % _IXMM_GROUP_SIZE
        ):
            raise ValueError("NVFP4 ixmm expert K dimensions must be divisible by 256")
        if self.validate_checkpoint:
            original_loader = extra_weight_attrs["weight_loader"]
            layer._ilu_nvfp4_loaded = {}

            def checked_loader(
                param, weight, name, shard_id, expert_id, return_success=False, **extra
            ):
                success = original_loader(
                    param,
                    weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                    **extra,
                )
                if success:
                    key = (name.rsplit(".", 1)[-1], shard_id)
                    layer._ilu_nvfp4_loaded.setdefault(key, set()).add(expert_id)
                return success if return_success else None

            extra_weight_attrs["weight_loader"] = checked_loader
        with torch.device("cpu") if self.stage_on_cpu else nullcontext():
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

    def _validate_loaded_experts(self, layer):
        if not self.validate_checkpoint:
            return
        loaded = layer._ilu_nvfp4_loaded
        expected = {
            (f"{'w2' if shard == 'w2' else 'w13'}_{kind}", shard)
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
            for shard in ("w1", "w2", "w3")
        }
        reference = loaded.get(("w13_weight", "w1"), set())
        if (
            set(loaded) != expected
            or len(reference) != layer.num_experts
            or any(experts != reference for experts in loaded.values())
        ):
            raise ValueError(
                "Incomplete NVFP4 experts: every local expert requires "
                "weight/block/global/input scales for w1/w2/w3"
            )

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
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        self._validate_loaded_experts(layer)
        if not _ixmm_backend_available():
            raise RuntimeError(
                "IluModelOptNvFp4MoEMethod requires the ixmm backend (ix30 "
                "group_gemm_fp4 only accepts the preprocessed uint8 layout). "
                "Install the ixmm wheel and rebuild ixformer with HAVE_IXMM on "
                "ivcore30."
            )

        # A single global scale per expert for w13 (w1 / w3 must match; community
        # warns otherwise at ``modelopt.py:1560-1566``).
        global13 = layer.w13_weight_scale_2
        if not torch.equal(global13[:, 0], global13[:, 1]):
            raise ValueError("NVFP4 gate/up global scales must match")
        w13_global = global13[:, 0].contiguous()
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

            if not torch.isfinite(g).all() or not (g > 0).all():
                raise ValueError(f"Invalid global scales for {w_name}")
            e, n, half_k = w.shape
            ixmm_weight = torch.empty(
                (e, half_k, n), dtype=torch.uint8, device=w.device
            )
            ixmm_scale = torch.empty(
                (e, half_k * 2 // _IXMM_GROUP_SIZE, n),
                dtype=torch.float32,
                device=w.device,
            )
            for begin in range(0, e, self.expert_chunk_size):
                end = min(begin + self.expert_chunk_size, e)
                packed, scale = preprocess_nvfp4_weight_for_ixmm(
                    w[begin:end],
                    s[begin:end],
                    g[begin:end],
                )
                if (
                    packed.shape != ixmm_weight[begin:end].shape
                    or scale.shape != ixmm_scale[begin:end].shape
                ):
                    raise ValueError(
                        f"Invalid NVFP4 ixmm conversion shapes for {w_name}"
                    )
                ixmm_weight[begin:end].copy_(packed)
                ixmm_scale[begin:end].copy_(scale)
                del packed, scale
            # Host-staged checkpoint parameters are restored to CPU by vLLM's
            # loader context; derived buffers must stay on the compute device.
            publish = publish_derived_weight if self.stage_on_cpu else replace_parameter
            publish(layer, w_name, ixmm_weight)
            publish(layer, s_name, ixmm_scale)

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

        layer._already_called_process_weights_after_loading = True
        layer._ilu_mxfp4_ixmm = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [ILU override] ModelOptNvFp4FusedMoE.apply — delta: wiring only. Community
    # delegates to ``self.moe_kernel.apply``; we call the plugin's ixmm compute
    # with explicit tensors, selecting the activation from ``layer.activation``.  NVFP4 experts
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
        activation = getattr(layer, "activation", "silu")
        activation = getattr(activation, "value", activation)
        alpha = beta = None
        if activation == "swigluoai_uninterleave":
            required = ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            if any(getattr(layer, name, None) is None for name in required):
                raise ValueError("SwiGLU-OAI requires explicit limit, alpha and beta")
            swiglu_limit = float(layer.swiglu_limit)
            alpha, beta = float(layer.swiglu_alpha), float(layer.swiglu_beta)
            variant = _SWIGLUOAI_VARIANT
        elif activation in ("silu", "swiglustep"):
            if activation == "swiglustep":
                if getattr(layer, "swiglu_limit", None) is None:
                    raise ValueError("NVFP4 swiglustep requires explicit swiglu_limit")
                swiglu_limit = float(layer.swiglu_limit)
            else:
                swiglu_limit = 0.0
            variant = _STEP3_SWIGLUSTEP_VARIANT
        else:
            raise NotImplementedError(
                f"Unsupported NVFP4 expert activation: {activation}"
            )
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
            swiglu_variant=variant,
            swiglu_alpha=alpha,
            swiglu_beta=beta,
        )
