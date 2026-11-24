# SPDX-License-Identifier: Apache-2.0
"""Iluvatar backend for ModelOpt NVFP4 routed experts.

Owns the NVFP4 conversion, layout helpers, and routed-expert method. The
quantization config and checkpoint selection remain in
``configs.modelopt.nvfp4``.

The conversion target is whatever ``select_four_bit_expert_compute`` picked --
the ix30 FP4 layout on ivcore30, IxInfer W4A8 on ivcore11, where there is no
ixmm. Everything after the conversion is the lifecycle in
``MoEComputeBindingMixin``, shared with block-FP8 and MXFP8.

Unlike those two, NVFP4 does not stream: a shard carries four checkpoint
tensors (values, block scale, per-expert global, input scale) and the paired
loader pairs exactly two, so this loads everything and converts in expert
chunks.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    MoEComputeBindingMixin,
)
from vllm_iluvatar.model_executor.layers.quantization.common.weight_utils import (
    publish_derived_weight,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.nvfp4_to_w4a8 import (
    convert_nvfp4_to_w4a8,
)

# Process this many experts per dequant/requant temporary. DSv4-scale w13 across
# all local experts at once would allocate several GB of fp32; mirror the MXFP4
# method's chunking (``ilu_mxfp4.py:93``).
_IXMM_EXPERT_CHUNK = 4
_NVFP4_GROUP_SIZE = 16
_IXMM_GROUP_SIZE = 256

# What each conversion target needs of K. The requirement comes from the
# converter, not the kernel: the ix30 repack groups 256 values per scale and
# the whole-tensor path does no padding, while ``fp8_block_to_w4a8`` packs
# nibbles in units of 32.
_CONVERSION_ALIGNMENT = {"ixmm_fp4fp8": _IXMM_GROUP_SIZE}
_W4A8_ALIGNMENT = 32

# Checkpoint scales no converted image references. The globals are folded into
# the conversion, and the activation is dynamically per-token quantized, so the
# W4A4 input scales are dead too.
_UNUSED_AFTER_CONVERSION = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)


def preprocess_nvfp4_weight_for_ixmm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one expert-chunk of NVFP4 weights to the ixmm ix30 format.

    Thin wrapper around IxFormer ``nvfp4_to_fp4fp8`` (native CUDA; same
    contract as the historical inlined torch path). Requires the matching
    IxFormer build with the native backend registered.

    Args:
        weight:       ``[E, n, k/2]`` uint8, two e2m1 fp4 packed along ``k``.
        weight_scale: ``[E, n, k/16]`` float8_e4m3fn per-16 block scale.
        global_scale: ``[E]`` float32 per-expert global scale (``weight_scale_2``).

    Returns:
        ixmm_weight: ``[E, k/2, n]`` uint8 in ix30 packed layout.
        ixmm_scale:  ``[E, k/256, n]`` float32 running-acc ratio scale.
    """
    # Lazy import + native-only, matching MXFP4 / block-FP8 loaders.
    from vllm_iluvatar.model_executor.layers.quantization.common.paired_moe_loader import (
        _native_converter,
    )

    return _native_converter("nvfp4_to_fp4fp8")(
        weight,
        weight_scale,
        global_scale,
        backend="ixformer_native",
    )


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


class IluModelOptNvFp4MoEMethod(MoEComputeBindingMixin, ModelOptNvFp4FusedMoE):
    """NVFP4 routed experts converted to the selected expert contract.

    The class name must keep containing ``ModelOpt``: community's
    ``FusedMoE.weight_loader`` -- mirrored in ``ilu_routed_experts`` -- selects
    the NVFP4 scale-loading protocol with ``"ModelOpt" in
    self.quant_method.__class__.__name__``. Without it the block
    ``w13_weight_scale`` falls through to the generic per-scale branch, which
    rejects it because community's ``create_weights`` never attaches a
    ``quant_method`` attribute to these parameters.
    """

    compute_source = "NVFP4"

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
        # No community backend is selected: the expert path is our own compute.
        self.nvfp4_backend = None
        self.moe_kernel = None
        self.selection = None
        self._compute = self._runtime_weights = None
        # ``create_weights`` reads this to size ``w*_input_scale``. False = the
        # per-local-expert layout MARLIN (the backend this checkpoint would
        # otherwise use) expects; the scales are dropped after loading anyway.
        self.use_global_sf = False

        assert not hasattr(self, "experts_cls"), (
            "IluModelOptNvFp4MoEMethod must not define experts_cls; community "
            "is_monolithic() would dereference it"
        )
        assert self.is_monolithic is False

    def _target_spec(self):
        """The weight contract the selected compute backend consumes."""
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            get_moe_backend,
        )

        selection = self._select_compute()
        return get_moe_backend(selection).weight_specs[selection.layout]

    @property
    def k_alignment(self):
        return _CONVERSION_ALIGNMENT.get(self._target_spec().format_id, _W4A8_ALIGNMENT)

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
            hidden_size % self.k_alignment
            or intermediate_size_per_partition % self.k_alignment
        ):
            raise ValueError(
                f"NVFP4 expert K dimensions must be divisible by {self.k_alignment}"
            )
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

    # ``is_monolithic``, ``supports_eplb``, ``maybe_make_prepare_finalize``,
    # ``get_fused_moe_quant_config``, ``select_gemm_impl`` and ``apply`` all
    # come from ``MoEComputeBindingMixin``: once the conversion has produced
    # the backend's contract, nothing downstream is NVFP4-specific.

    # [ILU override] ModelOptNvFp4FusedMoE.process_weights_after_loading —
    # delta: replace community's ``convert_to_nvfp4_moe_kernel_format`` +
    # kernel build with a conversion into the selected contract, then drop the
    # now-unused global / input scales so they do not stay resident.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            get_moe_backend,
        )

        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        self._validate_loaded_experts(layer)
        selection = self._select_compute()
        error = get_moe_backend(selection).support_error(
            self._compute_config(layer), selection
        )
        if error:
            raise ValueError(error)
        spec = self._target_spec()

        # A single global scale per expert for w13 (w1 / w3 must match; community
        # warns otherwise at ``modelopt.py:1560-1566``).
        global13 = layer.w13_weight_scale_2
        if not torch.equal(global13[:, 0], global13[:, 1]):
            raise ValueError("NVFP4 gate/up global scales must match")
        globals_by_group = {
            "w13": global13[:, 0].contiguous(),
            "w2": layer.w2_weight_scale_2.contiguous(),
        }
        # Host-staged checkpoint parameters stay on CPU; the derived image must
        # be built where the compute runs.
        source_device = layer.w13_weight.device
        device = (
            source_device
            if source_device.type == "cuda"
            else torch.device("cuda", torch.cuda.current_device())
        )

        for group, global_scale in globals_by_group.items():
            weight = getattr(layer, group + "_weight")
            scale = getattr(layer, group + "_weight_scale")
            if weight.dtype is not torch.uint8 or not weight.is_contiguous():
                raise ValueError(
                    f"{group}_weight expected contiguous uint8, got "
                    f"{weight.dtype} contiguous={weight.is_contiguous()}"
                )
            if scale.dtype is not torch.float8_e4m3fn:
                raise ValueError(
                    f"{group}_weight_scale expected float8_e4m3fn, got {scale.dtype}"
                )
            if not torch.isfinite(global_scale).all() or not (global_scale > 0).all():
                raise ValueError(f"Invalid global scales for {group}_weight")

            experts, n, packed_k = weight.shape
            weight_shape, scale_shape = spec.shapes(experts, n, packed_k * 2)
            out_weight = torch.empty(
                weight_shape, dtype=spec.weight_dtype, device=device
            )
            out_scale = torch.empty(scale_shape, dtype=spec.scale_dtype, device=device)
            for begin in range(0, experts, self.expert_chunk_size):
                end = min(begin + self.expert_chunk_size, experts)
                self._convert_chunk(
                    spec,
                    weight[begin:end].to(device),
                    scale[begin:end].to(device),
                    global_scale[begin:end].to(device),
                    out_weight[begin:end],
                    out_scale[begin:end],
                )
            if not torch.isfinite(out_scale).all():
                raise ValueError(f"Invalid or non-finite NVFP4 {group} weights")
            # Host-staged checkpoint parameters are restored to CPU by vLLM's
            # loader context; derived buffers must stay on the compute device.
            publish = publish_derived_weight if self.stage_on_cpu else replace_parameter
            publish(layer, group + "_weight", out_weight)
            publish(layer, group + "_weight_scale", out_scale)

        for name in _UNUSED_AFTER_CONVERSION:
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)

        self._bind_compute(layer)
        layer._already_called_process_weights_after_loading = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _convert_chunk(spec, weight, scale, global_scale, out_weight, out_scale):
        """Write one expert chunk into the target contract."""
        if spec.format_id != "ixmm_fp4fp8":
            convert_nvfp4_to_w4a8(
                weight,
                scale,
                global_scale,
                out_weight,
                out_scale,
                weight_format="ixmm_nn"
                if spec.format_id == "ixmm_w4a8_nn"
                else "ixinfer_" + spec.layout.lower(),
            )
            return
        packed, packed_scale = preprocess_nvfp4_weight_for_ixmm(
            weight, scale, global_scale
        )
        if packed.shape != out_weight.shape or packed_scale.shape != out_scale.shape:
            raise ValueError("Invalid NVFP4 ixmm conversion shapes")
        out_weight.copy_(packed)
        out_scale.copy_(packed_scale)
