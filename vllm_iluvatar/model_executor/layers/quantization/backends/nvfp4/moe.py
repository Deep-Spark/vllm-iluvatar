# SPDX-License-Identifier: Apache-2.0
"""ivcore11 NVFP4 routed experts executed as IxInfer W4A8.

ivcore30 keeps the ix30 FP4 method in ``backends/fp4/nvfp4_moe.py``; ivcore11
has no ixmm, and its default expert backend is ``ixinfer_w4a8_nn_v2``. NVFP4
reaches it by being restated as block-FP8 on a ``[1, 16]`` grid (see
``conversions/nvfp4_to_w4a8.py``) and handed to the same converter block-FP8
and MXFP8 use.

Unlike those two, NVFP4 does not stream: a shard carries four checkpoint
tensors (values, block scale, per-expert global, input scale) and the paired
loader pairs exactly two, so this keeps the existing load-everything-then-
convert-in-chunks shape and only changes what the conversion produces.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
    IluModelOptNvFp4MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    MoEComputeBindingMixin,
)
from vllm_iluvatar.model_executor.layers.quantization.common.weight_utils import (
    publish_derived_weight,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.nvfp4_to_w4a8 import (
    convert_nvfp4_to_w4a8,
)

logger = init_logger(__name__)

# The converter needs N and K to be multiples of 32; ix30 needed 256.
_W4A8_ALIGNMENT = 32

# Checkpoint scales the W4A8 image no longer references. The globals were
# folded into the conversion, and the activation is dynamically per-token
# quantized, so the W4A4 input scales are dead too.
_UNUSED_AFTER_CONVERSION = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)


class IluModelOptNvFp4ToW4A8MoEMethod(
    MoEComputeBindingMixin, IluModelOptNvFp4MoEMethod
):
    """NVFP4 routed experts converted to the selected IxInfer W4A8 contract.

    Takes ``create_weights`` and the checkpoint-coverage validation from the
    ix30 method, and the post-conversion lifecycle from the shared mixin.

    The class name must keep containing ``ModelOpt``: community's
    ``FusedMoE.weight_loader`` -- mirrored in ``ilu_routed_experts`` -- selects
    the NVFP4 scale-loading protocol with ``"ModelOpt" in
    self.quant_method.__class__.__name__``. Without it the block
    ``w13_weight_scale`` falls through to the generic per-scale branch, which
    rejects it because community's ``create_weights`` never attaches a
    ``quant_method`` attribute to these parameters.
    """

    compute_source = "NVFP4"
    k_alignment = _W4A8_ALIGNMENT

    def __init__(self, quant_config, moe_config, **kwargs):
        super().__init__(quant_config, moe_config, **kwargs)
        self.selection = None
        self._compute = self._runtime_weights = None
        logger.info_once("NVFP4 MoE execution backend: ixinfer_w4a8")

    # [ILU override] IluModelOptNvFp4MoEMethod.process_weights_after_loading --
    # delta: the conversion target is the selected W4A8 contract instead of the
    # ix30 FP4 layout, and binding the compute replaces the ixmm availability
    # check. The chunking, the global-scale validation and the parameter
    # cleanup are the same.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            get_moe_backend,
        )

        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        self._validate_loaded_experts(layer)
        selection = self._select_compute()
        backend = get_moe_backend(selection)
        error = backend.support_error(self._compute_config(layer), selection)
        if error:
            raise ValueError(error)
        spec = backend.weight_specs[selection.layout]
        weight_format = "ixinfer_" + spec.layout.lower()

        # One global scale per expert for w13: w1 and w3 must agree, as the
        # community loader only warns about it.
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
                convert_nvfp4_to_w4a8(
                    weight[begin:end].to(device),
                    scale[begin:end].to(device),
                    global_scale[begin:end].to(device),
                    out_weight[begin:end],
                    out_scale[begin:end],
                    weight_format=weight_format,
                )
            if not torch.isfinite(out_scale).all():
                raise ValueError(f"Invalid or non-finite NVFP4 {group} weights")
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


def nvfp4_w4a8_moe_method(quant_config, layer, prefix, **kwargs):
    """Claim NVFP4 routed experts on ivcore11; return None to defer."""
    from vllm_iluvatar.ilu_util.arch import IVCORE11, cuda_arch

    if not isinstance(layer, RoutedExperts):
        return None
    if cuda_arch() != IVCORE11:
        return None
    if quant_config.is_layer_excluded(prefix):
        return None
    return IluModelOptNvFp4ToW4A8MoEMethod(quant_config, layer.moe_config, **kwargs)


__all__ = ["IluModelOptNvFp4ToW4A8MoEMethod", "nvfp4_w4a8_moe_method"]
