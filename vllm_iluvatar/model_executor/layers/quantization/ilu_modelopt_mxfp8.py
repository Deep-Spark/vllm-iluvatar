# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP8 routed-expert quantization method.

Corresponds to: vllm/model_executor/layers/quantization/modelopt.py
                vllm/model_executor/layers/fused_moe/oracle/mxfp8.py

Target   : vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8Config
           (registered under the name ``modelopt_mxfp8``) and its
           ``ModelOptMxFp8FusedMoE`` routed-expert branch.
Why      : ``ModelOptMxFp8FusedMoE.__init__`` calls ``select_mxfp8_moe_backend``,
           whose auto priority is FLASHINFER_TRTLLM -> DEEPGEMM -> MARLIN -> ...
           ``MarlinExperts`` only gates on "CUDA SM75+" and ``IluPlatform._enum``
           is CUDA, so Marlin wrongly claims support and wins ahead of the
           EMULATION fallback. It then dies at load time in
           ``prepare_mxfp8_moe_layer_for_marlin`` ->
           ``torch.ops._C.gptq_marlin_repack``, which the Iluvatar build does not
           define. Same failure class as NVFP4/block-FP8 landing on MARLIN.
How      : ``@register_quantization_config("modelopt_mxfp8")`` on a subclass that
           only swaps ``FusedMoEMethodCls``; the inherited
           ``ModelOptQuantConfigBase.get_quant_method`` dispatches through that
           attribute, so no method body is copied. The MoE method skips the
           parent ``__init__`` (and therefore the backend selection), converts
           each [1,32] MXFP8 expert to the ix30 packed FP4 layout at load time,
           and dispatches both MoE GEMMs through ``ilu_mxfp4_fused_experts``.
           ``create_weights`` keeps the checkpoint MXFP8 layout (community
           weight loader, community sharding) but allocates it in host memory,
           because on EP4 the MXFP8 grid and its ix30 FP4 image do not both fit
           in HBM. ``process_weights_after_loading`` streams that grid through
           the GPU a few experts at a time and publishes only the FP4 result.
Scope    : MXFP8 routed experts on Iluvatar corex30 (MiniMax-M3 MXFP8:
           ``quant_method="mxfp8"``, ``weight_block_size=[1, 32]``, UE8M0
           scales). Dense MXFP8 linears are handled separately by
           ``IluMxfp8LinearKernel`` (per-channel requant, no FP4 detour).
Remove   : when community exposes a vendor MXFP8 MoE backend hook or ixmm gains
           a native [1,32] MXFP8 grouped GEMM that avoids load-time requant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptMxFp8FusedMoE,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    _e8m0_to_fp32 as e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _SWIGLUOAI_VARIANT,
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8 import (
    _IXMM_EXPERT_CHUNK,
    _IXMM_GROUP_SIZE,
    preprocess_fp8_block_weight_for_ixmm,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_modelopt_nvfp4 import (
    _get_ilu_output_scale,
)

# MiniMax-M3's routed experts are declared with community
# ``swigluoai_uninterleave``; ``_SWIGLUOAI_VARIANT`` reproduces exactly that
# math. Anything else must fail loudly rather than compute a different
# activation, so the accepted set is explicit.
_SUPPORTED_ACTIVATIONS = frozenset({"swigluoai_uninterleave"})


def preprocess_fp8_moe_weight_for_ixmm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block: tuple[int, int],
    *,
    scale_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    allow_k_padding: bool = False,
    compute_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a complete expert weight tensor to the ix30 FP4 layout.

    MXFP8 checkpoints live in host memory as ``[E, n, k]`` e4m3 plus UE8M0
    scales. This walks a few experts at a time so the dequant/repack never
    holds the whole grid in HBM, optionally decodes the source scale, and
    writes into a pre-allocated ixmm output.

    ``compute_device`` runs each chunk on that device (and publishes there)
    while the source stays where it is. The dequant is float32 elementwise
    math and the nibble pack is a 16-step gather; on CPU that dominates
    load time, so the MXFP8 method passes the current CUDA device.
    """
    k = int(weight.shape[2])
    padded_k = (
        (k + _IXMM_GROUP_SIZE - 1) // _IXMM_GROUP_SIZE * _IXMM_GROUP_SIZE
        if allow_k_padding
        else k
    )
    num_experts, n_dim = int(weight.shape[0]), int(weight.shape[1])
    device = compute_device if compute_device is not None else weight.device
    ixmm_weight = torch.empty(
        (num_experts, padded_k // 2, n_dim), dtype=torch.uint8, device=device
    )
    ixmm_scale = torch.empty(
        (num_experts, padded_k // _IXMM_GROUP_SIZE, n_dim),
        dtype=torch.float32,
        device=device,
    )

    for begin in range(0, num_experts, _IXMM_EXPERT_CHUNK):
        end = min(begin + _IXMM_EXPERT_CHUNK, num_experts)
        weight_chunk = weight[begin:end]
        scale_chunk = weight_scale[begin:end]
        if compute_device is not None:
            weight_chunk = weight_chunk.to(compute_device)
            scale_chunk = scale_chunk.to(compute_device)
        if scale_transform is not None:
            scale_chunk = scale_transform(scale_chunk)
        chunk_weight, chunk_scale = preprocess_fp8_block_weight_for_ixmm(
            weight_chunk,
            scale_chunk,
            block,
            allow_k_padding,
        )
        expected_weight_shape = (end - begin, padded_k // 2, n_dim)
        expected_scale_shape = (end - begin, padded_k // _IXMM_GROUP_SIZE, n_dim)
        if tuple(chunk_weight.shape) != expected_weight_shape:
            raise ValueError(
                "preprocess_fp8_block_weight_for_ixmm returned shape "
                f"{tuple(chunk_weight.shape)}, expected {expected_weight_shape}"
            )
        if tuple(chunk_scale.shape) != expected_scale_shape:
            raise ValueError(
                "preprocess_fp8_block_weight_for_ixmm returned scale shape "
                f"{tuple(chunk_scale.shape)}, expected {expected_scale_shape}"
            )
        ixmm_weight[begin:end].copy_(chunk_weight)
        ixmm_scale[begin:end].copy_(chunk_scale)

    return ixmm_weight, ixmm_scale


def _publish_derived_weight(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    """Attach a load-time-derived tensor under ``name``, as a buffer.

    Not ``replace_parameter``: the loader calls
    ``process_weights_after_loading`` inside ``device_loading_context``, which
    records every parameter that was on CPU on entry and, on exit, moves
    whatever still carries that *name* back to CPU. Since the ix30 FP4 image is
    published under the checkpoint's own names, keeping it a Parameter means the
    context evicts it to host memory the moment the context closes, and the
    first ``group_gemm_fp4`` then fails with "weight must be on the same CUDA
    device as a".

    A non-persistent buffer is both immune to that (the restore loop walks
    ``named_parameters()``) and the honest description of what this is: a
    derived, kernel-private tensor, not a checkpoint parameter, so it should
    also stay out of ``state_dict()``.
    """
    layer._parameters.pop(name, None)
    layer._buffers.pop(name, None)
    layer.register_buffer(name, tensor, persistent=False)


class IluModelOptMxFp8FusedMoE(ModelOptMxFp8FusedMoE):
    """MXFP8 routed experts converted to ix30 FP4 and run by ixmm group GEMM."""

    # [ILU override] ModelOptMxFp8FusedMoE.__init__ — delta: skip the parent body
    # (it calls select_mxfp8_moe_backend, which lands on MARLIN here) and mirror
    # only the field initialisation the rest of this class needs. Note
    # ``create_weights`` is overridden below rather than inherited, so the
    # community backend/kernel fields stay None on purpose.
    #
    # ``experts_cls`` is deliberately left unset (not None) so community
    # ``FusedMoEMethodBase.is_monolithic`` never dereferences it; the explicit
    # ``is_monolithic`` override below makes the answer independent of that
    # subtlety regardless. See ``ilu_modelopt_nvfp4.py`` / ``ilu_fp8.py``.
    def __init__(
        self,
        quant_config: ModelOptMxFp8Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        # Skip ModelOptMxFp8FusedMoE.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        # No community backend is selected: the ix30 path is our own compute.
        self.mxfp8_backend = None
        self.moe_kernel = None

        assert self.quant_config.is_checkpoint_mxfp8_serialized, (
            "IluModelOptMxFp8FusedMoE only handles serialized MXFP8 checkpoints"
        )
        assert not hasattr(self, "experts_cls"), (
            "IluModelOptMxFp8FusedMoE must not define experts_cls; community "
            "is_monolithic() would dereference it"
        )
        assert self.is_monolithic is False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        assert layer.intermediate_size_per_partition == intermediate_size_per_partition
        assert layer.hidden_size == hidden_size
        layer.orig_dtype = params_dtype

        if hidden_size % _IXMM_GROUP_SIZE != 0:
            raise ValueError(
                f"MXFP8 MoE FP4 path requires hidden_size divisible by "
                f"{_IXMM_GROUP_SIZE}, got {hidden_size}."
            )
        if intermediate_size_per_partition % _IXMM_GROUP_SIZE != 0:
            raise ValueError(
                "MXFP8 MoE FP4 path requires intermediate_size_per_partition "
                f"divisible by {_IXMM_GROUP_SIZE}, got "
                f"{intermediate_size_per_partition}."
            )

        layer.num_experts = num_experts
        weight_loader = extra_weight_attrs.get("weight_loader")
        w13_n = (2 if self.moe.is_act_and_mul else 1) * intermediate_size_per_partition

        # Checkpoint MXFP8 layout ([E, n, k] e4m3 + [E, n, k/32] e8m0), so the
        # community MoE weight loader stays in charge of sharding and of pairing
        # the w1 / w3 halves.
        #
        # The grid lives in host memory. On EP4 every layer's MXFP8 weights plus
        # its ix30 FP4 image do not all fit in HBM at once; allocating on the
        # host means the loader's ``device_loading_context`` stages exactly one
        # layer at a time onto the device for ``process_weights_after_loading``
        # and releases it again, so peak HBM is one layer's MXFP8 grid rather
        # than the whole model's.
        def _empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(*shape, dtype=dtype, device="cpu")

        w13_weight = ModelWeightParameter(
            data=_empty(num_experts, w13_n, hidden_size, dtype=MXFP8_VALUE_DTYPE),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=_empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=_empty(
                num_experts,
                w13_n,
                hidden_size // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=_empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # The generic MoE loader needs to know these are per-block scales.
        set_weight_attrs(
            layer.w13_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )
        set_weight_attrs(
            layer.w2_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )

    # [ILU override] FusedMoEMethodBase.is_monolithic — delta: constant False.
    # We always take the modular path (router first, then
    # ``RoutedExperts.forward_modular`` -> ``apply``).
    @property
    def is_monolithic(self) -> bool:
        return False

    # [ILU override] ModelOptMxFp8FusedMoE.supports_eplb — delta: community
    # returns True. The ixmm expert loop records no expert load, so claiming EPLB
    # support would silently disable rebalancing; matches the NVFP4/FP8 methods.
    @property
    def supports_eplb(self) -> bool:
        return False

    # [ILU override] ModelOptMxFp8FusedMoE.maybe_make_prepare_finalize — delta:
    # community assumes the modular-kernel init path; Iluvatar dispatch/combine
    # is the naive EP-group path, so defer to the base (returns None unless a
    # batched format is genuinely in play).
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        return FusedMoEMethodBase.maybe_make_prepare_finalize(self, routing_tables)

    # [ILU override] ModelOptMxFp8FusedMoE.get_fused_moe_quant_config — delta:
    # None. A ``FusedMoEQuantConfig`` only configures a community modular
    # kernel, which we do not build. Returning None also keeps
    # ``topk_indices_dtype`` at None so the router emits int32 ids -- the dtype
    # ``ilu_mxfp4_fused_experts`` expects.
    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    # [ILU override] ModelOptMxFp8FusedMoE.process_weights_after_loading — delta:
    # replace ``convert_to_fp8_moe_kernel_format`` (Marlin repack) + kernel build
    # with MXFP8 [1,32] UE8M0 -> ix30 packed FP4 conversion.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Community guards this too: mxfp8 reaches here more than once.
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if not _ixmm_backend_available():
            raise RuntimeError(
                "IluModelOptMxFp8FusedMoE requires the ixmm backend (ix30 "
                "group_gemm_fp4 only accepts the preprocessed uint8 layout). "
                "Install the ixmm wheel and rebuild ixformer with HAVE_IXMM on "
                "ivcore30."
            )

        for w_name, s_name in (
            ("w13_weight", "w13_weight_scale"),
            ("w2_weight", "w2_weight_scale"),
        ):
            w = getattr(layer, w_name)
            s = getattr(layer, s_name)
            assert w.dtype is MXFP8_VALUE_DTYPE
            assert s.dtype in (MXFP8_SCALE_DTYPE, torch.float8_e8m0fnu)
            # The loader's ``device_loading_context`` has already staged the
            # checkpoint parameters on the worker's device, so the parameter
            # itself names where the FP4 image belongs. Reading it beats
            # ``torch.cuda.current_device()``, which is not guaranteed to be
            # this worker's device at this point and would silently build every
            # rank's weights on cuda:0.
            device = w.device if w.device.type == "cuda" else None
            # ``compute_device`` requant on the GPU: the dequant/repack is
            # float32 elementwise math plus a 16-step gather, which on the host
            # costs minutes per shard for a 256-expert layer.
            ixmm_weight, ixmm_scale = preprocess_fp8_moe_weight_for_ixmm(
                w,
                s,
                (1, MXFP8_BLOCK_SIZE),
                scale_transform=e8m0_to_fp32,
                compute_device=device,
            )
            _publish_derived_weight(layer, w_name, ixmm_weight)
            _publish_derived_weight(layer, s_name, ixmm_scale)
        layer._already_called_process_weights_after_loading = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [ILU override] ModelOptMxFp8FusedMoE.apply — delta: wiring only. Community
    # delegates to ``self.moe_kernel.apply``; we call the plugin's ixmm compute
    # with explicit tensors. MiniMax-M3 routes through community
    # ``swigluoai_uninterleave``, so the pipeline uses its matching activation
    # before dynamically quantizing GEMM2 input.
    # Routed experts carry no bias and the shared output arrives through
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
        activation = getattr(layer, "activation", None)
        activation = getattr(activation, "value", activation)
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise NotImplementedError(
                "IluModelOptMxFp8FusedMoE supports "
                f"{sorted(_SUPPORTED_ACTIVATIONS)} routed experts; got "
                f"activation {activation!r}. Another variant needs a matching "
                "activation-and-requant implementation."
            )
        # All three activation parameters are required, none defaulted.
        # Community's own MXFP8 experts do the same: ``Mxfp8EmulationExperts``
        # raises on a missing clamp limit and forwards ``gemm1_alpha`` /
        # ``gemm1_beta`` (both ``float | None``) without substituting anything.
        # Defaulting here is the dangerous option in both directions: M3's
        # 1.702 / 1.0 / 7.0 would silently mis-run another checkpoint, and
        # ``SiluAndMulWithClamp``'s 1.0 / 0.0 would silently degrade M3 to
        # ``silu(gate) * up`` if the config wiring in ``MiniMaxM3MoE`` ever
        # broke. ``_SUPPORTED_ACTIVATIONS`` is a single variant that needs all
        # three, so demand all three.
        missing = [
            name
            for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            if getattr(layer, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"IluModelOptMxFp8FusedMoE requires {missing} on the layer for "
                f"{sorted(_SUPPORTED_ACTIVATIONS)} routed experts; the MoE "
                "config did not supply them."
            )
        swiglu_limit = layer.swiglu_limit
        swiglu_alpha = layer.swiglu_alpha
        swiglu_beta = layer.swiglu_beta
        return ilu_mxfp4_fused_experts(
            hidden_states=x,
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=layer.expert_map,
            output_scale=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
            swiglu_limit=float(swiglu_limit),
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            backend="ixmm",
            swiglu_variant=_SWIGLUOAI_VARIANT,
        )


@register_quantization_config("mxfp8")
@register_quantization_config("modelopt_mxfp8")
class IluModelOptMxFp8Config(ModelOptMxFp8Config):
    """MXFP8 config whose routed experts use the Iluvatar ix30 compute path.

    Only ``FusedMoEMethodCls`` differs from community: the inherited
    ``ModelOptQuantConfigBase.get_quant_method`` dispatches Linear layers through
    ``LinearMethodCls`` (unchanged ``ModelOptMxFp8LinearMethod``, which picks up
    ``IluMxfp8LinearKernel`` from the patched kernel list) and ``RoutedExperts``
    through ``FusedMoEMethodCls``.

    Both community keys must be shadowed. ``get_quantization_config`` builds a
    ``method_to_config`` map in which ``ModelOptMxFp8Config`` is reachable under
    *two* names: ``"modelopt_mxfp8"`` (ModelOpt checkpoints, via
    ``override_quantization_method``) and ``"mxfp8"`` (MiniMax-style checkpoints
    that tag ``quant_method: "mxfp8"`` directly -- which is what MiniMax-M3 does).
    Registering only the former leaves MiniMax-M3 on the community class and back
    on MARLIN.
    """

    def get_name(self) -> Any:
        # Keep the community name so model-config resolution and
        # ``override_quantization_method`` are unaffected by the shadowing.
        return "modelopt_mxfp8"


IluModelOptMxFp8Config.FusedMoEMethodCls = IluModelOptMxFp8FusedMoE
