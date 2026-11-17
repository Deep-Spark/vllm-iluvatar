# SPDX-License-Identifier: Apache-2.0
"""ivcore30 backends for serialized MXFP8 routed experts.

``IluMxFp8FusedMoE`` stages checkpoint tensors on the host and converts them
after loading. ``IluMxFp8ToFp4OnlineMoE`` converts shards as they arrive.
Both execute the resulting ix30 FP4 layout with the ixmm group GEMM backend.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.logger import init_logger
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

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
    _e8m0_to_fp32 as e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _DSV4_CLAMP_VARIANT,
    _SWIGLUOAI_VARIANT,
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
    _IXMM_EXPERT_CHUNK,
    _IXMM_GROUP_SIZE,
    preprocess_fp8_block_weight_for_ixmm,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
    _get_ilu_output_scale,
)
from vllm_iluvatar.model_executor.layers.quantization.common.weight_utils import (
    publish_derived_weight as _publish_derived_weight,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_fp4 import (
    IX30_GROUP_SIZE,
    Fp8SourceLayout,
    convert_to_ix30_fp4,
    ix30_fp4_shapes,
)

logger = init_logger(__name__)

# Community activation -> the ixformer SwiGLU variant that reproduces exactly
# its math. Anything else must fail loudly rather than compute a different
# activation, so the set is explicit and closed.
#
#   swigluoai_uninterleave (MiniMax-M3): packed gate/up halves, ``alpha`` inside
#       the sigmoid, ``beta`` offset on the up branch. Needs all three of
#       swiglu_limit / swiglu_alpha / swiglu_beta.
#   silu + a clamp limit (HY V4): community ``silu_and_mul_with_clamp`` computes
#       ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)``, which is exactly
#       ``_DSV4_CLAMP_VARIANT``. HY V4 expresses the clamp purely through
#       ``swiglu_limit`` and leaves ``activation`` at the "silu" default, so the
#       limit -- not the activation name -- is what distinguishes it from plain
#       SwiGLU. alpha/beta must be absent: M3's 1.702 / 1.0 would silently
#       compute a different activation.
_SWIGLUOAI_ACTIVATION = "swigluoai_uninterleave"
_CLAMPED_SILU_ACTIVATION = "silu"
_SUPPORTED_ACTIVATIONS = frozenset({_SWIGLUOAI_ACTIVATION, _CLAMPED_SILU_ACTIVATION})


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


class IluMxFp8FusedMoE(ModelOptMxFp8FusedMoE):
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
    # subtlety regardless. See ``nvfp4_moe.py`` / ``fp8_moe.py``.
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
            "IluMxFp8FusedMoE only handles serialized MXFP8 checkpoints"
        )
        assert not hasattr(self, "experts_cls"), (
            "IluMxFp8FusedMoE must not define experts_cls; community "
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
                "MXFP8 MoE FP4 path requires hidden_size divisible by "
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
                "IluMxFp8FusedMoE requires the ixmm backend (ix30 "
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
    # with explicit tensors, selecting the ixformer SwiGLU variant that matches
    # the layer's declared activation before dynamically quantizing GEMM2 input.
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
                "IluMxFp8FusedMoE supports "
                f"{sorted(_SUPPORTED_ACTIVATIONS)} routed experts; got "
                f"activation {activation!r}. Another variant needs a matching "
                "activation-and-requant implementation."
            )
        # All three activation parameters are required, none defaulted.
        # Community's own MXFP8 experts do the same: ``Mxfp8EmulationExperts``
        # raises on a missing clamp limit and forwards ``gemm1_alpha`` /
        # ``gemm1_beta`` (both ``float | None``) without substituting anything.
        # Defaulting here is the dangerous option in both directions: M3's
        # 1.702 / 1.0 / 7.0 would silently corrupt another checkpoint, and
        # ``SiluAndMulWithClamp``'s 1.0 / 0.0 would silently degrade M3 to
        # ``silu(gate) * up`` if the config wiring in ``MiniMaxM3MoE`` ever
        # broke. Each variant demands exactly the parameters it consumes.
        if activation == _SWIGLUOAI_ACTIVATION:
            required = ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            swiglu_variant = _SWIGLUOAI_VARIANT
        else:
            # Plain SwiGLU has no clamp and would need a different ixformer op,
            # so the limit is what makes "silu" acceptable here at all.
            required = ("swiglu_limit",)
            swiglu_variant = _DSV4_CLAMP_VARIANT
        missing = [name for name in required if getattr(layer, name, None) is None]
        if missing:
            raise ValueError(
                f"IluMxFp8FusedMoE requires {missing} on the layer for "
                f"{activation!r} routed experts; the MoE config did not supply "
                "them."
            )
        swiglu_limit = layer.swiglu_limit
        swiglu_alpha = layer.swiglu_alpha
        swiglu_beta = layer.swiglu_beta
        if swiglu_variant == _DSV4_CLAMP_VARIANT and (
            swiglu_alpha is not None or swiglu_beta is not None
        ):
            raise NotImplementedError(
                "IluMxFp8FusedMoE maps clamped-silu routed experts onto "
                "ixformer's DSv4 clamp SwiGLU, which has no alpha/beta; got "
                f"swiglu_alpha={swiglu_alpha!r}, swiglu_beta={swiglu_beta!r}. A "
                "checkpoint needing both the clamp and alpha/beta must declare "
                f"{_SWIGLUOAI_ACTIVATION!r}."
            )
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
            swiglu_variant=swiglu_variant,
        )


# How many half-loaded (weight without scale, or the reverse) expert shards may
# be outstanding before we call it a bug. Each one pins an MXFP8 staging buffer,
# so this is the knob that bounds the peak this method exists to reduce.
#
# Unlike the block-FP8 loader, the bound cannot be a small constant. Hy4-preview
# stores a whole layer's experts as one fused 3-D tensor
# (``experts.gate_up_proj [E, 2I, H]``) with the scales in a *separate* tensor
# (``experts.gate_up_proj_scale``), and the model's ``load_weights`` slices the
# weight tensor per expert. So every local expert's w1 and w3 weight lands before
# the first scale does: the true watermark is ``2 * num_local_experts``, and
# ``3 *`` leaves room for w2 overlapping the w13 group. Staging then peaks at one
# layer's MXFP8 (EP8 Hy4: ~1.2 GiB on device), still far below the whole-model
# host staging it replaces. A split per-expert checkpoint keeps arriving
# weight-then-scale and never gets near this.
_PENDING_SHARDS_PER_LOCAL_EXPERT = 3

# Loaders that hand tensors to ``weight_loader`` in a single sequential pass.
# Anything else may interleave or parallelise shard arrival, which would race
# on the staging buffers.
_SEQUENTIAL_LOAD_FORMATS = frozenset({"auto", "safetensors", "pt", "npcache"})

_W13 = "w13"
_W2 = "w2"

# The MXFP8 source block, as a ``(block_n, block_k)`` pair: one scale per row,
# one per 32 K elements. The converter's dequant is generic over the block
# shape; only ``scale_dtype`` has to be fp32, which is why ``_convert`` decodes
# the checkpoint's UE8M0 bytes first.
_MXFP8_SOURCE = Fp8SourceLayout(format="fp8-block", block_shape=(1, MXFP8_BLOCK_SIZE))


class IluMxFp8ToFp4OnlineMoE(IluMxFp8FusedMoE):
    """MXFP8 routed experts converted to ix30 FP4 as shards are loaded."""

    def __init__(self, *args, **kwargs) -> None:
        # None = derive from the local expert count in ``create_weights``.
        self.max_pending_shards: int | None = kwargs.pop("max_pending_shards", None)
        self.converter_backend = kwargs.pop("converter_backend", None)
        super().__init__(*args, **kwargs)
        # Per-layer loading state. A method instance is built per layer by
        # ``get_quant_method``, so this never crosses layers; ``_layer`` makes
        # that assumption checkable instead of implicit.
        self._layer: RoutedExperts | None = None
        self._fp4_scale_shapes: dict[str, tuple[int, ...]] = {}
        self._fp4_scales: dict[str, torch.Tensor] = {}
        self._staging: dict[tuple[str, int], torch.Tensor] = {}
        self._weight_ready: set[tuple[str, int, str]] = set()
        self._scale_ready: set[tuple[str, int, str]] = set()
        self._converted: set[tuple[str, int, str]] = set()
        self._pending_peak = 0
        self._conversions = 0

    # ------------------------------------------------------------------
    # Weight creation
    # ------------------------------------------------------------------
    # [ILU override] IluMxFp8FusedMoE.create_weights — delta: w13_weight /
    # w2_weight are allocated on the device in the ix30 FP4 layout, and the
    # MXFP8 checkpoint weights never get a resident parameter at all. The
    # checkpoint's UE8M0 block scales keep their community names, shapes and
    # loader (the expert-params mapping looks them up by name and the converter
    # reads them); they are replaced by the derived ix30 scales in
    # ``process_weights_after_loading``.
    #
    # The parent body is not reused even for validation, because it allocates
    # the full MXFP8 weights up front -- the exact allocation this method exists
    # to avoid. Its divisibility checks are repeated below instead.
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
        self._guard_supported(layer, hidden_size, intermediate_size_per_partition)
        self._layer = layer

        if hidden_size % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                "MXFP8 load-time FP4 needs hidden_size divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {hidden_size}."
            )
        if intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                "MXFP8 load-time FP4 needs intermediate_size_per_partition "
                f"divisible by {MXFP8_BLOCK_SIZE}, got "
                f"{intermediate_size_per_partition}."
            )

        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        self._num_local_experts = num_experts
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size_per_partition
        if self.max_pending_shards is None:
            self.max_pending_shards = _PENDING_SHARDS_PER_LOCAL_EXPERT * num_experts

        # FP4 targets. The ix30 layout transposes, so the checkpoint's K becomes
        # the packed dim and its N the last dim: w13 is [E, H/2, 2I] and w2 is
        # [E, align(I, 256)/2, H].
        w13_shape, w13_scale_shape = ix30_fp4_shapes(
            n=2 * intermediate_size_per_partition, k=hidden_size
        )
        w2_shape, w2_scale_shape = ix30_fp4_shapes(
            n=hidden_size, k=intermediate_size_per_partition
        )
        if w13_shape[0] * 2 != hidden_size:
            raise ValueError(
                "MXFP8 load-time FP4 needs hidden_size divisible by "
                f"{IX30_GROUP_SIZE} (it is the K dim of w13 and is not zero "
                f"padded), got {hidden_size}"
            )

        for name, shape, scale_shape in (
            ("w13_weight", w13_shape, w13_scale_shape),
            ("w2_weight", w2_shape, w2_scale_shape),
        ):
            weight = torch.nn.Parameter(
                torch.empty((num_experts, *shape), dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter(name, weight)
            set_weight_attrs(weight, extra_weight_attrs)
            # Overwrite the community loader: it would copy MXFP8 bytes straight
            # into this FP4 parameter.
            weight.weight_loader = self._make_fp4_weight_loader(layer, name)
            # The derived ix30 scale is held on the method, not the layer: no
            # checkpoint tensor maps to it, and the name it will eventually take
            # (``*_weight_scale``) is occupied by the checkpoint block scale for
            # the duration of loading. Allocated on first use so it lands on
            # whichever device the FP4 weight ended up on.
            self._fp4_scale_shapes[name.replace("_weight", "_weight_scale")] = (
                num_experts,
                *scale_shape,
            )

        # Checkpoint UE8M0 block scales, kept in their native uint8 dtype rather
        # than upcast: fp32 would cost 4x here and these are only read once per
        # shard. ``_convert`` decodes the narrow slice it needs.
        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            requires_grad=False,
        )
        w2_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            requires_grad=False,
        )
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        for name, param in (
            ("w13_weight_scale", w13_scale),
            ("w2_weight_scale", w2_scale),
        ):
            layer.register_parameter(name, param)
            set_weight_attrs(param, scale_attrs)
            param.weight_loader = self._make_scale_loader(layer, name)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _guard_supported(
        self, layer: RoutedExperts, hidden_size: int, intermediate_size: int
    ) -> None:
        """Reject every checkpoint or parallel layout this loader cannot do."""
        if not _ixmm_backend_available():
            raise RuntimeError(
                "MXFP8 load-time FP4 requires the ixmm backend (ix30 "
                "group_gemm_fp4 only accepts the preprocessed uint8 layout). "
                "Install the ixmm wheel and rebuild ixformer with HAVE_IXMM on "
                "ivcore30."
            )
        moe = layer.moe_config
        if moe.has_bias:
            raise ValueError(
                "MXFP8 load-time FP4 does not support biased routed experts"
            )
        if not moe.is_act_and_mul:
            raise ValueError(
                "MXFP8 load-time FP4 assumes a gate/up pair in w13 (is_act_and_mul)"
            )
        if moe.hidden_dim_unpadded != moe.hidden_dim:
            raise ValueError(
                "MXFP8 load-time FP4 does not support a padded hidden dim "
                f"({moe.hidden_dim_unpadded} -> {moe.hidden_dim}): the community "
                "loader would leave part of the staging buffer unwritten and "
                "the padding would be quantized as garbage"
            )
        unpadded_i = moe.intermediate_size_per_partition_unpadded
        if unpadded_i is not None and unpadded_i != intermediate_size:
            raise ValueError(
                "MXFP8 load-time FP4 does not support a padded intermediate "
                f"size ({unpadded_i} -> {intermediate_size}), for the same reason"
            )
        if getattr(layer, "enable_eplb", False):
            raise ValueError(
                "MXFP8 load-time FP4 does not support EPLB (the loader consumes "
                "each shard once and frees the MXFP8 source, so experts cannot "
                "be re-placed afterwards)"
            )

        load_format = self._current_load_format()
        if load_format is not None and load_format not in _SEQUENTIAL_LOAD_FORMATS:
            raise ValueError(
                "MXFP8 load-time FP4 requires a sequential weight loader; "
                f"load_format={load_format!r} may interleave shards across "
                "threads and race on the staging buffers. Supported: "
                f"{sorted(_SEQUENTIAL_LOAD_FORMATS)}"
            )

    @staticmethod
    def _current_load_format() -> str | None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except Exception:  # noqa: BLE001
            return None
        load_config = getattr(vllm_config, "load_config", None)
        load_format = getattr(load_config, "load_format", None)
        if load_format is None:
            return None
        return str(getattr(load_format, "value", load_format)).lower()

    def _make_fp4_weight_loader(self, layer: RoutedExperts, param_name: str):
        group = _W13 if param_name == "w13_weight" else _W2

        def fp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            local_expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert == -1:
                return False if return_success else None
            self._guard_shard(param, loaded_weight, weight_name, shard_id, group)
            staging = self._get_staging(layer, group, local_expert)
            tp_rank = layer.moe_config.tp_rank
            if group == _W13:
                layer._load_w13(
                    expert_data=staging,
                    shard_dim=0,
                    shard_id=shard_id,
                    loaded_weight=loaded_weight,
                    tp_rank=tp_rank,
                )
            else:
                layer._load_w2(
                    expert_data=staging,
                    shard_dim=1,
                    loaded_weight=loaded_weight,
                    tp_rank=tp_rank,
                )
            self._mark_ready(layer, self._weight_ready, group, local_expert, shard_id)
            return True if return_success else None

        return fp4_weight_loader

    def _make_scale_loader(self, layer: RoutedExperts, param_name: str):
        group = _W13 if param_name.startswith("w13") else _W2
        community_loader = layer.weight_loader

        def block_scale_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            loaded = community_loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            if loaded:
                local_expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
                self._mark_ready(
                    layer, self._scale_ready, group, local_expert, shard_id
                )
            return loaded if return_success else None

        return block_scale_loader

    def _guard_shard(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        group: str,
    ) -> None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")
        if (group == _W13) != (shard_id in ("w1", "w3")):
            raise ValueError(
                f"MXFP8 load-time FP4: shard_id={shard_id!r} does not belong to "
                f"{group}_weight"
            )
        if "bias" in weight_name:
            raise ValueError(
                "MXFP8 load-time FP4 does not support expert biases; "
                "create_weights should have rejected this layer"
            )
        if loaded_weight.dim() != 2:
            raise ValueError(
                "MXFP8 load-time FP4 expects one 2-D expert tensor per call, "
                f"got {loaded_weight.dim()}-D for {weight_name!r}. A fused 3-D "
                "checkpoint must be sliced per expert by the model's "
                "load_weights before reaching here."
            )
        if loaded_weight.dtype is not MXFP8_VALUE_DTYPE:
            raise ValueError(
                f"MXFP8 load-time FP4 expects {MXFP8_VALUE_DTYPE} expert "
                f"weights, got {loaded_weight.dtype} for {weight_name!r}"
            )
        if getattr(param, "is_transposed", False):
            raise ValueError(
                "MXFP8 load-time FP4 does not support is_transposed params"
            )
        if getattr(param, "use_bitsandbytes_4bit", False):
            raise ValueError("MXFP8 load-time FP4 does not support BitsAndBytes params")

    def _get_staging(
        self, layer: RoutedExperts, group: str, local_expert: int
    ) -> torch.Tensor:
        key = (group, local_expert)
        staging = self._staging.get(key)
        if staging is None:
            shape = (
                (2 * self._intermediate_size, self._hidden_size)
                if group == _W13
                else (self._hidden_size, self._intermediate_size)
            )
            staging = torch.empty(
                shape, dtype=MXFP8_VALUE_DTYPE, device=layer.w13_weight.device
            )
            self._staging[key] = staging
        return staging

    def _mark_ready(
        self,
        layer: RoutedExperts,
        ready: set[tuple[str, int, str]],
        group: str,
        local_expert: int,
        shard_id: str,
    ) -> None:
        unit = (group, local_expert, shard_id)
        ready.add(unit)
        if unit in self._weight_ready and unit in self._scale_ready:
            self._convert(layer, unit)
            return
        self._check_pending(layer)

    def _fp4_scale(self, name: str, like: torch.Tensor) -> torch.Tensor:
        scale = self._fp4_scales.get(name)
        if scale is None:
            scale = torch.empty(
                self._fp4_scale_shapes[name], dtype=torch.float32, device=like.device
            )
            self._fp4_scales[name] = scale
        return scale

    def _convert(self, layer: RoutedExperts, unit: tuple[str, int, str]) -> None:
        group, local_expert, shard_id = unit
        if group == _W13:
            size = self._intermediate_size
            offset = 0 if shard_id == "w1" else size
            src_weight = self._staging[(group, local_expert)].narrow(0, offset, size)
            # block_n == 1, so the scale has one row per output column and its
            # narrow offsets are the weight's.
            src_scale = layer.w13_weight_scale.data[local_expert].narrow(
                0, offset, size
            )
            out_weight = layer.w13_weight.data[local_expert].narrow(1, offset, size)
            out_scale = self._fp4_scale("w13_weight_scale", out_weight)[
                local_expert
            ].narrow(1, offset, size)
        else:
            src_weight = self._staging[(group, local_expert)]
            src_scale = layer.w2_weight_scale.data[local_expert]
            out_weight = layer.w2_weight.data[local_expert]
            out_scale = self._fp4_scale("w2_weight_scale", out_weight)[local_expert]

        convert_to_ix30_fp4(
            src_weight.unsqueeze(0),
            # UE8M0 exponents decoded to the fp32 multipliers the converter
            # expects. The slice is one expert shard, a few MiB at most.
            e8m0_to_fp32(src_scale).unsqueeze(0),
            source=_MXFP8_SOURCE,
            out_weight=out_weight.unsqueeze(0),
            out_scale=out_scale.unsqueeze(0),
            backend=self.converter_backend,
        )
        self._weight_ready.discard(unit)
        self._scale_ready.discard(unit)
        self._converted.add(unit)
        self._conversions += 1
        self._release_staging_if_done(group, local_expert)

    def _release_staging_if_done(self, group: str, local_expert: int) -> None:
        shards = ("w1", "w3") if group == _W13 else ("w2",)
        if all((group, local_expert, shard) in self._converted for shard in shards):
            self._staging.pop((group, local_expert), None)

    def _pending_units(self) -> list[tuple[str, int, str]]:
        return sorted(self._weight_ready ^ self._scale_ready)

    def _check_pending(self, layer: RoutedExperts) -> None:
        pending = self._pending_units()
        self._pending_peak = max(self._pending_peak, len(pending))
        if len(pending) > self.max_pending_shards:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: {len(pending)} "
                "half-loaded expert shards exceed max_pending_shards="
                f"{self.max_pending_shards} ({_PENDING_SHARDS_PER_LOCAL_EXPERT} x "
                f"{self._num_local_experts} local experts). Each one pins an "
                "MXFP8 staging buffer, so shards are accumulating well past a "
                "single layer's worth -- the arrival order is not what this "
                "loader assumes. Stuck units (group, local_expert, shard): "
                f"{pending[:16]}{' ...' if len(pending) > 16 else ''}"
            )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    # [ILU override] IluMxFp8FusedMoE.process_weights_after_loading — delta:
    # there is nothing left to convert, so this only verifies that every shard
    # did convert, then swaps the checkpoint block scales out for the derived
    # ix30 ones so the inherited ``apply`` finds them under the names it reads.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if self._layer is not layer:
            raise RuntimeError(
                "MXFP8 load-time FP4 method was created for a different layer; "
                "each RoutedExperts needs its own instance (it holds per-layer "
                "loading state)"
            )
        expected = {
            (_W13, expert, shard)
            for expert in range(self._num_local_experts)
            for shard in ("w1", "w3")
        } | {(_W2, expert, "w2") for expert in range(self._num_local_experts)}
        missing = sorted(expected - self._converted)
        if missing:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: {len(missing)} expert "
                "shards were never converted to FP4, so the parameter still "
                "holds uninitialised bytes. Missing (group, local_expert, "
                f"shard): {missing[:16]}"
                f"{' ...' if len(missing) > 16 else ''}"
            )
        if self._staging:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: MXFP8 staging buffers "
                f"are still held after loading: {sorted(self._staging)}"
            )

        # Publish as non-persistent buffers, for the reason spelled out in
        # ``common.weight_utils.publish_derived_weight``: a Parameter under a
        # name the loader saw on CPU gets evicted back to host when
        # ``device_loading_context`` closes.
        device = layer.w13_weight.device
        for public_name in ("w13_weight_scale", "w2_weight_scale"):
            scale = self._fp4_scales.pop(public_name)
            _publish_derived_weight(layer, public_name, scale.to(device))

        layer._already_called_process_weights_after_loading = True
        layer._ilu_mxfp4_ixmm = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug(
            "MXFP8 load-time FP4: %s converted %d expert shards (%d local "
            "experts), peak half-loaded shards %d",
            getattr(layer, "layer_name", "<moe>"),
            self._conversions,
            self._num_local_experts,
            self._pending_peak,
        )
