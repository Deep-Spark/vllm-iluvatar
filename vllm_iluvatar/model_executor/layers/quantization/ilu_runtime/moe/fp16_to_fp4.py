# SPDX-License-Identifier: Apache-2.0
"""Load-time native BF16 -> ix30 FP4 for MiniMax-M3 routed experts.

GLM's ``IluFp8ToFp4OnlineMoEMethod`` loads serialized block-FP8. MiniMax-M3's
checkpoint is native BF16 with no ``weight_scale_inv``. This subclass keeps the
GLM FP4 layout and ``weight_load`` staging, and quantizes each BF16 shard
straight to MXFP4 (no block-FP8 detour). Parent file is not modified.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _SWIGLUOAI_VARIANT,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp8_moe import (
    _SUPPORTED_ACTIVATIONS,
    _get_ilu_output_scale,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_fp4 import (
    IX30_GROUP_SIZE,
    Fp8SourceLayout,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.config import (
    dtype_name,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.moe.fp8_to_fp4 import (
    _DEFAULT_MAX_PENDING_SHARDS,
    _W13,
    IluFp8ToFp4OnlineMoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

logger = init_logger(__name__)

_DEFAULT_FP8_BLOCK = (128, 128)

# MXFP4 microscaling: 32 e2m1 values share one scale. ``_E2M1_LEVELS`` is the
# full signed code book indexed by nibble, so quantization is a nearest-level
# search rather than a hand-rolled exponent/mantissa split.
MXFP4_GROUP = 32
_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX_MAG = 6.0


def quantize_native_to_mxfp4(
    weight: torch.Tensor,
    group: int = MXFP4_GROUP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a native BF16 shard straight to MXFP4.

    Returns ``(packed [n, k // 2] uint8, scale [n, k // group] float32)`` in the
    layout ``ixformer.functions.preprocess_weight_for_ixmm`` consumes: two e2m1
    nibbles per byte, low nibble first (the order
    ``dequant_mxfp4_to_bf16`` unpacks).

    MiniMax-M3 ships native BF16, so there is no reason to detour through
    block-FP8: that would add a second lossy step and would pin the plugin to
    ``preprocess_fp8_block_weight_for_ixmm``, which the ixformer submodule does
    not export.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"quantize_native_to_mxfp4 expects a 2-D shard, got {weight.dim()}-D"
        )
    n, k = weight.shape
    if k % group != 0:
        raise ValueError(f"k={k} must be a multiple of the MXFP4 group {group}")

    w = weight.float().reshape(n, k // group, group)
    amax = w.abs().amax(dim=-1, keepdim=True)
    # An all-zero group has no scale; 1.0 keeps it finite and quantizes to 0.
    scale = torch.where(amax > 0, amax / _FP4_MAX_MAG, torch.ones_like(amax))
    normalized = (w / scale).clamp(-_FP4_MAX_MAG, _FP4_MAX_MAG)

    levels = torch.tensor(_E2M1_MAG, dtype=torch.float32, device=weight.device)
    mag = normalized.abs()
    # Nearest representable magnitude; ties go to the larger level, matching
    # round-half-away-from-zero on the code book.
    codes = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    codes = torch.where(normalized < 0, codes | 0x08, codes)

    codes = codes.reshape(n, k)
    low, high = codes[:, 0::2], codes[:, 1::2]
    packed = (low | (high << 4)).contiguous()
    return packed, scale.reshape(n, k // group).contiguous()


def convert_native_to_ix30_fp4(
    weight: torch.Tensor,
    *,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[n, k]`` shard -> ix30 ``([k/2, n] uint8, [k/256, n] float32)``."""
    import ixformer.functions as ixfops

    packed, scale = quantize_native_to_mxfp4(weight)
    ixmm_weight, ixmm_scale = ixfops.preprocess_weight_for_ixmm(
        packed.unsqueeze(0).view(torch.int8),
        scale.unsqueeze(0),
        backend=backend,
    )
    return ixmm_weight.squeeze(0), ixmm_scale.squeeze(0)


@IluvatarQuantRegistry.register_method
class IluFp16ToFp4OnlineMoEMethod(IluFp8ToFp4OnlineMoEMethod):
    """GLM load-time FP4 loader fed from MiniMax-M3 native BF16 expert shards."""

    @classmethod
    def from_ilu_runtime_config(cls, context: Any) -> IluFp16ToFp4OnlineMoEMethod:
        group = context.group
        gid = group.group_id
        weights = group.weights

        layer = context.layer
        if not isinstance(layer, RoutedExperts):
            raise TypeError(f"{gid}: load-time FP4 only supports RoutedExperts")
        if not isinstance(context.original_method, UnquantizedFusedMoEMethod):
            raise TypeError(
                f"{gid}: MiniMax-M3 load-time FP4 requires "
                f"UnquantizedFusedMoEMethod, got "
                f"{type(context.original_method).__name__}"
            )
        if group.require.original_disposition != "ignored":
            raise ValueError(
                f"{gid}: MiniMax-M3 load-time FP4 requires original_disposition=ignored"
            )
        if (
            weights.loading.mode != "transform"
            or weights.loading.stage != "weight_load"
        ):
            raise ValueError(
                f"{gid}: MiniMax-M3 load-time FP4 requires "
                "loading=transform@weight_load"
            )

        source_spec = weights.source
        if source_spec.format != "native":
            raise ValueError(
                f"{gid}: unsupported weights.source.format {source_spec.format!r}, "
                "expected 'native'"
            )
        if tuple(source_spec.dtypes or ()) not in ((), ("bfloat16",)):
            raise ValueError(
                f"{gid}: unsupported weights.source.dtypes "
                f"{tuple(source_spec.dtypes)}, expected ('bfloat16',)"
            )

        target = weights.target
        if (
            target.format != "ix30-fp4"
            or target.dtype != "uint8"
            or target.strategy != "group"
        ):
            raise ValueError(f"{gid}: unsupported load-time FP4 weight target")
        group_size = target.extra.get("group_size", IX30_GROUP_SIZE)
        if group_size != IX30_GROUP_SIZE:
            raise ValueError(
                f"{gid}: ix30 FP4 group_size must be {IX30_GROUP_SIZE}, got {group_size}"
            )
        scale_format = target.extra.get("scale_format", "running_acc")
        if scale_format != "running_acc":
            raise ValueError(
                f"{gid}: ix30 FP4 scale_format must be running_acc, got {scale_format!r}"
            )
        unknown_extra = set(target.extra) - {"group_size", "scale_format"}
        if unknown_extra:
            raise ValueError(
                f"{gid}: unsupported weights.target.extra keys: {sorted(unknown_extra)}"
            )

        activation = group.input_activations
        if (
            activation is None
            or activation.num_bits != 8
            or activation.type != "float"
            or activation.strategy != "token"
            or not activation.dynamic
            or group.output_activations is not None
        ):
            raise ValueError(
                f"{gid}: load-time FP4 requires dynamic per-token FP8 input "
                "activations and null output_activations"
            )

        allowed_options = {
            "converter_backend",
            "max_pending_shards",
            "fp8_block_shape",
        }
        unknown_options = set(group.options) - allowed_options
        if unknown_options:
            raise ValueError(f"{gid}: unsupported options: {sorted(unknown_options)}")
        backend = group.options.get("converter_backend", "auto")
        if backend not in ("auto", "torch", "ixformer_native"):
            raise ValueError(
                f"{gid}: converter_backend must be auto / torch / "
                f"ixformer_native, got {backend!r}"
            )
        max_pending = group.options.get(
            "max_pending_shards", _DEFAULT_MAX_PENDING_SHARDS
        )
        if not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError(
                f"{gid}: max_pending_shards must be a positive int, got {max_pending!r}"
            )

        block = group.options.get("fp8_block_shape", list(_DEFAULT_FP8_BLOCK))
        if isinstance(block, tuple):
            block = list(block)
        if (
            not isinstance(block, list)
            or len(block) != 2
            or any(not isinstance(v, int) or v <= 0 for v in block)
        ):
            raise ValueError(
                f"{gid}: fp8_block_shape must be two positive ints, got {block!r}"
            )
        block_shape = (int(block[0]), int(block[1]))

        return cls(
            context.base_quant_config,
            layer,
            ilu_params={"weight_block_size": [block_shape[0], block_shape[1]]},
            source=Fp8SourceLayout(format="fp8-block", block_shape=block_shape),
            converter_backend=None if backend == "auto" else backend,
            max_pending_shards=max_pending,
            runtime_context=context,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # Parent registers checkpoint scale_inv slots. MiniMax-M3 has none.
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)
        # Parent allocates ix30 scales with ``torch.empty``. Convert writes
        # every column, but a missed shard used to leave NaN scales that
        # ``group_gemm_fp4`` turns into NaN logits. Ones are a finite default
        # until convert overwrites them.
        layer.w13_weight_scale.data.fill_(1.0)
        layer.w2_weight_scale.data.fill_(1.0)
        logger.info_once(
            "MiniMax load-time FP4 created: w13 dtype=%s shape=%s "
            "w2 dtype=%s shape=%s (example %s)",
            layer.w13_weight.dtype,
            tuple(layer.w13_weight.shape),
            layer.w2_weight.dtype,
            tuple(layer.w2_weight.shape),
            getattr(layer, "layer_name", "<moe>"),
        )

    def _get_staging(
        self, layer: RoutedExperts, group: str, local_expert: int
    ) -> torch.Tensor:
        key = (group, local_expert)
        staging = self._staging.get(key)
        if staging is None:
            if group == _W13:
                shape = (2 * self._intermediate_size, self._hidden_size)
            else:
                shape = (self._hidden_size, self._intermediate_size)
            staging = torch.zeros(
                shape, dtype=torch.bfloat16, device=layer.w13_weight.device
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
        # Parent waits for a matching scale_inv. MiniMax has none, so we
        # convert as soon as the BF16 shard lands. The community loader can
        # invoke weight_loader more than once for the same (expert, shard);
        # a second pass would allocate a fresh zeros staging and overwrite
        # the FP4 image with Inf running-acc scales.
        if unit in self._converted:
            self._release_staging_if_done(group, local_expert)
            return
        ready.add(unit)
        if unit in self._weight_ready:
            self._convert(layer, unit)

    def _convert(self, layer: RoutedExperts, unit: tuple[str, int, str]) -> None:
        if unit in self._converted:
            return
        group, local_expert, shard_id = unit
        if group == _W13:
            size = self._intermediate_size
            offset = 0 if shard_id == "w1" else size
            native = self._staging[(group, local_expert)].narrow(0, offset, size)
            out_weight = layer.w13_weight.data[local_expert].narrow(1, offset, size)
            out_scale = layer.w13_weight_scale.data[local_expert].narrow(
                1, offset, size
            )
        else:
            native = self._staging[(group, local_expert)]
            out_weight = layer.w2_weight.data[local_expert]
            out_scale = layer.w2_weight_scale.data[local_expert]

        backends = [self.converter_backend]
        if self.converter_backend != "torch":
            backends.append("torch")
        converted_ok = False
        tmp_weight = tmp_scale = None
        for backend in backends:
            tmp_weight, tmp_scale = convert_native_to_ix30_fp4(
                native.contiguous(), backend=backend
            )
            if torch.isfinite(tmp_scale).all():
                converted_ok = True
                break
        if not converted_ok:
            n_bad = int((~torch.isfinite(tmp_scale)).sum().item())
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: ix30 convert produced "
                f"{n_bad} non-finite scales for local_expert={local_expert} "
                f"shard={shard_id} (native amax="
                f"{native.float().abs().max().item():.4g}, "
                f"backends={backends})"
            )
        out_weight.copy_(tmp_weight)
        out_scale.copy_(tmp_scale)

        self._weight_ready.discard(unit)
        self._scale_ready.discard(unit)
        self._converted.add(unit)
        self._conversions += 1
        if self._conversions == 1:
            logger.info_once(
                "MiniMax load-time FP4 converting shards: example %s %s "
                "expert=%s shard=%s",
                getattr(layer, "layer_name", "<moe>"),
                group,
                local_expert,
                shard_id,
            )
        self._release_staging_if_done(group, local_expert)

    def _guard_shard(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        group: str,
    ) -> None:
        super()._guard_shard(param, loaded_weight, weight_name, shard_id, group)
        if dtype_name(loaded_weight.dtype) != "bfloat16":
            raise ValueError(
                f"{getattr(self._layer, 'layer_name', '<moe>')}: MiniMax-M3 "
                f"load-time FP4 expected bfloat16, got "
                f"{dtype_name(loaded_weight.dtype)} for {weight_name!r}"
            )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        # Parent GLM method can install ``IluMoEBatchedExpertsFp8`` (silu /
        # swiglustep). MiniMax compute is ``ilu_mxfp4_fused_experts`` via
        # ``apply``, the same naive EP-group path MXFP8 uses.
        return None

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> FusedMoEExpertsModular:
        raise NotImplementedError(
            "IluFp16ToFp4OnlineMoEMethod uses apply() -> ilu_mxfp4_fused_experts; "
            "GLM batched experts are not the MiniMax compute path"
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            bad_mask = ~torch.isfinite(scale)
            if bad_mask.any():
                bad_experts = bad_mask.reshape(scale.shape[0], -1).any(dim=1)
                bad_ids = bad_experts.nonzero(as_tuple=False).flatten().tolist()
                raise RuntimeError(
                    f"{getattr(layer, 'layer_name', '<moe>')}: {name} has "
                    f"{int(bad_mask.sum().item())} non-finite values after "
                    f"load-time FP4 convert (local experts {bad_ids[:8]})"
                )

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
                "IluFp16ToFp4OnlineMoEMethod supports "
                f"{sorted(_SUPPORTED_ACTIVATIONS)} routed experts; got "
                f"activation {activation!r}"
            )
        missing = [
            name
            for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            if getattr(layer, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"IluFp16ToFp4OnlineMoEMethod requires {missing} on the layer "
                f"for {sorted(_SUPPORTED_ACTIVATIONS)} routed experts"
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
            swiglu_limit=float(layer.swiglu_limit),
            swiglu_alpha=layer.swiglu_alpha,
            swiglu_beta=layer.swiglu_beta,
            backend="ixmm",
            swiglu_variant=_SWIGLUOAI_VARIANT,
        )
