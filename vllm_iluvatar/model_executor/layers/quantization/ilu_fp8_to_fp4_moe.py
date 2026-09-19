# SPDX-License-Identifier: Apache-2.0
"""Load-time block-FP8 -> ix30 FP4 conversion for routed experts.

Corresponds to: ``vllm/model_executor/layers/quantization/fp8.py::Fp8MoEMethod``
(via this plugin's ``IluFp8MoEMethod``).

Target   : ``additional_config.ilu_quant`` groups whose weights contract is
           ``fp8-block -> ix30-fp4`` @ ``transform`` + ``weight_load``.
Baseline : main_9c12a81ac
Why      : ``IluFp8MoEMethod`` keeps the full FP8 routed-expert weights
           resident until ``process_weights_after_loading``, then converts
           them to ix30 FP4 in one pass. For GLM-5.2 that means every rank
           holds both the FP8 tensors and their FP4 replacements at the same
           moment, and the transient peak -- not the steady state -- is what
           makes TP8 impossible today. The FP4 form is ~4x smaller, so the
           steady state alone fits comfortably.
How      : create the parameters in the FP4 layout from the start, then
           convert each checkpoint shard as it lands: the community loader
           writes the FP8 shard into a single-expert staging buffer, the
           converter turns it into ix30 FP4 directly inside the resident
           parameter, and the staging buffer is released. Peak extra memory is
           one expert's worth of FP8, a few MiB, instead of the whole layer.
Scope    : ``RoutedExperts`` with block-FP8 checkpoints and dynamic activation
           scaling, the layout GLM-5.2 / DeepSeek-V3 style models use. Anything
           else (biased experts, padded hidden dim, fused 3D expert tensors,
           transposed params, BitsAndBytes, static input scales, EPLB) is
           rejected at ``create_weights`` rather than silently mishandled.
Remove   : when the FP4 form becomes the checkpoint format, or when community
           vLLM grows a load-time requantization hook that this can sit on.

Why per-shard conversion is exact: the ix30 group scale is an amax over 256
consecutive K elements *within one output column*, and the pack is likewise
per (column, K-group). Columns are therefore independent, so converting the w1
half of ``w13`` on its own produces the same bytes as converting all ``2I``
columns at once. w2 is loaded whole per expert, so it has nothing to split.
Unit tests assert this byte-for-byte rather than assuming it.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8 import (
    IluFp8MoEMethod,
    _ixmm_backend_available,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_to_fp4 import (
    IX30_GROUP_SIZE,
    Fp8SourceLayout,
    convert_to_ix30_fp4,
    ix30_fp4_shapes,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_quant_registry import (
    IluvatarQuantRegistry,
)

logger = init_logger(__name__)

# Sequential checkpoint iterators hand us a weight and its scale within a few
# tensors of each other, so only a couple of units are ever half-filled. A
# higher watermark means the arrival order is not what we assumed and the
# staging buffers are accumulating -- exactly the memory blowup this method
# exists to avoid -- so it is an error, not a warning.
_DEFAULT_MAX_PENDING_SHARDS = 8

# Loaders that hand tensors to ``weight_loader`` in a single sequential pass.
# Anything else may interleave or parallelise shard arrival, which would race
# on the staging buffers.
_SEQUENTIAL_LOAD_FORMATS = frozenset({"auto", "safetensors", "pt", "npcache"})

_W13 = "w13"
_W2 = "w2"


@IluvatarQuantRegistry.register_method
class IluFp8ToFp4OnlineMoEMethod(IluFp8MoEMethod):
    """``IluFp8MoEMethod`` that quantizes to ix30 FP4 during weight loading.

    Only the loading path differs from the parent; ``apply``,
    ``get_fused_moe_quant_config`` and the rest of the compute surface are
    inherited untouched, so both methods produce identical results from
    identical bytes.
    """

    def __init__(
        self,
        quant_config: Fp8Config | Any,
        layer: RoutedExperts,
        ilu_params: dict[str, Any] | None = None,
        *,
        source: Fp8SourceLayout | None = None,
        converter_backend: str | None = None,
        max_pending_shards: int = _DEFAULT_MAX_PENDING_SHARDS,
        runtime_context: Any = None,
    ) -> None:
        super().__init__(quant_config, layer, ilu_params)
        block = self.weight_block_size
        self.source = source or Fp8SourceLayout(
            format="fp8-block", block_shape=(block[0], block[1])
        )
        self.converter_backend = converter_backend
        self.max_pending_shards = max_pending_shards
        self._runtime_context = runtime_context

        # Per-layer loading state. A method instance is built per layer by
        # ``get_quant_method``, so this never crosses layers; ``_layer`` makes
        # that assumption checkable instead of implicit.
        self._layer: RoutedExperts | None = None
        self._staging: dict[tuple[str, int], torch.Tensor] = {}
        self._weight_ready: set[tuple[str, int, str]] = set()
        self._scale_ready: set[tuple[str, int, str]] = set()
        self._converted: set[tuple[str, int, str]] = set()
        self._pending_peak = 0
        self._conversions = 0

    # ------------------------------------------------------------------
    # Runtime-config contract
    # ------------------------------------------------------------------

    @classmethod
    def from_ilu_runtime_config(cls, context: Any) -> IluFp8ToFp4OnlineMoEMethod:
        group = context.group
        gid = group.group_id
        weights = group.weights

        layer = context.layer
        if not isinstance(layer, RoutedExperts):
            raise TypeError(f"{gid}: load-time FP4 only supports RoutedExperts")
        if not isinstance(context.original_method, IluFp8MoEMethod):
            raise TypeError(
                f"{gid}: load-time FP4 requires the checkpoint to already "
                f"dispatch to IluFp8MoEMethod, got "
                f"{type(context.original_method).__name__}"
            )
        if group.require.original_disposition != "quantized":
            raise ValueError(
                f"{gid}: load-time FP4 requires original_disposition=quantized"
            )
        if (
            weights.loading.mode != "transform"
            or weights.loading.stage != "weight_load"
        ):
            raise ValueError(
                f"{gid}: load-time FP4 requires loading=transform@weight_load"
            )

        source_spec = weights.source
        if source_spec.format != "fp8-block":
            raise ValueError(
                f"{gid}: unsupported weights.source.format {source_spec.format!r}, "
                f"expected 'fp8-block'"
            )
        if tuple(source_spec.dtypes or ()) not in ((), ("float8_e4m3fn",)):
            raise ValueError(
                f"{gid}: unsupported weights.source.dtypes "
                f"{tuple(source_spec.dtypes)}, expected ('float8_e4m3fn',)"
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
                f"activations and null output_activations"
            )

        allowed_options = {"converter_backend", "max_pending_shards"}
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

        base = context.base_quant_config
        block_size = getattr(base, "weight_block_size", None)
        if not block_size or len(block_size) != 2:
            raise ValueError(
                f"{gid}: load-time FP4 needs a 2-entry weight_block_size on the "
                f"base FP8 config, got {block_size!r}"
            )
        source_block = source_spec.extra.get("block_shape")
        if source_block is not None and tuple(source_block) != tuple(block_size):
            raise ValueError(
                f"{gid}: weights.source.extra.block_shape {tuple(source_block)} "
                f"disagrees with the checkpoint's weight_block_size "
                f"{tuple(block_size)}"
            )

        quant_config = copy.copy(base)
        return cls(
            quant_config,
            layer,
            source=Fp8SourceLayout(
                format="fp8-block", block_shape=(block_size[0], block_size[1])
            ),
            converter_backend=None if backend == "auto" else backend,
            max_pending_shards=max_pending,
            runtime_context=context,
        )

    # ------------------------------------------------------------------
    # Weight creation
    # ------------------------------------------------------------------

    # [ILU override] Fp8MoEMethod.create_weights — delta: w13_weight / w2_weight
    # are allocated in the ix30 FP4 layout and paired with resident
    # ``*_weight_scale`` outputs, instead of FP8 tensors that a later pass
    # replaces. The community ``*_weight_scale_inv`` block scales are kept with
    # their original shapes and loaders: ``make_expert_params_mapping`` looks
    # them up by name, and the converter reads them. They are dropped in
    # ``process_weights_after_loading``.
    #
    # The parent body is not reused even for validation, because it allocates
    # the full FP8 weights up front -- the exact allocation this method exists
    # to avoid. Its two divisibility checks are repeated below instead.
    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self._guard_supported(layer, hidden_size, intermediate_size_per_partition)
        self._layer = layer

        block_n, block_k = self.weight_block_size
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}."
            )
        tp_size = layer.moe_config.moe_parallel_config.tp_size
        if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
            raise ValueError(
                f"The input_size of down's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_k = {block_k}."
            )

        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        self._num_local_experts = num_experts
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size_per_partition

        # FP4 targets. w13 is [E, H/2, 2I] and w2 is [E, align(I,256)/2, H]:
        # the ix30 layout transposes, so the checkpoint's K becomes the packed
        # dim and its N becomes the last dim.
        w13_shape, w13_scale_shape = ix30_fp4_shapes(
            n=2 * intermediate_size_per_partition, k=hidden_size
        )
        w2_shape, w2_scale_shape = ix30_fp4_shapes(
            n=hidden_size, k=intermediate_size_per_partition
        )
        if w13_shape[0] * 2 != hidden_size:
            raise ValueError(
                f"load-time FP4 needs hidden_size divisible by {IX30_GROUP_SIZE} "
                f"(it is the K dim of w13 and is not zero padded), got {hidden_size}"
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
            # Overwrite the community loader: it would copy FP8 bytes straight
            # into this FP4 parameter.
            weight.weight_loader = self._make_fp4_weight_loader(layer, name)

            scale = torch.nn.Parameter(
                torch.empty((num_experts, *scale_shape), dtype=torch.float32),
                requires_grad=False,
            )
            # No checkpoint tensor maps to this parameter: the converter derives
            # it from the FP8 weight. The model's reverse check would otherwise
            # report it as never loaded. Completeness is instead enforced per
            # (expert, shard) in process_weights_after_loading.
            scale.ilu_derived_at_load = True
            layer.register_parameter(name.replace("_weight", "_weight_scale"), scale)

        # Block scales keep the community shapes, names and loader; we only
        # wrap the loader to notice when a shard's scale has landed.
        w13_scale_inv = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_scale_inv = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        for name, param in (
            ("w13_weight_scale_inv", w13_scale_inv),
            ("w2_weight_scale_inv", w2_scale_inv),
        ):
            layer.register_parameter(name, param)
            set_weight_attrs(param, scale_attrs)
            param.weight_loader = self._make_scale_loader(layer, name)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        runtime_context = self._runtime_context
        if runtime_context is not None:
            runtime_context.record_shape_result(transformed=True)

    def _guard_supported(
        self, layer: RoutedExperts, hidden_size: int, intermediate_size: int
    ) -> None:
        """Reject every checkpoint or parallel layout this loader cannot do."""
        if not _ixmm_backend_available():
            raise RuntimeError(
                "load-time FP4 requires the ixmm backend (ix30 group_gemm_fp4 "
                "only accepts the preprocessed uint8 layout). Install the ixmm "
                "wheel and rebuild ixformer with HAVE_IXMM on ivcore30."
            )
        if not self.block_quant or self.weight_block_size is None:
            raise ValueError("load-time FP4 only handles block-wise FP8 checkpoints")
        if self.quant_config.activation_scheme != "dynamic":
            raise ValueError(
                f"load-time FP4 requires dynamic activation scaling, got "
                f"{self.quant_config.activation_scheme!r} (static input scales "
                f"would need per-tensor params this method does not create)"
            )

        moe = layer.moe_config
        if moe.has_bias:
            raise ValueError("load-time FP4 does not support biased routed experts")
        if not moe.is_act_and_mul:
            raise ValueError(
                "load-time FP4 assumes a gate/up pair in w13 (is_act_and_mul)"
            )
        if moe.hidden_dim_unpadded != moe.hidden_dim:
            raise ValueError(
                f"load-time FP4 does not support a padded hidden dim "
                f"({moe.hidden_dim_unpadded} -> {moe.hidden_dim}): the community "
                f"loader would leave part of the staging buffer unwritten and "
                f"the padding would be quantized as garbage"
            )
        unpadded_i = moe.intermediate_size_per_partition_unpadded
        if unpadded_i is not None and unpadded_i != intermediate_size:
            raise ValueError(
                f"load-time FP4 does not support a padded intermediate size "
                f"({unpadded_i} -> {intermediate_size}), for the same reason"
            )
        if getattr(layer, "enable_eplb", False):
            raise ValueError(
                "load-time FP4 does not support EPLB (the loader consumes each "
                "shard once and frees the FP8 source, so experts cannot be "
                "re-placed afterwards)"
            )

        load_format = self._current_load_format()
        if load_format is not None and load_format not in _SEQUENTIAL_LOAD_FORMATS:
            raise ValueError(
                f"load-time FP4 requires a sequential weight loader; "
                f"load_format={load_format!r} may interleave shards across "
                f"threads and race on the staging buffers. Supported: "
                f"{sorted(_SEQUENTIAL_LOAD_FORMATS)}"
            )

    @staticmethod
    def _current_load_format() -> str | None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except Exception:  # noqa: BLE001 - no config outside engine init (tests)
            return None
        load_config = getattr(vllm_config, "load_config", None)
        load_format = getattr(load_config, "load_format", None)
        if load_format is None:
            return None
        return str(getattr(load_format, "value", load_format)).lower()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

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
                # Community narrowing, community padding rules, community
                # gate/up offsets -- the only difference is that the
                # destination is a small staging buffer, not the parameter.
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
            # The block scale param keeps its community shape and semantics, so
            # the community loader places it correctly; we only need to know
            # when it has landed.
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
                f"load-time FP4: shard_id={shard_id!r} does not belong to "
                f"{group}_weight"
            )
        if "bias" in weight_name:
            raise ValueError(
                "load-time FP4 does not support expert biases; create_weights "
                "should have rejected this layer"
            )
        if loaded_weight.dim() != 2:
            raise ValueError(
                f"load-time FP4 expects one 2-D expert tensor per call, got "
                f"{loaded_weight.dim()}-D for {weight_name!r}. Fused 3-D expert "
                f"checkpoints take a different community code path that this "
                f"loader does not mirror."
            )
        if getattr(param, "is_transposed", False):
            raise ValueError("load-time FP4 does not support is_transposed params")
        if getattr(param, "use_bitsandbytes_4bit", False):
            raise ValueError("load-time FP4 does not support BitsAndBytes params")

    # ------------------------------------------------------------------
    # Staging / conversion bookkeeping
    # ------------------------------------------------------------------

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
            staging = torch.empty(
                shape, dtype=torch.float8_e4m3fn, device=layer.w13_weight.device
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

    def _convert(self, layer: RoutedExperts, unit: tuple[str, int, str]) -> None:
        group, local_expert, shard_id = unit
        backend = self.converter_backend
        block_n = self.weight_block_size[0]

        if group == _W13:
            size = self._intermediate_size
            offset = 0 if shard_id == "w1" else size
            scale_rows = size // block_n
            src_weight = self._staging[(group, local_expert)].narrow(0, offset, size)
            src_scale = layer.w13_weight_scale_inv.data[local_expert].narrow(
                0, offset // block_n, scale_rows
            )
            out_weight = layer.w13_weight.data[local_expert].narrow(1, offset, size)
            out_scale = layer.w13_weight_scale.data[local_expert].narrow(
                1, offset, size
            )
        else:
            src_weight = self._staging[(group, local_expert)]
            src_scale = layer.w2_weight_scale_inv.data[local_expert]
            out_weight = layer.w2_weight.data[local_expert]
            out_scale = layer.w2_weight_scale.data[local_expert]

        convert_to_ix30_fp4(
            src_weight.unsqueeze(0),
            src_scale.unsqueeze(0),
            source=self.source,
            out_weight=out_weight.unsqueeze(0),
            out_scale=out_scale.unsqueeze(0),
            backend=backend,
        )

        self._weight_ready.discard(unit)
        self._scale_ready.discard(unit)
        self._converted.add(unit)
        self._conversions += 1
        self._release_staging_if_done(group, local_expert)

    def _release_staging_if_done(self, group: str, local_expert: int) -> None:
        """Free the FP8 staging once every shard sharing it has converted."""
        shards = ("w1", "w3") if group == _W13 else ("w2",)
        if all((group, local_expert, s) in self._converted for s in shards):
            self._staging.pop((group, local_expert), None)

    def _pending_units(self) -> list[tuple[str, int, str]]:
        return sorted(self._weight_ready ^ self._scale_ready)

    def _check_pending(self, layer: RoutedExperts) -> None:
        pending = self._pending_units()
        self._pending_peak = max(self._pending_peak, len(pending))
        if len(pending) > self.max_pending_shards:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: {len(pending)} half-loaded "
                f"expert shards exceed max_pending_shards="
                f"{self.max_pending_shards}. Each one pins an FP8 staging "
                f"buffer, so the checkpoint is not arriving weight-then-scale "
                f"as this loader assumes. Stuck units (group, local_expert, "
                f"shard): {pending}"
            )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    # [ILU override] IluFp8MoEMethod.process_weights_after_loading — delta: there
    # is nothing left to convert, so this only verifies that every shard did
    # convert and then drops the block scales. Skipping the parent is mandatory:
    # it would try to read FP8 ``w13_weight`` / ``w2_weight`` that no longer
    # exist in that dtype.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if self._layer is not layer:
            raise RuntimeError(
                "load-time FP4 method was created for a different layer; each "
                "RoutedExperts needs its own instance (it holds per-layer "
                "loading state)"
            )

        expected = {
            (_W13, e, shard)
            for e in range(self._num_local_experts)
            for shard in ("w1", "w3")
        } | {(_W2, e, "w2") for e in range(self._num_local_experts)}
        missing = sorted(expected - self._converted)
        if missing:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: {len(missing)} expert "
                f"shards were never converted to FP4, so the parameter still "
                f"holds uninitialised bytes. Missing (group, local_expert, "
                f"shard): {missing[:16]}"
                f"{' ...' if len(missing) > 16 else ''}"
            )
        if self._staging:
            raise RuntimeError(
                f"{getattr(layer, 'layer_name', '<moe>')}: FP8 staging buffers are "
                f"still held after loading: {sorted(self._staging)}"
            )

        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)

        layer._ilu_mxfp4_ixmm = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug(
            "load-time FP4: %s converted %d expert shards (%d local experts), "
            "peak half-loaded shards %d, backend=%s",
            getattr(layer, "layer_name", "<moe>"),
            self._conversions,
            self._num_local_experts,
            self._pending_peak,
            self.converter_backend or "auto",
        )
