# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Corresponds to: vllm/model_executor/layers/fused_moe/routed_experts.py.

Iluvatar OOT replacement for routed expert weight loading.

Target     : vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts
Baseline   : main_88e5e2c57be8 ``RoutedExperts.weight_loader`` and
             ``RoutedExperts.load_weights``
Why        : preserve the fork's MoE loading deltas without editing community
             vLLM source.
How        : @PluggableLayer.register_oot subclass. Copied methods are pinned
             to the community baseline; Iluvatar-only deltas are marked inline.
Scope      : imported during ``apply_oot_registrations()``.
Fingerprint: ``RoutedExperts.weight_loader`` and ``RoutedExperts.load_weights``
             community sources are checked at import time; gated by
             VLLM_ILU_PATCH_CHECK_FINGERPRINT.
Remove     : when upstream supports ``IluAutoGPTQMoEMethod`` intermediate-size
             parameters and parameter-provided MoE shard dimensions natively.
"""

import inspect
from collections.abc import Iterable
from typing import Literal, overload

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    logger,
)

from vllm_iluvatar.ilu_patch import general_patch as p

_ROUTED_EXPERTS_WEIGHT_LOADER_BASELINE = """\
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        quant_config_name = self.quant_config and self.quant_config.get_name()
        if quant_config_name == "gpt_oss_mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        use_global_sf = (
            getattr(self.quant_method, "use_global_sf", False)
            and "input_scale" in weight_name
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if quant_method_name in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
            "CompressedTensorsWNA16RDNA3MoEMethod",
            "CompressedTensorsW4A16FlydslMoEMethod",
        ):
            if is_transposed:
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                # BnB params are stored as flat packed tensors (e.g.
                # (packed_size, 1)), not in the logical weight layout.
                # Narrowing packed data for hidden-dim padding is not
                # meaningful, so require an exact shape match.
                if expert_data.shape != loaded_weight.shape:
                    raise ValueError(
                        "BitsAndBytes quantization with padded hidden_size "
                        "(e.g., from DeepEP) is not supported. "
                        f"Parameter shape {tuple(expert_data.shape)} != "
                        f"checkpoint shape {tuple(loaded_weight.shape)}"
                    )
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BnB stores weights as flat packed tensors.  _load_w13 is
                # still used to split the w1/w3 portions along shard_dim.
                # _narrow_expert_data_for_padding will be a no-op since
                # packed sizes should already match; if DeepEP padding
                # causes a mismatch the copy_() will fail with a clear
                # shape error.
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            # ModelOpt NVFP4 stores w13 input scales as two logical shards.
            # The generic assignment below would broadcast w1/w3 into the
            # whole expert row, so the second shard would overwrite the first.
            if (
                "ModelOpt" in quant_method_name
                and param.data.ndim == 2
                and shard_id in ("w1", "w3")
            ):
                scale_expert_id = global_expert_id if use_global_sf else expert_id
                scale_shard_id = 0 if shard_id == "w1" else 1
                param.data[scale_expert_id][scale_shard_id] = self._to_scalar(
                    loaded_weight
                )
                return True if return_success else None

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()
            quant_method = getattr(param, "quant_method", None)

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            # NOTE: ModelOpt MXFP8 MoE uses block scales in weight_scale
            # tensors (quant_method=BLOCK), so those must not be treated
            # as per-tensor scalars here.
            is_block_weight_scale = (
                "weight_scale" in weight_name
                and quant_method == FusedMoeWeightScaleSupported.BLOCK.value
            )
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            is_per_tensor = is_per_tensor and not is_block_weight_scale
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.moe_config.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.moe_config.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None
"""

_ROUTED_EXPERTS_LOAD_WEIGHTS_BASELINE = """\
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        expert_mapping = self.get_expert_mapping(include_fused=True)
        unpadded_hidden = self.moe_config.hidden_dim_unpadded
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            # Fused expert weights can be identified by their 3D tensors
            is_fused = loaded_weight.dim() == 3
            matched = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    if matched and is_fused:
                        break
                    continue
                matched = True
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                if is_fused:
                    # w1 and w3 share one fused tensor; use a local copy so the
                    # transpose below doesn't mutate loaded_weight across
                    # iterations (else w3 is transposed twice and wrongly chunked)
                    fused_weight = loaded_weight
                    if shard_id in {"w1", "w3"}:
                        if fused_weight.shape[-1] != unpadded_hidden:
                            # [..., hidden, intermediate] -> [..., intermediate, hidden]
                            fused_weight = fused_weight.transpose(-1, -2)
                        # Repurpose expert_id for deconcatenating w1 and w3
                        experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                    else:
                        if fused_weight.shape[-2] != unpadded_hidden:
                            # [..., intermediate, hidden] -> [..., hidden, intermediate]
                            fused_weight = fused_weight.transpose(-1, -2)
                        experts_shard = fused_weight
                    start = 0
                else:
                    # loaded_weight is a single expert weight, so we add a dummy expert
                    # dimension to unify the loading logic with the fused case
                    experts_shard = loaded_weight.unsqueeze(0)
                    start = expert_id

                # Unified loading logic for fused and non-fused experts
                loaded_experts = experts_shard.unbind()
                for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                    success = param.weight_loader(
                        param=param,
                        loaded_weight=loaded_expert,
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        logger.debug(
                            "Loaded expert %d of shard %s into %s for layer %s",
                            expert_id,
                            shard_id,
                            param_name,
                            self.layer_name,
                        )
                        yield param_name
"""


def _check_routed_experts_fingerprints() -> None:
    p.check_fingerprint(
        "RoutedExperts.weight_loader",
        inspect.getsource(RoutedExperts.weight_loader),
        _ROUTED_EXPERTS_WEIGHT_LOADER_BASELINE,
        resync_hint=(
            "Resync IluRoutedExperts.weight_loader with the community "
            "baseline or replace the copy with an upstream shard-dim hook."
        ),
    )
    p.check_fingerprint(
        "RoutedExperts.load_weights",
        inspect.getsource(RoutedExperts.load_weights),
        _ROUTED_EXPERTS_LOAD_WEIGHTS_BASELINE,
        resync_hint=(
            "Resync IluRoutedExperts.load_weights with the community "
            "baseline or move the fused-scale normalization to an upstream hook."
        ),
    )


_check_routed_experts_fingerprints()


@RoutedExperts.register_oot()
class IluRoutedExperts(RoutedExperts):
    """RoutedExperts with the fork's minimal MoE loading deltas."""

    def _needs_intermediate_size_param(self, quant_method) -> bool:
        # fork routed_experts.py: IluAutoGPTQMoEMethod needs the same full
        # intermediate-size metadata as the upstream packed MoE methods.
        return (
            quant_method.__class__.__name__ == "IluAutoGPTQMoEMethod"
            or super()._needs_intermediate_size_param(quant_method)
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        expert_mapping = self.get_expert_mapping(include_fused=True)
        unpadded_hidden = self.moe_config.hidden_dim_unpadded
        fused_w13_intermediate_dim = None
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            # Fused expert weights can be identified by their 3D tensors
            is_fused = loaded_weight.dim() == 3
            matched = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    if matched and is_fused:
                        break
                    continue
                matched = True
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                if is_fused:
                    # w1 and w3 share one fused tensor; use a local copy so the
                    # transpose below doesn't mutate loaded_weight across
                    # iterations (else w3 is transposed twice and wrongly chunked)
                    fused_weight = loaded_weight
                    # >>> ILU MODIFICATION >>>
                    # record fused w13 full dim
                    if "weight_scale" not in param_name:
                        if "w13_weight" in param_name:
                            fused_w13_intermediate_dim = (
                                fused_weight.shape[-2]
                                if fused_weight.shape[-1] == unpadded_hidden
                                else fused_weight.shape[-1]
                            )
                    # <<< ILU MODIFICATION <<<
                    if shard_id in {"w1", "w3"}:
                        if fused_weight.shape[-1] != unpadded_hidden:
                            # [..., hidden, intermediate] -> [..., intermediate, hidden]
                            fused_weight = fused_weight.transpose(-1, -2)
                        # >>> ILU MODIFICATION >>>
                        # trim fused w13 scale dim
                        if (
                            "w13_weight_scale" in param_name
                            and fused_weight.shape[-2] == 2
                            and fused_w13_intermediate_dim is not None
                            and fused_weight.shape[-1] == fused_w13_intermediate_dim
                            and fused_weight.shape[-1] % 2 == 0
                        ):
                            fused_weight = fused_weight.narrow(
                                -1, 0, fused_weight.shape[-1] // 2
                            )
                        # <<< ILU MODIFICATION <<<
                        # Repurpose expert_id for deconcatenating w1 and w3
                        experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                    else:
                        # >>> ILU MODIFICATION >>>
                        # handle fused w2 scale layout
                        if fused_weight.shape[-2] != unpadded_hidden or (
                            "w2_weight_scale" in param_name
                            and fused_weight.shape[-1] == 1
                        ):
                            # <<< ILU MODIFICATION <<<
                            # [..., intermediate, hidden] -> [..., hidden, intermediate]
                            fused_weight = fused_weight.transpose(-1, -2)
                        experts_shard = fused_weight
                    start = 0
                else:
                    # loaded_weight is a single expert weight, so we add a dummy expert
                    # dimension to unify the loading logic with the fused case
                    experts_shard = loaded_weight.unsqueeze(0)
                    start = expert_id

                # Unified loading logic for fused and non-fused experts
                loaded_experts = experts_shard.unbind()
                for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                    success = param.weight_loader(
                        param=param,
                        loaded_weight=loaded_expert,
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        logger.debug(
                            "Loaded expert %d of shard %s into %s for layer %s",
                            expert_id,
                            shard_id,
                            param_name,
                            self.layer_name,
                        )
                        yield param_name

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        quant_config_name = self.quant_config and self.quant_config.get_name()
        if quant_config_name == "gpt_oss_mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        use_global_sf = (
            getattr(self.quant_method, "use_global_sf", False)
            and "input_scale" in weight_name
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if quant_method_name in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
            "CompressedTensorsWNA16RDNA3MoEMethod",
            "CompressedTensorsW4A16FlydslMoEMethod",
        ):
            if is_transposed:
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                # BnB params are stored as flat packed tensors (e.g.
                # (packed_size, 1)), not in the logical weight layout.
                # Narrowing packed data for hidden-dim padding is not
                # meaningful, so require an exact shape match.
                if expert_data.shape != loaded_weight.shape:
                    raise ValueError(
                        "BitsAndBytes quantization with padded hidden_size "
                        "(e.g., from DeepEP) is not supported. "
                        f"Parameter shape {tuple(expert_data.shape)} != "
                        f"checkpoint shape {tuple(loaded_weight.shape)}"
                    )
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BnB stores weights as flat packed tensors.  _load_w13 is
                # still used to split the w1/w3 portions along shard_dim.
                # _narrow_expert_data_for_padding will be a no-op since
                # packed sizes should already match; if DeepEP padding
                # causes a mismatch the copy_() will fail with a clear
                # shape error.
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)
        # >>> ILU MODIFICATION >>>
        # fork routed_experts.py: Iluvatar quantized MoE parameters may carry
        # their physical shard dimension because packed CoreX layouts do not
        # always match upstream's logical w1/w2/w3 shard table.
        shard_dim_force = getattr(param, "shard_dim", None)
        shard_dim = shard_dim_force if shard_dim_force is not None else shard_dim
        # <<< ILU MODIFICATION <<<

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            # ModelOpt NVFP4 stores w13 input scales as two logical shards.
            # The generic assignment below would broadcast w1/w3 into the
            # whole expert row, so the second shard would overwrite the first.
            if (
                "ModelOpt" in quant_method_name
                and param.data.ndim == 2
                and shard_id in ("w1", "w3")
            ):
                scale_expert_id = global_expert_id if use_global_sf else expert_id
                scale_shard_id = 0 if shard_id == "w1" else 1
                param.data[scale_expert_id][scale_shard_id] = self._to_scalar(
                    loaded_weight
                )
                return True if return_success else None

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()
            quant_method = getattr(param, "quant_method", None)

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            # NOTE: ModelOpt MXFP8 MoE uses block scales in weight_scale
            # tensors (quant_method=BLOCK), so those must not be treated
            # as per-tensor scalars here.
            is_block_weight_scale = (
                "weight_scale" in weight_name
                and quant_method == FusedMoeWeightScaleSupported.BLOCK.value
            )
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            is_per_tensor = is_per_tensor and not is_block_weight_scale
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.moe_config.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.moe_config.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None


# OOT method copies do not inherit function-object attributes from upstream.
# Keep the MoE loader marker so vLLM passes weight_name/shard_id/expert_id.
IluRoutedExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
