# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GPT-OSS loader override.

Corresponds to: vllm/model_executor/models/gpt_oss.py
Target   : vllm.model_executor.models.gpt_oss.GptOssForCausalLM
           (arch "GptOssForCausalLM")
Baseline : main_dcfebf93f4e
Why      : Iluvatar W4A8/W8A8 GPT-OSS checkpoints use padded routed-expert
           tensors and plugin env-controlled layouts that community GPT-OSS
           loaders do not understand.
How      : Register an OOT subclass, keep community construction unchanged, then
           swap the inner ``GptOssModel`` instance to a subclass that overrides
           only ``_load_weights_other``. The padding-copy helper stays local to
           this plugin instead of patching community ``weight_utils`` or
           ``parameter.py``.
Scope    : GPT-OSS non-MXFP4/non-Quark expert weight loading on Iluvatar.
Remove   : when community GPT-OSS exposes a model-specific padded expert loader
           hook or supports the Iluvatar layouts natively.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import vllm.envs as envs
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM as _GptOssForCausalLM,
)
from vllm.model_executor.models.gpt_oss import GptOssModel as _GptOssModel
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.utils.math_utils import cdiv


def padding_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Copy overlapping 2D/3D slices for padded expert tensors."""
    if param.shape == loaded_weight.shape:
        param.data.copy_(loaded_weight)
        return

    if param.ndim != loaded_weight.ndim:
        raise ValueError(
            f"Cannot load weight with different ndim: param.ndim={param.ndim}, "
            f"loaded.ndim={loaded_weight.ndim}, param.shape={tuple(param.shape)}, "
            f"loaded.shape={tuple(loaded_weight.shape)}"
        )
    if param.ndim not in (2, 3):
        raise ValueError(
            f"padding_weight_loader only supports 2D/3D tensors, got {param.ndim}D. "
            f"param.shape={tuple(param.shape)}, loaded.shape={tuple(loaded_weight.shape)}"
        )
    if param.shape[0] != loaded_weight.shape[0]:
        raise AssertionError(
            "Mismatch in number of experts: "
            f"param={param.shape[0]}, loaded={loaded_weight.shape[0]}"
        )

    copy_d1 = min(param.shape[1], loaded_weight.shape[1])
    if param.ndim == 2:
        param.data[:, :copy_d1].copy_(loaded_weight[:, :copy_d1])
        return

    copy_d2 = min(param.shape[2], loaded_weight.shape[2])
    param.data[:, :copy_d1, :copy_d2].copy_(loaded_weight[:, :copy_d1, :copy_d2])


def _route_expert_name(name: str) -> str:
    return name.replace(".mlp.experts.", ".mlp.experts.routed_experts.", 1)


def _weight_slice_for_expert(
    *,
    use_ep: bool,
    ep_rank_start: int,
    ep_rank_end: int,
    tp_rank_start: int,
    tp_rank_end: int,
    pack_factor: int = 1,
    is_w13: bool,
) -> tuple[slice | type(Ellipsis), ...]:
    if use_ep:
        return (slice(ep_rank_start, ep_rank_end), ...)
    if is_w13:
        return (slice(None), slice(None), slice(2 * tp_rank_start, 2 * tp_rank_end))
    return (
        slice(None),
        slice(tp_rank_start // pack_factor, tp_rank_end // pack_factor),
        slice(None),
    )


class GptOssModel(_GptOssModel):
    """GPT-OSS model with Iluvatar padded routed-expert weight loading."""

    def _load_weights_other(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        def handle_weight(
            weight_name: str,
            weight: torch.Tensor,
            param_name: str,
            *,
            permute_dims: tuple[int, ...] | None = None,
            slice_dims: tuple[slice | type(Ellipsis), ...] | None = None,
            contiguous: bool = True,
        ) -> None:
            param = params_dict[param_name]
            if slice_dims is not None:
                weight = weight[slice_dims]
            if permute_dims is not None:
                weight = weight.permute(*permute_dims)
            if contiguous:
                weight = weight.contiguous()
            padding_weight_loader(param, weight)
            loaded_params.add(param_name)

        use_ep = self.parallel_config.enable_expert_parallel
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=get_tensor_model_parallel_world_size(),
            dp_size=get_dp_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
            pcp_size=get_pcp_group().world_size,
            pcp_rank=get_pcp_group().rank_in_group,
        )

        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        pack_factor = 2 if envs.VLLM_W8A8_MOE_USE_W4A8 else 1
        w4a8_enabled = envs.VLLM_W8A8_MOE_USE_W4A8
        gemm_format = envs.VLLM_W8A8_FORMAT
        expert_permute_dims = None if gemm_format == "NN" else (0, 2, 1)
        scale_permute_dims = None if w4a8_enabled else (0, 2, 1)

        for name, weight in weights:
            if is_pp_missing_parameter(name, self):
                continue

            if (
                ".experts.w13_weight" in name
                and "scale" not in name
                and "bias" not in name
            ):
                handle_weight(
                    name,
                    weight,
                    _route_expert_name(name),
                    permute_dims=expert_permute_dims,
                    slice_dims=_weight_slice_for_expert(
                        use_ep=use_ep,
                        ep_rank_start=ep_rank_start,
                        ep_rank_end=ep_rank_end,
                        tp_rank_start=tp_rank_start,
                        tp_rank_end=tp_rank_end,
                        is_w13=True,
                    ),
                )
            elif (
                ".experts.w2_weight" in name
                and "scale" not in name
                and "bias" not in name
            ):
                handle_weight(
                    name,
                    weight,
                    _route_expert_name(name),
                    permute_dims=expert_permute_dims,
                    slice_dims=_weight_slice_for_expert(
                        use_ep=use_ep,
                        ep_rank_start=ep_rank_start,
                        ep_rank_end=ep_rank_end,
                        tp_rank_start=tp_rank_start,
                        tp_rank_end=tp_rank_end,
                        pack_factor=pack_factor,
                        is_w13=False,
                    ),
                )
            elif ".experts.gate_up_proj_scale" in name:
                new_name = _route_expert_name(
                    name.replace("gate_up_proj_scale", "w13_weight_scale")
                )
                handle_weight(
                    name,
                    weight,
                    new_name,
                    permute_dims=scale_permute_dims,
                    slice_dims=_weight_slice_for_expert(
                        use_ep=use_ep,
                        ep_rank_start=ep_rank_start,
                        ep_rank_end=ep_rank_end,
                        tp_rank_start=tp_rank_start,
                        tp_rank_end=tp_rank_end,
                        is_w13=True,
                    ),
                    contiguous=w4a8_enabled,
                )
            elif ".experts.down_proj_scale" in name:
                new_name = _route_expert_name(
                    name.replace("down_proj_scale", "w2_weight_scale")
                )
                slice_dims = (
                    (slice(ep_rank_start, ep_rank_end), ...) if use_ep else None
                )
                handle_weight(
                    name,
                    weight,
                    new_name,
                    permute_dims=scale_permute_dims,
                    slice_dims=slice_dims,
                    contiguous=w4a8_enabled,
                )
            elif ".experts.w13_bias" in name:
                slice_dims = (
                    (slice(ep_rank_start, ep_rank_end), ...)
                    if use_ep
                    else (slice(None), slice(2 * tp_rank_start, 2 * tp_rank_end))
                )
                handle_weight(
                    name,
                    weight,
                    _route_expert_name(name),
                    slice_dims=slice_dims,
                    contiguous=False,
                )
            elif ".experts.w2_bias" in name:
                routed_name = _route_expert_name(name)
                param = params_dict[routed_name]
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                elif tp_rank != 0:
                    weight.zero_()
                param.data.copy_(weight)
                loaded_params.add(routed_name)
            elif "sinks" in name:
                name = name.replace("self_attn", "attn")
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                shard_id = "q" if "q_proj" in name else "k" if "k_proj" in name else "v"
                name = name.replace("self_attn", "attn")
                param_name = name.replace(f"{shard_id}_proj", "qkv_proj")
                param = params_dict[param_name]
                param.weight_loader(param, weight, loaded_shard_id=shard_id)
                loaded_params.add(param_name)
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(name)

        return loaded_params


class GptOssForCausalLM(_GptOssForCausalLM):
    """GPT-OSS causal LM wired for Iluvatar expert loading."""

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model.__class__ = GptOssModel


def register_model() -> None:
    """Model registration hook: point GPT-OSS arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "vllm_iluvatar.model_executor.models.gpt_oss:GptOssForCausalLM",
    )
