# SPDX-License-Identifier: Apache-2.0
"""Grouped W8A16 BMM linear method for the DeepSeek-V4/V4.1 ``attn.wo_a``.

Corresponds to: no upstream file (Iluvatar-only quantization method).

Registered through ``IluvatarQuantRegistry`` and selected by the
``compressed-tensors`` config group that names ``IluV4W8A16BMMLinearMethod``
(see ``compressed_tensors.py``). Discovery happens by importing this module
from ``_OOT_OP_PACKAGES``; the decorator does the registration.

``wo_a`` is not a normal linear: the attention wrapper consumes it as a
grouped BMM, so this method only lays out and Marlin-repacks the weight and
leaves ``apply`` unreachable.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry


@IluvatarQuantRegistry.register_method
class IluV4W8A16BMMLinearMethod(LinearMethodBase):
    """W8A16 grouped-BMM method for DeepSeek-V4 ``attn.wo_a``."""

    def __init__(self, parent_config, ilu_params: dict[str, Any]) -> None:
        self.parent_config = parent_config
        try:
            self.gemm_format = ilu_params.get("bmm_w8a16_gemm_format", "")
            assert self.gemm_format != ""
        except KeyError as exc:
            raise ValueError(
                "IluV4W8A16BMMLinearMethod requires "
                "`bmm_w8a16_gemm_format` in its config group"
            ) from exc

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.parameter import (
            ChannelQuantScaleParameter,
            ModelWeightParameter,
        )

        out_total = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(out_total, input_size_per_partition, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((out_total, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Reshape and Marlin-repack ``wo_a.weight`` for grouped W8A16."""
        import ixformer.inference.functions as ixf_ops

        bmm_geom = getattr(layer, "bmm_geom", None)
        assert bmm_geom is not None, (
            "IluV4W8A16BMMLinearMethod requires layer.bmm_geom to be set "
            "by the attention wrapper before process_weights_after_loading."
        )
        n_local_groups: int = bmm_geom.n_local_groups
        o_lora_rank: int = bmm_geom.o_lora_rank
        k_per_group: int = bmm_geom.heads_per_group * bmm_geom.head_dim

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        assert weight.shape == (n_local_groups * o_lora_rank, k_per_group), (
            f"wo_a weight shape {tuple(weight.shape)} does not match "
            f"({n_local_groups} * {o_lora_rank}, {k_per_group}) implied by "
            f"bmm_geom={bmm_geom!r}"
        )
        assert weight_scale.shape == (n_local_groups * o_lora_rank, 1), (
            f"wo_a weight_scale shape {tuple(weight_scale.shape)} != "
            f"({n_local_groups * o_lora_rank}, 1)"
        )

        w_grouped = (
            weight.view(n_local_groups, o_lora_rank, k_per_group)
            .permute(0, 2, 1)
            .contiguous()
        )
        s_grouped = weight_scale.view(n_local_groups, o_lora_rank, 1).contiguous()

        layer.W_repacked = ixf_ops.marlin_w8_weight_repack(
            weights=w_grouped,
            weight_format="int8",
            reformat=self.gemm_format,
        )
        layer.W_scale_repacked = s_grouped
        layer.W_format = self.gemm_format

        delattr(layer, "weight")
        delattr(layer, "weight_scale")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "IluV4W8A16BMMLinearMethod.apply must not be called; "
            "wo_a is consumed by IluDeepseekV4MultiHeadLatentAttentionWrapper "
            "via marlin_w8a16(W_repacked, W_scale_repacked)."
        )
