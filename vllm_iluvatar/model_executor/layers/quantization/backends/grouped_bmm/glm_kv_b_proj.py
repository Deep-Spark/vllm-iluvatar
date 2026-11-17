# SPDX-License-Identifier: Apache-2.0
"""Grouped W8A16 method for GLM-5.2 ``kv_b_proj``.

``kv_b_proj`` is never run as a flat linear: the attention wrapper sets
``layer.bmm_geom`` before weights are processed and consumes the split
``W_UK`` / ``W_UV`` Marlin tensors left on the layer.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry


@IluvatarQuantRegistry.register_method
class IluGlmKVBProjW8A16Method(LinearMethodBase):
    """W8A16 grouped-BMM method for GLM-5.2 ``kv_b_proj``."""

    def __init__(self, parent_config, ilu_params: dict[str, Any]) -> None:
        self.parent_config = parent_config

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
        """Split ``kv_b_proj`` into W_UK/W_UV and Marlin-repack each."""
        import ixformer.inference.functions as ixf_ops

        geom = getattr(layer, "bmm_geom", None)
        assert geom is not None, (
            "IluGlmKVBProjW8A16Method requires layer.bmm_geom to be set by "
            "the attention wrapper before process_weights_after_loading."
        )
        num_heads: int = geom.num_heads
        nope: int = geom.qk_nope_head_dim
        v: int = geom.v_head_dim
        kv_lora: int = geom.kv_lora_rank

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        assert weight.shape == (num_heads * (nope + v), kv_lora), (
            f"kv_b_proj weight shape {tuple(weight.shape)} does not match "
            f"({num_heads} * ({nope} + {v}), {kv_lora}) implied by "
            f"bmm_geom={geom!r}"
        )
        assert weight_scale.shape == (num_heads * (nope + v), 1), (
            f"kv_b_proj weight_scale shape {tuple(weight_scale.shape)} != "
            f"({num_heads * (nope + v)}, 1)"
        )

        w = weight.view(num_heads, nope + v, kv_lora)
        s = weight_scale.view(num_heads, nope + v, 1)
        w_uk, w_uv = w.split([nope, v], dim=1)
        w_uk_s, w_uv_s = s.split([nope, v], dim=1)

        layer.W_UK = ixf_ops.marlin_w8_weight_repack(
            weights=w_uk.contiguous(),
            weight_format="int8",
            reformat="k16n16_grouped_n",
        )
        layer.W_UK_S = w_uk_s.contiguous()
        layer.W_UV = ixf_ops.marlin_w8_weight_repack(
            weights=w_uv.permute(0, 2, 1).contiguous(),
            weight_format="int8",
            reformat="k16n16",
        )
        layer.W_UV_S = w_uv_s.contiguous()

        delattr(layer, "weight")
        delattr(layer, "weight_scale")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "IluGlmKVBProjW8A16Method.apply must not be called; kv_b_proj is "
            "consumed by IluGlmMoeDsaSparseMLA via "
            "marlin_w8a16(W_UK/W_UV, W_UK_S/W_UV_S)."
        )


__all__ = ["IluGlmKVBProjW8A16Method"]
