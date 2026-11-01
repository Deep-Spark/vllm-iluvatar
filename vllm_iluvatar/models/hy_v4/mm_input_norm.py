# SPDX-License-Identifier: Apache-2.0
"""Device-side rescale and normalise for the HY V4 VL vision tower.

Corresponds to: vllm/model_executor/layers/fusion/mm_input_norm.py (upstream)

The Iluvatar vLLM build has no ``mm_device_do_normalize`` switch and no
``layers/fusion`` package, so the processor cannot hand raw pixels to the
upstream fused kernel. The processor leaves pixels in the 0–255 range, and
this module applies the same affine the HF processor used to run on CPU::

    output = (pixel * rescale_factor - image_mean) / image_std

which is ``pixel * weight + bias`` with ``weight = rescale_factor / std`` and
``bias = -image_mean / std``. Arithmetic is fp32 on the pixel tensor's device;
the result is cast to the vision dtype. The Triton kernel from day0 is not
used: this platform dispatches custom ops through the out-of-tree path, and
the affine is a few elementwise operations.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn


class IdentityInputNorm(nn.Module):
    """Cast pixels to the visual dtype without rescale or normalise."""

    @property
    def input_dtype(self) -> torch.dtype | None:
        return None

    def forward(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        return pixel_values.to(visual_dtype, copy=False)


class GpuInputNorm(nn.Module):
    """Per-channel rescale and normalise on the pixel tensor's device.

    ``pixel_values`` is ``(num_patches, channel * patch_size ** 2)`` in
    channel-major patch layout, values still in the 0–255 range (uint8 or
    float). Channel ``c`` is scaled by ``weight[c]`` and shifted by ``bias[c]``.
    """

    def __init__(
        self,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
    ):
        super().__init__()
        if len(image_mean) != len(image_std) or not image_mean:
            raise ValueError(
                "image_mean and image_std must be non-empty and the same length, "
                f"got {len(image_mean)} and {len(image_std)}"
            )
        if rescale_factor == 0.0 or any(s == 0.0 for s in image_std):
            raise ValueError("rescale_factor and image_std entries must be non-zero")
        mean = torch.tensor(image_mean, dtype=torch.float32)
        std = torch.tensor(image_std, dtype=torch.float32)
        self.channel = len(image_mean)
        self.register_buffer("weight", rescale_factor / std)
        self.register_buffer("bias", -mean / std)

    @property
    def input_dtype(self) -> torch.dtype | None:
        return torch.uint8

    def forward(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        if pixel_values.ndim != 2:
            raise ValueError(
                "pixel_values must be (num_patches, patch_dim), "
                f"got shape {tuple(pixel_values.shape)}"
            )
        patches, size = pixel_values.shape
        if size % self.channel != 0:
            raise ValueError(
                f"patch dim {size} is not divisible by channel {self.channel}"
            )
        spatial = size // self.channel
        values = pixel_values.reshape(patches, self.channel, spatial).to(torch.float32)
        weight = self.weight.to(device=values.device, dtype=torch.float32).view(
            1, self.channel, 1
        )
        bias = self.bias.to(device=values.device, dtype=torch.float32).view(
            1, self.channel, 1
        )
        return (values * weight + bias).reshape(patches, size).to(visual_dtype)


def _image_processor_config(model_config) -> dict:
    """Processor fields that define the affine, from vLLM or the checkpoint."""
    config = getattr(model_config, "hf_image_processor_config", None) or {}
    config = config.get("image_processor", config)
    if "image_mean" in config and "image_std" in config:
        return config
    model = getattr(model_config, "model", None)
    if not model:
        return config
    path = os.path.join(os.fspath(model), "preprocessor_config.json")
    if not os.path.isfile(path):
        return config
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    loaded.update({key: value for key, value in config.items() if value is not None})
    return loaded


def build_mm_input_norm(model_config) -> nn.Module:
    """Build the vision-tower input norm for HY V4 VL.

    Always the GPU affine. The paired processor leaves pixels unnormalised,
    and this vLLM build cannot request that through ``mm_device_do_normalize``.
    """
    config = _image_processor_config(model_config)
    image_mean = config.get("image_mean", [0.5, 0.5, 0.5])
    image_std = config.get("image_std", [0.5, 0.5, 0.5])
    rescale_factor = config.get("rescale_factor", 1 / 255)
    return GpuInputNorm(
        image_mean=[float(v) for v in image_mean],
        image_std=[float(v) for v in image_std],
        rescale_factor=float(rescale_factor),
    )


__all__ = ["IdentityInputNorm", "GpuInputNorm", "build_mm_input_norm"]
