# SPDX-License-Identifier: Apache-2.0
"""Configured VL resource limits must also constrain profiling dummy inputs."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm_iluvatar.models.hy_v4.mm_preprocess import (
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
    HYV4VLProcessingInfo,
)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_hy4_mm_only_uses_full_hf_processor(modality):
    """The 0.31 multimodal-only entry must retain video metadata fields."""
    ctx = Mock()
    hf = ctx.get_hf_processor.return_value
    values = {f"{modality}_grid_thw": torch.tensor([[1, 2, 2]])}
    if modality == "video":
        values["video_timestamps"] = torch.tensor([0.0])
        data = (np.zeros((1, 32, 32, 3), dtype=np.uint8), {"fps": 1.0})
    else:
        data = Image.new("RGB", (32, 32))
    ctx.call_hf_processor.return_value = BatchFeature(values)
    info = HYV4VLProcessingInfo(ctx)
    processor = HYV4VLMultiModalProcessor(info, HYV4VLDummyInputsBuilder(info))
    items = processor.data_parser.parse_mm_data({modality: [data]})
    options = {"max_pixels": 1024}
    result = processor._apply_hf_processor_main(items, options)
    args = ctx.call_hf_processor.call_args.args
    assert args[0] is hf
    assert ("images" if modality == "image" else "videos") in args[1]
    assert args[2] == options
    assert set(values) <= result.keys()


def test_hy4_dummy_budgets_respect_modality_processor_overrides():
    config = SimpleNamespace(
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        use_image_newline_token=True,
        cat_extra_token=False,
    )
    image = SimpleNamespace(
        patch_size=16,
        merge_size=2,
        size={"shortest_edge": 4096, "longest_edge": 4194304},
    )
    video = SimpleNamespace(
        max_frames=100,
        video_max_image_size=1024,
        timestamp_token_per_frame=4,
        timestamp_token_reserve=2000,
        size={"shortest_edge": 4096, "longest_edge": 33554432},
    )
    options = {
        "image": {"max_pixels": 262144},
        "video": {
            "max_pixels": 262144,
            "max_frames": 4,
            "video_max_image_size": 256,
            "timestamp_token_reserve": 0,
        },
    }
    ctx = Mock()
    ctx.get_hf_processor.return_value = SimpleNamespace(
        image_processor=image, video_processor=video
    )
    ctx.get_hf_config.return_value = SimpleNamespace(vision_config=config)
    ctx.get_merged_mm_kwargs.side_effect = lambda kwargs, modality: options[modality]
    info = HYV4VLProcessingInfo(ctx)
    assert info.get_max_frames_per_video() == 4
    size = info.get_image_size_with_most_features()
    assert size.height * size.width <= 262144
    frames, height, width = info.get_dummy_video_shape(8192, {"video": 1})
    assert 1 <= frames <= 4 and height <= 256 and width <= 256
    budgets = info.get_mm_max_tokens_per_item(8192, {"image": 1, "video": 1})
    assert budgets["video"] == 2 * (262144 // 32**2 + 4)
    assert budgets["image"] < 1024
