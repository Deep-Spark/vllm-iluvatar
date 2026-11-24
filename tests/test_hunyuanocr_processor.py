# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU checks for the HunyuanOCR processor default and its actual patch layout."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import vllm.envs  # noqa: F401 - initialize vLLM before its OOT platform module
from PIL import Image
from transformers import AutoImageProcessor

from vllm_iluvatar.platforms.ilu_platform import _configure_model_processor


def model_config(model_type="hunyuan_vl", kwargs=None):
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type),
        multimodal_config=SimpleNamespace(mm_processor_kwargs=kwargs),
    )


@pytest.mark.parametrize("kwargs", [None, {}, {"use_fast": None}])
def test_default_processor_emits_raster_patches(tmp_path, kwargs):
    pytest.importorskip(
        "transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl"
    )
    config = model_config(kwargs=kwargs)
    _configure_model_processor(config)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "HunYuanVLImageProcessor",
                "patch_size": 2,
                "merge_size": 2,
                "temporal_patch_size": 1,
                "do_resize": False,
                "do_rescale": False,
                "do_normalize": False,
            }
        )
    )
    processor = AutoImageProcessor.from_pretrained(
        tmp_path, **config.multimodal_config.mm_processor_kwargs
    )
    # Multiple merge blocks in both axes distinguish raster order from 2x2 blocks.
    pixels = np.arange(8 * 16 * 3, dtype=np.int64).reshape(8, 16, 3) % 251
    pixels = pixels.astype(np.uint8)
    result = processor(images=Image.fromarray(pixels), return_tensors="np")
    expected = np.stack(
        [
            pixels[y : y + 2, x : x + 2].transpose(2, 0, 1).reshape(-1)
            for y in range(0, 8, 2)
            for x in range(0, 16, 2)
        ]
    )
    np.testing.assert_array_equal(result["pixel_values"], expected)
    np.testing.assert_array_equal(result["image_grid_thw"], [[1, 4, 8]])


def test_processor_default_preserves_other_options_and_does_not_mutate_input():
    original = {"min_image_size": 512}
    config = model_config(kwargs=original)
    _configure_model_processor(config)
    assert config.multimodal_config.mm_processor_kwargs == {
        "min_image_size": 512,
        "use_fast": False,
    }
    assert original == {"min_image_size": 512}


@pytest.mark.parametrize("use_fast", [True, False])
def test_explicit_processor_choice_is_preserved(use_fast):
    config = model_config(kwargs={"use_fast": use_fast})
    _configure_model_processor(config)
    assert config.multimodal_config.mm_processor_kwargs["use_fast"] is use_fast


def test_other_models_keep_their_processor_default():
    config = model_config(model_type="qwen2_vl")
    _configure_model_processor(config)
    assert config.multimodal_config.mm_processor_kwargs is None
    _configure_model_processor(None)
