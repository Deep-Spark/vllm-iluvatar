# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.
"""HY V4 VL multimodal processing glue, ported from the day0 reference.

Corresponds to: vllm/models/hy_v4/nvidia/mm_preprocess.py (upstream #54160)

Target   : ``HYV4VLProcessingInfo`` / ``HYV4VLMultiModalProcessor`` /
           ``HYV4VLDummyInputsBuilder`` -- the ``MULTIMODAL_REGISTRY`` wiring the
           VL model registers with.
Why      : the Iluvatar vLLM baseline has no ``hy_v4`` multimodal processor.
How      : faithful copy of the upstream module, with two baseline deltas:

           * config/processor imports point at the plugin-local ``.config`` and
             ``.processor`` modules.
           * profiling options use the baseline context's
             ``get_merged_mm_kwargs`` with an explicit modality. The baseline
             processor already calls the full HF processor on multimodal-only
             inputs, so the day0 four-argument forwarding override is omitted.
Remove   : when the Iluvatar baseline carries the upstream processor glue.
"""

import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)

from .config import HYV4VLConfig
from .processor import (
    HYV4VLProcessor,
    adaptive_resize,
    smart_resize,
    vision_token_count,
)


class HYV4VLDataParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed image embeddings")
        return super()._parse_image_data(data)

    def _parse_video_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed video embeddings")
        return super()._parse_video_data(data)


class HYV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> HYV4VLConfig:
        return self.ctx.get_hf_config(HYV4VLConfig)

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self.ctx.get_hf_processor(HYV4VLProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_max_frames_per_video(self) -> int:
        processor = self.get_hf_processor().video_processor
        options = self.ctx.get_merged_mm_kwargs({}, modality="video")
        return int(options.get("max_frames", processor.max_frames))

    def get_data_parser(self):
        return HYV4VLDataParser(video_needs_metadata=True)

    def get_image_size_with_most_features(self):
        processor = self.get_hf_processor().image_processor
        options = self.ctx.get_merged_mm_kwargs({}, modality="image")
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = max(1, maximum // factor**2)
        candidates = []
        for rows in range(1, math.isqrt(cells * 200) + 1):
            cols = min(cells // rows, rows * 200)
            if cols and rows <= cols * 200:
                candidates.append((rows * (cols + 1), rows, cols))
        _, rows, cols = max(candidates)
        height, width = smart_resize(
            rows * factor, cols * factor, factor, minimum, maximum
        )
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(self, seq_len, mm_counts):
        processor = self.get_hf_processor().video_processor
        options = self.ctx.get_merged_mm_kwargs({}, modality="video")
        config = self.get_hf_config().vision_config
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        max_frames = options.get("max_frames", processor.max_frames)
        side = options.get("video_max_image_size", processor.video_max_image_size)
        available = max(1, seq_len // max(1, sum(mm_counts.values())))
        best = (
            0,
            1,
            config.patch_size * config.spatial_merge_size,
            config.patch_size * config.spatial_merge_size,
        )
        for frames in range(1, min(max_frames, available) + 1):
            height, width = adaptive_resize(
                side,
                side,
                frames,
                config.patch_size,
                config.spatial_merge_size,
                maximum,
                side,
                options.get(
                    "timestamp_token_per_frame", processor.timestamp_token_per_frame
                ),
                options.get(
                    "timestamp_token_reserve", processor.timestamp_token_reserve
                ),
            )
            grid = (frames, height // config.patch_size, width // config.patch_size)
            count = vision_token_count(grid, config, "video")
            overhead = frames * (processor.timestamp_token_per_frame + 2) + 2
            if count + overhead <= available and count > best[0]:
                best = (count, frames, height, width)
        return best[1:]

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        config = self.get_hf_config().vision_config
        size = self.get_image_size_with_most_features()
        image_tokens = vision_token_count(
            (1, size.height // config.patch_size, size.width // config.patch_size),
            config,
            "image",
        )
        video = self.get_hf_processor().video_processor
        options = self.ctx.get_merged_mm_kwargs({}, modality="video")
        factor = config.patch_size * config.spatial_merge_size
        max_frames = options.get("max_frames", video.max_frames)
        max_pixels = options.get("max_pixels", video.size["longest_edge"])
        video_tokens = 2 * (max_pixels // factor**2 + max_frames)
        video_tokens += 2 * max_frames * int(bool(config.cat_extra_token))
        return {"image": image_tokens, "video": video_tokens}


class HYV4VLDummyInputsBuilder(BaseDummyInputsBuilder[HYV4VLProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        processor = self.info.get_hf_processor()
        return "".join(
            # For string prompts the HYV4 HF processor expands its internal
            # ``[start, pad, end]`` placeholder itself.  The wrapper in
            # ``apply`` leaves strings untouched and only canonicalizes raw
            # token-id prompts for the baseline prompt-update path.
            processor.get_placeholder(modality) * mm_counts.get(modality, 0)
            for modality in ("image", "video")
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        frames, height, width = self.info.get_dummy_video_shape(seq_len, mm_counts)
        videos = self._get_dummy_videos(
            width=width,
            height=height,
            num_frames=frames,
            num_videos=mm_counts.get("video", 0),
            overrides=mm_options.get("video"),
        )
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": [
                (
                    video,
                    {
                        "fps": 2.0,
                        "duration": len(video) / 2,
                        "total_num_frames": len(video),
                        "frames_indices": list(range(len(video))),
                        "do_sample_frames": False,
                        "video_backend": "opencv",
                    },
                )
                for video in videos
            ],
        }


class HYV4VLMultiModalProcessor(BaseMultiModalProcessor[HYV4VLProcessingInfo]):
    def apply(self, inputs, timing_ctx):
        options = self.info.ctx.get_merged_mm_kwargs(inputs.hf_processor_mm_kwargs)
        intervals = options.get("second_per_grid_ts")
        if intervals is not None:
            videos = inputs.mm_data_items.get("video", ())
            if len(intervals) != len(videos):
                raise ValueError("second_per_grid_ts must match video count")
            normalized = []
            for item, interval in zip(videos, intervals):
                if item is None:
                    raise ValueError("Video timing overrides require media metadata")
                frames, metadata = item
                normalized.append(
                    (frames, {**metadata, "second_per_grid_t": float(interval)})
                )
            mm_items = MultiModalDataItems(
                {**inputs.mm_data_items, "video": VideoProcessorItems(normalized)}
            )
            kwargs = dict(inputs.hf_processor_mm_kwargs)
            kwargs.pop("second_per_grid_ts", None)
            inputs = replace(
                inputs, mm_data_items=mm_items, hf_processor_mm_kwargs=kwargs
            )
        return super().apply(inputs, timing_ctx)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        result = {}
        for modality, pixel_key in (
            ("image", "pixel_values"),
            ("video", "pixel_values_videos"),
        ):
            grid_key = f"{modality}_grid_thw"
            grid = hf_inputs.get(grid_key, torch.empty((0, 3), dtype=torch.long))
            result[pixel_key] = MultiModalFieldConfig.flat_from_sizes(
                modality, grid.prod(-1)
            )
            result[grid_key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
            if modality == "video":
                tau = self.info.get_hf_config().vision_config.temporal_patch_size
                result["video_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", (grid[:, 0] + tau - 1) // tau, keep_on_cpu=True
                )
        return result

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def replacement(index: int, modality: str) -> PromptUpdateDetails:
            item = out_mm_kwargs[modality][index]
            grid = item[f"{modality}_grid_thw"].data
            timestamps = item["video_timestamps"].data if modality == "video" else None
            tokens = [
                getattr(processor, f"{modality}_start_token_id"),
                *processor.get_replacement(modality, grid, timestamps),
                getattr(processor, f"{modality}_end_token_id"),
            ]
            token_id = getattr(processor, f"{modality}_token_id")
            return PromptUpdateDetails.select_token_id(tokens, token_id)

        # The chat template and dummy text both render a placeholder as the
        # unexpanded ``[start, pad, end]`` triplet (see chat_template.jinja and
        # ``HYV4VLProcessor.get_placeholder``). Match that triplet directly so
        # both the HF-expanded path and the cached text-only path resolve.
        return [
            PromptReplacement(
                modality=modality,
                target=[
                    getattr(processor, f"{modality}_start_token_id"),
                    getattr(processor, f"{modality}_token_id"),
                    getattr(processor, f"{modality}_end_token_id"),
                ],
                replacement=partial(replacement, modality=modality),
            )
            for modality in ("image", "video")
        ]
