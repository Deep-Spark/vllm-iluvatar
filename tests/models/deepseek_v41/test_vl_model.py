# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the DeepSeek-V4.1 vision wrapper."""

from __future__ import annotations

import torch

from vllm_iluvatar.models.deepseek_v41 import vl_model
from vllm_iluvatar.models.deepseek_v41.vl_model import IluDeepseekV41VLForCausalLM

# Representative checkpoint names, taken verbatim from the index of
# DeepSeek-V4.1-Flash-w4a8-TN.
_MAPPING = {
    "embed.weight": "language_model.model.embed_tokens.weight",
    "head.weight": "language_model.lm_head.weight",
    "norm.weight": "language_model.model.norm.weight",
    "layers.0.ffn.gate.weight": "language_model.model.layers.0.ffn.gate.weight",
    "layers.0.ffn.gate.bias": (
        "language_model.model.layers.0.ffn.gate.e_score_correction_bias"
    ),
    # Image routing bias: a real parameter, so it must survive the mapping
    # under its own name (the text model used to skip it).
    "layers.0.ffn.gate.bias_vl": "language_model.model.layers.0.ffn.gate.bias_vl",
    "layers.2.attn.compressor.wgate.weight": (
        "language_model.model.layers.2.attn.compressor.wgate.weight"
    ),
    # The tower is named the same on both sides.
    "vision.patch_embed.proj.weight": "vision.patch_embed.proj.weight",
    "vision.blocks.0.attn.wqkv.weight": "vision.blocks.0.attn.wqkv.weight",
    "aligner.w1.weight": "aligner.w1.weight",
    "image_start": "image_start",
}

# The DSpark draft loads from the checkpoint itself, so the target drops its
# weights entirely -- including the draft's own bias_vl.
_DROPPED = (
    "mtp.0.ffn.gate.bias_vl",
    "mtp.0.ffn.experts.0.w1.weight",
    "mtp.0.ffn_norm.weight",
)


def _apply(name: str) -> list[str]:
    mapper = vl_model._make_v41_vl_weights_mapper()
    return [mapped for mapped, _ in mapper.apply([(name, None)])]


def test_weights_mapper_reroots_text_weights_under_language_model():
    for name, expected in _MAPPING.items():
        assert _apply(name) == [expected], name


def test_weights_mapper_drops_dspark_draft_weights():
    for name in _DROPPED:
        assert _apply(name) == [], name


def test_build_image_span_places_delimiters_around_aligner_rows():
    model = IluDeepseekV41VLForCausalLM.__new__(IluDeepseekV41VLForCausalLM)
    model.image_start = torch.tensor([1.0, 1.0])
    model.image_end = torch.tensor([2.0, 2.0])
    model.image_newline = torch.tensor([3.0, 3.0])

    # One 2x1 image: start, (row, newline) * 2, end.
    types = torch.tensor(
        [
            vl_model.IMAGE_START,
            vl_model.IMAGE,
            vl_model.IMAGE_NEW_LINE,
            vl_model.IMAGE,
            vl_model.IMAGE_NEW_LINE,
            vl_model.IMAGE_END,
        ]
    )
    image_embeds = torch.tensor([[10.0, 11.0], [12.0, 13.0]])

    span = model._build_image_span(image_embeds, types)

    assert torch.equal(
        span,
        torch.tensor(
            [
                [1.0, 1.0],
                [10.0, 11.0],
                [3.0, 3.0],
                [12.0, 13.0],
                [3.0, 3.0],
                [2.0, 2.0],
            ]
        ),
    )


def test_register_model_points_every_arch_name_at_the_vl_wrapper(monkeypatch):
    import vllm

    registered: dict[str, str] = {}
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: registered.__setitem__(arch, path),
    )

    vl_model.register_model()

    target = "vllm_iluvatar.models.deepseek_v41.vl_model:IluDeepseekV41VLForCausalLM"
    assert registered == {
        "DeepseekV41ForCausalLM": target,
        "IluDeepseekV41ForCausalLM": target,
        "IluDeepseekV41VLForCausalLM": target,
    }
