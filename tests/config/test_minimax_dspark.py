# SPDX-License-Identifier: Apache-2.0
"""DSpark config normalization must be model-import-free and repeatable."""

from types import SimpleNamespace

from transformers import PretrainedConfig

from vllm_iluvatar.config.minimax_dspark import normalize_dspark_config
from vllm_iluvatar.config.speculative import make_hf_config_override, make_post_init


def checkpoint():
    return PretrainedConfig(
        architectures=["DSparkMiniMaxDraftModel"],
        vision_config={"hidden_size": 32},
        text_config=dict(
            num_hidden_layers=5,
            hidden_size=6144,
            rope_theta=10000,
            partial_rotary_factor=0.5,
            dspark_markov_rank=128,
            dspark_noise_token_id=123,
            quantization_config={"quant_method": "mxfp8", "ignored_layers": ["old"]},
        ),
    )


def test_normalization_preserves_source_and_required_fields():
    source = checkpoint()
    before = source.to_dict()
    result = normalize_dspark_config(source)
    assert source.to_dict() == before
    assert result.model_type == "minimax_m3_dspark"
    assert result.hidden_size == 6144 and result.num_hidden_layers == 5
    assert result.num_local_experts == 0 and result.n_shared_experts == 0
    assert result.markov_rank == 128 and result.mask_token_id == 123
    assert result.rope_parameters["partial_rotary_factor"] == 0.5
    assert result.quantization_config["ignored_layers"] == [
        "old",
        "model.fc",
        "markov_head",
        "confidence_head",
        "lm_head",
        "embed_tokens",
    ]
    assert not hasattr(result, "vision_config")
    assert normalize_dspark_config(result) is result
    assert len(result.quantization_config["ignored_layers"]) == 6


def test_config_route_and_post_init_restore_draft_architecture():
    def unused(config):
        raise AssertionError("MiniMax DSpark uses its checkpoint adapter")

    result = make_hf_config_override(unused)(checkpoint())

    def community(owner):
        owner.draft_model_config.hf_config.architectures = ["DSparkDraftModel"]

    updates = []
    owner = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=result),
        draft_parallel_config=SimpleNamespace(enable_expert_parallel=True),
        update_arch_=lambda: updates.append(True),
    )
    make_post_init(community)(owner)
    assert result.architectures == ["DSparkMiniMaxDraftModel"]
    assert updates == [True] and not owner.draft_parallel_config.enable_expert_parallel
