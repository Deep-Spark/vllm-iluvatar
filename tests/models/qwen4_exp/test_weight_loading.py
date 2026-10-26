# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

from vllm_iluvatar.models.qwen4_exp.model import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    _remap_qsa_cache_scale_name,
)


def test_quantized_loading_rejects_missing_scale_even_if_weight_loaded() -> None:
    from types import SimpleNamespace
    from vllm_iluvatar.models.qwen4_exp.loading import verify_quantized_weights_loaded

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(4, 4)))
    model.register_parameter("weight_scale", torch.nn.Parameter(torch.ones(4, 1)))
    config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    with pytest.raises(RuntimeError, match="weight_scale"):
        verify_quantized_weights_loaded(model, {"weight"}, config)
    verify_quantized_weights_loaded(model, {"weight", "weight_scale"}, config)


def test_quantized_loading_resolves_real_moe_runner_parameter_paths() -> None:
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm_iluvatar.models.qwen4_exp.loading import verify_quantized_weights_loaded

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = torch.nn.Module()
    runner.routed_experts.register_parameter("w13_weight", torch.nn.Parameter(torch.ones(1)))
    runner.routed_experts.register_parameter("w13_weight_scale", torch.nn.Parameter(torch.ones(1)))
    model = torch.nn.Module()
    model.experts = runner
    config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    with pytest.raises(RuntimeError, match="w13_weight_scale"):
        verify_quantized_weights_loaded(model, {"experts.w13_weight"}, config)
    verify_quantized_weights_loaded(model, {"experts.w13_weight", "experts.w13_weight_scale"}, config)


def test_conditional_generation_quantized_loading_handles_interleaved_visual_weights() -> None:
    from types import SimpleNamespace
    from vllm_iluvatar.models.qwen4_exp.model import Qwen4ExpForCausalLM

    lm = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    torch.nn.Module.__init__(lm)
    lm.config = SimpleNamespace(ple_layer_ids=[])
    lm.quant_config = SimpleNamespace(
        get_name=lambda: "compressed-tensors",
        get_cache_scale_mapper=lambda: Qwen4ExpModel.hf_to_vllm_mapper,
        get_checkpoint_weight_mapper=lambda: Qwen4ExpModel.hf_to_vllm_mapper,
        _ignore_unexpected_suffixes=[],
    )
    lm._verify_quantized_weights = False
    lm.model = torch.nn.Module()
    lm.model.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, 2)))
    lm.model.register_parameter("weight_scale", torch.nn.Parameter(torch.zeros(2, 1)))
    lm.lm_head = torch.nn.Module()
    lm.lm_head.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, 2)))

    visual = torch.nn.Module()
    visual.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, 2)))

    vl = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(vl)
    vl.language_model_only = False
    vl.language_model = lm
    vl.visual = visual

    # Shard 1 has language_model weights followed by visual weights; a later
    # shard has lm_head.weight, splitting language_model into two groupby chunks.
    interleaved = [
        ("model.language_model.weight", torch.ones(2, 2)),
        ("model.language_model.weight_scale", torch.ones(2, 1)),
        ("model.visual.weight", torch.ones(2, 2)),
        ("lm_head.weight", torch.ones(2, 2)),
    ]
    loaded = vl.load_weights(interleaved)
    assert loaded == {
        "language_model.model.weight",
        "language_model.model.weight_scale",
        "visual.weight",
        "language_model.lm_head.weight",
    }


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            (0, 1, 2),
        ),
        (
            "layers.1.linear_attn.in_proj_z.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            3,
        ),
        (
            "layers.1.linear_attn.in_proj_b.weight",
            "layers.1.linear_attn.in_proj_ba.weight",
            0,
        ),
        (
            "layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.1.mlp.experts.0.gate_proj.weight",
            "layers.1.mlp.experts.0.gate_proj.weight",
            None,
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            None,
        ),
        # HC merged down-and-injection projection.
        (
            "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        (
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        ),
        (
            "layers.1.ple.ple_embedding.layer_multipliers",
            "layers.1.ple.ple_embedding.layer_multipliers",
            None,
        ),
    ],
)
def test_text_checkpoint_mapper_preserves_qwen4_exp_specific_weights(
    checkpoint_name: str,
    model_name: str,
    shard_id: str | int | tuple[int, ...] | None,
) -> None:
    assert Qwen4ExpModel.hf_to_vllm_mapper._map_name_with_shard(checkpoint_name) == (
        model_name,
        shard_id,
    )


def test_vl_checkpoint_mapper_composes_language_and_vision_paths() -> None:
    outer_mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
    assert outer_mapper._map_name(
        "model.language_model.layers.0.ple.key_proj.weight"
    ) == ("language_model.model.layers.0.ple.key_proj.weight")
    assert outer_mapper._map_name("lm_head.weight") == ("language_model.lm_head.weight")

    visual_name = outer_mapper._map_name("model.visual.blocks.0.attn.q.weight")
    assert visual_name == "visual.blocks.0.attn.q.weight"
    child_name = visual_name.removeprefix("visual.")
    assert Qwen3_VisionTransformer.hf_to_vllm_mapper._map_name_with_shard(
        child_name
    ) == ("blocks.0.attn.qkv.weight", "q")

    packed_visual_name = outer_mapper._map_name("model.visual.blocks.0.attn.qkv.weight")
    assert packed_visual_name == "visual.blocks.0.attn.qkv.weight"
    assert Qwen3_VisionTransformer.hf_to_vllm_mapper._map_name_with_shard(
        packed_visual_name.removeprefix("visual.")
    ) == ("blocks.0.attn.qkv.weight", None)


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        (
            "layers.0.self_attn.k_proj.k_scale",
            "layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.v_proj.output_scale",
            "layers.0.self_attn._v_scale",
        ),
        (
            "language_model.model.layers.0.self_attn.attn.k_scale",
            "language_model.model.layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
        ),
        (
            "layers.1.self_attn.k_proj.k_scale",
            "layers.1.self_attn.k_proj.k_scale",
        ),
    ],
)
def test_only_qsa_main_cache_scales_move_to_the_merged_owner(
    checkpoint_name: str,
    model_name: str,
) -> None:
    assert _remap_qsa_cache_scale_name(checkpoint_name, frozenset({0})) == model_name


def test_ple_backbone_only_skips_ple_weight_names() -> None:
    from types import SimpleNamespace

    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_iluvatar.models.qwen4_exp.model import _ple_skip_substrs

    assert _ple_skip_substrs(SimpleNamespace(ple_layer_ids=[2])) == ()
    assert _ple_skip_substrs(SimpleNamespace(ple_layer_ids=[])) == (".ple.",)

    skip = WeightsMapper(
        orig_to_new_substr={
            substr: None
            for substr in _ple_skip_substrs(SimpleNamespace(ple_layer_ids=[]))
        }
    )
    keep = WeightsMapper(
        orig_to_new_substr={
            substr: None
            for substr in _ple_skip_substrs(SimpleNamespace(ple_layer_ids=[2]))
        }
    )
    ple_name = "layers.1.ple.ple_embedding.ngram_embedding.weight"
    qsa_name = "layers.0.self_attn.indexer.index_qk_proj.weight"
    assert skip._map_name(ple_name) is None
    assert skip._map_name(qsa_name) == qsa_name
    assert keep._map_name(ple_name) == ple_name


def test_requires_raw_input_tokens_follows_ple_layer_ids() -> None:
    from types import SimpleNamespace

    from vllm_iluvatar.models.qwen4_exp.model import _requires_raw_input_tokens

    assert _requires_raw_input_tokens(SimpleNamespace(ple_layer_ids=[2])) is True
    assert _requires_raw_input_tokens(SimpleNamespace(ple_layer_ids=[])) is False
    assert _requires_raw_input_tokens(SimpleNamespace()) is False
