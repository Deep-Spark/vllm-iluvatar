# SPDX-License-Identifier: Apache-2.0
"""The draft block's quant exclusions, rebased from checkpoint names."""

from dataclasses import dataclass, field

from vllm_iluvatar.models.hy_v4.mtp import _remap_mtp_quant_exclusions


@dataclass
class _ModelOptConfig:
    exclude_modules: list[str] = field(default_factory=list)


@dataclass
class _CompressedTensorsConfig:
    ignored_layers: list[str] = field(default_factory=list)


def _remap(config, layers=78, mtp=1):
    return _remap_mtp_quant_exclusions(config, layers, mtp)


def test_checkpoint_named_mtp_modules_are_rebased_onto_draft_prefixes():
    config = _ModelOptConfig(
        [
            "lm_head",
            "model.layers.3.mlp.gate",
            "model.mtp_layers.0.eh_proj",
            "model.mtp_layers.0.mlp.gate",
        ]
    )

    remapped = _remap(config)

    assert remapped is not config, "the target model shares the original object"
    assert config.exclude_modules[-1] == "model.mtp_layers.0.mlp.gate"
    assert remapped.exclude_modules[-3:] == [
        "model.layers.78.eh_proj",
        "model.layers.78.mlp.gate",
        "model.layers.78.head",
    ]


def test_the_draft_head_follows_the_targets_lm_head_exclusion():
    """``SharedHead`` builds a head the checkpoint has no tensor for.

    ``load_weights`` ties it to the target's unquantized ``lm_head.weight``, so
    an opt-out config quantizing it (nothing told it not to) would create the
    parameter in the checkpoint's format and leave its scale uninitialised.
    """
    remapped = _remap(_ModelOptConfig(["lm_head"]))

    assert remapped.exclude_modules == ["lm_head", "model.layers.78.head"]


def test_every_mtp_layer_gets_its_own_head_exclusion():
    remapped = _remap(_ModelOptConfig(["re:.*lm_head.*"]), mtp=3)

    assert remapped.exclude_modules[1:] == [
        "model.layers.78.head",
        "model.layers.79.head",
        "model.layers.80.head",
    ]


def test_a_quantized_target_head_leaves_the_draft_head_quantized():
    config = _ModelOptConfig(["model.layers.3.mlp.gate"])

    assert _remap(config) is config


def test_compressed_tensors_exclusions_use_their_own_attribute():
    remapped = _remap(_CompressedTensorsConfig(["lm_head"]))

    assert remapped.ignored_layers == ["lm_head", "model.layers.78.head"]


def test_an_unquantized_model_has_nothing_to_remap():
    assert _remap(None) is None
    assert _remap(_ModelOptConfig([])).exclude_modules == []
