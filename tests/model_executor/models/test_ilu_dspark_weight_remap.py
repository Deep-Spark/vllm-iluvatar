# SPDX-License-Identifier: Apache-2.0
"""Pure-function tests for the ilu DSpark checkpoint name remapping."""

from __future__ import annotations

import pytest

from vllm_iluvatar.models.deepseek_v4.dspark import remap_dspark_name

NUM_HIDDEN_LAYERS = 43


@pytest.mark.parametrize(
    ("ckpt_name", "expected"),
    [
        # ModuleList indices are 0/1/2 (community nvidia/dspark convention),
        # not num_hidden_layers + stage (MTP-on-target convention).
        ("mtp.0.attn.wq_a.weight", "model.layers.0.attn.wq_a.weight"),
        ("mtp.0.attn.attn_sink", "model.layers.0.attn.attn_sink"),
        ("mtp.1.attn.wkv.weight", "model.layers.1.attn.wkv.weight"),
        ("mtp.2.ffn.experts.7.w1.weight", "model.layers.2.ffn.experts.7.w1.weight"),
        ("mtp.0.main_proj.weight", "model.main_proj.weight"),
        ("mtp.0.main_norm.weight", "model.main_norm.weight"),
        ("mtp.2.norm.weight", "model.norm.weight"),
        ("mtp.2.hc_head_fn", "model.hc_head_fn"),
        ("mtp.2.markov_head.markov_w1.weight", "model.markov_head.markov_w1.weight"),
    ],
)
def test_remap_known_keys(ckpt_name, expected):
    assert remap_dspark_name(ckpt_name, NUM_HIDDEN_LAYERS) == expected


@pytest.mark.parametrize(
    "ckpt_name",
    [
        "layers.0.attn.wq_a.weight",  # target weight, not draft
        "embed.weight",
        "head.weight",
        "mtp.2.confidence_head.proj.weight",  # not wired into inference
    ],
)
def test_remap_skips_non_draft_keys(ckpt_name):
    assert remap_dspark_name(ckpt_name, NUM_HIDDEN_LAYERS) is None


def test_remap_does_not_use_mtp_layer_offset():
    # Regression: an earlier revision remapped to layers.{43+i}, which
    # KeyError'd against the ModuleList's layers.0/1/2 parameter names.
    assert (
        remap_dspark_name("mtp.0.attn.attn_sink", NUM_HIDDEN_LAYERS)
        != f"model.layers.{NUM_HIDDEN_LAYERS}.attn.attn_sink"
    )
