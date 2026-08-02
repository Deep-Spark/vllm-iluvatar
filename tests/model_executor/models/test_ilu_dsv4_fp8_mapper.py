# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the DeepSeek-V4 FP8 ``WeightsMapper``.

Covers ``_make_fp8_weights_mapper()``, the plugin-side equivalent of the
community ``_make_deepseek_v4_weights_mapper("fp4")``
(``vllm/models/deepseek_v4/nvidia/model.py:1311-1341``).

The mapper is a pure string transform: no GPU, no distributed init, no config.

``SAMPLES`` holds exactly one representative checkpoint name per distinct
pattern (82 patterns, ``mtp.*`` included).  ``EXTRA_SAMPLES`` adds variants
that probe tricky interactions (routed ``w2`` vs shared ``w2``, ``hc_*_scale``
vs ``.scale``).  Every expected name is hard-coded -- never read back from the
code under test.

``hc_head`` uses the dot-less prefix form (``"hc_head": "model.hc_head"``);
a trailing dot would turn ``hc_head_fn`` into ``model.hc_head._fn``.
"""

from __future__ import annotations

import importlib
import re

import pytest

FP8_MODEL_MODULE = "vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model"

# ---------------------------------------------------------------------------
# Expected mapping tables.
# ---------------------------------------------------------------------------

# ``orig_to_new_regex`` must be an *ordered* mapping: the MXFP4 experts rule has
# to run before the generic block-FP8 rule, otherwise ``.scale`` is rewritten to
# ``.weight_scale_inv`` first and the experts rule can never match.
EXPECTED_REGEX_ORDER = [
    (r"(\.experts\.\d+\.w[123])\.scale$", r"\1.weight_scale"),
    (r"\.scale$", ".weight_scale_inv"),
]
EXPECTED_PREFIX = {
    "layers.": "model.layers.",
    "embed.": "model.embed.",
    "norm.": "model.norm.",
    "hc_head": "model.hc_head",
    "mtp.": "model.mtp.",
}
EXPECTED_SUFFIX = {
    "head.weight": "lm_head.weight",
    "embed.weight": "embed_tokens.weight",
    ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
}
EXPECTED_SUBSTR = {
    ".shared_experts.w2": ".shared_experts.down_proj",
}

# Acceptance baseline: "distinct patterns: 82"
EXPECTED_PATTERN_COUNT = 82

# ---------------------------------------------------------------------------
# Sample set: (pattern, checkpoint name, expected mapped name)
# ---------------------------------------------------------------------------

# routed_expert_weight (2 patterns)
_ROUTED_EXPERT_WEIGHT = [
    (
        "layers.*.ffn.experts.*.w*.weight",
        "layers.0.ffn.experts.0.w1.weight",
        "model.layers.0.ffn.experts.0.w1.weight",
    ),
    (
        "mtp.*.ffn.experts.*.w*.weight",
        "mtp.0.ffn.experts.0.w1.weight",
        "model.mtp.0.ffn.experts.0.w1.weight",
    ),
]

# routed_expert_scale (2 patterns) -> ``.weight_scale`` (no ``_inv``)
_ROUTED_EXPERT_SCALE = [
    (
        "layers.*.ffn.experts.*.w*.scale",
        "layers.42.ffn.experts.255.w3.scale",
        "model.layers.42.ffn.experts.255.w3.weight_scale",
    ),
    (
        "mtp.*.ffn.experts.*.w*.scale",
        "mtp.0.ffn.experts.0.w2.scale",
        "model.mtp.0.ffn.experts.0.w2.weight_scale",
    ),
]

# shared_expert_weight (2 patterns) -> only ``w2`` is renamed (``down_proj``);
# ``w1`` / ``w3`` stay put and are fused into ``gate_up_proj`` by the loader's
# stacked mapping, not by the mapper.
_SHARED_EXPERT_WEIGHT = [
    (
        "layers.*.ffn.shared_experts.w*.weight",
        "layers.0.ffn.shared_experts.w2.weight",
        "model.layers.0.ffn.shared_experts.down_proj.weight",
    ),
    (
        "mtp.*.ffn.shared_experts.w*.weight",
        "mtp.0.ffn.shared_experts.w1.weight",
        "model.mtp.0.ffn.shared_experts.w1.weight",
    ),
]

# shared_expert_scale (2 patterns) -> ``.weight_scale_inv``
_SHARED_EXPERT_SCALE = [
    (
        "layers.*.ffn.shared_experts.w*.scale",
        "layers.0.ffn.shared_experts.w2.scale",
        "model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv",
    ),
    (
        "mtp.*.ffn.shared_experts.w*.scale",
        "mtp.0.ffn.shared_experts.w3.scale",
        "model.mtp.0.ffn.shared_experts.w3.weight_scale_inv",
    ),
]

# other_fp8_weight (13 patterns)
_OTHER_FP8_WEIGHT = [
    (
        "layers.*.attn.wkv.weight",
        "layers.0.attn.wkv.weight",
        "model.layers.0.attn.wkv.weight",
    ),
    (
        "layers.*.attn.wo_a.weight",
        "layers.0.attn.wo_a.weight",
        "model.layers.0.attn.wo_a.weight",
    ),
    (
        "layers.*.attn.wo_b.weight",
        "layers.0.attn.wo_b.weight",
        "model.layers.0.attn.wo_b.weight",
    ),
    (
        "layers.*.attn.wq_a.weight",
        "layers.0.attn.wq_a.weight",
        "model.layers.0.attn.wq_a.weight",
    ),
    (
        "layers.*.attn.wq_b.weight",
        "layers.0.attn.wq_b.weight",
        "model.layers.0.attn.wq_b.weight",
    ),
    (
        "layers.*.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.weight",
        "model.layers.2.attn.indexer.wq_b.weight",
    ),
    (
        "mtp.*.attn.wkv.weight",
        "mtp.0.attn.wkv.weight",
        "model.mtp.0.attn.wkv.weight",
    ),
    (
        "mtp.*.attn.wo_a.weight",
        "mtp.0.attn.wo_a.weight",
        "model.mtp.0.attn.wo_a.weight",
    ),
    (
        "mtp.*.attn.wo_b.weight",
        "mtp.0.attn.wo_b.weight",
        "model.mtp.0.attn.wo_b.weight",
    ),
    (
        "mtp.*.attn.wq_a.weight",
        "mtp.0.attn.wq_a.weight",
        "model.mtp.0.attn.wq_a.weight",
    ),
    (
        "mtp.*.attn.wq_b.weight",
        "mtp.0.attn.wq_b.weight",
        "model.mtp.0.attn.wq_b.weight",
    ),
    (
        "mtp.*.e_proj.weight",
        "mtp.0.e_proj.weight",
        "model.mtp.0.e_proj.weight",
    ),
    (
        "mtp.*.h_proj.weight",
        "mtp.0.h_proj.weight",
        "model.mtp.0.h_proj.weight",
    ),
]

# other_fp8_scale (13 patterns) -> ``.weight_scale_inv``
_OTHER_FP8_SCALE = [
    (
        "layers.*.attn.wkv.scale",
        "layers.0.attn.wkv.scale",
        "model.layers.0.attn.wkv.weight_scale_inv",
    ),
    (
        "layers.*.attn.wo_a.scale",
        "layers.0.attn.wo_a.scale",
        "model.layers.0.attn.wo_a.weight_scale_inv",
    ),
    (
        "layers.*.attn.wo_b.scale",
        "layers.0.attn.wo_b.scale",
        "model.layers.0.attn.wo_b.weight_scale_inv",
    ),
    (
        "layers.*.attn.wq_a.scale",
        "layers.0.attn.wq_a.scale",
        "model.layers.0.attn.wq_a.weight_scale_inv",
    ),
    (
        "layers.*.attn.wq_b.scale",
        "layers.0.attn.wq_b.scale",
        "model.layers.0.attn.wq_b.weight_scale_inv",
    ),
    (
        "layers.*.attn.indexer.wq_b.scale",
        "layers.2.attn.indexer.wq_b.scale",
        "model.layers.2.attn.indexer.wq_b.weight_scale_inv",
    ),
    (
        "mtp.*.attn.wkv.scale",
        "mtp.0.attn.wkv.scale",
        "model.mtp.0.attn.wkv.weight_scale_inv",
    ),
    (
        "mtp.*.attn.wo_a.scale",
        "mtp.0.attn.wo_a.scale",
        "model.mtp.0.attn.wo_a.weight_scale_inv",
    ),
    (
        "mtp.*.attn.wo_b.scale",
        "mtp.0.attn.wo_b.scale",
        "model.mtp.0.attn.wo_b.weight_scale_inv",
    ),
    (
        "mtp.*.attn.wq_a.scale",
        "mtp.0.attn.wq_a.scale",
        "model.mtp.0.attn.wq_a.weight_scale_inv",
    ),
    (
        "mtp.*.attn.wq_b.scale",
        "mtp.0.attn.wq_b.scale",
        "model.mtp.0.attn.wq_b.weight_scale_inv",
    ),
    (
        "mtp.*.e_proj.scale",
        "mtp.0.e_proj.scale",
        "model.mtp.0.e_proj.weight_scale_inv",
    ),
    (
        "mtp.*.h_proj.scale",
        "mtp.0.h_proj.scale",
        "model.mtp.0.h_proj.weight_scale_inv",
    ),
]

# plain_tensor (48 patterns)
_PLAIN_TENSOR = [
    # global tensors
    ("embed.weight", "embed.weight", "model.embed_tokens.weight"),
    ("head.weight", "head.weight", "lm_head.weight"),
    ("norm.weight", "norm.weight", "model.norm.weight"),
    ("hc_head_base", "hc_head_base", "model.hc_head_base"),
    ("hc_head_fn", "hc_head_fn", "model.hc_head_fn"),
    # ``hc_head_scale`` ends with ``_scale``, not ``.scale`` -> regex must miss.
    ("hc_head_scale", "hc_head_scale", "model.hc_head_scale"),
    # main-trunk decoder layers
    (
        "layers.*.attn.attn_sink",
        "layers.0.attn.attn_sink",
        "model.layers.0.attn.attn_sink",
    ),
    (
        "layers.*.attn.kv_norm.weight",
        "layers.0.attn.kv_norm.weight",
        "model.layers.0.attn.kv_norm.weight",
    ),
    (
        "layers.*.attn.q_norm.weight",
        "layers.0.attn.q_norm.weight",
        "model.layers.0.attn.q_norm.weight",
    ),
    (
        "layers.*.attn_norm.weight",
        "layers.0.attn_norm.weight",
        "model.layers.0.attn_norm.weight",
    ),
    (
        "layers.*.ffn.gate.tid*eid",
        "layers.0.ffn.gate.tid2eid",
        "model.layers.0.ffn.gate.tid2eid",
    ),
    (
        "layers.*.ffn.gate.weight",
        "layers.0.ffn.gate.weight",
        "model.layers.0.ffn.gate.weight",
    ),
    (
        "layers.*.ffn.gate.bias",
        "layers.3.ffn.gate.bias",
        "model.layers.3.ffn.gate.e_score_correction_bias",
    ),
    (
        "layers.*.ffn_norm.weight",
        "layers.0.ffn_norm.weight",
        "model.layers.0.ffn_norm.weight",
    ),
    (
        "layers.*.hc_attn_base",
        "layers.0.hc_attn_base",
        "model.layers.0.hc_attn_base",
    ),
    (
        "layers.*.hc_attn_fn",
        "layers.0.hc_attn_fn",
        "model.layers.0.hc_attn_fn",
    ),
    (
        "layers.*.hc_attn_scale",
        "layers.0.hc_attn_scale",
        "model.layers.0.hc_attn_scale",
    ),
    (
        "layers.*.hc_ffn_base",
        "layers.0.hc_ffn_base",
        "model.layers.0.hc_ffn_base",
    ),
    (
        "layers.*.hc_ffn_fn",
        "layers.0.hc_ffn_fn",
        "model.layers.0.hc_ffn_fn",
    ),
    (
        "layers.*.hc_ffn_scale",
        "layers.0.hc_ffn_scale",
        "model.layers.0.hc_ffn_scale",
    ),
    # compressor / indexer sub-modules (non-dense layers only)
    (
        "layers.*.attn.compressor.ape",
        "layers.2.attn.compressor.ape",
        "model.layers.2.attn.compressor.ape",
    ),
    (
        "layers.*.attn.compressor.norm.weight",
        "layers.2.attn.compressor.norm.weight",
        "model.layers.2.attn.compressor.norm.weight",
    ),
    (
        "layers.*.attn.compressor.wgate.weight",
        "layers.2.attn.compressor.wgate.weight",
        "model.layers.2.attn.compressor.wgate.weight",
    ),
    (
        "layers.*.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wkv.weight",
        "model.layers.2.attn.compressor.wkv.weight",
    ),
    (
        "layers.*.attn.indexer.compressor.ape",
        "layers.2.attn.indexer.compressor.ape",
        "model.layers.2.attn.indexer.compressor.ape",
    ),
    (
        "layers.*.attn.indexer.compressor.norm.weight",
        "layers.2.attn.indexer.compressor.norm.weight",
        "model.layers.2.attn.indexer.compressor.norm.weight",
    ),
    (
        "layers.*.attn.indexer.compressor.wgate.weight",
        "layers.2.attn.indexer.compressor.wgate.weight",
        "model.layers.2.attn.indexer.compressor.wgate.weight",
    ),
    (
        "layers.*.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.wkv.weight",
        "model.layers.2.attn.indexer.compressor.wkv.weight",
    ),
    (
        "layers.*.attn.indexer.weights_proj.weight",
        "layers.2.attn.indexer.weights_proj.weight",
        "model.layers.2.attn.indexer.weights_proj.weight",
    ),
    # MTP layer (skipped at load time, but the mapper still has to rewrite it)
    (
        "mtp.*.attn.attn_sink",
        "mtp.0.attn.attn_sink",
        "model.mtp.0.attn.attn_sink",
    ),
    (
        "mtp.*.attn.kv_norm.weight",
        "mtp.0.attn.kv_norm.weight",
        "model.mtp.0.attn.kv_norm.weight",
    ),
    (
        "mtp.*.attn.q_norm.weight",
        "mtp.0.attn.q_norm.weight",
        "model.mtp.0.attn.q_norm.weight",
    ),
    (
        "mtp.*.attn_norm.weight",
        "mtp.0.attn_norm.weight",
        "model.mtp.0.attn_norm.weight",
    ),
    (
        "mtp.*.enorm.weight",
        "mtp.0.enorm.weight",
        "model.mtp.0.enorm.weight",
    ),
    (
        "mtp.*.ffn.gate.bias",
        "mtp.0.ffn.gate.bias",
        "model.mtp.0.ffn.gate.e_score_correction_bias",
    ),
    (
        "mtp.*.ffn.gate.weight",
        "mtp.0.ffn.gate.weight",
        "model.mtp.0.ffn.gate.weight",
    ),
    (
        "mtp.*.ffn_norm.weight",
        "mtp.0.ffn_norm.weight",
        "model.mtp.0.ffn_norm.weight",
    ),
    ("mtp.*.hc_attn_base", "mtp.0.hc_attn_base", "model.mtp.0.hc_attn_base"),
    ("mtp.*.hc_attn_fn", "mtp.0.hc_attn_fn", "model.mtp.0.hc_attn_fn"),
    ("mtp.*.hc_attn_scale", "mtp.0.hc_attn_scale", "model.mtp.0.hc_attn_scale"),
    ("mtp.*.hc_ffn_base", "mtp.0.hc_ffn_base", "model.mtp.0.hc_ffn_base"),
    ("mtp.*.hc_ffn_fn", "mtp.0.hc_ffn_fn", "model.mtp.0.hc_ffn_fn"),
    ("mtp.*.hc_ffn_scale", "mtp.0.hc_ffn_scale", "model.mtp.0.hc_ffn_scale"),
    ("mtp.*.hc_head_base", "mtp.0.hc_head_base", "model.mtp.0.hc_head_base"),
    ("mtp.*.hc_head_fn", "mtp.0.hc_head_fn", "model.mtp.0.hc_head_fn"),
    ("mtp.*.hc_head_scale", "mtp.0.hc_head_scale", "model.mtp.0.hc_head_scale"),
    ("mtp.*.hnorm.weight", "mtp.0.hnorm.weight", "model.mtp.0.hnorm.weight"),
    ("mtp.*.norm.weight", "mtp.0.norm.weight", "model.mtp.0.norm.weight"),
]

SAMPLES: list[tuple[str, str, str]] = [
    *_ROUTED_EXPERT_WEIGHT,
    *_ROUTED_EXPERT_SCALE,
    *_SHARED_EXPERT_WEIGHT,
    *_SHARED_EXPERT_SCALE,
    *_OTHER_FP8_WEIGHT,
    *_OTHER_FP8_SCALE,
    *_PLAIN_TENSOR,
]

# Same patterns as above, extra names that probe the rule interactions.
EXTRA_SAMPLES: list[tuple[str, str, str]] = [
    # routed ``w2`` must NOT be caught by the ``.shared_experts.w2`` substr rule
    (
        "layers.*.ffn.experts.*.w*.weight",
        "layers.0.ffn.experts.0.w2.weight",
        "model.layers.0.ffn.experts.0.w2.weight",
    ),
    (
        "layers.*.ffn.experts.*.w*.scale",
        "layers.0.ffn.experts.0.w2.scale",
        "model.layers.0.ffn.experts.0.w2.weight_scale",
    ),
    # shared experts ``w1`` / ``w3`` are left untouched (loader-side stacking)
    (
        "layers.*.ffn.shared_experts.w*.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "model.layers.0.ffn.shared_experts.w1.weight",
    ),
    (
        "layers.*.ffn.shared_experts.w*.weight",
        "layers.0.ffn.shared_experts.w3.weight",
        "model.layers.0.ffn.shared_experts.w3.weight",
    ),
    (
        "layers.*.ffn.shared_experts.w*.scale",
        "layers.0.ffn.shared_experts.w1.scale",
        "model.layers.0.ffn.shared_experts.w1.weight_scale_inv",
    ),
    # routed experts scale on the last layer / last expert id
    (
        "layers.*.ffn.experts.*.w*.scale",
        "layers.42.ffn.experts.255.w1.scale",
        "model.layers.42.ffn.experts.255.w1.weight_scale",
    ),
]

ALL_SAMPLES = SAMPLES + EXTRA_SAMPLES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mapper():
    """Build the plugin mapper (import deferred: ``fp8_model`` lands in 2.2)."""
    fp8_model = importlib.import_module(FP8_MODEL_MODULE)
    return fp8_model._make_fp8_weights_mapper()


def _map_one(mapper, name: str) -> str | None:
    """Map a single checkpoint name through the public ``apply_list`` API."""
    mapped = mapper.apply_list([name])
    return mapped[0] if mapped else None


def _normalise_pattern(name: str) -> str:
    """``normalise_pattern``: digit runs become ``*``."""
    return re.sub(r"\d+", "*", name)


# ---------------------------------------------------------------------------
# sample-set self checks (no dependency on the code under test)
# ---------------------------------------------------------------------------


def test_sample_set_covers_every_baseline_pattern():
    patterns = [pattern for pattern, _, _ in SAMPLES]
    assert len(set(patterns)) == EXPECTED_PATTERN_COUNT
    # one representative per pattern, no accidental duplicates
    assert len(patterns) == EXPECTED_PATTERN_COUNT


def test_sample_set_includes_mtp_patterns():
    mtp_patterns = {p for p, _, _ in SAMPLES if p.startswith("mtp.")}
    # baseline report: 1 + 1 + 1 + 1 + 7 + 7 + 19 mtp patterns
    assert len(mtp_patterns) == 37


@pytest.mark.parametrize(("pattern", "name", "_expected"), ALL_SAMPLES)
def test_sample_name_normalises_to_its_pattern(pattern, name, _expected):
    assert _normalise_pattern(name) == pattern


# ---------------------------------------------------------------------------
# mapping tables (Requirement 2.2)
# ---------------------------------------------------------------------------


def test_mapper_declares_the_four_mapping_kinds():
    mapper = _mapper()

    assert [
        (pattern.pattern, repl)
        for pattern, repl in mapper.orig_to_new_regex.items()
    ] == EXPECTED_REGEX_ORDER
    assert dict(mapper.orig_to_new_prefix) == EXPECTED_PREFIX
    assert dict(mapper.orig_to_new_suffix) == EXPECTED_SUFFIX
    assert dict(mapper.orig_to_new_substr) == EXPECTED_SUBSTR


def test_experts_scale_regex_is_ordered_before_generic_scale_regex():
    """Requirement 2.2: the experts rule must be evaluated first."""
    mapper = _mapper()

    keys = list(mapper.orig_to_new_regex)
    experts_idx = next(
        i for i, k in enumerate(keys) if ".experts" in k.pattern
    )
    generic_idx = next(
        i for i, k in enumerate(keys) if k.pattern == r"\.scale$"
    )
    assert experts_idx < generic_idx

    # ...and the observable consequence of that order:
    assert (
        _map_one(mapper, "layers.0.ffn.experts.0.w1.scale")
        == "model.layers.0.ffn.experts.0.w1.weight_scale"
    )
    assert (
        _map_one(mapper, "layers.0.attn.wkv.scale")
        == "model.layers.0.attn.wkv.weight_scale_inv"
    )


# ---------------------------------------------------------------------------
# full name set (Requirement 2.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_pattern", "name", "expected"), ALL_SAMPLES)
def test_mapper_maps_checkpoint_name_to_design_table(_pattern, name, expected):
    assert _map_one(_mapper(), name) == expected


def test_no_checkpoint_name_is_dropped():
    """``apply_list`` filters out ``None`` targets; nothing here may vanish."""
    mapper = _mapper()
    names = [name for _, name, _ in ALL_SAMPLES]
    assert len(mapper.apply_list(names)) == len(names)


def test_suffix_rules():
    """Requirement 2.3: ``head.weight`` / ``embed.weight`` / ``.ffn.gate.bias``."""
    mapper = _mapper()

    assert _map_one(mapper, "head.weight") == "lm_head.weight"
    assert _map_one(mapper, "embed.weight") == "model.embed_tokens.weight"
    assert (
        _map_one(mapper, "layers.3.ffn.gate.bias")
        == "model.layers.3.ffn.gate.e_score_correction_bias"
    )
    # ``lm_head`` must not be re-prefixed with ``model.``
    assert not _map_one(mapper, "head.weight").startswith("model.")


def test_shared_experts_w2_substr_rule():
    """Requirement 2.3: only shared ``w2`` becomes ``down_proj``."""
    mapper = _mapper()

    assert (
        _map_one(mapper, "layers.0.ffn.shared_experts.w2.weight")
        == "model.layers.0.ffn.shared_experts.down_proj.weight"
    )
    assert (
        _map_one(mapper, "layers.0.ffn.shared_experts.w2.scale")
        == "model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv"
    )
    for wid in ("w1", "w3"):
        name = f"layers.0.ffn.shared_experts.{wid}.weight"
        assert _map_one(mapper, name) == f"model.{name}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("layers.0.attn_norm.weight", "model.layers.0.attn_norm.weight"),
        ("embed.weight", "model.embed_tokens.weight"),
        ("norm.weight", "model.norm.weight"),
        ("hc_head_fn", "model.hc_head_fn"),
        ("mtp.0.attn_norm.weight", "model.mtp.0.attn_norm.weight"),
    ],
)
def test_five_prefix_promotions(name, expected):
    """Requirement 2.3: ``layers.`` / ``embed.`` / ``norm.`` / ``hc_head`` / ``mtp.``."""
    assert _map_one(_mapper(), name) == expected


def test_underscore_scale_names_are_not_touched_by_the_scale_regex():
    """``hc_*_scale`` has no dot before ``scale`` -> stays a plain tensor name."""
    mapper = _mapper()

    assert _map_one(mapper, "hc_head_scale") == "model.hc_head_scale"
    assert (
        _map_one(mapper, "layers.0.hc_attn_scale")
        == "model.layers.0.hc_attn_scale"
    )
    assert (
        _map_one(mapper, "layers.0.hc_ffn_scale")
        == "model.layers.0.hc_ffn_scale"
    )


def test_apply_preserves_tensor_objects():
    """The loader path uses ``apply()`` over (name, tensor) pairs."""
    import torch

    mapper = _mapper()
    tensors = {name: torch.empty(0) for _, name, _ in SAMPLES}
    mapped = dict(mapper.apply(list(tensors.items())))

    expected_names = {expected for _, _, expected in SAMPLES}
    assert set(mapped) == expected_names
    assert all(t.numel() == 0 for t in mapped.values())
