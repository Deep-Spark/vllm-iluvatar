# SPDX-License-Identifier: Apache-2.0
"""CPU-only test for the load-statistics classifier.

Pins the classifier (``fp8_load_stats.classify``) against the acceptance
baseline.  Every row of ``SCAN_REPORT_ROWS`` below is a (pattern, dtype,
category) triple covering all seven categories.

Three properties are checked per row:

1. ``normalise_pattern(sample_name)`` reproduces the report's ``pattern``;
2. ``classify`` puts the row in the expected category using the safetensors
   dtype spelling;
3. the same row still lands in that category with the torch dtype spelling and
   the mapped name the loader actually sees.

No torch, no vLLM, no GPU: ``fp8_load_stats`` is torch-free, so everything
here is plain Python (one test uses real ``torch.dtype`` objects and skips if
torch is unavailable).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_load_stats import (
    CATEGORY_KEYS,
    EXPECTED_TRUNK_COUNTS,
    STATS_KEYS,
    STATS_LOG_TAG,
    TOTAL_KEY,
    LoadStats,
    classify,
    classify_tensor,
    dtype_name,
    normalise_pattern,
)

# ---------------------------------------------------------------------------
# Rows copied from the acceptance baseline report.
#
# (sample_name, pattern, safetensors_dtype, torch_dtype, category)
#
# `sample_name` and `pattern` are the report's own "sample" and "pattern"
# columns; `category` is the section the row appears under. `torch_dtype` is the
# corresponding torch spelling, which is what the loader sees.
# ---------------------------------------------------------------------------

SCAN_REPORT_ROWS: list[tuple[str, str, str, str, str]] = [
    # -- routed_expert_weight (2 patterns) ---------------------------------
    (
        "layers.0.ffn.experts.0.w1.weight",
        "layers.*.ffn.experts.*.w*.weight",
        "I8",
        "int8",
        "routed_expert_weight",
    ),
    (
        "mtp.0.ffn.experts.0.w1.weight",
        "mtp.*.ffn.experts.*.w*.weight",
        "I8",
        "int8",
        "routed_expert_weight",
    ),
    # -- routed_expert_scale (2 patterns) ----------------------------------
    (
        "layers.0.ffn.experts.0.w1.scale",
        "layers.*.ffn.experts.*.w*.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "routed_expert_scale",
    ),
    (
        "mtp.0.ffn.experts.0.w1.scale",
        "mtp.*.ffn.experts.*.w*.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "routed_expert_scale",
    ),
    # -- shared_expert_weight (2 patterns) ---------------------------------
    (
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.*.ffn.shared_experts.w*.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "shared_expert_weight",
    ),
    (
        "mtp.0.ffn.shared_experts.w1.weight",
        "mtp.*.ffn.shared_experts.w*.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "shared_expert_weight",
    ),
    # -- shared_expert_scale (2 patterns) ----------------------------------
    (
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.*.ffn.shared_experts.w*.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "shared_expert_scale",
    ),
    (
        "layers.0.ffn.shared_experts.w2.scale",
        "layers.*.ffn.shared_experts.w*.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "shared_expert_scale",
    ),
    # -- other_fp8_weight (13 patterns) ------------------------------------
    (
        "layers.0.attn.wkv.weight",
        "layers.*.attn.wkv.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "other_fp8_weight",
    ),
    (
        "layers.0.attn.wq_a.weight",
        "layers.*.attn.wq_a.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "other_fp8_weight",
    ),
    (
        "layers.0.attn.wo_a.weight",
        "layers.*.attn.wo_a.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "other_fp8_weight",
    ),
    (
        "layers.2.attn.indexer.wq_b.weight",
        "layers.*.attn.indexer.wq_b.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "other_fp8_weight",
    ),
    (
        "mtp.0.e_proj.weight",
        "mtp.*.e_proj.weight",
        "F8_E4M3",
        "float8_e4m3fn",
        "other_fp8_weight",
    ),
    # -- other_fp8_scale (13 patterns) -------------------------------------
    (
        "layers.0.attn.wkv.scale",
        "layers.*.attn.wkv.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "other_fp8_scale",
    ),
    (
        "layers.0.attn.wq_b.scale",
        "layers.*.attn.wq_b.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "other_fp8_scale",
    ),
    (
        "layers.2.attn.indexer.wq_b.scale",
        "layers.*.attn.indexer.wq_b.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "other_fp8_scale",
    ),
    (
        "mtp.0.h_proj.scale",
        "mtp.*.h_proj.scale",
        "F8_E8M0",
        "float8_e8m0fnu",
        "other_fp8_scale",
    ),
    # -- plain_tensor (48 patterns) ----------------------------------------
    ("embed.weight", "embed.weight", "BF16", "bfloat16", "plain_tensor"),
    ("head.weight", "head.weight", "BF16", "bfloat16", "plain_tensor"),
    ("norm.weight", "norm.weight", "BF16", "bfloat16", "plain_tensor"),
    ("hc_head_fn", "hc_head_fn", "F32", "float32", "plain_tensor"),
    ("hc_head_scale", "hc_head_scale", "F32", "float32", "plain_tensor"),
    (
        "layers.0.attn.attn_sink",
        "layers.*.attn.attn_sink",
        "F32",
        "float32",
        "plain_tensor",
    ),
    (
        "layers.0.attn.q_norm.weight",
        "layers.*.attn.q_norm.weight",
        "BF16",
        "bfloat16",
        "plain_tensor",
    ),
    (
        "layers.0.ffn.gate.weight",
        "layers.*.ffn.gate.weight",
        "BF16",
        "bfloat16",
        "plain_tensor",
    ),
    (
        "layers.3.ffn.gate.bias",
        "layers.*.ffn.gate.bias",
        "F32",
        "float32",
        "plain_tensor",
    ),
    (
        "layers.0.ffn.gate.tid2eid",
        "layers.*.ffn.gate.tid*eid",
        "I64",
        "int64",
        "plain_tensor",
    ),
    (
        "layers.0.hc_attn_scale",
        "layers.*.hc_attn_scale",
        "F32",
        "float32",
        "plain_tensor",
    ),
    (
        "layers.2.attn.compressor.wkv.weight",
        "layers.*.attn.compressor.wkv.weight",
        "BF16",
        "bfloat16",
        "plain_tensor",
    ),
    (
        "layers.2.attn.compressor.ape",
        "layers.*.attn.compressor.ape",
        "F32",
        "float32",
        "plain_tensor",
    ),
    (
        "layers.2.attn.indexer.compressor.norm.weight",
        "layers.*.attn.indexer.compressor.norm.weight",
        "BF16",
        "bfloat16",
        "plain_tensor",
    ),
    (
        "layers.2.attn.indexer.weights_proj.weight",
        "layers.*.attn.indexer.weights_proj.weight",
        "BF16",
        "bfloat16",
        "plain_tensor",
    ),
    ("mtp.0.enorm.weight", "mtp.*.enorm.weight", "BF16", "bfloat16", "plain_tensor"),
    (
        "mtp.0.attn.attn_sink",
        "mtp.*.attn.attn_sink",
        "F32",
        "float32",
        "plain_tensor",
    ),
]

ROW_IDS = [f"{row[4]}:{row[0]}" for row in SCAN_REPORT_ROWS]


def _mapped_name(name: str) -> str:
    """Apply the loader's ``WeightsMapper`` rules to a checkpoint key.

    A local transcription of ``_make_fp8_weights_mapper()`` -- importing
    ``fp8_model`` would drag in torch and vLLM, and the rules are four lines.
    The mapper itself is already covered by ``test_ilu_dsv4_fp8_mapper.py``;
    what matters here is only that classification is invariant under it.
    """
    name = re.sub(r"(\.experts\.\d+\.w[123])\.scale$", r"\1.weight_scale", name)
    if name.endswith(".scale"):
        name = name[: -len(".scale")] + ".weight_scale_inv"
    name = name.replace(".shared_experts.w2", ".shared_experts.down_proj", 1)
    for prefix, new in (
        ("layers.", "model.layers."),
        ("embed.", "model.embed."),
        ("norm.", "model.norm."),
        ("hc_head", "model.hc_head"),
        ("mtp.", "model.mtp."),
    ):
        if name.startswith(prefix):
            name = name.replace(prefix, new, 1)
            break
    for suffix, new in (
        ("head.weight", "lm_head.weight"),
        ("embed.weight", "embed_tokens.weight"),
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
    ):
        if name.endswith(suffix):
            name = new.join(name.rsplit(suffix, 1))
            break
    return name


# ---------------------------------------------------------------------------
# The口径 guard: report pattern + report category, both spellings of dtype.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample,pattern,st_dtype,torch_dtype,category", SCAN_REPORT_ROWS, ids=ROW_IDS
)
def test_normalise_pattern_reproduces_report_pattern(
    sample, pattern, st_dtype, torch_dtype, category
):
    assert normalise_pattern(sample) == pattern


@pytest.mark.parametrize(
    "sample,pattern,st_dtype,torch_dtype,category", SCAN_REPORT_ROWS, ids=ROW_IDS
)
def test_classify_matches_report_with_safetensors_dtype(
    sample, pattern, st_dtype, torch_dtype, category
):
    assert classify(pattern, st_dtype) == category


@pytest.mark.parametrize(
    "sample,pattern,st_dtype,torch_dtype,category", SCAN_REPORT_ROWS, ids=ROW_IDS
)
def test_classify_matches_report_with_torch_dtype(
    sample, pattern, st_dtype, torch_dtype, category
):
    assert classify(pattern, torch_dtype) == category


@pytest.mark.parametrize(
    "sample,pattern,st_dtype,torch_dtype,category", SCAN_REPORT_ROWS, ids=ROW_IDS
)
def test_classification_survives_the_weights_mapper(
    sample, pattern, st_dtype, torch_dtype, category
):
    """What the loader actually passes: mapped name + torch dtype name."""
    assert classify_tensor(_mapped_name(sample), torch_dtype) == category


def test_every_category_has_a_representative_row():
    """A row table that lost a category would pass vacuously."""
    covered = {row[4] for row in SCAN_REPORT_ROWS}
    assert covered == set(CATEGORY_KEYS)


def test_mapper_actually_rewrites_the_scale_rows():
    """Guard for the test above: the mapped names must really differ."""
    assert (
        _mapped_name("layers.0.ffn.experts.0.w1.scale")
        == "model.layers.0.ffn.experts.0.w1.weight_scale"
    )
    assert (
        _mapped_name("layers.0.ffn.shared_experts.w1.scale")
        == "model.layers.0.ffn.shared_experts.w1.weight_scale_inv"
    )
    assert (
        _mapped_name("layers.0.ffn.shared_experts.w2.scale")
        == "model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv"
    )
    assert _mapped_name("head.weight") == "lm_head.weight"


# ---------------------------------------------------------------------------
# dtype spelling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        ("F8_E4M3", "F8_E4M3"),
        ("F8_E8M0", "F8_E8M0"),
        ("I8", "I8"),
        ("torch.float8_e4m3fn", "float8_e4m3fn"),
        ("torch.float8_e8m0fnu", "float8_e8m0fnu"),
        ("torch.uint8", "uint8"),
        ("float8_e4m3fn", "float8_e4m3fn"),
    ],
)
def test_dtype_name_accepts_both_spellings(given, expected):
    assert dtype_name(given) == expected


def test_dtype_name_accepts_real_torch_dtypes():
    torch = pytest.importorskip("torch")

    assert dtype_name(torch.float8_e4m3fn) == "float8_e4m3fn"
    assert dtype_name(torch.float8_e8m0fnu) == "float8_e8m0fnu"
    assert dtype_name(torch.uint8) == "uint8"
    assert dtype_name(torch.bfloat16) == "bfloat16"


def test_uint8_scale_is_still_a_scale():
    """MoE scales reach the parameter as ``uint8`` after the ``view``; a re-scan
    of a *loaded* model must not demote them to weights."""
    assert (
        classify_tensor("model.layers.0.ffn.experts.0.w1.weight_scale", "uint8")
        == "routed_expert_scale"
    )


# ---------------------------------------------------------------------------
# LoadStats: the log contract downstream parsers rely on.
# ---------------------------------------------------------------------------


def _record_rows(stats: LoadStats, rows=SCAN_REPORT_ROWS) -> None:
    for sample, _pattern, _st_dtype, torch_dtype, _category in rows:
        stats.record(_mapped_name(sample), torch_dtype)


def test_counts_add_up_to_the_total():
    stats = LoadStats()
    _record_rows(stats)

    assert stats.total == len(SCAN_REPORT_ROWS)
    assert sum(stats.counts.values()) == stats.total
    assert stats.unknown == 0
    assert stats.describe_unknown() is None


def test_log_lines_are_eight_parseable_key_values():
    stats = LoadStats()
    _record_rows(stats)

    lines = stats.log_lines()
    assert len(lines) == 8

    parsed = {}
    for line in lines:
        match = re.fullmatch(rf"\[{re.escape(STATS_LOG_TAG)}\] (\w+)=(\d+)", line)
        assert match is not None, line
        parsed[match.group(1)] = int(match.group(2))

    # Key spelling and order are frozen.
    assert list(parsed) == list(STATS_KEYS)
    assert parsed == stats.as_dict()
    assert parsed[TOTAL_KEY] == sum(parsed[key] for key in CATEGORY_KEYS)


def test_per_category_counts_match_the_recorded_rows():
    stats = LoadStats()
    _record_rows(stats)

    expected = {key: 0 for key in CATEGORY_KEYS}
    for row in SCAN_REPORT_ROWS:
        expected[row[4]] += 1

    assert {key: stats.counts[key] for key in CATEGORY_KEYS} == expected


def test_unclassifiable_tensor_is_counted_and_named():
    stats = LoadStats()
    # An unrecognised dtype is the only way into `unknown`: every dtype this
    # checkpoint uses is covered.
    stats.record("model.layers.0.attn.mystery", "F4_E2M1")

    assert stats.unknown == 1
    note = stats.describe_unknown()
    assert note is not None
    assert "model.layers.0.attn.mystery" in note
    # The total still counts it, so the seven printed categories stop summing to
    # the total -- a discrepancy the statistics consumer should flag.
    assert stats.total == 1
    assert sum(stats.as_dict()[key] for key in CATEGORY_KEYS) == 0


def test_expected_trunk_counts_are_baseline_table():
    """The baseline constant, pinned against the acceptance report."""
    assert EXPECTED_TRUNK_COUNTS == {
        "routed_expert_weight": 33024,
        "routed_expert_scale": 33024,
        "shared_expert_weight": 129,
        "shared_expert_scale": 129,
        "other_fp8_weight": 236,
        "other_fp8_scale": 236,
        "plain_tensor": 834,
        TOTAL_KEY: 67612,
    }
    assert (
        sum(EXPECTED_TRUNK_COUNTS[key] for key in CATEGORY_KEYS)
        == EXPECTED_TRUNK_COUNTS[TOTAL_KEY]
    )


# ---------------------------------------------------------------------------
# Port fidelity: the branch logic must stay byte-identical to the scanner's.
# ---------------------------------------------------------------------------

_SCANNER_RELATIVE = Path(
    "my_tests/deepseek/deepseek-v4/fp8原始权重适配/weight_inspect/scan_v4_weights.py"
)


def _find_scanner() -> Path | None:
    """Locate the scanner in the surrounding workspace, if it is checked out.

    It lives outside this repository (``vllm_plugin_dev/vllm-iluvatar``), so the
    guard is best-effort: on a machine that only has the plugin repo the test
    skips rather than failing.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _SCANNER_RELATIVE
        if candidate.is_file():
            return candidate
    return None


#: Stand-in for the one statement that is spelled differently on purpose: the
#: scanner inlines the "everything unquantised" dtype set, the port hoists it to
#: a module constant (checked separately by value).
_PLAIN_SET_TOKEN = "dtype in <PLAIN_DTYPES>"


def _statements(function_source_or_node) -> list[str]:
    """Normalised statement list of a function body, docstring excluded.

    Round-tripping through ``ast.unparse`` erases every formatting difference
    (line wrapping, quote style, spacing), so what is compared is the code
    itself rather than the layout the formatter happened to pick.
    """
    import ast
    import textwrap

    node = function_source_or_node
    if isinstance(node, str):
        node = ast.parse(textwrap.dedent(node)).body[0]

    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    statements = []
    for stmt in body:
        text = ast.unparse(stmt)
        text = re.sub(r"dtype in \{[^{}]*\}", _PLAIN_SET_TOKEN, text)
        text = text.replace("dtype in _PLAIN_DTYPE_NAMES", _PLAIN_SET_TOKEN)
        statements.append(text)
    return statements


def _scanner_toplevel(path: Path) -> dict[str, object]:
    """``name -> ast node`` for the scanner's top-level functions and assignments."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found[target.id] = node.value
    return found


@pytest.fixture
def scanner_toplevel():
    scanner = _find_scanner()
    if scanner is None:
        pytest.skip(f"scanner not checked out next to this repo: {_SCANNER_RELATIVE}")
    return _scanner_toplevel(scanner)


def test_classify_body_matches_the_scanner_source(scanner_toplevel):
    """The port is verbatim: statement for statement, branch for branch."""
    original = scanner_toplevel.get("classify")
    assert original is not None, "no classify() in the scanner source"

    assert _statements(inspect.getsource(classify)) == _statements(original)


def test_normalise_pattern_body_matches_the_scanner_source(scanner_toplevel):
    original = scanner_toplevel.get("normalise_pattern")
    assert original is not None, "no normalise_pattern() in the scanner source"

    assert _statements(inspect.getsource(normalise_pattern)) == _statements(original)


def test_dtype_sets_match_the_scanner_source(scanner_toplevel):
    """Including the hoisted plain-dtype set, compared by value."""
    import ast

    from vllm_iluvatar.model_executor.models.deepseek_v4 import fp8_load_stats

    for name in ("FP8_E4M3_NAMES", "FP8_E8M0_NAMES"):
        node = scanner_toplevel.get(name)
        assert node is not None, f"no {name} in the scanner source"
        assert getattr(fp8_load_stats, name) == ast.literal_eval(node)

    classify_node = scanner_toplevel["classify"]
    inlined = [
        ast.literal_eval(sub)
        for sub in ast.walk(classify_node)
        if isinstance(sub, ast.Set) and len(sub.elts) > 10
    ]
    assert len(inlined) == 1, "expected exactly one large inline dtype set"
    assert fp8_load_stats._PLAIN_DTYPE_NAMES == inlined[0]
