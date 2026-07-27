# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-side tensor classification + load statistics for the raw FP8 loader.

Spec ``deepseek-v4-fp8-weight-loading``, task 4.5 (design §4.5.6,
requirements 3.4-3.10).

Provenance -- do not "improve" the classifier
---------------------------------------------
``normalise_pattern`` and ``classify`` are **ported verbatim** from the offline
scanner that produced the acceptance baseline::

    my_tests/deepseek/deepseek-v4/fp8原始权重适配/weight_inspect/scan_v4_weights.py

The numbers the 8-card run is checked against (requirement 3.7: 33024 / 33024 /
129 / 129 / 236 / 236 / 834, total 67612) are that scanner's per-category counts.
Any change to a branch of ``classify`` -- reordering the routed/shared tests,
widening a dtype set, touching the suffix tests -- silently re-buckets tensors and
makes the comparison meaningless. **The branch logic must not be modified.**
``tests/model_executor/models/test_ilu_dsv4_fp8_load_stats.py`` pins the mapping
against ``weight_inspect/scan_report.md`` for exactly that reason.

The only deliberate difference from the scanner is the *input format*, and it is
additive rather than a change of logic:

* the scanner feeds safetensors dtype names (``F8_E4M3`` / ``F8_E8M0`` / ``I8``
  / ``BF16`` / ...), taken from ``safe_open(...).get_slice(k).get_dtype()``;
* the loader only has ``torch.dtype`` objects, whose names are
  ``float8_e4m3fn`` / ``float8_e8m0fnu`` / ``uint8`` / ``bfloat16`` / ...

Both spellings were already accepted by the scanner's dtype sets, so no branch of
``classify`` needed an edit; ``dtype_name()`` below just strips the ``torch.``
prefix off ``str(torch.float8_e4m3fn)`` before handing the string over. That also
keeps this module **torch-free**, so the unit test is plain Python.

The single cosmetic deviation is that the last dtype set was hoisted out of
``classify`` into ``_PLAIN_DTYPE_NAMES`` (same members) so the formatter has
nothing to reflow inside the ported function. The unit test compares the ported
function to the scanner's, statement by statement via ``ast.unparse``, and the
hoisted set by value -- so any drift on either side fails.

Which names get classified
--------------------------
Checkpoint-side names *after* the ``WeightsMapper`` and *before* the loader's
stacked / expert renaming, i.e. exactly what ``load_weights`` iterates. The
mapper only rewrites suffixes and lifts prefixes (``.scale`` ->
``.weight_scale[_inv]``, ``layers.`` -> ``model.layers.``, ``.shared_experts.w2``
-> ``.shared_experts.down_proj``), none of which crosses a category boundary:
``classify`` already tests all three scale suffixes, and the ``model.`` prefix is
inert for every branch. Post-fusion names would *not* work -- ``fused_wqa_wkv``
and ``w13_weight`` are many-to-one and would undercount.

Log format (contract for task 5.2's ``verify_load_stats.py``)
-------------------------------------------------------------
``LoadStats.log_lines()`` emits one ``key=value`` line per item, each tagged with
``[ilu_fp8_load_stats]`` so a script can grep them out of an 8-rank log::

    [ilu_fp8_load_stats] routed_expert_weight=33024
    [ilu_fp8_load_stats] routed_expert_scale=33024
    [ilu_fp8_load_stats] shared_expert_weight=129
    [ilu_fp8_load_stats] shared_expert_scale=129
    [ilu_fp8_load_stats] other_fp8_weight=236
    [ilu_fp8_load_stats] other_fp8_scale=236
    [ilu_fp8_load_stats] plain_tensor=834
    [ilu_fp8_load_stats] total_trunk_tensor=67612

Exactly eight keys, spelled as in ``STATS_KEYS``; the eighth is the total. The
key names are frozen -- task 5.2 parses them.

Counting口径 (requirements 3.8 / 3.9): these are **checkpoint-side trunk tensor
counts** -- how many ``mtp.``-free checkpoint tensors the loader consumed. They
are TP-rank independent (every rank iterates all of them and keeps its own
slice), so all 8 ranks must print identical numbers. They are *not* parameter
counts: stacked fusion and expert merging are many-to-one, so
``len(named_parameters())`` can never equal 67612 and the two must never be
compared.

``total_trunk_tensor`` is the number of recorded tensors across *all* buckets,
including ``unknown``. ``unknown`` has no ``key=value`` line (it must be 0 on
this checkpoint); if it is ever non-zero the total stops matching the sum of the
seven printed categories and ``describe_unknown()`` supplies a warning line
naming the offenders.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Ported verbatim from weight_inspect/scan_v4_weights.py -- see module docstring.
# ---------------------------------------------------------------------------

#: Category order, as in the scanner's ``CATEGORY_ORDER``.
CATEGORY_ORDER: tuple[str, ...] = (
    "routed_expert_weight",
    "routed_expert_scale",
    "shared_expert_weight",
    "shared_expert_scale",
    "other_fp8_weight",
    "other_fp8_scale",
    "plain_tensor",
    "unknown",
)

FP8_E4M3_NAMES = {"F8_E4M3", "F8_E4M3FN", "float8_e4m3fn"}
FP8_E8M0_NAMES = {"F8_E8M0", "F8_E8M0FNU", "float8_e8m0fnu"}

#: The scanner's final "everything unquantised" dtype set, hoisted out of
#: ``classify``'s body verbatim (same members, same order) so the formatter has
#: nothing to reflow inside the ported function. No branch changed.
_PLAIN_DTYPE_NAMES = {
    "BF16",
    "F16",
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
    "BOOL",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
}


def normalise_pattern(name: str) -> str:
    """Collapse a tensor name to its pattern by replacing every digit run with ``*``.

    Ported verbatim from ``scan_v4_weights.py``.
    """
    return re.sub(r"\d+", "*", name)


def classify(pattern: str, dtype: str) -> str:
    """Bucket a (pattern, dtype) pair into one of ``CATEGORY_ORDER``.

    Ported verbatim from ``scan_v4_weights.py``. **Do not change any branch.**
    """
    is_fp8_w = dtype in FP8_E4M3_NAMES
    is_fp8_s = dtype in FP8_E8M0_NAMES
    is_uint8 = dtype in {"U8", "uint8"}
    is_int8 = dtype in {"I8", "int8"}

    is_scale_name = (
        pattern.endswith("weight_scale")
        or pattern.endswith("weight_scale_inv")
        or pattern.endswith(".scale")
    )
    is_weight_name = pattern.endswith(".weight")

    is_routed = (
        "experts.*" in pattern or re.search(r"\.experts\.\*\.", pattern) is not None
    ) and "shared" not in pattern
    is_shared = "shared_experts" in pattern or "shared_expert" in pattern

    if is_routed:
        if is_scale_name or is_fp8_s or (is_uint8 and "scale" in pattern):
            return "routed_expert_scale"
        if is_weight_name or is_uint8 or is_int8 or is_fp8_w:
            return "routed_expert_weight"
    if is_shared:
        if is_scale_name or is_fp8_s:
            return "shared_expert_scale"
        if is_weight_name or is_fp8_w:
            return "shared_expert_weight"

    if is_fp8_w:
        return "other_fp8_weight"
    if is_fp8_s or (is_scale_name and (is_uint8 or is_int8)):
        return "other_fp8_scale"

    if dtype in _PLAIN_DTYPE_NAMES:
        return "plain_tensor"

    return "unknown"


# ---------------------------------------------------------------------------
# Loader-side adapters (additive: the classifier above stays untouched).
# ---------------------------------------------------------------------------


def dtype_name(dtype: object) -> str:
    """Normalise a dtype to the string spelling ``classify`` expects.

    Accepts either style, per task 4.5:

    * safetensors names, straight through: ``"F8_E4M3"``, ``"F8_E8M0"``, ``"I8"``;
    * torch names, with the ``torch.`` prefix stripped: a ``torch.dtype`` object
      or its ``str()`` (``"torch.float8_e4m3fn"`` -> ``"float8_e4m3fn"``).

    Deliberately does *not* import torch, so this module (and its unit test) stay
    runnable without a torch build.
    """
    text = dtype if isinstance(dtype, str) else str(dtype)
    prefix = "torch."
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text


def classify_tensor(name: str, dtype: object) -> str:
    """``classify(normalise_pattern(name), dtype_name(dtype))`` in one call."""
    return classify(normalise_pattern(name), dtype_name(dtype))


# ---------------------------------------------------------------------------
# Statistics accumulator + log contract.
# ---------------------------------------------------------------------------

#: Grep tag that prefixes every statistics line (task 5.2 parses on it).
STATS_LOG_TAG = "ilu_fp8_load_stats"

#: The seven per-category keys, in report order. Spelling is frozen.
CATEGORY_KEYS: tuple[str, ...] = CATEGORY_ORDER[:-1]

#: Key for the grand total (all buckets, ``unknown`` included).
TOTAL_KEY = "total_trunk_tensor"

#: The eight keys emitted as ``key=value``, in order.
STATS_KEYS: tuple[str, ...] = (*CATEGORY_KEYS, TOTAL_KEY)

#: Requirement 3.7's acceptance baseline, for reference by tests and scripts.
EXPECTED_TRUNK_COUNTS: dict[str, int] = {
    "routed_expert_weight": 33024,
    "routed_expert_scale": 33024,
    "shared_expert_weight": 129,
    "shared_expert_scale": 129,
    "other_fp8_weight": 236,
    "other_fp8_scale": 236,
    "plain_tensor": 834,
    TOTAL_KEY: 67612,
}


class LoadStats:
    """Per-category counter over the checkpoint tensors a load consumed.

    Only names and counts are kept -- never tensors -- so this adds no memory
    pressure on top of the weight stream.
    """

    #: How many offending names ``describe_unknown()`` spells out.
    max_reported_unknown = 20

    def __init__(self) -> None:
        self._counts: dict[str, int] = {key: 0 for key in CATEGORY_ORDER}
        self._unknown_names: list[str] = []

    def record(self, name: str, dtype: object) -> str:
        """Classify one consumed checkpoint tensor and count it."""
        category = classify_tensor(name, dtype)
        self._counts[category] += 1
        if category == "unknown":
            if len(self._unknown_names) < self.max_reported_unknown:
                self._unknown_names.append(name)
        return category

    def record_all(self, items: Iterable[tuple[str, object]]) -> None:
        for name, dtype in items:
            self.record(name, dtype)

    @property
    def counts(self) -> dict[str, int]:
        """Per-category counts, ``unknown`` included."""
        return dict(self._counts)

    @property
    def total(self) -> int:
        """Every recorded tensor, ``unknown`` included (see module docstring)."""
        return sum(self._counts.values())

    @property
    def unknown(self) -> int:
        return self._counts["unknown"]

    def as_dict(self) -> dict[str, int]:
        """The eight published ``key -> value`` pairs, in ``STATS_KEYS`` order."""
        published = {key: self._counts[key] for key in CATEGORY_KEYS}
        published[TOTAL_KEY] = self.total
        return published

    def log_lines(self) -> list[str]:
        """One tagged ``key=value`` line per published key (requirement 3.6)."""
        return [
            f"[{STATS_LOG_TAG}] {key}={value}" for key, value in self.as_dict().items()
        ]

    def describe_unknown(self) -> str | None:
        """A warning line naming unclassifiable tensors, or ``None`` if there are none.

        Intentionally *not* tagged with ``STATS_LOG_TAG``: it is prose, and the
        eight parseable keys must stay eight.
        """
        if not self.unknown:
            return None
        listed = ", ".join(self._unknown_names)
        return (
            f"{self.unknown} checkpoint tensor(s) could not be classified, so "
            f"{TOTAL_KEY} no longer equals the sum of the seven categories; "
            f"first {len(self._unknown_names)}: {listed}"
        )
