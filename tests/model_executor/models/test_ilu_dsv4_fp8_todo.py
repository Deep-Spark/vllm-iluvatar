# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests (T9) for the Stage 1 ``TODO(ilu_fp8_*)`` numbering contract.

This file is a pure *source* check: it locates ``fp8_attention.py`` /
``fp8_model.py`` on disk and inspects them with ``ast`` + a regex over the raw
text. Nothing is imported from ``torch`` / ``vllm`` / ``vllm_iluvatar``, so the
test runs identically on a laptop and inside the Iluvatar container -- which is
the whole point: it is the cheapest possible regression fence around design §7.

Why ``ast`` rather than plain text matching for the custom-op prohibition
------------------------------------------------------------------------
Requirement 5.8 (no torch custom op) is about *code*, not about prose: both
source files carry long docstrings explaining why no op is registered, and a text
grep would flag those explanations as violations. Imports / calls / attribute
chains are therefore matched on the parse tree, where docstrings and comments are
not reachable.

Requirement 5.5 (``fp8_model.py`` must not reference ``ixformer``) was a Stage-1
fence and is gone: module 5 put the MHC kernels in that file, exactly as the W4A8
fork does.

The TODO numbers themselves are collected *both* ways on purpose:

* ``ast`` gives the authoritative set -- every number must sit in the message of
  a ``raise NotImplementedError(...)`` inside a known method (requirement 5.1);
* the text regex is the wider net -- if a number ever leaks into a comment, a
  log line or a second ``raise``, the two sets diverge and
  ``test_no_todo_number_outside_not_implemented_raise`` fails. That is what
  keeps "编号唯一" (requirement 5.3) honest, since ``ast`` alone would happily
  ignore a duplicate hidden in a comment.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.8
"""

from __future__ import annotations

import ast
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Source location -- no absolute path is hard-coded.
# ---------------------------------------------------------------------------
_PKG_NAME = "vllm_iluvatar"
_REL_DIR = Path("model_executor") / "models" / "deepseek_v4"
_SOURCE_FILES = ("fp8_attention.py", "fp8_model.py")


def _package_root() -> Path:
    """Return the ``vllm_iluvatar`` package directory.

    Walking up from this test file covers the normal source checkout (the
    package sits next to ``tests/``). ``find_spec`` on the *top-level* package
    is the fallback for a site-packages install; it resolves through the path
    finders only and does not execute the package, so it stays torch-free.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _PKG_NAME
        if (candidate / _REL_DIR).is_dir():
            return candidate

    spec = importlib.util.find_spec(_PKG_NAME)
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if not locations:
        raise AssertionError(
            f"cannot locate the {_PKG_NAME} package from {__file__} nor via "
            "importlib.find_spec"
        )
    return Path(locations[0])


def _source_path(file_name: str) -> Path:
    path = _package_root() / _REL_DIR / file_name
    assert path.is_file(), f"missing Stage 1 source file: {path}"
    return path


# ---------------------------------------------------------------------------
# Expected numbering.
# ---------------------------------------------------------------------------
# After Module 4 (Stage 2 FFN/MoE) no ``TODO(ilu_fp8_*)`` stub remains in these
# two files: A1-A4 (attention forward / decode+prefill / o_proj / fused
# kv-insert) and B2 (lightning indexer forward) are wired via ixformer ``ds4_*``
# kernels (Module 3), E1-E5 (model orchestration) were wired in Module 1, and C1
# (gate / top-k / routed experts + shared-expert MLP) is wired in Module 4 as a
# torch-equivalent FFN.
EXPECTED_NUMBERS = frozenset()

# Numbers reserved but not spent:
#   B1 -- compressor forward (the ilu compressor forward is inherited/overridden
#         in the sibling W4A8 module, not stubbed here).
#   D1 -- FP8 block-scaled GEMM kernel, delivered in Module 2 (lives in
#         model_executor/kernels, not in these two model files).
# Asserting their absence makes an accidental new number fail loudly.
FORBIDDEN_NUMBERS = frozenset({"B1", "D1"})

# number -> (source file, enclosing class, enclosing method).
# Empty after Module 4: every Stage-1 number has graduated to a wired path.
EXPECTED_LOCATIONS: dict[str, tuple[str, str, str]] = {}

# Two Stage-1 fences graduated in Module 5 and are deliberately *not* replaced:
#
# * "no community activation CustomOp": design §4.4 / R10 assumed
#   ``SiluAndMulWithClamp.__init__`` would die looking up ``torch.ops._C`` on
#   Iluvatar. The plugin's OOT ``IluSiluAndMulWithClamp`` binds
#   ``vllm._custom_ops`` instead and the W4A8 fork ships it, so the shared-expert
#   MLP now builds it like every other model.
# * "``fp8_model.py`` is ixformer-free": the MHC ops are ixformer kernels called
#   from this file (as in the W4A8 fork), so the vendor import is expected here
#   exactly like it already was in ``fp8_attention.py``.

# Requirement 5.1: the exact message shape.
TODO_RE = re.compile(r"TODO\(ilu_fp8_([A-E]\d+)\)")

FORBIDDEN_OP_HELPER = "direct_register_custom_op"
# Requirement 5.8 -- the low-level registration entry points, in case someone
# bypasses vLLM's helper.
FORBIDDEN_TORCH_LIBRARY_CALLS = frozenset({"define", "impl", "custom_op"})


# ---------------------------------------------------------------------------
# Collectors.
# ---------------------------------------------------------------------------
def _iter_scoped_nodes(node, cls_name=None, func_name=None):
    """Yield ``(node, enclosing class, enclosing function)`` for the whole tree.

    Only the innermost class / function is tracked, which is all design §7 needs
    (every stub lives directly in a method of a top-level class).
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            next_cls, next_func = child.name, func_name
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            next_cls, next_func = cls_name, child.name
        else:
            next_cls, next_func = cls_name, func_name
        yield child, cls_name, func_name
        yield from _iter_scoped_nodes(child, next_cls, next_func)


def _raise_message(node: ast.Raise) -> str | None:
    """Return the message of ``raise NotImplementedError("...")``, else None.

    Implicitly concatenated literals (all Stage 1 stubs wrap their message over
    several lines) are folded by the parser into a single ``ast.Constant``, so no
    manual joining is needed.
    """
    exc = node.exc
    if not isinstance(exc, ast.Call):
        return None
    func = exc.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if name != "NotImplementedError" or not exc.args:
        return None
    first = exc.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _collect_stubs(file_name: str) -> list[tuple[str, str, str, str]]:
    """``(number, file, class, method)`` for every NotImplementedError stub."""
    path = _source_path(file_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    stubs: list[tuple[str, str, str, str]] = []
    for node, cls_name, func_name in _iter_scoped_nodes(tree):
        if not isinstance(node, ast.Raise):
            continue
        message = _raise_message(node)
        if message is None:
            continue
        match = TODO_RE.search(message)
        assert match is not None, (
            f"{file_name}:{node.lineno} raises NotImplementedError without a "
            f"TODO(ilu_fp8_XX) number: {message!r}"
        )
        assert message.startswith(match.group(0) + ":"), (
            f"{file_name}:{node.lineno} message must start with "
            f"'{match.group(0)}: ', got {message!r}"
        )
        stubs.append((match.group(1), file_name, cls_name or "", func_name or ""))
    return stubs


def _all_stubs() -> list[tuple[str, str, str, str]]:
    return [stub for name in _SOURCE_FILES for stub in _collect_stubs(name)]


def _root_name(node) -> str | None:
    """Walk an attribute chain down to its root ``Name`` (``a.b.c`` -> ``a``)."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_todo_numbers_are_unique():
    """Requirement 5.3: one number, one place -- across both source files."""
    numbers = [stub[0] for stub in _all_stubs()]
    duplicates = {num: cnt for num, cnt in Counter(numbers).items() if cnt > 1}
    assert not duplicates, f"TODO numbers used more than once: {duplicates}"


def test_todo_number_set_matches_design_section_7():
    """Requirement 5.2: exactly the design §7 Stage 1 entries, nothing else."""
    found = {stub[0] for stub in _all_stubs()}
    assert found == set(EXPECTED_NUMBERS), (
        f"missing: {sorted(EXPECTED_NUMBERS - found)}; "
        f"unexpected: {sorted(found - EXPECTED_NUMBERS)}"
    )


@pytest.mark.parametrize("number", sorted(FORBIDDEN_NUMBERS))
def test_reserved_numbers_are_not_used_in_stage_1(number):
    """B1 (compressor reused as-is) and D1 (Stage 2 kernel) stay unspent."""
    found = {stub[0] for stub in _all_stubs()}
    assert number not in found, (
        f"TODO(ilu_fp8_{number}) must not appear in Stage 1 sources: design §7 "
        "keeps B1 unused (community compressor is reused verbatim) and D1 for "
        "Stage 2's FP8 block GEMM kernel"
    )


@pytest.mark.parametrize("number", sorted(EXPECTED_LOCATIONS))
def test_todo_number_sits_on_its_design_entry(number):
    """Requirements 5.1 / 5.4: each number guards the entry design §7 assigns."""
    by_number = {stub[0]: stub[1:] for stub in _all_stubs()}
    assert by_number.get(number) == EXPECTED_LOCATIONS[number], (
        f"TODO(ilu_fp8_{number}) expected at {EXPECTED_LOCATIONS[number]}, "
        f"found at {by_number.get(number)}"
    )


@pytest.mark.parametrize("file_name", _SOURCE_FILES)
def test_no_todo_number_outside_not_implemented_raise(file_name):
    """Every textual occurrence must be a stub message, not a comment or log.

    Guards the uniqueness check above: a number parked in a comment would be
    invisible to the ``ast`` pass yet would show up in a ``grep`` audit.
    """
    text = _source_path(file_name).read_text(encoding="utf-8")
    text_numbers = Counter(TODO_RE.findall(text))
    ast_numbers = Counter(stub[0] for stub in _collect_stubs(file_name))
    assert text_numbers == ast_numbers, (
        "TODO(ilu_fp8_*) occurrences outside a raise NotImplementedError "
        f"message in {file_name}: {text_numbers - ast_numbers}"
    )


@pytest.mark.parametrize("file_name", _SOURCE_FILES)
def test_no_custom_op_registration(file_name):
    """Requirement 5.8: this model registers no torch custom op.

    ``direct_register_custom_op`` is checked as an import, as a call and as a
    bare reference; ``torch.library.{define,impl,custom_op}`` covers the route
    around vLLM's helper. Prose mentions in the docstrings are unaffected.
    """
    path = _source_path(file_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == FORBIDDEN_OP_HELPER:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Name) and node.id == FORBIDDEN_OP_HELPER:
            offenders.append(f"line {node.lineno}: {FORBIDDEN_OP_HELPER}")
        elif isinstance(node, ast.Attribute):
            if node.attr == FORBIDDEN_OP_HELPER:
                offenders.append(f"line {node.lineno}: .{FORBIDDEN_OP_HELPER}")
            elif node.attr in FORBIDDEN_TORCH_LIBRARY_CALLS and isinstance(
                node.value, ast.Attribute
            ):
                # torch.library.define / .impl / .custom_op
                if node.value.attr == "library" and _root_name(node.value) == "torch":
                    offenders.append(f"line {node.lineno}: torch.library.{node.attr}")

    assert not offenders, (
        f"{file_name} registers a torch custom op: {offenders}"
    )
