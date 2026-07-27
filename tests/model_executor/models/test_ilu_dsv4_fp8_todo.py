# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests (T9) for the Stage 1 ``TODO(ilu_fp8_*)`` numbering contract.

This file is a pure *source* check: it locates ``fp8_attention.py`` /
``fp8_model.py`` on disk and inspects them with ``ast`` + a regex over the raw
text. Nothing is imported from ``torch`` / ``vllm`` / ``vllm_iluvatar``, so the
test runs identically on a laptop and inside the Iluvatar container -- which is
the whole point: it is the cheapest possible regression fence around design §7.

Why ``ast`` rather than plain text matching for the two prohibitions
-------------------------------------------------------------------
Requirements 5.5 (no ``ixformer`` reference) and 5.8 (no torch custom op) are
about *code*, not about prose. Both source files carry long docstrings that
explain exactly why the vendor kernel package is not imported and why no op is
registered, and Stage 2 notes will keep mentioning both words. A text grep would
flag those explanations as violations, so imports / calls / attribute chains are
matched on the parse tree instead, where docstrings and comments simply are not
reachable.

The TODO numbers themselves are collected *both* ways on purpose:

* ``ast`` gives the authoritative set -- every number must sit in the message of
  a ``raise NotImplementedError(...)`` inside a known method (requirement 5.1);
* the text regex is the wider net -- if a number ever leaks into a comment, a
  log line or a second ``raise``, the two sets diverge and
  ``test_no_todo_number_outside_not_implemented_raise`` fails. That is what
  keeps "编号唯一" (requirement 5.3) honest, since ``ast`` alone would happily
  ignore a duplicate hidden in a comment.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.8
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
# Expected numbering, hard-coded from design §7.
# ---------------------------------------------------------------------------
# ``B2`` is part of the set: risk R1 was confirmed on device, so the
# self-written indexer shell (``IluDeepseekV4Fp8IndexerShell.forward``) exists
# and design §7 marks B2 as "已启用".
EXPECTED_NUMBERS = frozenset(
    {"A1", "A2", "A3", "A4", "B2", "C1", "E1", "E2", "E3", "E4", "E5"}
)

# Numbers design §7 reserves but Stage 1 must NOT spend:
#   B1 -- compressor forward. The community ``DeepseekCompressor`` is reused
#         verbatim (design §4.2), so there is no plugin-side wrapper to stub.
#   D1 -- FP8 block-scaled GEMM kernel, Stage 2 only.
# Asserting their absence makes an accidental new number fail loudly instead of
# silently widening Stage 1's scope.
FORBIDDEN_NUMBERS = frozenset({"B1", "D1"})

# number -> (source file, enclosing class, enclosing method), design §7 table.
EXPECTED_LOCATIONS = {
    "A1": ("fp8_attention.py", "IluDeepseekV4Fp8Attention", "forward"),
    "A2": ("fp8_attention.py", "IluDeepseekV4Fp8Attention", "forward_mqa"),
    "A3": ("fp8_attention.py", "IluDeepseekV4Fp8Attention", "_o_proj"),
    "A4": (
        "fp8_attention.py",
        "IluDeepseekV4Fp8Attention",
        "_fused_qnorm_rope_kv_insert",
    ),
    "B2": ("fp8_attention.py", "IluDeepseekV4Fp8IndexerShell", "forward"),
    # C1 is raised from a module-level helper, not inline in the MoE forward.
    # Three entry points share the group-C gap (design §7 / §4.4): the MoE
    # forward, the shared-expert MLP forward, and the activation placeholder that
    # replaced the community SwiGLU CustomOp -- Stage 1 constructs no activation
    # module because ``SiluAndMulWithClamp.__init__`` binds
    # ``torch.ops._C.silu_and_mul_with_clamp`` on any CUDA-like platform and
    # Iluvatar has no such symbol. Requirement 5.3 wants each number to exist
    # exactly once in the source, so the raise lives in the helper and the three
    # entries delegate; ``C1_DELEGATES`` below pins that they do.
    "C1": ("fp8_model.py", "", "_raise_ffn_not_wired"),
    "E1": ("fp8_model.py", "IluDeepseekV4Fp8ForCausalLM", "forward"),
    "E2": ("fp8_model.py", "IluDeepseekV4Fp8ForCausalLM", "compute_logits"),
    "E3": ("fp8_model.py", "IluDeepseekV4Fp8Model", "forward"),
    "E4": ("fp8_model.py", "IluDeepseekV4Fp8DecoderLayer", "forward"),
    "E5": ("fp8_model.py", "IluDeepseekV4Fp8Model", "hc_head"),
}

# The helper holding the single C1 raise, and every entry that must route to it.
C1_HELPER = "_raise_ffn_not_wired"
C1_DELEGATES = frozenset(
    {
        ("fp8_model.py", "IluDeepseekV4Fp8MoE", "forward"),
        ("fp8_model.py", "IluDeepseekV4Fp8MLP", "forward"),
        ("fp8_model.py", "_Fp8ActivationNotWired", "forward"),
    }
)

# Construction-time landmine fence (design §4.4 / risk R10): the community SwiGLU
# CustomOps resolve their kernel in ``__init__``
# (``activation.py:131-137`` / ``:186-195``) and the ``is_cuda_alike()`` branch has
# no fallback, so instantiating one on Iluvatar raises AttributeError before any
# weight exists. Stage 1 therefore constructs no activation module at all -- not
# even through the plugin's OOT override, which would bind an ixformer kernel.
FORBIDDEN_ACTIVATION_CLASSES = frozenset({"SiluAndMul", "SiluAndMulWithClamp"})

# Requirement 5.1: the exact message shape.
TODO_RE = re.compile(r"TODO\(ilu_fp8_([A-E]\d+)\)")

FORBIDDEN_VENDOR_ROOT = "ixformer"
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


def test_group_c_entries_delegate_to_the_single_raise_site():
    """Requirements 5.1 / 5.2 for C1, across the indirection.

    ``test_todo_number_sits_on_its_design_entry`` can only pin the helper, so this
    is what still guarantees that each design §7 group-C entry actually raises:
    all three call the helper, and nothing else does.
    """
    path = _source_path("fp8_model.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    callers = set()
    for node, cls_name, func_name in _iter_scoped_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == C1_HELPER:
            callers.add(("fp8_model.py", cls_name or "", func_name or ""))

    assert callers == set(C1_DELEGATES), (
        f"missing delegates: {sorted(C1_DELEGATES - callers)}; "
        f"unexpected callers of {C1_HELPER}(): {sorted(callers - C1_DELEGATES)}"
    )


@pytest.mark.parametrize("file_name", _SOURCE_FILES)
def test_no_community_activation_module_is_constructed(file_name):
    """Stage 1 builds no activation module (design §4.4).

    Imports *and* bare name uses are both rejected: the failure mode this guards
    is a construction-time ``torch.ops._C`` lookup, so merely referencing the
    class in code is enough to reintroduce it. Docstring prose that explains the
    ban is untouched -- only the parse tree is inspected.
    """
    path = _source_path(file_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_ACTIVATION_CLASSES:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_ACTIVATION_CLASSES:
            offenders.append(f"line {node.lineno}: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ACTIVATION_CLASSES:
                offenders.append(f"line {node.lineno}: .{node.attr}")

    assert not offenders, (
        f"{file_name} references a community activation CustomOp: {offenders}; "
        "these bind torch.ops._C in __init__ and have no fallback on Iluvatar"
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
def test_no_ixformer_reference(file_name):
    """Requirement 5.5: no vendor kernel import or attribute use.

    Docstrings that *talk about* ixformer are fine -- only ``Import`` /
    ``ImportFrom`` / attribute chains are inspected.
    """
    path = _source_path(file_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == FORBIDDEN_VENDOR_ROOT:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == FORBIDDEN_VENDOR_ROOT:
                offenders.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Attribute):
            if _root_name(node) == FORBIDDEN_VENDOR_ROOT:
                offenders.append(f"line {node.lineno}: {FORBIDDEN_VENDOR_ROOT} attr")

    assert not offenders, f"{file_name} references ixformer: {offenders}"


@pytest.mark.parametrize("file_name", _SOURCE_FILES)
def test_no_custom_op_registration(file_name):
    """Requirement 5.8: Stage 1 registers no torch custom op.

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
