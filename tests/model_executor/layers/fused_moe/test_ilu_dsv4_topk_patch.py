# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the ``can_use_dsv4_topk`` patch.

The patch closes the Triton ``dsv4_topk`` fast path on Iluvatar so
``fused_topk_bias`` falls through to ``vllm_topk_softplus_sqrt`` ->
``ops.topk_hash_softplus_sqrt``, which the plugin's ``vllm._custom_ops`` splat
points at ixformer.

Why the patch needs its own test file, separate from any MoE-layer test: the
mini DS-V4 config used elsewhere has ``topk == 2`` and 8 experts, so community
``can_use_dsv4_topk`` returns ``False`` for it anyway (``_TOPK == 6`` and
``shape[1] in (256, 384)``). A mini-MoE end-to-end test therefore cannot
distinguish "patched" from "never eligible". The tests below drive the predicate
with real-model-shaped arguments instead.

The rebind target is the **defining** module ``router/dsv4_topk.py``, because
``fused_topk_bias_router.py:17-20`` holds a ``from ... import`` copy and that
copy is the only call site that matters -- ``test_register_propagates_to_the_from_import_consumer``
is the assertion that actually protects production behaviour.

Pure CPU: no GPU, no vendor kernel library.
"""

from __future__ import annotations

import sys

import pytest
import torch

DEFINING_MODULE = "vllm.model_executor.layers.fused_moe.router.dsv4_topk"
CONSUMER_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
SYMBOL = "can_use_dsv4_topk"

IMPL_MODULE = "vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_dsv4_topk"
PATCH_MODULE = "vllm_iluvatar.ilu_patch.general_patches.patch_dsv4_topk"

FACTORY = "make_can_use_dsv4_topk"

# DeepSeek-V4-Flash router shapes: 256 routed experts, top-6, fp32 logits, fp32
# ``[256]`` correction bias, int32 indices, renormalize on. Community
# ``can_use_dsv4_topk`` answers True for exactly this combination, which is why
# the 40 bias layers would take the Triton branch unpatched.
_NUM_EXPERTS = 256
_TOPK = 6


def _defining_module():
    import importlib

    return importlib.import_module(DEFINING_MODULE)


def _consumer_module():
    import importlib

    return importlib.import_module(CONSUMER_MODULE)


def _impl_module():
    import importlib

    return importlib.import_module(IMPL_MODULE)


def _patch_module():
    import importlib

    return importlib.import_module(PATCH_MODULE)


def _real_model_args() -> tuple:
    return (
        torch.zeros(4, _NUM_EXPERTS, dtype=torch.float32),
        torch.zeros(_NUM_EXPERTS, dtype=torch.float32),
        _TOPK,
        True,
        torch.int32,
    )


def _symbol_holders(original) -> list:
    """Every imported module whose ``SYMBOL`` attribute *is* ``original``.

    ``rebind_module_func`` rebinds exactly this set, so it is also the set to
    restore afterwards.
    """
    return [
        m for m in list(sys.modules.values()) if getattr(m, SYMBOL, None) is original
    ]


@pytest.fixture()
def community_symbol():
    """Yield the pristine community predicate and restore every holder after."""
    defining = _defining_module()
    consumer = _consumer_module()

    original = getattr(defining, SYMBOL)
    holders = _symbol_holders(original)
    assert consumer in holders, (
        f"{CONSUMER_MODULE} no longer holds a from-import copy of {SYMBOL}; "
        "the patch's rebind target needs revisiting"
    )
    try:
        yield original
    finally:
        for mod in holders:
            setattr(mod, SYMBOL, original)


@pytest.fixture()
def unregistered_patch(community_symbol):
    """Make ``register()`` behave as if it had never run in this process."""
    from vllm_iluvatar.ilu_patch import general_patch

    was_applied = PATCH_MODULE in general_patch._APPLIED
    general_patch._APPLIED.discard(PATCH_MODULE)
    try:
        yield community_symbol
    finally:
        general_patch._APPLIED.discard(PATCH_MODULE)
        if was_applied:
            general_patch._APPLIED.add(PATCH_MODULE)


# ---------------------------------------------------------------------------
# the premise: community really would take the Triton branch on these shapes
# ---------------------------------------------------------------------------


def test_community_predicate_is_true_for_real_dsv4_bias_layers(community_symbol):
    """If this ever goes False the patch has become unnecessary -- and this test
    is the place that says so, instead of the patch silently doing nothing."""
    assert community_symbol(*_real_model_args()) is True


# ---------------------------------------------------------------------------
# wrapper: always False, never calls the community predicate
# ---------------------------------------------------------------------------


def test_wrapper_returns_false_without_calling_the_original():
    """``make_can_use_dsv4_topk`` unconditionally closes the Triton path."""

    def original(*args, **kwargs):
        raise AssertionError("community predicate must not run after the patch")

    wrapper = getattr(_impl_module(), FACTORY)(original)
    assert wrapper(*_real_model_args()) is False


# ---------------------------------------------------------------------------
# the rebind: both references point at the wrapper
# ---------------------------------------------------------------------------


def test_register_rebinds_the_defining_module(unregistered_patch):
    original = unregistered_patch
    _patch_module().register()

    patched = getattr(_defining_module(), SYMBOL)
    assert patched is not original, f"{DEFINING_MODULE}.{SYMBOL} was not rebound"
    assert getattr(patched, "__wrapped__", None) is original, (
        "wrapper must expose the community predicate via __wrapped__"
    )


def test_register_propagates_to_the_from_import_consumer(unregistered_patch):
    """The branch that matters: ``fused_topk_bias`` calls the *copy* in
    ``fused_topk_bias_router.py``, not the attribute on ``dsv4_topk``."""
    original = unregistered_patch
    _patch_module().register()

    defining_ref = getattr(_defining_module(), SYMBOL)
    consumer_ref = getattr(_consumer_module(), SYMBOL)

    assert consumer_ref is not original, (
        f"{CONSUMER_MODULE}.{SYMBOL} still holds the community predicate; "
        "rebind_module_func was not used"
    )
    assert consumer_ref is defining_ref, (
        "both references must be the same wrapper object, otherwise the two "
        "call sites can disagree"
    )


def test_patched_symbol_returns_false(unregistered_patch):
    """End-to-end through the rebound attribute on both holders."""
    _patch_module().register()

    for holder in (_defining_module(), _consumer_module()):
        assert getattr(holder, SYMBOL)(*_real_model_args()) is False


# ---------------------------------------------------------------------------
# idempotence
# ---------------------------------------------------------------------------


def test_register_twice_applies_only_once(unregistered_patch, monkeypatch):
    from vllm_iluvatar.ilu_patch import general_patch

    patch_mod = _patch_module()
    calls: list[int] = []
    real_apply = patch_mod._apply

    def counting_apply() -> None:
        calls.append(1)
        real_apply()

    monkeypatch.setattr(patch_mod, "_apply", counting_apply)

    patch_mod.register()
    after_first = getattr(_defining_module(), SYMBOL)
    patch_mod.register()
    after_second = getattr(_defining_module(), SYMBOL)

    assert len(calls) == 1, f"_apply ran {len(calls)} times, expected exactly 1"
    assert after_second is after_first, "second register() replaced the wrapper"
    assert PATCH_MODULE in general_patch._APPLIED


def test_repeated_register_does_not_stack_wrappers(unregistered_patch):
    original = unregistered_patch
    patch_mod = _patch_module()

    patch_mod.register()
    patch_mod.register()
    patch_mod.register()

    chain = [getattr(_defining_module(), SYMBOL)]
    while (inner := getattr(chain[-1], "__wrapped__", None)) is not None:
        chain.append(inner)

    assert chain[-1] is original
    assert len(chain) == 2, (
        "expected exactly one wrapper layer over the community predicate, got a "
        f"chain of {len(chain)}: {[getattr(f, '__qualname__', f) for f in chain]}"
    )


def test_wrapper_keeps_the_community_identity_metadata():
    import inspect

    impl = _impl_module()
    original = getattr(_defining_module(), SYMBOL)
    wrapper = getattr(impl, FACTORY)(original)

    assert wrapper.__name__ == SYMBOL
    # ``functools.wraps`` sets ``__wrapped__``, and ``inspect.signature``
    # follows it by default -- comparing the two that way is a tautology.
    # What matters is that the wrapper still *accepts* the community call, so
    # bind the community signature's parameters against the raw wrapper.
    assert wrapper.__wrapped__ is original
    raw = inspect.signature(wrapper, follow_wrapped=False)
    community = inspect.signature(original)
    raw.bind(
        *[
            object()
            for p in community.parameters.values()
            if p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
    )
    assert any(
        p.kind is p.VAR_POSITIONAL for p in raw.parameters.values()
    ), (
        "the wrapper reads none of its arguments, so it should take "
        "*args/**kwargs and survive upstream adding a parameter"
    )
