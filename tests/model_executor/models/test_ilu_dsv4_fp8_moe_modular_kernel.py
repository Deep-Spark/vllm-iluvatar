# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the P3 modular-kernel-init skip patch.

Design §6.3 / §9.2: with P1 pinning the MXFP4 MoE backend to
``Mxfp4MoeBackend.NONE``, ``Mxfp4MoEMethod.process_weights_after_loading``
(``quantization/mxfp4.py:735``) early-returns, ``_setup_kernel`` never runs, and
both ``moe_kernel`` and ``experts_cls`` stay ``None``. Community
``MoERunner.maybe_init_modular_kernel`` (``moe_runner.py:852-857``) then asks
``quant_method.is_monolithic``, and community
``FusedMoEMethodBase.is_monolithic`` (``fused_moe_method_base.py:163-170``)
guards only with ``hasattr(self, "experts_cls")`` — the attribute exists and is
``None`` — so it dereferences ``None.is_monolithic()``.

That call site is reached **only under expert parallelism**:
``BaseDeviceCommunicator.prepare_communication_buffer_for_model``
(``base_device_communicator.py:331-339``) returns immediately unless
``is_ep_communicator``. Hence the 8-card TP8+EP8 run is where it first surfaced,
on every rank, immediately after the weights had loaded.

P3 is a **wrapper** patch on the class method: on Iluvatar, when the routed
quant method is an unwired MXFP4 method (backend ``NONE`` + ``experts_cls is
None`` + ``moe_kernel is None``), build nothing and return; otherwise delegate.
Skipping is the Stage 1 semantics, not a workaround — a modular kernel is a
compute kernel and Stage 1 wires none (requirement 5.5, design §2 C2).

What this file pins
-------------------

* the patch is idempotent (``register_once`` keyed on the module name) and does
  not stack wrappers;
* the wrapper is installed on the class that *defines* the method, resolved via
  the MRO, so it is independent of whether ``patch_fused_moe_layer`` has already
  rebound the ``moe_runner.MoERunner`` module global to ``IluMoERunner``;
* under backend ``NONE`` the call is a no-op: no raise, no modular method swap,
  no MoE quant config, no kernel;
* every other configuration still runs the community body: a real backend, a
  quant method without ``mxfp4_backend``, a method that already owns a
  ``moe_kernel``, and any non-Iluvatar platform;
* the community implementation really does crash on the NONE backend (so the
  patch is not guarding against a hypothetical).

Pure CPU: no GPU, no vendor kernel library. One test builds a real mini
``IluDeepseekV4Fp8MoE`` on CPU (~1 MB) to exercise the true community objects
end to end; the branch matrix uses stubs that borrow the community property
objects, so it cannot drift away from community semantics either.

Validates: Requirements 6.4, 6.5
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

# Reused from the P1 test module: the ``community_symbol`` / ``unregistered_patch``
# fixture chain that installs + restores P1, the mini-MoE builder, and the
# session-scoped TP1 distributed env (imported for fixture injection).
from test_ilu_dsv4_fp8_moe_patch import (  # noqa: F401  (fixture injection)
    BIAS_LAYER_ID,
    PLATFORM_GATE,
    _build_mini_moe,
    _impl_module,
    _patch_module,
    _routed_experts,
    _tp1_distributed_env,
    community_symbol,
    unregistered_patch,
)

RUNNER_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
P3_MODULE = "vllm_iluvatar.ilu_patch.general_patches.patch_dsv4_moe_modular_kernel"

METHOD = "maybe_init_modular_kernel"
P3_PLATFORM_GATE = "_is_iluvatar_platform"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _p3_module():
    return importlib.import_module(P3_MODULE)


def _runner_class():
    """The community ``MoERunner``, i.e. the class that defines the method.

    Read through the MRO for the same reason the patch does: after
    ``patch_fused_moe_layer`` the module global is ``IluMoERunner``.
    """
    module = importlib.import_module(RUNNER_MODULE)
    return _p3_module()._defining_class(getattr(module, "MoERunner"), METHOD)


def _installed_method():
    return vars(_runner_class())[METHOD]


def _backend(name: str):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return getattr(Mxfp4MoeBackend, name)


def _stub_mxfp4_method(*, backend=None, experts_cls=None, moe_kernel=None):
    """Stand-in for ``Mxfp4MoEMethod`` carrying the community properties.

    ``is_monolithic`` / ``supports_internal_mk`` are the community property
    objects themselves, so the stub reproduces community semantics exactly --
    including the ``experts_cls is None`` dereference this patch exists for.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    class _StubMxfp4Method:
        is_monolithic = FusedMoEMethodBase.is_monolithic
        supports_internal_mk = FusedMoEMethodBase.supports_internal_mk

    stub = _StubMxfp4Method()
    if backend is not None:
        stub.mxfp4_backend = backend
    stub.experts_cls = experts_cls
    stub.moe_kernel = moe_kernel
    return stub


def _stub_unquantized_method():
    """A quant method with no ``mxfp4_backend`` attribute at all."""
    return SimpleNamespace(
        moe_kernel=None, supports_internal_mk=False, is_monolithic=False
    )


def _stub_runner(quant_method):
    """Minimal ``MoERunner`` stand-in: only the attribute path the guard reads."""
    return SimpleNamespace(routed_experts=SimpleNamespace(quant_method=quant_method))


def _unwired_runner():
    """Exactly the 8-card state: backend NONE, no experts class, no kernel."""
    return _stub_runner(_stub_mxfp4_method(backend=_backend("NONE")))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def p3_unregistered():
    """Restore the class method and ``_APPLIED`` so ``register()`` can be tested."""
    from vllm_iluvatar.ilu_patch import general_patch

    target_cls = _runner_class()
    original = vars(target_cls)[METHOD]
    was_applied = P3_MODULE in general_patch._APPLIED
    general_patch._APPLIED.discard(P3_MODULE)
    try:
        yield original
    finally:
        setattr(target_cls, METHOD, original)
        general_patch._APPLIED.discard(P3_MODULE)
        if was_applied:
            general_patch._APPLIED.add(P3_MODULE)


@pytest.fixture()
def p3_on_iluvatar(p3_unregistered, monkeypatch):
    """Apply P3 and force the Iluvatar branch; yields the community method."""
    original = p3_unregistered
    _p3_module().register()
    monkeypatch.setattr(_p3_module(), P3_PLATFORM_GATE, lambda: True)
    yield original


# ---------------------------------------------------------------------------
# the bug itself: community code crashes on the NONE backend
# ---------------------------------------------------------------------------


def test_community_method_crashes_on_the_unwired_none_backend(p3_unregistered):
    """The 8-card traceback, reproduced without a GPU.

    If this ever stops raising, community ``is_monolithic`` grew a ``None``
    guard and P3 can be dropped -- so the failure message points there.
    """
    community = p3_unregistered

    with pytest.raises(AttributeError) as excinfo:
        community(_unwired_runner())

    assert "is_monolithic" in str(excinfo.value), (
        "expected the community experts_cls dereference to fail; if community "
        "fused_moe_method_base.is_monolithic now handles experts_cls is None, "
        "patch P3 is obsolete"
    )


# ---------------------------------------------------------------------------
# Requirement 6.4: idempotence
# ---------------------------------------------------------------------------


def test_register_twice_applies_only_once(p3_unregistered, monkeypatch):
    from vllm_iluvatar.ilu_patch import general_patch

    patch_mod = _p3_module()
    calls: list[int] = []
    real_apply = patch_mod._apply

    def counting_apply() -> None:
        calls.append(1)
        real_apply()

    # register() resolves ``_apply`` as a module global at call time.
    monkeypatch.setattr(patch_mod, "_apply", counting_apply)

    patch_mod.register()
    after_first = _installed_method()
    patch_mod.register()
    after_second = _installed_method()

    assert len(calls) == 1, f"_apply ran {len(calls)} times, expected exactly 1"
    assert after_second is after_first, "second register() replaced the wrapper"
    assert P3_MODULE in general_patch._APPLIED


def test_repeated_register_does_not_stack_wrappers(p3_unregistered):
    original = p3_unregistered
    patch_mod = _p3_module()

    patch_mod.register()
    patch_mod.register()
    patch_mod.register()

    installed = _installed_method()
    assert installed is not original
    assert getattr(installed, "__wrapped__", None) is original, (
        "the wrapper must expose the community method via __wrapped__ and there "
        "must be exactly one layer of wrapping"
    )


def test_wrapper_lands_on_the_defining_class(p3_unregistered):
    """Not on the module global: ``patch_fused_moe_layer`` may have rebound it to
    ``IluMoERunner``, which inherits this method rather than defining it."""
    _p3_module().register()

    target_cls = _runner_class()
    assert target_cls.__name__ == "MoERunner"
    assert METHOD in vars(target_cls)
    # Subclasses (IluMoERunner) pick it up through inheritance.
    subclass = type("_Sub", (target_cls,), {})
    assert getattr(subclass, METHOD) is getattr(target_cls, METHOD)


# ---------------------------------------------------------------------------
# NONE backend: early return, nothing built
# ---------------------------------------------------------------------------


def test_none_backend_returns_without_running_the_community_body(p3_on_iluvatar):
    patch_mod = _p3_module()

    def community(_self):
        raise AssertionError("community body must not run on the NONE backend")

    wrapper = patch_mod.make_maybe_init_modular_kernel(community)

    assert wrapper(_unwired_runner()) is None


def test_patched_class_method_is_a_no_op_on_the_none_backend(p3_on_iluvatar):
    """Through the installed attribute, not the factory."""
    runner = _unwired_runner()
    quant_method = runner.routed_experts.quant_method

    assert _installed_method()(runner) is None
    assert quant_method.moe_kernel is None
    assert runner.routed_experts.quant_method is quant_method


# ---------------------------------------------------------------------------
# every other configuration keeps community behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quant_method_factory",
    [
        pytest.param(
            lambda: _stub_mxfp4_method(backend=_backend("TRITON"), experts_cls=object),
            id="real-backend",
        ),
        pytest.param(
            lambda: _stub_mxfp4_method(backend=_backend("NONE"), moe_kernel=object()),
            id="already-owns-a-kernel",
        ),
        pytest.param(_stub_unquantized_method, id="not-an-mxfp4-method"),
    ],
)
def test_other_quant_methods_delegate_to_community(
    p3_on_iluvatar, quant_method_factory
):
    patch_mod = _p3_module()
    seen: list[object] = []

    def community(self):
        seen.append(self)

    wrapper = patch_mod.make_maybe_init_modular_kernel(community)
    runner = _stub_runner(quant_method_factory())

    wrapper(runner)

    assert seen == [runner], "community body must run for anything but backend NONE"


def test_non_iluvatar_platform_delegates_even_on_backend_none(
    p3_unregistered, monkeypatch
):
    patch_mod = _p3_module()
    monkeypatch.setattr(patch_mod, P3_PLATFORM_GATE, lambda: False)

    seen: list[object] = []

    def community(self):
        seen.append(self)

    wrapper = patch_mod.make_maybe_init_modular_kernel(community)
    runner = _unwired_runner()

    wrapper(runner)

    assert seen == [runner], (
        "off Iluvatar the community body must run unchanged, backend NONE or not"
    )


def test_platform_gate_is_a_zero_arg_predicate():
    gate = getattr(_p3_module(), P3_PLATFORM_GATE)
    assert isinstance(gate(), bool)


# ---------------------------------------------------------------------------
# end to end on real community objects (the EP call site, minus the GPU)
# ---------------------------------------------------------------------------


def test_real_mini_moe_survives_maybe_init_modular_kernel(
    unregistered_patch, p3_unregistered, monkeypatch
):
    """The actual 8-card sequence on one CPU rank.

    P1 + P3 applied, mini MoE built, ``process_weights_after_loading`` run (the
    NONE early-return), then the EP-only call that used to raise. Nothing may be
    built: no kernel, no MoE quant config (which is what proves the community
    body was skipped rather than merely surviving), and the quant method object
    must not be swapped for ``FusedMoEModularMethod``.
    """
    _patch_module().register()  # P1
    _p3_module().register()  # P3
    monkeypatch.setattr(_impl_module(), PLATFORM_GATE, lambda: True)
    monkeypatch.setattr(_p3_module(), P3_PLATFORM_GATE, lambda: True)

    with _build_mini_moe(BIAS_LAYER_ID) as (moe, _):
        experts = _routed_experts(moe)
        quant_method = experts.quant_method
        quant_method.process_weights_after_loading(experts)

        moe.experts.maybe_init_modular_kernel()

        assert experts.quant_method is quant_method, (
            "quant method was replaced -- a modular kernel method was installed"
        )
        assert quant_method.moe_kernel is None
        assert quant_method.moe_quant_config is None, (
            "moe_quant_config was built, so the community body ran instead of "
            "the P3 early return"
        )
