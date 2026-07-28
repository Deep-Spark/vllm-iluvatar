# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests (T8) for the P1 MXFP4-backend-selection patch.

Design §4.4 / §6.3: community
``vllm.model_executor.layers.fused_moe.oracle.mxfp4.select_deepseek_v4_mxfp4_moe_backend``
(``oracle/mxfp4.py:566``) can never return ``Mxfp4MoeBackend.NONE`` off-ROCm —
``_get_priority_backends()`` yields TRTLLM / DEEPGEMM / MARLIN / BATCHED_MARLIN and,
when none of them supports the config, the function ends in
``raise NotImplementedError("No MXFP4 MoE backend supports the deployment
configuration.")``. Stage 1 wants ``NONE`` precisely because
``Mxfp4MoEMethod.process_weights_after_loading`` early-returns on it, leaving the
four routed MXFP4 params byte-identical to the checkpoint.

So P1 is a **wrapper** patch: on the Iluvatar platform return
``(Mxfp4MoeBackend.NONE, None)``, otherwise delegate to the community function.
Because ``quantization/mxfp4.py:21-27`` is a ``from ... import
select_deepseek_v4_mxfp4_moe_backend`` consumer, patching only the defining
module is a no-op for the code path that actually calls it
(``Mxfp4MoEMethod.__init__``, ``quantization/mxfp4.py:484``) — hence
``general_patch.rebind_module_func``.

Contract this file pins for Task 3.3 (implementation)
-----------------------------------------------------

``vllm_iluvatar/model_executor/layers/fused_moe/oracle/ilu_mxfp4.py``:

* ``make_select_deepseek_v4_mxfp4_moe_backend(orig) -> wrapper`` — factory taking
  the community function and returning a ``functools.wraps(orig)``-decorated
  wrapper (mirrors ``vllm_iluvatar/config/speculative.py::make_hf_config_override``).
* ``_is_iluvatar_platform() -> bool`` — module-level platform gate, read as a
  **module global at call time** (so tests / other callers can monkeypatch it,
  mirroring ``distributed/parallel_state.py::_is_ivcore30_fn``). Its production
  body is expected to be ``current_platform.device_name == "iluvatar"``:
  ``IluPlatform`` (``platforms/ilu_platform.py:115``) sets that class attribute and
  it stays a cheap string compare — ``isinstance(current_platform, IluPlatform)``
  would need importing ``platforms/ilu_platform.py``, which builds the platform
  class as an import side effect. ``is_cuda()`` is unusable as a gate because
  ``IluPlatform`` subclasses ``CudaPlatform`` (design §4.2 makes the same point).

``vllm_iluvatar/ilu_patch/general_patches/patch_dsv4_mxfp4_backend.py``:

* ``register()`` delegating to ``general_patch.register_once(__name__, _apply, ...)``
* ``_apply()`` doing the ``rebind_module_func`` on the oracle module.

Test-hygiene notes:

* Plugin imports are deferred into helpers/fixtures (``importlib.import_module``),
  as in the 1.x / 2.x test files, so collection of this file does not blow up
  before Task 3.3 lands — the tests fail individually with ``ModuleNotFoundError``
  instead.
* The patch mutates community module globals, so every test that applies it goes
  through the ``unregistered_patch`` fixture, which snapshots *all* holders of the
  original symbol (``rebind_module_func`` walks ``sys.modules``) and writes them
  back afterwards, and also clears this patch's entry from
  ``general_patch._APPLIED``. Without that, one test applying the patch would
  leak into every later test in the same pytest process.
* Pure CPU: no GPU, no vendor kernel library, no model construction.

Validates: Requirements 6.4, 6.5
"""

from __future__ import annotations

import contextlib
import importlib
import sys

import pytest
import torch

# Task 3.4 only. Reused from task 1.2: ``_tp1_distributed_env`` is a
# session-scoped autouse fixture and importing it here is what makes it apply to
# this module's tests too (parallel linear layers read the TP world size).
# ``make_mini_vllm_config`` carries the mini DS-V4 hf_config plus the
# ``DeepseekV4FP8Config`` that routes the experts to ``Mxfp4MoEMethod``.
from test_ilu_dsv4_fp8_attention import (  # noqa: F401  (fixture injection)
    BF16,
    E8M0,
    FP8,
    FP32,
    MINI_CONFIG_FIELDS,
    MINI_HIDDEN_SIZE,
    MINI_TP_SIZE,
    MINI_VOCAB_SIZE,
    _tp1_distributed_env,
    make_mini_vllm_config,
)

# ---------------------------------------------------------------------------
# module / symbol names under test
# ---------------------------------------------------------------------------

ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
QUANT_MODULE = "vllm.model_executor.layers.quantization.mxfp4"
SYMBOL = "select_deepseek_v4_mxfp4_moe_backend"

ILU_IMPL_MODULE = "vllm_iluvatar.model_executor.layers.fused_moe.oracle.ilu_mxfp4"
PATCH_MODULE = "vllm_iluvatar.ilu_patch.general_patches.patch_dsv4_mxfp4_backend"

FACTORY = "make_select_deepseek_v4_mxfp4_moe_backend"
PLATFORM_GATE = "_is_iluvatar_platform"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _oracle_module():
    return importlib.import_module(ORACLE_MODULE)


def _quant_module():
    return importlib.import_module(QUANT_MODULE)


def _backend_none():
    return _oracle_module().Mxfp4MoeBackend.NONE


def _impl_module():
    """Import the P1 implementation module (lands in Task 3.3)."""
    return importlib.import_module(ILU_IMPL_MODULE)


def _patch_module():
    """Import the P1 patch wiring module (lands in Task 3.3)."""
    return importlib.import_module(PATCH_MODULE)


def _symbol_holders(original) -> list:
    """Every imported module whose ``SYMBOL`` attribute *is* ``original``.

    ``rebind_module_func`` rebinds exactly this set (identity match inside the
    ``vllm`` top-level package), so it is also the set to restore.
    """
    return [
        m
        for m in list(sys.modules.values())
        if getattr(m, SYMBOL, None) is original
    ]


def _wrapper_chain(func) -> list:
    """``func`` plus its ``__wrapped__`` ancestors (``functools.wraps`` chain)."""
    chain = [func]
    while (inner := getattr(chain[-1], "__wrapped__", None)) is not None:
        chain.append(inner)
    return chain


class _SentinelConfig:
    """Stand-in for ``FusedMoEConfig``.

    The wrapper must not introspect the config on the Iluvatar branch, and must
    forward it untouched on the delegation branch, so an opaque object is enough.
    """


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def community_symbol():
    """Yield the pristine community function and restore every holder after."""
    oracle_mod = _oracle_module()
    _quant_module()  # ensure the from-import consumer is in sys.modules

    original = getattr(oracle_mod, SYMBOL)
    holders = _symbol_holders(original)
    # Sanity: the consumer that actually calls it must be part of the snapshot,
    # otherwise this test file would silently stop covering Requirement 6.3/6.5.
    assert _quant_module() in holders, (
        f"{QUANT_MODULE} no longer holds a from-import copy of {SYMBOL}; "
        "update this test and design §6.3"
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
# Requirement 6.5 / 6.3: both references point at the wrapper
# ---------------------------------------------------------------------------


def test_register_rebinds_the_defining_module(unregistered_patch):
    original = unregistered_patch
    _patch_module().register()

    patched = getattr(_oracle_module(), SYMBOL)
    assert patched is not original, f"{ORACLE_MODULE}.{SYMBOL} was not rebound"
    assert getattr(patched, "__wrapped__", None) is original, (
        "wrapper must expose the community function via __wrapped__ "
        "(functools.wraps(orig))"
    )


def test_register_propagates_to_the_from_import_consumer(unregistered_patch):
    """The branch that matters: ``Mxfp4MoEMethod.__init__`` calls the *copy* in
    ``quantization/mxfp4.py``, not the attribute on the oracle module."""
    original = unregistered_patch
    _patch_module().register()

    oracle_ref = getattr(_oracle_module(), SYMBOL)
    quant_ref = getattr(_quant_module(), SYMBOL)

    assert quant_ref is not original, (
        f"{QUANT_MODULE}.{SYMBOL} still holds the community function; "
        "rebind_module_func was not used"
    )
    assert quant_ref is oracle_ref, (
        "both references must be the same wrapper object, otherwise the two "
        "call sites can disagree"
    )


# ---------------------------------------------------------------------------
# Requirement 6.4: idempotence
# ---------------------------------------------------------------------------


def test_register_twice_applies_only_once(unregistered_patch, monkeypatch):
    """``register_once`` keys on the module name in ``general_patch._APPLIED``;
    the second call must be a no-op, not a second wrap."""
    from vllm_iluvatar.ilu_patch import general_patch

    patch_mod = _patch_module()
    calls: list[int] = []
    real_apply = patch_mod._apply

    def counting_apply() -> None:
        calls.append(1)
        real_apply()

    # register() resolves ``_apply`` as a module global at call time.
    monkeypatch.setattr(patch_mod, "_apply", counting_apply)

    patch_mod.register()
    after_first = getattr(_oracle_module(), SYMBOL)
    patch_mod.register()
    after_second = getattr(_oracle_module(), SYMBOL)

    assert len(calls) == 1, f"_apply ran {len(calls)} times, expected exactly 1"
    assert after_second is after_first, "second register() replaced the wrapper"
    assert PATCH_MODULE in general_patch._APPLIED


def test_repeated_register_does_not_stack_wrappers(unregistered_patch):
    original = unregistered_patch
    patch_mod = _patch_module()

    patch_mod.register()
    patch_mod.register()
    patch_mod.register()

    chain = _wrapper_chain(getattr(_oracle_module(), SYMBOL))
    assert chain[-1] is original
    assert len(chain) == 2, (
        "expected exactly one wrapper layer over the community function, got a "
        f"chain of {len(chain)}: {[getattr(f, '__qualname__', f) for f in chain]}"
    )


# ---------------------------------------------------------------------------
# Requirement 6.2 / 6.5: platform branching
# ---------------------------------------------------------------------------


def test_iluvatar_branch_returns_backend_none_without_calling_original(monkeypatch):
    impl = _impl_module()
    monkeypatch.setattr(impl, PLATFORM_GATE, lambda: True)

    def original(config):
        raise AssertionError("community function must not run on Iluvatar")

    wrapper = getattr(impl, FACTORY)(original)

    assert wrapper(_SentinelConfig()) == (_backend_none(), None)


def test_non_iluvatar_branch_delegates_to_the_original(monkeypatch):
    impl = _impl_module()
    monkeypatch.setattr(impl, PLATFORM_GATE, lambda: False)

    config = _SentinelConfig()
    seen: list[object] = []
    delegated = ("delegated-backend", "delegated-cls")

    def original(cfg):
        seen.append(cfg)
        return delegated

    wrapper = getattr(impl, FACTORY)(original)

    assert wrapper(config) is delegated, "non-Iluvatar must return the community result"
    assert seen == [config], "the config must be forwarded unchanged"


def test_patched_symbol_honours_the_platform_gate(unregistered_patch, monkeypatch):
    """End-to-end through the rebound attribute: the gate has to be read as a
    module global at call time, not captured when the wrapper was built."""
    impl = _impl_module()
    _patch_module().register()
    monkeypatch.setattr(impl, PLATFORM_GATE, lambda: True)

    for holder in (_oracle_module(), _quant_module()):
        assert getattr(holder, SYMBOL)(_SentinelConfig()) == (_backend_none(), None)


def test_platform_gate_is_a_zero_arg_predicate():
    gate = getattr(_impl_module(), PLATFORM_GATE)
    assert isinstance(gate(), bool)


def test_wrapper_keeps_the_community_identity_metadata(monkeypatch):
    """``functools.wraps`` is what lets ``check_signatures`` and any debug output
    still see the community name/signature after the rebind."""
    import inspect

    impl = _impl_module()
    original = getattr(_oracle_module(), SYMBOL)
    wrapper = getattr(impl, FACTORY)(original)

    assert wrapper.__name__ == SYMBOL
    assert inspect.signature(wrapper) == inspect.signature(original)


# ###########################################################################
# Task 3.4 -- ``IluDeepseekV4Fp8MoE``: the parameters P1 exists for
# ###########################################################################
#
# These live in this file rather than in a new one on purpose: every assertion
# below only holds *because* P1 is applied. Without it,
# ``Mxfp4MoEMethod.__init__`` raises before a single parameter is created, so
# the construction fixture is the end-to-end proof that the patch does its job.
#
# Ordering contract (design §4.4, task 3.3): ``register()`` first, then build.
# ``_moe_layer`` enforces it structurally -- the MoE is only constructed inside
# the fixture body, after both the rebind and the platform-gate monkeypatch.
#
# Mini-config folding of design §5.3
# ----------------------------------
# ``Mxfp4MoEMethod.create_weights`` (``quantization/mxfp4.py:530-622``) builds
# the four parameters from three numbers and a fixed 32-element scale block:
#
#     mxfp4_block = 32
#     E_local     = n_routed_experts // ep_size
#     H           = hidden_size
#     I_local     = moe_intermediate_size // moe_tp_size
#
#     w13_weight        [E_local, 2 * I_local, H // 2]
#     w13_weight_scale  [E_local, 2 * I_local, H // mxfp4_block]
#     w2_weight         [E_local, H,           I_local // 2]
#     w2_weight_scale   [E_local, H,           I_local // mxfp4_block]
#
# (``H // 2`` / ``I_local // 2`` because MXFP4 packs two 4-bit values per byte.)
#
# ==================  ============================  =====================
# quantity            design §5.3 (8 ranks)         mini config (TP1)
# ==================  ============================  =====================
# E_local             256 // 8   = 32               8 // 1   = 8
# H                   4096                          256
# I_local             2048                          256
# w13_weight          [32, 4096, 2048]              [8, 512, 128]
# w13_weight_scale    [32, 4096,  128]              [8, 512,   8]
# w2_weight           [32, 4096, 1024]              [8, 256, 128]
# w2_weight_scale     [32, 4096,   64]              [8, 256,   8]
# ==================  ============================  =====================
#
# ``maybe_roundup_sizes`` is the identity on both columns: under
# ``Mxfp4MoeBackend.NONE`` it reduces to ``round_up(I_local, 64)``, and 2048 /
# 256 are both multiples of 64 (that is what task 3.1's T6 pins).
#
# The 8-rank column assumes the experts are split expert-wise (32 local experts
# holding the full intermediate size), which is what ``FusedMoE`` produces when
# EP covers the 8 ranks. A TP-only launch would instead give
# ``[256, 512, 2048]`` / ``[256, 4096, 128]`` -- the same element count on a
# different axis. At TP1 the two collapse into one, so nothing here depends on
# the distinction; task 5.3's 8-card run is where it has to be settled.

MXFP4_BLOCK = 32
UINT8 = torch.uint8

_N_ROUTED_EXPERTS = MINI_CONFIG_FIELDS["n_routed_experts"]
_MOE_INTERMEDIATE_SIZE = MINI_CONFIG_FIELDS["moe_intermediate_size"]
_N_SHARED_EXPERTS = MINI_CONFIG_FIELDS["n_shared_experts"]
_TOP_K = MINI_CONFIG_FIELDS["num_experts_per_tok"]
_NUM_HASH_LAYERS = MINI_CONFIG_FIELDS["num_hash_layers"]
_FP8_BLOCK = MINI_CONFIG_FIELDS["quantization_config"]["weight_block_size"]

# TP1 / EP1: every expert is local and the intermediate size is unsplit.
_E_LOCAL = _N_ROUTED_EXPERTS // MINI_TP_SIZE
_I_LOCAL = _MOE_INTERMEDIATE_SIZE // MINI_TP_SIZE
_H = MINI_HIDDEN_SIZE

ROUTED_PARAM_PREFIX = "experts.routed_experts."

# design §5.3, folded per the table above. Written out rather than computed in a
# loop so a wrong formula shows up as a literal mismatch.
EXPECTED_ROUTED_PARAMS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "w13_weight": ((_E_LOCAL, 2 * _I_LOCAL, _H // 2), UINT8),
    "w13_weight_scale": ((_E_LOCAL, 2 * _I_LOCAL, _H // MXFP4_BLOCK), UINT8),
    "w2_weight": ((_E_LOCAL, _H, _I_LOCAL // 2), UINT8),
    "w2_weight_scale": ((_E_LOCAL, _H, _I_LOCAL // MXFP4_BLOCK), UINT8),
}

# Guard the folding itself: 8 / 256 / 256 is what the mini config must supply,
# and the literal shapes must be reachable from the same formulas as the TP8
# column. If someone shrinks the mini config further, this fails first and
# points at the table above instead of at the implementation.
assert (_E_LOCAL, _H, _I_LOCAL) == (8, 256, 256)
assert EXPECTED_ROUTED_PARAMS["w13_weight"][0] == (8, 512, 128)
assert EXPECTED_ROUTED_PARAMS["w13_weight_scale"][0] == (8, 512, 8)
assert EXPECTED_ROUTED_PARAMS["w2_weight"][0] == (8, 256, 128)
assert EXPECTED_ROUTED_PARAMS["w2_weight_scale"][0] == (8, 256, 8)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# design §5.3 shared-expert rows. Block-FP8 scale dims are
# ``ceil(size / block)`` with ``block = [128, 128]``; the intermediate round-up
# (design §4.5.4) is a no-op here exactly as it is at TP8.
_SHARED_I = (_MOE_INTERMEDIATE_SIZE * _N_SHARED_EXPERTS) // MINI_TP_SIZE
EXPECTED_SHARED_PARAMS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "gate_up_proj.weight": ((2 * _SHARED_I, _H), FP8),
    "gate_up_proj.weight_scale_inv": (
        (_ceil_div(2 * _SHARED_I, _FP8_BLOCK[0]), _ceil_div(_H, _FP8_BLOCK[1])),
        E8M0,
    ),
    "down_proj.weight": ((_H, _SHARED_I), FP8),
    "down_proj.weight_scale_inv": (
        (_ceil_div(_H, _FP8_BLOCK[0]), _ceil_div(_SHARED_I, _FP8_BLOCK[1])),
        E8M0,
    ),
}
assert EXPECTED_SHARED_PARAMS["gate_up_proj.weight"][0] == (512, 256)
assert EXPECTED_SHARED_PARAMS["gate_up_proj.weight_scale_inv"][0] == (4, 2)
assert EXPECTED_SHARED_PARAMS["down_proj.weight"][0] == (256, 256)
assert EXPECTED_SHARED_PARAMS["down_proj.weight_scale_inv"][0] == (2, 2)

# A hash layer and a non-hash layer of the mini config (num_hash_layers == 3).
HASH_LAYER_ID = 0
BIAS_LAYER_ID = _NUM_HASH_LAYERS


@contextlib.contextmanager
def _build_mini_moe(layer_id: int):
    """Build one ``IluDeepseekV4Fp8MoE`` on real CPU memory.

    CPU, not ``meta``: ``data_ptr()`` is meaningless without storage, and the
    whole point of the P1 patch is that those pointers survive
    ``process_weights_after_loading``. The mini config keeps the allocation
    under ~1 MB.
    """
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        IluDeepseekV4Fp8MoE,
    )

    vllm_config = make_mini_vllm_config()
    prefix = f"model.layers.{layer_id}.ffn"
    with set_default_torch_dtype(BF16), set_current_vllm_config(vllm_config):
        with torch.device("cpu"):
            moe = IluDeepseekV4Fp8MoE(vllm_config, prefix=prefix)
        yield moe, vllm_config


@pytest.fixture()
def moe_factory(unregistered_patch, monkeypatch):
    """Apply P1, force the Iluvatar branch, and only then hand out a builder.

    The gate is monkeypatched because the test host is not an Iluvatar device;
    the wrapper reads ``_is_iluvatar_platform`` as a module global at call time,
    so patching the module attribute is enough (see
    ``test_patched_symbol_honours_the_platform_gate``).
    """
    _patch_module().register()
    monkeypatch.setattr(_impl_module(), PLATFORM_GATE, lambda: True)
    yield _build_mini_moe


def _param_table(module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        name: (tuple(param.shape), param.dtype)
        for name, param in module.named_parameters()
    }


def _routed_experts(moe):
    """The ``RoutedExperts`` child that owns the four MXFP4 parameters.

    ``FusedMoE`` is a factory returning a ``MoERunner``; the parameters hang off
    its ``routed_experts`` attribute, hence the
    ``experts.routed_experts.w*`` qualified names.
    """
    return moe.experts.routed_experts


# ---------------------------------------------------------------------------
# Requirement 7.1: the four routed parameters exist, uint8, at design shapes
# ---------------------------------------------------------------------------


def test_routed_experts_use_the_community_mxfp4_method_on_backend_none(moe_factory):
    """Precondition for everything below: the NONE backend really was selected.

    If this fails, the remaining routed assertions are meaningless -- another
    backend would have swizzled or freed the weights.
    """
    backend_none = _backend_none()
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        quant_method = _routed_experts(moe).quant_method
        assert type(quant_method).__name__ == "Mxfp4MoEMethod", (
            "routed experts must land on the community Mxfp4MoEMethod, got "
            f"{type(quant_method).__name__}"
        )
        assert quant_method.mxfp4_backend == backend_none


def test_four_routed_params_are_present_in_named_parameters(moe_factory):
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        names = set(_param_table(moe))

    expected = {ROUTED_PARAM_PREFIX + name for name in EXPECTED_ROUTED_PARAMS}
    missing = sorted(expected - names)
    assert not missing, (
        f"routed MXFP4 parameters absent from named_parameters(): {missing}; "
        f"present: {sorted(names)}"
    )


def test_routed_param_shapes_and_dtypes_match_design_5_3(moe_factory):
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        actual = _param_table(moe)

    mismatches = {
        name: {
            "expected": spec,
            "actual": actual.get(ROUTED_PARAM_PREFIX + name),
        }
        for name, spec in EXPECTED_ROUTED_PARAMS.items()
        if actual.get(ROUTED_PARAM_PREFIX + name) != spec
    }
    assert not mismatches, f"routed MXFP4 shape/dtype mismatches: {mismatches}"


# ---------------------------------------------------------------------------
# Requirement 7.2: process_weights_after_loading moves nothing
# ---------------------------------------------------------------------------


def test_process_weights_after_loading_keeps_data_ptr_shape_and_params(moe_factory):
    """The NONE early-return (``quantization/mxfp4.py:727-738``) in action.

    Any other backend would run ``_setup_kernel`` ->
    ``convert_weight_to_mxfp4_moe_kernel_format`` (new storage, new pointer) and
    the TRITON backends additionally ``del layer.w13_weight_scale``. Comparing
    ``data_ptr()`` is what distinguishes "left alone" from "rebuilt with the same
    shape".
    """
    names = list(EXPECTED_ROUTED_PARAMS)

    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        experts = _routed_experts(moe)

        before = {
            name: (
                getattr(experts, name).data_ptr(),
                tuple(getattr(experts, name).shape),
                getattr(experts, name).dtype,
            )
            for name in names
        }

        experts.quant_method.process_weights_after_loading(experts)

        still_there = [name for name in names if getattr(experts, name, None) is None]
        assert not still_there, (
            "process_weights_after_loading dropped routed parameters "
            f"(w13_weight_scale must survive on backend NONE): {still_there}"
        )

        after = {
            name: (
                getattr(experts, name).data_ptr(),
                tuple(getattr(experts, name).shape),
                getattr(experts, name).dtype,
            )
            for name in names
        }

    moved = {
        name: {"before": before[name], "after": after[name]}
        for name in names
        if before[name] != after[name]
    }
    assert not moved, f"routed MXFP4 parameters were moved or reshaped: {moved}"


def test_w13_weight_scale_survives_as_a_registered_parameter(moe_factory):
    """Stronger than the ``getattr`` check above: it must still be a *parameter*,
    not a leftover plain attribute, or the loader has nothing to write into."""
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        experts = _routed_experts(moe)
        experts.quant_method.process_weights_after_loading(experts)
        names = {name for name, _ in experts.named_parameters()}

    assert "w13_weight_scale" in names
    assert "w2_weight_scale" in names


# ---------------------------------------------------------------------------
# Requirement 7.5: shared experts
# ---------------------------------------------------------------------------


def test_shared_expert_params_match_design_5_3(moe_factory):
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        actual = _param_table(moe)

    mismatches = {
        name: {"expected": spec, "actual": actual.get("shared_experts." + name)}
        for name, spec in EXPECTED_SHARED_PARAMS.items()
        if actual.get("shared_experts." + name) != spec
    }
    assert not mismatches, f"shared-expert shape/dtype mismatches: {mismatches}"


def test_shared_expert_intermediate_round_up_is_a_no_op(moe_factory):
    """Requirement 7.4's construction-side half: the block-FP8 round-up must not
    change the intermediate size on this checkpoint's numbers."""
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        assert moe.shared_experts.intermediate_size == _SHARED_I


# ---------------------------------------------------------------------------
# design §4.4.1: no activation module is constructed
# ---------------------------------------------------------------------------


def test_shared_expert_activation_is_a_torch_swiglu(moe_factory):
    """The ``act_fn`` slot is a parameter-free torch SwiGLU (Module 4), not a
    community activation ``CustomOp`` (design §4.4.1 / R10).

    Community ``SiluAndMulWithClamp.__init__``
    (``vllm/model_executor/layers/activation.py:186-195``) resolves
    ``torch.ops._C.silu_and_mul_with_clamp`` while it is being constructed, and
    the ``is_cuda_alike()`` branch has no fallback. ``IluPlatform`` derives from
    ``CudaPlatform`` so that branch always wins, and Iluvatar has no such symbol
    -- building one would die with ``AttributeError`` before a single parameter
    existed. So the slot holds ``_Fp8SwiGLU``, a plain ``nn.Module`` that does
    the clamped SwiGLU in torch: no ``torch.ops._C`` handle, no parameter, no
    buffer, and it actually computes (it no longer raises the group-C stub).

    Asserted on the *class*, not on ``isinstance``, so the community classes stay
    unimported here -- importing them is harmless, instantiating them is not.
    """
    from vllm.model_executor.custom_op import CustomOp

    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        act_fn = moe.shared_experts.act_fn

        assert not isinstance(act_fn, CustomOp), (
            "act_fn must not be a community activation CustomOp -- those bind "
            f"torch.ops._C in __init__; got {type(act_fn).__name__}"
        )
        # A kernel handle would mean something was resolved at construction time.
        assert getattr(act_fn, "op", None) is None
        # No parameter / buffer, so the loader and the shape tables are unaffected.
        assert list(act_fn.named_parameters()) == []
        assert list(act_fn.named_buffers()) == []

        # Wired: pure torch, runs on CPU. gate/up halves -> [., _SHARED_I].
        out = act_fn(torch.zeros(2, 2 * _SHARED_I, dtype=BF16))
        assert tuple(out.shape) == (2, _SHARED_I)
        assert out.dtype is BF16
        # silu(0) * 0 == 0, a cheap sanity value that needs no kernel.
        assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Requirement 7.6: gate branches on layer_id
# ---------------------------------------------------------------------------


def test_hash_layer_gate_has_tid2eid_and_no_correction_bias(moe_factory):
    with moe_factory(HASH_LAYER_ID) as (moe, _):
        assert moe.layer_id == HASH_LAYER_ID
        assert moe.gate.e_score_correction_bias is None
        tid2eid = moe.gate.tid2eid
        assert tid2eid is not None
        assert tuple(tid2eid.shape) == (MINI_VOCAB_SIZE, _TOP_K)
        names = set(_param_table(moe))

    assert "gate.tid2eid" in names
    assert "gate.e_score_correction_bias" not in names


def test_non_hash_layer_gate_has_correction_bias_and_no_tid2eid(moe_factory):
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        assert moe.layer_id == BIAS_LAYER_ID
        assert moe.gate.tid2eid is None
        bias = moe.gate.e_score_correction_bias
        assert bias is not None
        assert tuple(bias.shape) == (_N_ROUTED_EXPERTS,)
        assert bias.dtype is FP32
        names = set(_param_table(moe))

    assert "gate.e_score_correction_bias" in names
    assert "gate.tid2eid" not in names


def test_gate_weight_shape_and_router_output_dtype(moe_factory):
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        assert tuple(moe.gate.weight.shape) == (_N_ROUTED_EXPERTS, _H)
        # design §5.3 / community nvidia/model.py:545-551: the router runs fp32
        assert moe.gate.out_dtype is FP32
