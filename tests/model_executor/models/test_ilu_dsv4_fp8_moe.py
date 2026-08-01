# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``IluDeepseekV4Fp8MoE`` construction and its MoE method.

Successor of ``test_ilu_dsv4_fp8_moe_patch.py``. That file paired two things:
the P1 backend-selection patch and the parameters it existed for. Module 4.1
retired P1 -- the routed experts now land on ``IluMxfp4MoEMethod`` because
``IluDeepseekV4FP8Config`` returns it, with no community module globals rewritten
-- so the patch half is gone and the construction half moved here under a name
that no longer claims to test a patch.

What is asserted
----------------
1. **Method selection** -- the routed experts get ``IluMxfp4MoEMethod``, the
   method reports ``mxfp4_backend == NONE``, defines **no** ``experts_cls``, and
   answers ``is_monolithic``/``supports_eplb``/``get_fused_moe_quant_config``/
   ``maybe_make_prepare_finalize`` the way the runner needs. These are the exact
   properties that let patches P1 and P3 be deleted, so they are pinned here
   rather than left implicit.
2. **Parameter shapes** (design §5.3) -- the four routed ``uint8`` MXFP4
   parameters and the two shared-expert block-FP8 pairs, at the mini-config
   folding of the §5.3 numbers.
3. **``process_weights_after_loading`` moves nothing** -- same ``data_ptr()``,
   shape and dtype before and after, and ``w13/w2_weight_scale`` survive as
   registered parameters. This is what Phase 2's zero-copy
   ``.view(torch.int8)`` -> ``group_gemm_fp4`` depends on.
4. **Gate branching on ``layer_id``** (requirement 7.6) -- hash layers get
   ``tid2eid``, the rest get ``e_score_correction_bias``, never both.
5. **The shared-expert activation stays a plain torch module** (design §4.4.1).

Pure CPU: no GPU, no vendor kernel library, no patch registration.

Validates: Requirements 7.1, 7.2, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import contextlib

import pytest
import torch

# Reused from task 1.2: ``_tp1_distributed_env`` is a session-scoped autouse
# fixture and importing it here is what makes it apply to this module's tests too
# (parallel linear layers read the TP world size). ``make_mini_vllm_config``
# carries the mini DS-V4 hf_config plus ``IluDeepseekV4FP8Config``, which is what
# routes the routed experts to ``IluMxfp4MoEMethod``.
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


def _backend_none():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return Mxfp4MoeBackend.NONE


# ###########################################################################
# Mini-config folding of design §5.3
# ###########################################################################
#
# ``Mxfp4MoEMethod.create_weights`` (``quantization/mxfp4.py:530-622``, inherited
# unchanged by ``IluMxfp4MoEMethod``) builds the four parameters from three
# numbers and a fixed 32-element scale block:
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
# 256 are both multiples of 64 (that is what ``test_ilu_dsv4_fp8_mxfp4_roundup``
# pins). ``IluMxfp4MoEMethod`` keeps ``mxfp4_backend = NONE`` precisely so this
# stays true.
#
# The 8-rank column assumes the experts are split expert-wise (32 local experts
# holding the full intermediate size), which is what ``FusedMoE`` produces when
# EP covers the 8 ranks. A TP-only launch would instead give
# ``[256, 512, 2048]`` / ``[256, 4096, 128]`` -- the same element count on a
# different axis. At TP1 the two collapse into one, so nothing here depends on
# the distinction.


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
def moe_factory():
    """Hand out the MoE builder.

    No patch registration and no platform-gate monkeypatching since module 4.1:
    ``make_mini_vllm_config`` supplies ``IluDeepseekV4FP8Config``, whose
    ``get_quant_method`` returns ``IluMxfp4MoEMethod`` unconditionally for routed
    MXFP4 experts. Nothing consults ``current_platform``, so construction behaves
    identically on an Iluvatar device and on this CI host.
    """
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


def test_routed_experts_use_the_ilu_mxfp4_method(moe_factory):
    """Precondition for everything below: our method really was selected.

    If this fails the remaining routed assertions are meaningless -- a community
    backend would have swizzled or freed the weights, and construction would most
    likely have raised in ``select_deepseek_v4_mxfp4_moe_backend`` first.
    """
    from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 import (
        IluMxfp4MoEMethod,
    )

    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        quant_method = _routed_experts(moe).quant_method
        assert isinstance(quant_method, IluMxfp4MoEMethod), (
            "routed experts must land on IluMxfp4MoEMethod, got "
            f"{type(quant_method).__name__}"
        )
        # NONE is load-bearing: maybe_roundup_sizes / skip_forward_padding /
        # has_unpadded_output all read it (see the folding table above).
        assert quant_method.mxfp4_backend == _backend_none()


def test_method_defines_no_experts_cls(moe_factory):
    """Why patch P3 could be deleted.

    Community ``FusedMoEMethodBase.is_monolithic``
    (``fused_moe_method_base.py:163-170``) does
    ``if hasattr(self, "experts_cls"): return self.experts_cls.is_monolithic()``
    when ``moe_kernel is None``. ``Mxfp4MoEMethod.__init__`` (``mxfp4.py:481``)
    assigns ``None`` there, so the guard passes and the attribute access raises
    ``AttributeError`` -- the EP-only crash P3 worked around. Our ``__init__``
    must leave the attribute undefined.
    """
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        quant_method = _routed_experts(moe).quant_method
        assert not hasattr(quant_method, "experts_cls")
        assert quant_method.moe_kernel is None
        # The property that P3 used to crash on must simply answer.
        assert quant_method.is_monolithic is False


def test_method_opts_out_of_community_kernel_and_modular_wrapping(moe_factory):
    """The four answers ``IluMoERunner`` needs to take the modular ILU path.

    * ``get_fused_moe_quant_config -> None`` keeps ``topk_indices_dtype`` at
      ``None`` (``fused_moe_method_base.py:135-140``), so the router emits int32
      ids -- the dtype ``ilu_mxfp4_fused_experts`` expects.
    * ``maybe_make_prepare_finalize -> None`` stops
      ``MoERunner.maybe_init_modular_kernel`` from wrapping us into a
      ``FusedMoEModularMethod`` whose ``apply`` would expect a real kernel.
    * ``supports_internal_mk`` stays False, so the runner asks for the
      prepare/finalize at all rather than assuming we own a kernel.
    * ``supports_eplb -> False`` makes an EPLB launch fail loudly instead of
      silently not rebalancing (Phase 1 records no expert load).
    """
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        experts = _routed_experts(moe)
        quant_method = experts.quant_method
        assert quant_method.get_fused_moe_quant_config(experts) is None
        assert quant_method.maybe_make_prepare_finalize() is None
        assert quant_method.supports_internal_mk is False
        assert quant_method.supports_eplb is False
        assert quant_method.topk_indices_dtype is None


def test_maybe_init_modular_kernel_is_a_no_op_without_p3(moe_factory):
    """End-to-end proof that P3 is unnecessary.

    ``BaseDeviceCommunicator.prepare_communication_buffer_for_model`` calls this
    on every MoE module under expert parallelism, and it is where the P3 crash
    happened. With our method it has to return without touching anything.
    """
    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        experts = _routed_experts(moe)
        before = type(experts.quant_method)
        moe.experts.maybe_init_modular_kernel()
        assert type(experts.quant_method) is before, (
            "quant_method was replaced -- maybe_init_modular_kernel wrapped us "
            "into a modular method, which Phase 1 cannot execute"
        )


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
    """``IluMxfp4MoEMethod.process_weights_after_loading`` is a no-op.

    Any community backend would run ``_setup_kernel`` ->
    ``convert_weight_to_mxfp4_moe_kernel_format`` (new storage, new pointer) and
    the TRITON backends additionally ``del layer.w13_weight_scale``. Comparing
    ``data_ptr()`` is what distinguishes "left alone" from "rebuilt with the same
    shape" -- and "left alone" is what Phase 1's dequant reads and what Phase 2
    needs for a zero-copy ``.view(torch.int8)`` into ``group_gemm_fp4``.
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
            f"(w13_weight_scale must survive): {still_there}"
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
# Module 5: the shared-expert activation is the kernel-backed CustomOp
# ---------------------------------------------------------------------------


def test_shared_expert_activation_is_the_oot_clamped_swiglu(moe_factory):
    """``act_fn`` is ``SiluAndMulWithClamp``, i.e. the plugin's OOT subclass.

    Design §4.4.1 / R10 assumed constructing a community activation ``CustomOp``
    would die on Iluvatar, because ``SiluAndMulWithClamp.__init__`` resolves its
    kernel while being built and the ``is_cuda_alike()`` branch has no fallback.
    That is only true without the plugin: ``IluSiluAndMulWithClamp``
    (``vllm_iluvatar/model_executor/layers/activation.py``) overrides that branch
    to bind ``vllm._custom_ops.silu_and_mul_with_clamp``, ``ilu_register`` applies
    it before any model module is imported, and the W4A8 fork has been running it
    in production. Module 5 therefore dropped the torch stand-in.

    ``fuse_quant`` must stay off: that path returns a ``(q, scale, dtype)`` tuple
    for a downstream int8 linear, which the block-FP8 ``down_proj`` cannot take.
    """
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    with moe_factory(BIAS_LAYER_ID) as (moe, _):
        act_fn = moe.shared_experts.act_fn

        assert isinstance(act_fn, SiluAndMulWithClamp)
        # The OOT subclass, not the community class itself.
        assert type(act_fn) is not SiluAndMulWithClamp
        assert act_fn.swiglu_limit == float(MINI_CONFIG_FIELDS["swiglu_limit"])
        assert act_fn.fuse_quant is False
        # No parameter / buffer, so the loader and the shape tables are unaffected.
        assert list(act_fn.named_parameters()) == []
        assert list(act_fn.named_buffers()) == []


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
