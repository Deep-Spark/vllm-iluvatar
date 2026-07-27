# SPDX-License-Identifier: Apache-2.0
"""CPU-only test (T6) for "MXFP4 shapes get no padding under backend NONE".

Design §4.4 ("padding 核实") establishes that ``Mxfp4MoEMethod.maybe_roundup_sizes``
(community ``quantization/mxfp4.py:513-528``) forwards to
``mxfp4_round_up_hidden_size_and_intermediate_size``
(community ``fused_moe/oracle/mxfp4.py:635``), and that ``Mxfp4MoeBackend.NONE``
matches none of that function's ``if``/``elif`` arms (EMULATION / DEEPGEMM /
MARLIN / TRTLLM / FLASHINFER_CUTLASS / ``is_rocm()`` / CPU), so it lands in::

    else:
        intermediate_size = round_up(intermediate_size, 64)

``hidden_size`` is left untouched on that arm, and ``round_up(2048, 64) == 2048``
because 2048 is already a multiple of 64 — hence DeepSeek-V4-Flash's
``(hidden_size, moe_intermediate_size) = (4096, 2048)`` survives unpadded, which
is what makes the four routed MXFP4 params match the checkpoint shapes exactly.

Scope notes:

* This file exercises **community code only**. It does not depend on the P1
  backend-selection patch (Task 3.3) nor on the plugin's MoE implementation in
  ``fp8_model.py`` — so it is expected to pass as-is, before either exists.
* Pure CPU: no GPU, no distributed init, no model construction.

Beyond the single value required by Requirements 7.3, the cases below pin down
*why* there is no padding (identity comes from the 64-multiple input, not from
the function being a no-op), so a future community change to the fallback arm
fails here with a readable diagnosis instead of silently repadding the experts.

Validates: Requirements 7.3
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------
# Expected values, hard-coded from design §4.4 / config.json.
# --------------------------------------------------------------------------

# config.json: hidden_size = 4096, moe_intermediate_size = 2048
CONFIG_HIDDEN_SIZE = 4096
CONFIG_INTERMEDIATE_SIZE = 2048

# The fallback arm's alignment for intermediate_size (oracle/mxfp4.py else branch).
FALLBACK_INTERMEDIATE_ALIGNMENT = 64

# mini config used by the other Stage 1 tests shrinks moe_intermediate_size to 256,
# which is also what the real model sees per TP8 partition (2048 / 8 = 256).
MINI_INTERMEDIATE_SIZE = 256


def _round_up_fn():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        mxfp4_round_up_hidden_size_and_intermediate_size,
    )

    return mxfp4_round_up_hidden_size_and_intermediate_size


def _backend_none():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return Mxfp4MoeBackend.NONE


# --------------------------------------------------------------------------
# Requirements 7.3: the real config shape is returned untouched.
# --------------------------------------------------------------------------


def test_backend_none_leaves_real_config_shapes_unpadded():
    assert _round_up_fn()(
        _backend_none(), CONFIG_HIDDEN_SIZE, CONFIG_INTERMEDIATE_SIZE
    ) == (CONFIG_HIDDEN_SIZE, CONFIG_INTERMEDIATE_SIZE)


@pytest.mark.parametrize(
    "intermediate_size", [CONFIG_INTERMEDIATE_SIZE, MINI_INTERMEDIATE_SIZE]
)
def test_backend_none_no_padding_for_64_multiple_intermediate(intermediate_size):
    """Both the real (2048) and the mini / per-TP8-partition (256) values."""
    # Precondition of the identity: already 64-aligned.
    assert intermediate_size % FALLBACK_INTERMEDIATE_ALIGNMENT == 0

    assert _round_up_fn()(
        _backend_none(), CONFIG_HIDDEN_SIZE, intermediate_size
    ) == (CONFIG_HIDDEN_SIZE, intermediate_size)


# --------------------------------------------------------------------------
# Why it is unpadded: the fallback arm does round intermediate_size up to 64,
# and does not touch hidden_size. Without these two cases, the assertion above
# would also hold for a function that did nothing at all.
# --------------------------------------------------------------------------


def test_backend_none_rounds_non_aligned_intermediate_up_to_64():
    intermediate_size = CONFIG_INTERMEDIATE_SIZE - 48  # 2000, not a 64-multiple
    assert intermediate_size % FALLBACK_INTERMEDIATE_ALIGNMENT != 0

    hidden, intermediate = _round_up_fn()(
        _backend_none(), CONFIG_HIDDEN_SIZE, intermediate_size
    )

    assert intermediate == CONFIG_INTERMEDIATE_SIZE  # round_up(2000, 64) == 2048
    assert hidden == CONFIG_HIDDEN_SIZE


def test_backend_none_never_rounds_hidden_size():
    # 4097 is aligned to nothing; the fallback arm still returns it verbatim.
    hidden, intermediate = _round_up_fn()(
        _backend_none(), CONFIG_HIDDEN_SIZE + 1, CONFIG_INTERMEDIATE_SIZE
    )

    assert hidden == CONFIG_HIDDEN_SIZE + 1
    assert intermediate == CONFIG_INTERMEDIATE_SIZE
