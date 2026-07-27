# SPDX-License-Identifier: Apache-2.0
"""CPU-only test (T3) for "e8m0 ``view`` and ``copy_`` are NOT equivalent".

Design §4.5.2 turns a community comment into a hard rule for the FP8 loader.
Community ``vllm/models/deepseek_v4/nvidia/model.py:1200-1211`` (the ``.experts.``
branch of ``load_weights``) reads::

    # E8M0 scales are stored as float8_e8m0fnu in checkpoints but the MoE param
    # is uint8. copy_() would do a numeric conversion (e.g. 2^-7 -> 0), destroying
    # the raw exponent bytes.
    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)

This file is the executable evidence for that comment, so the rule cannot be
"optimised" back into a ``copy_()`` by a later change without a red test.

Byte <-> value mapping (UE8M0)
------------------------------
``torch.float8_e8m0fnu`` is a *pure exponent* format: 8 exponent bits, zero
mantissa bits, zero sign bits, exponent bias 127. A stored byte ``e`` therefore
denotes the value ``2 ** (e - 127)`` (and ``e == 255`` is NaN, which this file
stays away from). Consequences used below:

===========  =====================  ==================================
raw byte     value                  ``uint8`` after a numeric ``copy_``
===========  =====================  ==================================
120          ``2 ** -7`` = 0.0078   0   (truncated towards zero)
124          ``2 ** -3`` = 0.125    0
127          ``2 ** 0``  = 1.0      1
128          ``2 ** 1``  = 2.0      2
129          ``2 ** 2``  = 4.0      4
130          ``2 ** 3``  = 8.0      8
===========  =====================  ==================================

Why ``copy_`` destroys the bytes: ``Tensor.copy_`` is defined as a *value* copy
with dtype conversion, so writing a ``float8_e8m0fnu`` source into a ``uint8``
destination converts the represented number (e.g. ``2 ** -7 == 0.0078125``) into
an integer, truncating towards zero. Every ``|value| < 1`` exponent point
(i.e. every raw byte below 127) therefore collapses to ``0``, and the points at
or above 1.0 land on the *value* (1, 2, 4, 8) rather than on the raw byte
(127, 128, 129, 130). ``view(torch.uint8)`` instead reinterprets the same
storage and is exact for every point.

Reading a failure here: if the ``view`` cases go red, the reinterpretation
semantics of ``Tensor.view`` or the e8m0 bit layout changed; if only the
``copy_`` cases go red, torch changed its conversion/rounding behaviour (that
would make ``copy_`` *differently* wrong, not correct — the rule still holds).

Scope notes:

* Pure torch semantics: no GPU, no model construction, no distributed init, no
  ``ixformer``. The tensors are built by viewing a ``uint8`` byte pattern as
  ``float8_e8m0fnu``, so the raw exponent bytes are exact by construction and
  the test never depends on float32 -> e8m0 rounding.
* Exercises no plugin code, so it is expected to pass before Task 4.4 lands.

Validates: Requirements 1.2, 1.3
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------
# UE8M0 exponent points. Each entry is (raw_byte, value, uint8_after_copy_).
#
# value            = 2 ** (raw_byte - E8M0_BIAS)
# uint8_after_copy_ = int(value) truncated towards zero
#
# Deliberately bounded: no byte >= 255 (NaN) and no exponent whose value
# overflows uint8 (>= 256), since out-of-range float -> uint8 casts are not
# well defined and would make the copy_ expectations non-deterministic.
# --------------------------------------------------------------------------

E8M0_BIAS = 127

EXPONENT_POINTS = [
    (120, 2.0**-7, 0),  # the exact case named in the community comment
    (124, 2.0**-3, 0),
    (127, 2.0**0, 1),
    (128, 2.0**1, 2),
    (129, 2.0**2, 4),
    (130, 2.0**3, 8),
]

RAW_BYTES = [byte for byte, _, _ in EXPONENT_POINTS]

# The byte the community comment calls out explicitly: 2 ** -7 -> 127 - 7 == 120.
BYTE_OF_2_POW_MINUS_7 = E8M0_BIAS - 7


def _e8m0_from_bytes(raw_bytes: list[int]) -> torch.Tensor:
    """A ``float8_e8m0fnu`` tensor holding exactly ``raw_bytes``."""
    return torch.tensor(raw_bytes, dtype=torch.uint8).view(torch.float8_e8m0fnu)


# --------------------------------------------------------------------------
# The byte <-> value mapping itself. Without this, the copy_ expectations below
# would be magic numbers.
# --------------------------------------------------------------------------


def test_byte_of_two_pow_minus_seven_is_120():
    assert BYTE_OF_2_POW_MINUS_7 == 120
    assert EXPONENT_POINTS[0][0] == BYTE_OF_2_POW_MINUS_7


@pytest.mark.parametrize(("raw_byte", "value", "_expected_uint8"), EXPONENT_POINTS)
def test_raw_byte_denotes_two_pow_exponent_minus_bias(raw_byte, value, _expected_uint8):
    """Raw byte ``e`` reads back as ``2 ** (e - 127)`` (pure exponent format)."""
    assert value == 2.0 ** (raw_byte - E8M0_BIAS)

    decoded = _e8m0_from_bytes([raw_byte]).to(torch.float32)
    assert decoded.item() == value


# --------------------------------------------------------------------------
# Requirements 1.2: view(torch.uint8) is byte-exact for every exponent point.
# --------------------------------------------------------------------------


def test_view_uint8_is_bytewise_identical_for_all_exponent_points():
    scale = _e8m0_from_bytes(RAW_BYTES)

    viewed = scale.view(torch.uint8)

    assert viewed.dtype == torch.uint8
    assert viewed.tolist() == RAW_BYTES
    # Zero-copy: same storage, so no numeric conversion can have happened.
    assert viewed.data_ptr() == scale.data_ptr()
    assert viewed.shape == scale.shape


def test_view_uint8_preserves_shape_of_a_moe_scale_block():
    """Mini stand-in for ``w13_weight_scale`` ``[local_experts, N, K // 32]``."""
    numel = 2 * 3 * len(RAW_BYTES)
    pattern = (RAW_BYTES * numel)[:numel]
    scale = _e8m0_from_bytes(pattern).reshape(2, 3, len(RAW_BYTES))

    viewed = scale.view(torch.uint8)

    assert viewed.shape == (2, 3, len(RAW_BYTES))
    assert viewed.stride() == scale.stride()
    assert viewed.flatten().tolist() == pattern


# --------------------------------------------------------------------------
# Requirements 1.3: copy_ into a uint8 param is a numeric conversion and
# destroys the exponent bytes. This is the counter-example that pins the rule.
# --------------------------------------------------------------------------


def test_copy_into_uint8_param_zeroes_the_two_pow_minus_seven_byte():
    scale = _e8m0_from_bytes([BYTE_OF_2_POW_MINUS_7])
    param = torch.nn.Parameter(torch.empty(1, dtype=torch.uint8), requires_grad=False)

    param.data.copy_(scale)

    # 2 ** -7 == 0.0078125 truncates to the integer 0: the raw byte 120 is gone.
    assert param.data.item() == 0
    assert param.data.item() != BYTE_OF_2_POW_MINUS_7


def test_copy_into_uint8_param_never_preserves_any_exponent_byte():
    scale = _e8m0_from_bytes(RAW_BYTES)
    param = torch.nn.Parameter(
        torch.empty(len(RAW_BYTES), dtype=torch.uint8), requires_grad=False
    )

    param.data.copy_(scale)

    expected_after_copy = [expected for _, _, expected in EXPONENT_POINTS]
    assert param.data.tolist() == expected_after_copy

    # Every point below 1.0 collapses to 0 ...
    for (raw_byte, value, expected), got in zip(
        EXPONENT_POINTS, param.data.tolist(), strict=True
    ):
        if abs(value) < 1.0:
            assert got == 0, f"byte {raw_byte} (= {value}) should truncate to 0"
        # ... and no point, below or above 1.0, keeps its raw byte.
        assert got != raw_byte
        assert got == expected

    # Restating the headline of Requirements 1.3: the two paths disagree.
    assert param.data.tolist() != scale.view(torch.uint8).tolist()


# --------------------------------------------------------------------------
# The correct posture for writing an e8m0 checkpoint scale into a uint8 param.
# This is the shape Task 4.4's ``.experts.`` branch has to use.
# --------------------------------------------------------------------------


def test_copy_of_viewed_tensor_is_bytewise_faithful():
    scale = _e8m0_from_bytes(RAW_BYTES)
    param = torch.nn.Parameter(
        torch.empty(len(RAW_BYTES), dtype=torch.uint8), requires_grad=False
    )
    param_ptr = param.data.data_ptr()

    # view() first, then copy_ -> uint8 -> uint8, i.e. a pure byte copy.
    param.data.copy_(scale.view(torch.uint8))

    assert param.data.tolist() == RAW_BYTES
    assert param.data.dtype == torch.uint8
    assert param.data.data_ptr() == param_ptr  # copied in place, no realloc


def test_assigning_viewed_tensor_keeps_bytes_and_shares_storage():
    scale = _e8m0_from_bytes(RAW_BYTES)
    param = torch.nn.Parameter(
        torch.empty(len(RAW_BYTES), dtype=torch.uint8), requires_grad=False
    )

    param.data = scale.view(torch.uint8)

    assert param.data.dtype == torch.uint8
    assert param.data.tolist() == RAW_BYTES
    assert param.data.data_ptr() == scale.data_ptr()
