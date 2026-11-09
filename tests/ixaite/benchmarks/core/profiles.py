# SPDX-License-Identifier: Apache-2.0
"""Shared runtime batch / CUDA Graph T profiles for Iluvatar serving."""

from __future__ import annotations

# Decode effective T in [1, max_num_seqs=128] -- first-version gate covers all.
DECODE_REQUIRED_T: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# CUDA Graph capture discrete sizes (full table). Every capture size is a
# required gate point: the off-power-of-two multiples of 8 land on the
# safe-default pad path (no tuned key) and previously hid a pad-to-256
# regression (2026-08-18 _hgemm_pad_m fix).
GRAPH_FULL_T: tuple[int, ...] = (1, 2, 4) + tuple(range(8, 256, 8)) + (256,)
GRAPH_REQUIRED_T: tuple[int, ...] = GRAPH_FULL_T
GRAPH_OPTIONAL_T: tuple[int, ...] = ()

# Prefill / mixed anchors under max_num_batched_tokens=2048.
PREFILL_REQUIRED_T: tuple[int, ...] = (512, 1024, 2048)
PREFILL_OPTIONAL_T: tuple[int, ...] = (256,)


def linear_required_t(*, full_profile: bool = False) -> tuple[int, ...]:
    """Default gate T set for dense linear (decode full + graph top + prefill anchors)."""
    required = DECODE_REQUIRED_T + GRAPH_REQUIRED_T + PREFILL_REQUIRED_T
    if not full_profile:
        return tuple(sorted(set(required)))
    extra = GRAPH_FULL_T + GRAPH_OPTIONAL_T + PREFILL_OPTIONAL_T
    return tuple(sorted(set(required + extra)))
