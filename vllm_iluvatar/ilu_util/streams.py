# SPDX-License-Identifier: Apache-2.0
"""Shared CUDA auxiliary streams for opt-free prefill compute overlap."""

from __future__ import annotations

from functools import cache

import torch


@cache
def get_prefill_aux_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    """Return one cached auxiliary CUDA stream per device for prefill overlap.

    Sharing a single auxiliary stream across MLA and KDA prefill projections
    lets KDA reuse the CUDACachingAllocator stream pool already warmed by MLA
    during initial memory profiling instead of creating a second unprofiled pool.
    """
    return torch.cuda.Stream(device=device)
