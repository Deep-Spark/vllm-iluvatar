# SPDX-License-Identifier: Apache-2.0
"""Align router padding masks with DP-gathered token rows."""

from contextlib import contextmanager

import torch
from vllm import envs


@contextmanager
def _gathered_router_padding(ctx, dist_group, sizes):
    local_padding = ctx.is_padding
    if not envs.VLLM_MOE_SKIP_PADDING or local_padding is None:
        yield
        return
    if getattr(ctx, "_ilu_is_padding_gathered", False):
        yield
        return
    cache = getattr(ctx, "_ilu_gathered_router_padding", None)
    if cache is None:
        cache = {}
        ctx._ilu_gathered_router_padding = cache
    key = (id(dist_group), tuple(sizes), local_padding.data_ptr())
    if key not in cache:
        # One gather per forward context, reused by every MoE layer.
        local_size = sizes[dist_group.rank_in_group]
        mask = local_padding[:local_size].to(torch.int8)
        cache[key] = dist_group.all_gatherv(mask, dim=0, sizes=sizes).bool()
    prev_gathered = getattr(ctx, "_ilu_is_padding_gathered", False)
    ctx.is_padding = cache[key]
    ctx._ilu_is_padding_gathered = True
    try:
        yield
    finally:
        ctx.is_padding = local_padding
        ctx._ilu_is_padding_gathered = prev_gathered
