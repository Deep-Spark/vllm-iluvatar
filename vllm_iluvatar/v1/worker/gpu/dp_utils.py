# SPDX-License-Identifier: Apache-2.0
"""Publish the all-ranks decode-only bit from the V2 DP sync.

Corresponds to: vllm/v1/worker/gpu/dp_utils.py
                vllm/v1/worker/gpu/cudagraph_utils.py
Baseline : v0.30.0rc0
Why      : hybrid EP selects the LL decode backend from
           ``DPMetadata.all_ranks_decode_only``. The V1 coordinator publishes
           that bit; the V2 runner never calls it. 0.30 already all-reduces
           token counts inside ``sync_cudagraph_and_dp_padding`` and passes a
           per-rank ``uniform_decode`` flag, but does not store the agreed bit.
How      : wrap the community sync and all-reduce ``uniform_decode`` on the
           same CPU group, then ``set_last_all_ranks_decode_only``. CUDA graph
           capture does not go through that sync, so ``capture`` publishes the
           bit for each descriptor (``uniform_token_count == decode_query_len``).
Experiment: VLLM_ILU_MERGE_DP_SYNC=1 appends the vote to the six-row
           0.30 agreement in merged_dp_sync; default keeps the original path.
Scope    : V2 DP+EP MoE forward and CUDA graph capture.
Remove   : when community DP sync writes this bit into forward metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import os

import torch
import torch.distributed as dist
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState

from vllm_iluvatar.v1.worker.dp_utils import set_last_all_ranks_decode_only

_ORIGINAL_SYNC: Callable[..., Any] | None = None


def _is_uniform_decode(
    cudagraph_manager: CudaGraphManager | None,
    uniform_token_count: int | None,
) -> bool:
    return (
        cudagraph_manager is not None
        and uniform_token_count == cudagraph_manager.decode_query_len
    )


def _all_ranks_set(bits: torch.Tensor) -> bool:
    return bool(torch.all(bits == 1).item())


def _publish_all_ranks_decode_only(
    uniform_decode: bool,
    dp_size: int,
    dp_rank: int,
) -> None:
    """Agree the per-rank uniform-decode vote. One bit, same group as V2 DP sync."""
    bits = torch.zeros(dp_size, dtype=torch.int32)
    bits[dp_rank] = int(bool(uniform_decode))
    if should_skip_dp_coordination():
        bits[:] = bits[dp_rank]
    else:
        dist.all_reduce(bits, group=get_dp_group().cpu_group)
    set_last_all_ranks_decode_only(_all_ranks_set(bits))


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    parallel_config: ParallelConfig | None = None,
    allow_ubatching: bool = False,
    uniform_decode: bool = False,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Community V2 DP sync, plus the all-ranks decode-only bit."""
    if _ORIGINAL_SYNC is None:
        raise RuntimeError("sync_cudagraph_and_dp_padding patch is not installed")
    if os.environ.get("VLLM_ILU_MERGE_DP_SYNC", "0") == "1":
        from .merged_dp_sync import sync_cudagraph_and_dp_padding as sync
    else:
        _publish_all_ranks_decode_only(uniform_decode, dp_size, dp_rank)
        sync = _ORIGINAL_SYNC
    result = sync(
        cudagraph_manager,
        desired_batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
    )

    return result


def make_capture(original: Callable[..., None]) -> Callable[..., None]:
    """Publish decode-only state for each CUDA graph descriptor."""

    @wraps(original)
    def capture(
        self: Any,
        create_forward_fn: Callable[..., Any],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn_for_desc(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Any:
            set_last_all_ranks_decode_only(
                _is_uniform_decode(self, desc.uniform_token_count)
            )
            return create_forward_fn(desc, warmup=warmup)

        return original(self, create_forward_fn_for_desc, progress_bar_desc)

    return capture
