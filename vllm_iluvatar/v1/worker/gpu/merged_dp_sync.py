# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""0.30 DP sync with the decode-only vote appended to its existing collective.

Keep all six upstream metadata rows and DPSyncState semantics intact.
Derived from the paired vLLM source; only row 6 and bit publication are added.
"""
from __future__ import annotations

from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm_iluvatar.v1.worker.dp_utils import set_last_all_ranks_decode_only

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds, get_num_ubatches


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
    """Coordinates the batch descriptor and DP padding across all ranks.

    `parallel_config` is only needed to decide whether to microbatch, so callers
    that never do (`allow_ubatching=False`) can leave it out.

    Returns (synced_batch_desc, sync). `sync` is None when no rank has work.
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(7, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    tensor[4][dp_rank] = int(allow_ubatching)
    tensor[5][dp_rank] = num_reqs
    tensor[6][dp_rank] = int(uniform_decode)
    if should_skip_dp_coordination():
        tensor[:] = tensor[:, dp_rank, None].clone()
    else:
        dist.all_reduce(tensor, group=group)

    set_last_all_ranks_decode_only(bool(torch.all(tensor[6] == 1).item()))

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]
    allow_ubatching_across_dp = tensor[4]
    num_reqs_across_dp = tensor[5]

    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    synced_uniform_token_count: int | None = int(uniform_token_counts_across_dp[0])
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    if torch.all(allow_ubatching_across_dp == 1):
        # This rank voted too, so its caller passed a config.
        assert parallel_config is not None
        # A uniform decode only if every rank runs the same uniform query
        # length, so every rank reaches the same answer here.
        uniform_decode_across_dp = uniform_decode and torch.all(
            uniform_token_counts_across_dp == (uniform_token_count or 0)
        )
        if check_ubatch_thresholds(
            parallel_config,
            # Thresholds only grow with the token count, so holding the smallest
            # rank to them holds every rank to them.
            int(num_tokens_across_dp.min()),
            uniform_decode=uniform_decode_across_dp,
        ):
            # Expert all-to-all requires every rank to split and pad equally.
            # Empty microbatches run as dummy batches.
            ubatch_num_tokens = int(num_tokens_across_dp.max())
            num_ubatches = get_num_ubatches(parallel_config)
            ubatch_desc = None
            if cudagraph_manager is not None:
                # Match a FULL microbatched CUDA graph, falling back to an eager
                # descriptor if no graph matches.
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    ubatch_num_tokens,
                    synced_uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
                if 2 * int(num_tokens_across_dp.min()) < ubatch_desc.num_tokens:
                    # If one rank has an empty second microbatch, run without
                    # CUDA graphs.
                    ubatch_desc = None
            if ubatch_desc is None:
                ubatch_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=ubatch_num_tokens,
                    num_reqs=num_reqs,
                    num_ubatches=num_ubatches,
                )
            # Refresh the token count to include CUDA graph padding.
            ubatch_num_tokens = ubatch_desc.num_tokens
            num_reqs = int(num_reqs_across_dp.max())
            if ubatch_desc.cg_mode == CUDAGraphMode.FULL:
                assert ubatch_desc.num_reqs is not None
                num_reqs = ubatch_desc.num_reqs
            return ubatch_desc, DPSyncState(
                num_tokens_across_dp=torch.full_like(
                    num_tokens_across_dp, ubatch_num_tokens
                ),
                uniform_token_count=synced_uniform_token_count,
                eager=ubatch_desc.cg_mode == CUDAGraphMode.NONE,
                num_reqs=num_reqs,
            )

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=desired_batch_desc.num_active_loras,
            ),
            DPSyncState(
                num_tokens_across_dp=num_tokens_across_dp,
                uniform_token_count=synced_uniform_token_count,
                eager=True,
                num_reqs=int(num_reqs_across_dp.max()),
            ),
        )

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1).item()):
        synced_max_query_len = int(max_query_lens_across_dp.max().item())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, DPSyncState(
        num_tokens_across_dp=num_tokens_across_dp,
        uniform_token_count=synced_uniform_token_count,
        eager=False,
        num_reqs=(
            synced_desc.num_reqs
            if synced_desc.cg_mode == CUDAGraphMode.FULL
            and synced_desc.num_reqs is not None
            else int(num_reqs_across_dp.max())
        ),
    )
