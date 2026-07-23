# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/dp_utils.py

ILU override implementation for DP batch coordination. This module holds only
the upstream-mirrored implementation; the runtime wiring, source fingerprint,
and signature checks live in
``ilu_patch.general_patches.patch_dp_utils`` and are auto-discovered during
``register_general``.

Target   : vllm.v1.worker.dp_utils.coordinate_batch_across_dp
Baseline : plugin/vllm 0.24.0 runtime source
Why      : DeepSeek-V4 hybrid DP+EP must know whether every DP rank is in
           decode-only mode before switching decode to the LL collective path.
           The same coordination also disables CUDA graph replay when any DP
           rank is still in prefill, keeping collective paths consistent.
How      : full-function copy of the upstream coordinator plus small optional
           extensions: sync the all-ranks decode-only bit and the pre-CUDA-graph
           token count, optionally return the decode-only bit to callers, and
           preserve the upstream 3-value return contract for existing consumers.
Scope    : DP batch coordination metadata on Iluvatar.
Remove   : when upstream DPMetadata carries all-ranks decode-only state or
           exposes an equivalent hook.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

logger = init_logger(__name__)

# [ILU addition] Last synchronized all-DP decode-only bit.  The
# ``DPMetadata.make`` / ``set_forward_context`` wrappers attach this value to
# forward metadata without changing the community return contract by default.
_ILU_LAST_ALL_RANKS_DECODE_ONLY = False


def get_last_all_ranks_decode_only() -> bool:
    return _ILU_LAST_ALL_RANKS_DECODE_ONLY


def _get_device_and_group(parallel_config: ParallelConfig):
    device = get_dp_group().device
    group = get_dp_group().device_group
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks.",
        )
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    pre_cudagraph_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    uniform_decode: bool | None,
    sync_decode_only: bool,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    # >>> ILU MODIFICATION — add pre-cudagraph token count and optional
    # all-ranks decode-only synchronization to the community DP all-reduce
    # payload.  Community synchronizes five rows; ILU uses row 1 for the
    # pre-CUDA-graph token count and row 5 for the decode-only bit when needed.
    tensor_cpu = torch.zeros(6 if sync_decode_only else 5, dp_size, dtype=torch.int32)
    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor_cpu[1][dp_rank] = pre_cudagraph_num_tokens_per_ubatch
    tensor_cpu[2][dp_rank] = padded_num_tokens_per_ubatch
    tensor_cpu[3][dp_rank] = 1 if should_ubatch else 0
    tensor_cpu[4][dp_rank] = cudagraph_mode
    if sync_decode_only:
        tensor_cpu[5][dp_rank] = 1 if uniform_decode else 0
    # <<< ILU MODIFICATION <<<
    tensor = tensor_cpu.to(device, non_blocking=True)
    dist.all_reduce(tensor, group=group)
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    should_ubatch = bool(torch.all(tensor[3] == 1).item())
    if not should_ubatch:
        return False
    orig_min_num_tokens = int(tensor[0, :].min().item())
    padded_max_num_tokens = int(tensor[2, :].max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[2 if should_dp_pad else 1, :]
    if should_dp_pad:
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    return int(tensor[4, :].min().item())


def _post_process_all_ranks_decode_only(tensor: torch.Tensor) -> bool:
    # [ILU addition] Row 5 is present only when ``sync_decode_only`` is true.
    return bool(torch.all(tensor[5] == 1).item())


# [ILU override] coordinate_batch_across_dp — delta: keep the community
# positional prefix, then append optional ILU parameters for pre-CUDA-graph
# padding, all-ranks decode-only sync, and the opt-in 4-value return.
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
    num_tokens_pre_cudagraph_padding: int | None = None,
    sync_decode_only: bool = True,
    return_all_ranks_decode_only: bool = False,
) -> (
    tuple[bool, torch.Tensor | None, int] | tuple[bool, torch.Tensor | None, int, bool]
):
    global _ILU_LAST_ALL_RANKS_DECODE_ONLY

    # >>> ILU MODIFICATION — LL already requires all ranks to use the same
    # decode collective path, so skip the extra decode-only row for that backend.
    if parallel_config.all2all_backend == "deepep_low_latency":
        sync_decode_only = False
    # <<< ILU MODIFICATION <<<

    # >>> ILU MODIFICATION — initialize and publish all-ranks decode-only state
    # even for the community early-exit case where DP size is one.
    all_ranks_decode_only = bool(uniform_decode)
    if parallel_config.data_parallel_size == 1:
        _ILU_LAST_ALL_RANKS_DECODE_ONLY = all_ranks_decode_only
        if return_all_ranks_decode_only:
            return False, None, cudagraph_mode, all_ranks_decode_only
        return False, None, cudagraph_mode
    # <<< ILU MODIFICATION <<<

    should_attempt_ubatching = False
    if allow_microbatching:
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded
    # >>> ILU MODIFICATION — preserve the token count before CUDA graph padding
    # so mixed prefill/decode ranks can still agree on the safe DP padding size.
    if num_tokens_pre_cudagraph_padding is None:
        num_tokens_pre_cudagraph_padding = num_tokens_padded
    assert num_tokens_padded >= num_tokens_unpadded
    assert num_tokens_pre_cudagraph_padding >= num_tokens_unpadded
    # <<< ILU MODIFICATION <<<

    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        pre_cudagraph_num_tokens_per_ubatch=num_tokens_pre_cudagraph_padding,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        uniform_decode=uniform_decode,
        sync_decode_only=sync_decode_only,
        parallel_config=parallel_config,
    )

    # >>> ILU MODIFICATION — synchronize whether every DP rank is decode-only;
    # mixed prefill/decode disables CUDA graph replay to keep collective choices
    # consistent across ranks.
    if sync_decode_only:
        all_ranks_decode_only = _post_process_all_ranks_decode_only(tensor)

    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
    if sync_decode_only and uniform_decode is not None and not all_ranks_decode_only:
        synced_cudagraph_mode = 0
    # <<< ILU MODIFICATION <<<

    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)
    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
    num_tokens_after_padding = _post_process_dp_padding(tensor, should_dp_pad)

    # >>> ILU MODIFICATION — store the synchronized bit for forward metadata and
    # optionally expose it to callers that have been updated to consume it.
    _ILU_LAST_ALL_RANKS_DECODE_ONLY = all_ranks_decode_only
    if return_all_ranks_decode_only:
        return (
            should_ubatch,
            num_tokens_after_padding,
            synced_cudagraph_mode,
            all_ranks_decode_only,
        )
    # <<< ILU MODIFICATION <<<
    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode
