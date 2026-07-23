# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar DP coordination helpers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from vllm_iluvatar.v1.worker import dp_utils


def test_coordinate_batch_signature_preserves_community_prefix():
    params = list(inspect.signature(dp_utils.coordinate_batch_across_dp).parameters)
    assert params[:6] == [
        "num_tokens_unpadded",
        "allow_microbatching",
        "parallel_config",
        "num_tokens_padded",
        "uniform_decode",
        "cudagraph_mode",
    ]
    assert params[6:] == [
        "num_tokens_pre_cudagraph_padding",
        "sync_decode_only",
        "return_all_ranks_decode_only",
    ]


def test_post_process_helpers_use_cpu_tensors():
    tensor = torch.tensor(
        [
            [8, 10],
            [8, 10],
            [12, 14],
            [1, 1],
            [2, 1],
            [1, 0],
        ],
        dtype=torch.int32,
    )

    assert dp_utils._post_process_ubatch(tensor, num_ubatches=2) is True
    assert dp_utils._post_process_dp_padding(tensor, should_dp_pad=False).tolist() == [
        8,
        10,
    ]
    assert dp_utils._post_process_dp_padding(tensor, should_dp_pad=True).tolist() == [
        14,
        14,
    ]
    assert dp_utils._post_process_cudagraph_mode(tensor) == 1
    assert dp_utils._post_process_all_ranks_decode_only(tensor) is False


def test_coordinate_single_rank_returns_all_ranks_decode_only():
    parallel_config = SimpleNamespace(
        all2all_backend="allgather_reducescatter",
        data_parallel_size=1,
    )

    result = dp_utils.coordinate_batch_across_dp(
        num_tokens_unpadded=4,
        allow_microbatching=False,
        parallel_config=parallel_config,
        uniform_decode=True,
        cudagraph_mode=2,
        return_all_ranks_decode_only=True,
    )

    assert result == (False, None, 2, True)
    assert dp_utils.get_last_all_ranks_decode_only() is True
