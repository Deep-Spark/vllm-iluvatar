# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar V2 DP decode-only coordination."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils as community_dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_iluvatar.v1.worker.dp_utils import (
    get_last_all_ranks_decode_only,
    set_last_all_ranks_decode_only,
)
from vllm_iluvatar.v1.worker.gpu import dp_utils


def test_sync_signature_matches_community():
    assert list(
        inspect.signature(dp_utils.sync_cudagraph_and_dp_padding).parameters
    ) == list(
        inspect.signature(community_dp_utils.sync_cudagraph_and_dp_padding).parameters
    )


def test_uniform_decode_requires_the_decode_query_len():
    manager = SimpleNamespace(decode_query_len=6)

    assert dp_utils._is_uniform_decode(manager, 6) is True
    assert dp_utils._is_uniform_decode(manager, 1) is False
    assert dp_utils._is_uniform_decode(manager, None) is False
    assert dp_utils._is_uniform_decode(None, 6) is False


def test_all_ranks_set_requires_every_rank():
    bits = torch.tensor([1, 1], dtype=torch.int32)
    assert dp_utils._all_ranks_set(bits) is True
    bits[1] = 0
    assert dp_utils._all_ranks_set(bits) is False


def test_publish_uses_the_local_vote_when_coordination_is_skipped(monkeypatch):
    monkeypatch.setattr(dp_utils, "should_skip_dp_coordination", lambda: True)
    dp_utils._publish_all_ranks_decode_only(True, dp_size=2, dp_rank=0)
    assert get_last_all_ranks_decode_only() is True
    dp_utils._publish_all_ranks_decode_only(False, dp_size=2, dp_rank=1)
    assert get_last_all_ranks_decode_only() is False


def test_capture_publishes_the_bit_of_each_descriptor():
    descs = [
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=12,
            num_reqs=2,
            uniform_token_count=6,
        ),
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=12,
            num_reqs=2,
            uniform_token_count=None,
        ),
    ]
    published: list[bool] = []

    def original(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        for desc in descs:
            create_forward_fn(desc, warmup=True)
            published.append(get_last_all_ranks_decode_only())

    capture = dp_utils.make_capture(original)
    set_last_all_ranks_decode_only(False)
    capture(SimpleNamespace(decode_query_len=6), lambda desc, warmup: None)

    assert published == [True, False]
