# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for opt-in PP queue depth and async prefill coalescing."""

from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace


def test_scaled_mq_limits_tracks_pp_batches():
    from vllm_iluvatar.ilu_patch.general_patches.patch_message_queue import (
        _scaled_mq_limits,
    )

    chunks, _ = _scaled_mq_limits(8, 10, 8 << 20, "", min_chunks=0)
    assert chunks == 24
    chunks_deep, _ = _scaled_mq_limits(8, 10, 8 << 20, "", min_chunks=32)
    assert chunks_deep == 32


def test_pp_prefill_coalesce_async_and_early_done():
    from vllm_iluvatar.v1.engine.core import _make_pp_prefill_coalesce

    calls = []

    class FakeReq:
        def __init__(self, rid, prompt, computed):
            self.request_id = rid
            self.num_tokens = prompt
            self.num_prompt_tokens = prompt
            self.num_computed_tokens = computed
            self.num_output_placeholders = 0
            self.resumable = False

    class FakeExec:
        def __init__(self, is_done):
            self._done = is_done

        def done(self):
            return self._done

    def orig(engine):
        calls.append("orig")
        return (None, True)

    wrapped = _make_pp_prefill_coalesce(orig, 0.05)

    # 1. Async + kv producer + 1 in-flight batch stops early once uncomputed >= budget.
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=True),
            speculative_config=None,
            kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        ),
        is_pooling_model=False,
        batch_queue=deque([(None, None, FakeExec(False))]),
        batch_queue_size=14,
        scheduler=SimpleNamespace(
            running=[FakeReq("r1", 10000, 9000)],
            waiting=[],
            max_num_scheduled_tokens=2560,
        ),
        input_queue=queue.Queue(),
        is_running=lambda: True,
        _process_input_queue=lambda: None,
        _ilu_pp_initial_fill=True,
        _ilu_pp_coalesce_remaining_s=0.05,
        _ilu_pp_coalesce_last=None,
    )
    engine.input_queue.put(("add", FakeReq("r2", 10000, 0)))

    def handle(_op, req):
        engine.scheduler.waiting.append(req)

    engine._handle_client_request = handle
    wrapped(engine)
    assert len(engine.scheduler.waiting) == 1
    assert calls == ["orig"]

    # 2. If oldest batch is already done, coalesce exits immediately without draining queue.
    engine2 = SimpleNamespace(
        vllm_config=engine.vllm_config,
        is_pooling_model=False,
        batch_queue=deque([(None, None, FakeExec(True))]),
        batch_queue_size=14,
        scheduler=SimpleNamespace(
            running=[FakeReq("r1", 10000, 9000)],
            waiting=[],
            max_num_scheduled_tokens=2560,
        ),
        input_queue=queue.Queue(),
        is_running=lambda: True,
        _process_input_queue=lambda: None,
        _ilu_pp_initial_fill=True,
        _ilu_pp_coalesce_remaining_s=0.05,
        _ilu_pp_coalesce_last=None,
        _handle_client_request=lambda *_a: None,
    )
    wrapped(engine2)
    assert engine2._ilu_pp_coalesce_remaining_s == 0.05
