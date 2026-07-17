# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu_worker.py

IluWorker — subclass of the community Worker that overrides
``execute_dummy_batch`` for Iluvatar DP idle-rank optimisation.

Why  : The community ``execute_dummy_batch`` calls ``_dummy_run`` without
       ``profile_seq_lens``, which defaults to ``max_model_len``.  Under
       data-parallelism, idle-rank dummy runs execute every step in DP
       lockstep; scanning the full context makes the dummy forward ~15 ms
       slower than a real decode step, pacing *all* DP ranks at that speed
       via the per-step DP barrier.  Passing ``profile_seq_lens=1`` makes
       the dummy forward as cheap as possible.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker


class IluWorker(Worker):

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )
