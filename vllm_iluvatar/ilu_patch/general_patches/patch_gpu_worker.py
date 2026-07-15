# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar Worker dummy-batch optimisation.

Corresponds to: vllm/v1/worker/gpu_worker.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.v1.worker.gpu_worker.Worker.execute_dummy_batch
Baseline : main_88e5e2c57be8
Why      : The community ``execute_dummy_batch`` calls ``_dummy_run`` without
           ``profile_seq_lens``, which defaults to ``max_model_len``.  Under
           data-parallelism, idle-rank dummy runs execute every step in DP
           lockstep; scanning the full context makes the dummy forward ~15 ms
           slower than a real decode step, pacing *all* DP ranks at that speed
           via the per-step DP barrier.
How      : direct class-method replacement via ``setattr``; passes
           ``profile_seq_lens=1`` so the dummy forward is as cheap as possible.
Scope    : Worker (formerly GPUWorker) on Iluvatar when data-parallelism is enabled.
Remove   : when upstream adds a configurable ``profile_seq_lens`` for idle DP
           ranks or optimises the dummy-batch path natively.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register Iluvatar Worker.execute_dummy_batch override."""
    p.register_once(
        __name__,
        _apply,
        detail="Worker.execute_dummy_batch override",
    )


def _apply() -> None:
    import types

    from vllm.v1.worker.gpu_worker import Worker

    def _ilu_execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(
            num_tokens, uniform_decode=True, profile_seq_lens=1
        )

    _ilu_mod = types.SimpleNamespace(
        execute_dummy_batch=_ilu_execute_dummy_batch,
    )
    p.check_signatures(
        Worker, _ilu_mod, ["execute_dummy_batch"], strict=True
    )

    Worker.execute_dummy_batch = _ilu_execute_dummy_batch
