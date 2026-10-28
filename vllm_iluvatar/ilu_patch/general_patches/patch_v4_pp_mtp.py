# SPDX-License-Identifier: Apache-2.0
"""Publish batch-bound MTP proposals for synchronous DeepSeek V4 PP.

Corresponds to: vllm/v1/worker/gpu_worker.py, vllm/v1/core/sched/scheduler.py,
               vllm/v1/engine/core.py
Target   : Worker.sample_tokens, Scheduler.update_from_output, EngineCore.post_step
Baseline : vLLM feat/v0.31.0 (341bf7d4d1244424c94296ce4fc5626ad14b827a)
Why      : V2 synchronous PP can queue several batches before sampling; reading
           a shared draft handler later can attach proposals to the wrong batch.
How      : Delegate to the original methods and carry proposals with each output
           until its accepted/rejected token accounting has completed.
Scope    : DeepSeek V4, MTP, PP>1, V2 runner, synchronous scheduling only.
Remove   : When upstream binds draft proposals to each queued sampled output.
"""

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="V4 V2 PP batch-bound MTP proposals")


def _apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_iluvatar.v1.engine.ilu_pp_mtp import (
        wrap_post_step,
        wrap_sample_tokens,
        wrap_update_from_output,
    )

    # [ILU override] Worker.sample_tokens - delta: bind proposals to its output.
    Worker.sample_tokens = wrap_sample_tokens(Worker.sample_tokens)
    # [ILU override] Scheduler.update_from_output - delta: publish after accounting.
    Scheduler.update_from_output = wrap_update_from_output(Scheduler.update_from_output)
    # [ILU override] EngineCore.post_step - delta: avoid batch-unbound publication.
    EngineCore.post_step = wrap_post_step(EngineCore.post_step)
