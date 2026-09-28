# SPDX-License-Identifier: Apache-2.0
"""Carry real DSpark draft ids through the PP model-output contract.

Corresponds to:
    vllm/v1/outputs.py :: ModelRunnerOutput
    vllm/v1/engine/core.py :: EngineCore.post_step
    vllm/v1/core/sched/scheduler.py :: Scheduler.update_from_output
    vllm/v1/worker/gpu/model_runner.py :: prepare_inputs / sample_tokens
Baseline : v0.30.1rc0
Why      : sync + PP batch_queue makes ``take_draft_token_ids()`` return
           ``[-1]*K`` (or a later step's drafts). Workers then verify the wrong
           lookahead and acceptance collapses toward 0.
How      : install ``draft_token_ids`` on ``ModelRunnerOutput``; attach real
           ids after ``sample_tokens``; apply them in ``update_from_output``;
           skip stale ``post_step`` RPC; stage scheduled ids in
           ``prepare_inputs``.
Scope    : sync scheduling, PP>1, speculative decode (esp. DSpark).
Remove   : when community ships the same PP draft transport.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="PP+spec real draft ids on model output + staging",
    )


def _scheduler_uses_async_scheduling(scheduler: Any) -> bool:
    sched = getattr(scheduler, "scheduler_config", None)
    if sched is None:
        vcfg = getattr(scheduler, "vllm_config", None)
        sched = getattr(vcfg, "scheduler_config", None) if vcfg is not None else None
    return bool(getattr(sched, "async_scheduling", False))


def make_scheduler_update_from_output(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original)
    def _ilu_update_from_output(self, scheduler_output, model_runner_output):
        outputs = original(self, scheduler_output, model_runner_output)
        draft_token_ids = getattr(model_runner_output, "draft_token_ids", None)
        if draft_token_ids is not None and not _scheduler_uses_async_scheduling(self):
            self.update_draft_token_ids(draft_token_ids)
        return outputs

    return _ilu_update_from_output


def _apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_iluvatar.v1.engine import core as core_impl
    from vllm_iluvatar.v1.outputs import install_draft_token_ids_field
    from vllm_iluvatar.v1.worker.gpu import model_runner as runner_impl

    install_draft_token_ids_field(ModelRunnerOutput)

    if not getattr(Scheduler.update_from_output, "_ilu_pp_spec_drafts_patched", False):
        wrapped = make_scheduler_update_from_output(Scheduler.update_from_output)
        wrapped._ilu_pp_spec_drafts_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = wrapped  # type: ignore[method-assign]

    if not getattr(EngineCore.__init__, "_ilu_pp_spec_drafts_patched", False):
        init = core_impl.make_init(EngineCore.__init__)
        init._ilu_pp_spec_drafts_patched = True  # type: ignore[attr-defined]
        EngineCore.__init__ = init  # type: ignore[method-assign]

    if not getattr(EngineCore.post_step, "_ilu_pp_spec_drafts_patched", False):
        post = core_impl.make_post_step(EngineCore.post_step)
        post._ilu_pp_spec_drafts_patched = True  # type: ignore[attr-defined]
        EngineCore.post_step = post  # type: ignore[method-assign]

    if not getattr(GPUModelRunner.prepare_inputs, "_ilu_pp_spec_drafts_patched", False):
        prep = runner_impl.make_prepare_inputs(GPUModelRunner.prepare_inputs)
        prep._ilu_pp_spec_drafts_patched = True  # type: ignore[attr-defined]
        GPUModelRunner.prepare_inputs = prep  # type: ignore[method-assign]

    if not getattr(GPUModelRunner.sample_tokens, "_ilu_pp_spec_drafts_patched", False):
        sample = runner_impl.make_sample_tokens(GPUModelRunner.sample_tokens)
        sample._ilu_pp_spec_drafts_patched = True  # type: ignore[attr-defined]
        GPUModelRunner.sample_tokens = sample  # type: ignore[method-assign]
