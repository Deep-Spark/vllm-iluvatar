# SPDX-License-Identifier: Apache-2.0
"""Wire Iluvatar's temporary PP + speculative-decode compatibility patch.

Corresponds to:
    vllm/v1/outputs.py
    vllm/v1/core/sched/scheduler.py
    vllm/v1/engine/core.py
    vllm/v1/worker/gpu_model_runner.py

Target   : PP batch queue with speculative decode / MTP.
Baseline : main_6e448d0
Why      : a pipeline batch queue schedules the next batch before the prior
           output updates request state. Draft tokens must therefore travel in
           ModelRunnerOutput and non-last PP ranks must avoid drafter paths.
How      : patch the transport contract in place, wrap narrow EngineCore /
           Scheduler methods, and copy only runner/scheduler methods requiring
           a middle-of-method delta.
Scope    : Iluvatar vLLM processes. The semantic fix is generic and should be
           removed once adopted by community vLLM.
Enable   : opt-in via ``VLLM_ILU_ENABLE_PP_MTP=1``; default off leaves the
           community PP / speculative-decode path untouched.
Limits   : async scheduling is not supported together with PP + MTP. Keep
           async scheduling disabled when this patch is enabled.
Remove   : after the equivalent PP + speculative-decode fixes are upstream.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the PP + MTP compatibility patch once per process."""
    p.register_once(
        __name__,
        _apply,
        detail="PP batch-queue speculative-decode / MTP compatibility",
    )


def _bind_method(
    implementation: object, community_globals: dict[str, object], name: str
):
    """Recreate a copied method with the community module's globals."""
    rebound = types.FunctionType(
        implementation.__code__,
        community_globals,
        name=name,
        argdefs=implementation.__defaults__,
        closure=implementation.__closure__,
    )
    rebound.__kwdefaults__ = implementation.__kwdefaults__
    rebound.__annotations__ = implementation.__annotations__
    rebound.__doc__ = implementation.__doc__
    return rebound


def _check_method(
    community_cls: type,
    community_name: str,
    implementation: object,
    baseline: str,
) -> None:
    """Run the required drift and signature checks for a copied method."""
    p.check_fingerprint(
        f"{community_cls.__name__}.{community_name}",
        inspect.getsource(getattr(community_cls, community_name)),
        baseline,
        resync_hint=(
            "Re-sync the PP+MTP copied implementation and baseline in "
            "vllm_iluvatar before using a newer community vLLM revision."
        ),
    )
    p.check_signatures(
        community_cls,
        types.SimpleNamespace(**{community_name: implementation}),
        [community_name],
        strict=True,
    )


def _pp_mtp_enabled() -> bool:
    """Return True when the PP + MTP compatibility patch is opted in.

    Gate read (default off): ``VLLM_ILU_ENABLE_PP_MTP``. When off, the
    plugin leaves community vLLM untouched, so PP + MTP / speculative
    decode follows the upstream path. PP + MTP does not support async
    scheduling.
    """
    import vllm.envs as envs

    return bool(envs.VLLM_ILU_ENABLE_PP_MTP)


def _apply() -> None:
    # Opt-in gate: skip all wiring (community path) unless enabled.
    if not _pp_mtp_enabled():
        return
    _wire_patches()


def _wire_patches() -> None:
    from vllm.v1.core.sched import scheduler as scheduler_mod
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker import gpu_model_runner as runner_mod

    from vllm_iluvatar.v1.core.sched import scheduler as scheduler_impl
    from vllm_iluvatar.v1.engine import core as core_impl
    from vllm_iluvatar.v1.outputs import install_draft_token_ids_field
    from vllm_iluvatar.v1.worker import gpu_model_runner as runner_impl

    install_draft_token_ids_field(ModelRunnerOutput)

    _check_method(
        scheduler_mod.Scheduler,
        "schedule",
        scheduler_impl._ilu_schedule,
        scheduler_impl.SCHEDULE_BASELINE,
    )
    # [ILU override] Scheduler.schedule - delta: permit negative token budget
    # from PP output timing instead of treating it as schedulable work.
    scheduler_mod.Scheduler.schedule = _bind_method(
        scheduler_impl._ilu_schedule,
        scheduler_mod.__dict__,
        "schedule",
    )

    original_update_from_output = scheduler_mod.Scheduler.update_from_output

    def _ilu_update_from_output(self, scheduler_output, model_runner_output):
        outputs = original_update_from_output(
            self, scheduler_output, model_runner_output
        )
        draft_token_ids = getattr(model_runner_output, "draft_token_ids", None)
        if draft_token_ids is not None:
            self.update_draft_token_ids(draft_token_ids)
        return outputs

    # [ILU override] Scheduler.update_from_output - delta: ingest inline drafts.
    scheduler_mod.Scheduler.update_from_output = _ilu_update_from_output

    EngineCore.__init__ = core_impl.make_init(EngineCore.__init__)
    EngineCore.post_step = core_impl.make_post_step(EngineCore.post_step)

    runner_cls = runner_mod.GPUModelRunner
    for name, baseline in (
        ("_update_states", runner_impl._UPDATE_STATES_BASELINE),
        ("_build_attention_metadata", runner_impl._BUILD_ATTENTION_METADATA_BASELINE),
        ("_dummy_run", runner_impl._DUMMY_RUN_BASELINE),
    ):
        implementation = getattr(runner_impl, f"_ilu_{name}")
        _check_method(runner_cls, name, implementation, baseline)
        # [ILU override] GPUModelRunner copied method - delta: PP + MTP.
        setattr(
            runner_cls,
            name,
            _bind_method(implementation, runner_mod.__dict__, name),
        )

    runner_cls.__init__ = runner_impl.make_runner_init(runner_cls.__init__)
    runner_cls.sample_tokens = runner_impl.make_sample_tokens(runner_cls.sample_tokens)
    # [ILU override] GPUModelRunner._check_and_update_cudagraph_mode - delta:
    # skip the drafter-only sub-step on non-last PP ranks (no drafter there).
    runner_cls._check_and_update_cudagraph_mode = (
        runner_impl.make_check_and_update_cudagraph_mode(
            runner_cls._check_and_update_cudagraph_mode
        )
    )
    # [ILU override] GPUModelRunner.initialize_metadata_builders - delta: skip
    # the drafter attn-backend init on non-last PP ranks (no drafter there).
    runner_cls.initialize_metadata_builders = (
        runner_impl.make_initialize_metadata_builders(
            runner_cls.initialize_metadata_builders
        )
    )
