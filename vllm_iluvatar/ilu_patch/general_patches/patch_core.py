# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar DP EngineCore scheduling wrappers.

Corresponds to: vllm/v1/engine/core.py
Target   : DPEngineCoreProc.{_should_throttle_prefills,_process_engine_step}
Baseline : main_dcfebf93f4e
Why      : Align replacement prefills across DP ranks before EP execution and
           avoid entering a dummy batch during the short replacement round trip.
           The patch is installed only when VLLM_ILU_DP_ENABLE_SYNC_PREFILLS
           is enabled and VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS is positive.
           VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS bounds the fail-open delay.
How      : Delegation wrappers installed on DPEngineCoreProc only.
           Both community method bodies are fingerprinted before wrapping so
           upstream scheduling changes fail fast and require a wrapper review.
Scope    : Env-gated Iluvatar hybrid DP+EP serving.
Remove   : When upstream provides equivalent synchronized prefill scheduling.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p


SHOULD_THROTTLE_PREFILLS = '''\
def _should_throttle_prefills(self) -> bool:
    # Throttle new prefills to cadence-aligned steps for DP balancing.
    # step_counter is identical across DP ranks. On a fresh wave the
    # counter is 0, so prefills are admitted immediately after idle.
    return (
        self.prefill_schedule_interval > 1
        and self.step_counter % self.prefill_schedule_interval != 0
    )
'''

PROCESS_ENGINE_STEP = '''\
def _process_engine_step(self) -> bool:
    """Called only when there are unfinished local requests."""

    # Step the engine core.
    outputs, model_executed = self.step_fn()
    # Put EngineCoreOutputs into the output queue.
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)
    # Post-step hook.
    self.post_step(model_executed)

    # If no model execution happened but there is still scheduler work
    # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
    # the GIL briefly to allow background transfer threads to make progress.
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)

    return model_executed
'''

HANDLE_CLIENT_REQUEST = '''\
def _handle_client_request(
    self, request_type: EngineCoreRequestType, request: Any
) -> None:
    if request_type == EngineCoreRequestType.START_DP_WAVE:
        if self.ignore_start_dp_wave:
            return
        new_wave, exclude_eng_index = request
        if exclude_eng_index != self.engine_index and (
            new_wave >= self.current_wave
        ):
            self.current_wave = new_wave
            if not self.engines_running:
                logger.debug(
                    "EngineCore starting idle loop for wave %d.",
                    new_wave,
                )
                self.engines_running = True
    else:
        super()._handle_client_request(request_type, request)
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DPEngineCoreProc prefill sync and replacement wait wrappers",
    )


def _apply() -> None:
    import vllm.envs as envs
    from vllm.v1.engine.core import DPEngineCoreProc

    import vllm_iluvatar.v1.engine.core as impl

    sync_prefills = bool(envs.VLLM_ILU_DP_ENABLE_SYNC_PREFILLS)
    replacement_wait_ms = float(envs.VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS)
    if not sync_prefills or replacement_wait_ms <= 0:
        return

    p.check_fingerprint(
        "DPEngineCoreProc._should_throttle_prefills",
        inspect.getsource(DPEngineCoreProc._should_throttle_prefills),
        SHOULD_THROTTLE_PREFILLS,
        resync_hint=(
            "Review _make_should_throttle_prefills against the new community "
            "policy and refresh SHOULD_THROTTLE_PREFILLS."
        ),
    )
    p.check_fingerprint(
        "DPEngineCoreProc._process_engine_step",
        inspect.getsource(DPEngineCoreProc._process_engine_step),
        PROCESS_ENGINE_STEP,
        resync_hint=(
            "Review _make_process_engine_step against the new community method "
            "and refresh PROCESS_ENGINE_STEP."
        ),
    )
    p.check_fingerprint(
        "DPEngineCoreProc._handle_client_request",
        inspect.getsource(DPEngineCoreProc._handle_client_request),
        HANDLE_CLIENT_REQUEST,
        resync_hint=(
            "Review _make_handle_client_request against the new community "
            "wave-start policy and refresh HANDLE_CLIENT_REQUEST."
        ),
    )

    should_throttle = impl._make_should_throttle_prefills(
        DPEngineCoreProc._should_throttle_prefills
    )
    process_step = impl._make_process_engine_step(
        DPEngineCoreProc._process_engine_step
    )
    handle_client_request = impl._make_handle_client_request(
        DPEngineCoreProc._handle_client_request
    )
    ilu_methods = types.SimpleNamespace(
        _should_throttle_prefills=should_throttle,
        _process_engine_step=process_step,
        _handle_client_request=handle_client_request,
    )
    p.check_signatures(
        DPEngineCoreProc,
        ilu_methods,
        [
            "_should_throttle_prefills",
            "_process_engine_step",
            "_handle_client_request",
        ],
        strict=True,
    )

    DPEngineCoreProc._should_throttle_prefills = should_throttle  # type: ignore[method-assign]
    DPEngineCoreProc._process_engine_step = process_step  # type: ignore[method-assign]
    DPEngineCoreProc._handle_client_request = handle_client_request  # type: ignore[method-assign]
