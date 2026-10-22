# SPDX-License-Identifier: Apache-2.0
"""Gate remote image ranges without encoder allocations or KV-wait rewinds.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target: Scheduler._try_schedule_encoder_inputs and Scheduler.update_from_output
Why: a P-only decoder must await exact producer image rows, while the normal
     scheduler would allocate an encoder cache entry and run the entire ViT.
How: use the existing native connector's background receiver readiness; leave
     requests pending with zero new tokens, without re-entering remote KV wait.
Scope: explicitly enabled DeepSeek V4.1 PD deployments; text is a direct pass.
Remove: when upstream exposes a range-based remote multimodal scheduler hook.
"""

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register():
    p.register_once(__name__, _apply, detail="native PD multimodal range readiness")


def _apply():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_schedule
    from vllm_iluvatar.v1.worker.mm_replay.scheduler import find_replay, logger

    # Preserve the source rewrite used by PP and even128/129 remote replay.
    patch_pp_spec_schedule.register()
    if getattr(Scheduler._try_schedule_encoder_inputs, "_ilu_mm_replay", False):
        return
    original_encoder = Scheduler._try_schedule_encoder_inputs
    original_update = Scheduler.update_from_output

    @wraps(original_encoder)
    def try_encoder(
        self,
        request,
        num_computed_tokens,
        num_new_tokens,
        encoder_compute_budget,
        shift_computed_tokens=0,
    ):
        replay = (
            find_replay(getattr(self, "connector", None))
            if any(f.modality == "image" for f in request.mm_features)
            else None
        )
        if replay is not None:
            if not replay.can_schedule(
                request, num_computed_tokens, num_new_tokens, shift_computed_tokens
            ):
                return [], 0, encoder_compute_budget, []
            if replay.consumer:
                return [], num_new_tokens, encoder_compute_budget, []
        return original_encoder(
            self,
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens,
        )

    @wraps(original_update)
    def update_from_output(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        replay = find_replay(getattr(self, "connector", None))
        if replay is not None and replay.records:
            from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs

            failed = replay.failures()
            for rid, error in failed.items():
                if rid in self.requests:
                    logger.error(
                        "Remote image replay failed request=%s: %s", rid, error
                    )
            for request in self.finish_requests(
                list(failed), RequestStatus.FINISHED_ERROR
            ):
                client = request.client_index
                output = result.setdefault(client, EngineCoreOutputs())
                output.outputs.append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                if output.finished_requests is None:
                    output.finished_requests = set()
                output.finished_requests.add(request.request_id)
                # The normal method already emitted this step's finished map.
                # Do not leave a duplicate terminal notice for the next step.
                finished = self.finished_req_ids_dict
                if finished is not None and client in finished:
                    finished[client].discard(request.request_id)
                    if not finished[client]:
                        finished.pop(client)
        return result

    try_encoder._ilu_mm_replay = True
    Scheduler._try_schedule_encoder_inputs = try_encoder
    Scheduler.update_from_output = update_from_output
