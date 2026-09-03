# SPDX-License-Identifier: Apache-2.0
"""Wire V2 PP + MTP compatibility behind an Iluvatar opt-in flag.

Corresponds to:
    vllm/v1/worker/gpu/pp_utils.py
    vllm/v1/worker/gpu/model_runner.py

Target   : V2 GPU model runner under pipeline parallelism + MTP.
Baseline : main_6e448d0
Why      : non-last PP ranks need sampled-token broadcasts with a fixed width
           and need the proposed draft tokens relayed from the last PP rank.
How      : wrap the small PPHandler methods and GPUModelRunner hooks at
           register-time instead of modifying community vLLM in place.
Layout   : this file only wires patches; implementations are split between
           vllm_iluvatar/v1/worker/gpu/pp_utils.py and model_runner.py.
Scope    : Iluvatar V2 PP + MTP runs.
Enable   : opt-in via ``VLLM_ILU_ENABLE_V2_PP_MTP=1``.
Remove   : after the equivalent V2 PP + MTP fixes are upstream.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

RECEIVE_BASELINE = r'''
    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())
'''


def register() -> None:
    """Register the V2 PP + MTP compatibility patch once per process."""
    p.register_once(
        __name__,
        _apply,
        detail="V2 PP + MTP sampled/draft-token broadcast compatibility",
    )


def _v2_pp_mtp_enabled() -> bool:
    import vllm.envs as envs

    return bool(envs.VLLM_ILU_ENABLE_V2_PP_MTP)


def _apply() -> None:
    if not _v2_pp_mtp_enabled():
        return

    from vllm.v1.worker.gpu import model_runner as runner_mod
    from vllm.v1.worker.gpu import pp_utils as pp_utils_mod

    import vllm_iluvatar.v1.worker.gpu.model_runner as runner_impl
    import vllm_iluvatar.v1.worker.gpu.pp_utils as pp_impl

    handler_cls = pp_utils_mod.PPHandler
    p.check_fingerprint(
        "PPHandler.receive",
        inspect.getsource(handler_cls.receive),
        RECEIVE_BASELINE,
        resync_hint=(
            "Re-sync vllm_iluvatar/v1/worker/gpu/pp_utils.py and the "
            "RECEIVE_BASELINE in patch_v2_pp_mtp.py to the current community "
            "PPHandler.receive source."
        ),
    )
    p.check_signatures(
        handler_cls,
        types.SimpleNamespace(
            __init__=pp_impl.make_pphandler_init(handler_cls.__init__),
            get_prev_sampled_outputs=pp_impl.make_get_prev_sampled_outputs(
                handler_cls.get_prev_sampled_outputs
            ),
            receive=pp_impl.make_receive(pp_utils_mod),
            broadcast=pp_impl.make_broadcast(handler_cls.broadcast),
        ),
        ["__init__", "get_prev_sampled_outputs", "receive", "broadcast"],
        strict=True,
    )
    runner_cls = runner_mod.GPUModelRunner
    p.check_signatures(
        runner_cls,
        types.SimpleNamespace(
            __init__=runner_impl.make_runner_init(runner_cls.__init__),
            update_pp_decode_requests=runner_impl.make_update_pp_decode_requests(
                runner_cls.update_pp_decode_requests
            ),
            sample_tokens=runner_impl.make_sample_tokens(runner_cls.sample_tokens),
        ),
        ["__init__", "update_pp_decode_requests", "sample_tokens"],
        strict=True,
    )

    handler_cls.__init__ = pp_impl.make_pphandler_init(handler_cls.__init__)
    handler_cls.get_prev_sampled_outputs = pp_impl.make_get_prev_sampled_outputs(
        handler_cls.get_prev_sampled_outputs
    )
    handler_cls.receive = pp_impl.make_receive(pp_utils_mod)
    handler_cls.broadcast = pp_impl.make_broadcast(handler_cls.broadcast)
    handler_cls.broadcast_draft = pp_impl.make_broadcast_draft(pp_utils_mod)

    runner_cls.__init__ = runner_impl.make_runner_init(runner_cls.__init__)
    runner_cls.update_pp_decode_requests = runner_impl.make_update_pp_decode_requests(
        runner_cls.update_pp_decode_requests
    )
    runner_cls.sample_tokens = runner_impl.make_sample_tokens(runner_cls.sample_tokens)
