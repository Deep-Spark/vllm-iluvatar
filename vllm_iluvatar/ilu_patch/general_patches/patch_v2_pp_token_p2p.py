# SPDX-License-Identifier: Apache-2.0
"""Wire the V2 PP sampled-token send/recv transport behind an Iluvatar opt-in.

Corresponds to:
    vllm/v1/worker/gpu/pp_utils.py

Target   : vllm.v1.worker.gpu.pp_utils.PPHandler
Baseline : v0.31.0 (db9527a468)
Why      : IXCCL has no copy-engine broadcast, so the community collective
           leaves a resident ``ixcclDevKernel_Broadcast_RING_LL`` pending on a
           non-last PP rank for the whole rank skew. Per SWDRV-5168 that pins
           the 256th compute kernel dispatched after it starts until it
           retires, which serialises the pipeline.
How      : replace the PPHandler collectives with ixfd send/recv at
           register time instead of modifying community vLLM in place.
Layout   : this file only wires the patch; the implementation lives in
           vllm_iluvatar/v1/worker/gpu/pp_utils.py.
Scope    : Iluvatar V2 pipeline-parallel runs.
Enable   : opt-in via ``VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P=1``.
Remove   : on CQ silicon, where the hardware event table allocates dynamically
           and a resident collective kernel no longer blocks compute.
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
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())
'''

BROADCAST_BASELINE = r'''
    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
'''

BROADCAST_DRAFTS_BASELINE = r'''
    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
'''


def register() -> None:
    """Register the V2 PP sampled-token transport patch once per process."""
    p.register_once(
        __name__,
        _apply,
        detail="V2 PP sampled-token relay over ixfd send/recv",
    )


def _enabled() -> bool:
    import vllm.envs as envs

    return bool(envs.VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P)


def _apply() -> None:
    if not _enabled():
        return

    from vllm.v1.worker.gpu import pp_utils as pp_utils_mod

    import vllm_iluvatar.v1.worker.gpu.pp_utils as pp_impl

    handler_cls = pp_utils_mod.PPHandler
    for symbol, method, baseline in (
        ("PPHandler.receive", handler_cls.receive, RECEIVE_BASELINE),
        ("PPHandler.broadcast", handler_cls.broadcast, BROADCAST_BASELINE),
        (
            "PPHandler.broadcast_drafts",
            handler_cls.broadcast_drafts,
            BROADCAST_DRAFTS_BASELINE,
        ),
    ):
        p.check_fingerprint(
            symbol,
            inspect.getsource(method),
            baseline,
            resync_hint=(
                "Re-sync vllm_iluvatar/v1/worker/gpu/pp_utils.py and the "
                f"baseline for {symbol} in patch_v2_pp_token_p2p.py to the "
                "current community source."
            ),
        )

    overrides = {
        "__init__": pp_impl.make_pphandler_init(handler_cls.__init__),
        "receive": pp_impl.make_receive(pp_utils_mod),
        "broadcast": pp_impl.make_broadcast(pp_utils_mod),
        "broadcast_drafts": pp_impl.make_broadcast_drafts(pp_utils_mod),
    }
    p.check_signatures(
        handler_cls,
        types.SimpleNamespace(**overrides),
        sorted(overrides),
        strict=True,
    )
    for name, override in overrides.items():
        setattr(handler_cls, name, override)
