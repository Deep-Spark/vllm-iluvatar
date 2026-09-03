# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/pp_utils.py

ILU wrapper implementation for V2 PPHandler MTP token relay. Runtime wiring and
signature checks live in
``ilu_patch.general_patches.patch_v2_pp_mtp`` and are auto-discovered during
``register_general``.

Target   : vllm.v1.worker.gpu.pp_utils.PPHandler
Baseline : main_6e448d0
Why      : Under PP + MTP, non-last PP ranks must receive the last rank's
           sampled tokens and proposed draft tokens before their next decode
           verification step. NCCL broadcast also requires every rank to enter
           collectives in the same order with identical element counts.
How      : call the community PPHandler methods, padding sampled-token payloads
           before broadcast and appending one draft-token collective after the
           community receive/broadcast sequence when V2 PP+MTP is opted in.
Scope    : V2 pipeline-parallel MTP/speculative decoding on Iluvatar.
Enable   : opt-in via ``VLLM_ILU_ENABLE_V2_PP_MTP=1``.
Remove   : when upstream carries equivalent V2 PP + MTP sampled/draft-token
           broadcast compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch


def make_pphandler_init(original_init: Callable[..., None]) -> Callable[..., None]:
    @wraps(original_init)
    def _ilu_init(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        relay_draft_tokens: bool = False,
    ) -> None:
        original_init(self, max_num_reqs, num_speculative_steps, device)
        # V2 PP+MTP needs an extra draft-token collective. Keep it opt-in per
        # runner so normal PP decode preserves the upstream collective sequence.
        self.relay_draft_tokens = relay_draft_tokens

    return _ilu_init


def make_get_prev_sampled_outputs(
    original_get_prev_sampled_outputs: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original_get_prev_sampled_outputs)
    def _ilu_get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        # Keep a reference to the slot before the community method pops it. The
        # community method owns stale-request filtering; we only attach the draft
        # tensor that receive() stored on the same pending slot.
        slot = self.queue[0] if self.queue else None
        outputs = original_get_prev_sampled_outputs(self)
        if outputs is not None:
            outputs["draft_tokens"] = getattr(slot, "draft_tokens", None)
        return outputs

    return _ilu_get_prev_sampled_outputs


def make_receive(pp_utils_mod: Any) -> Callable[..., bool]:
    def _ilu_receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = pp_utils_mod.compute_need_sampled_mask(input_batch)
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
            # >>> ILU MODIFICATION >>>
            # Match the last-rank third broadcast when MTP draft relay is enabled.
            draft_tokens = None
            if getattr(self, "relay_draft_tokens", False):
                draft_tokens = torch.empty(
                    num_reqs,
                    self.max_sample_len - 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            # <<< ILU MODIFICATION <<<
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)

        pending = pp_utils_mod.PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        # >>> ILU MODIFICATION >>>
        # Attach the optional draft tensor without changing the community dataclass.
        pending.draft_tokens = draft_tokens
        # <<< ILU MODIFICATION <<<
        self.queue[-1] = pending
        return bool(need_sampled_mask.all())

    return _ilu_receive


def make_broadcast(original_broadcast: Callable[..., None]) -> Callable[..., None]:
    @wraps(original_broadcast)
    def _ilu_broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        # The receiver always posts [num_reqs, max_sample_len]. Under spec decode
        # the sampler can emit width 1 before rejection sampling has run, so pad
        # the source columns to keep NCCL element counts identical.
        width = sampled_token_ids.shape[-1]
        if width != self.max_sample_len:
            assert width < self.max_sample_len
            padded = sampled_token_ids.new_full(
                (sampled_token_ids.shape[0], self.max_sample_len), -1
            )
            padded[:, :width] = sampled_token_ids
            sampled_token_ids = padded

        original_broadcast(
            self, sampled_token_ids, num_sampled, num_rejected, input_batch
        )

    return _ilu_broadcast


def make_broadcast_draft(pp_utils_mod: Any) -> Callable[..., None]:
    def _ilu_broadcast_draft(self, draft_tokens: torch.Tensor, input_batch: InputBatch):
        """Relay proposed draft tokens from the last PP rank to non-last ranks."""
        assert self.is_last_rank
        if not self.relay_draft_tokens:
            return
        if pp_utils_mod.compute_need_sampled_mask(input_batch) is None:
            # `broadcast` skipped too, so skip here to keep op counts matched.
            return
        draft_tokens = draft_tokens.to(torch.int64).contiguous()
        with torch.cuda.stream(self.broadcast_stream):
            # wait_stream so the side-stream broadcast sees propose()'s output.
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                draft_tokens, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens.record_stream(self.broadcast_stream)

    return _ilu_broadcast_draft
