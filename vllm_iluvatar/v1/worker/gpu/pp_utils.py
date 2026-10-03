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

``compute_need_sampled_mask`` / ``make_prepare_inputs_max_seq_len`` are wired
separately by ``ilu_patch.general_patches.patch_pp_finishing_relay``:

Target   : vllm.v1.worker.gpu.pp_utils.compute_need_sampled_mask
Baseline : v0.30.1
Why      : v0.27.1 left requests that finish at this step out of the
           sampled-token relay; v0.30.1 relays every sampling row.  A PD
           prefiller's requests all finish on their first token, so each of
           them now leaves a broadcast receive pending on PP 0..PP n-2 until the
           last stage samples.
How      : copy of the community function plus the v0.27.1 finishing
           exclusion, fed by ``max_seq_len`` (prompt_len + max_tokens) that
           ``prepare_inputs`` attaches to the batch.  Every PP rank derives it
           from the same scheduler output, so the collectives stay matched.
Scope    : PP>1 without speculative decoding (whose optimistic computed
           count could exclude a row that does not finish);
           VLLM_ILU_PP_SKIP_FINISHING_RELAY=0 restores the community mask.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

MAX_SEQ_LEN_ATTR = "ilu_max_seq_len_np"


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # >>> ILU MODIFICATION >>>
    # Exclude requests that we know are finished.
    max_seq_len = getattr(input_batch, MAX_SEQ_LEN_ATTR, None)
    if max_seq_len is not None:
        not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
        produces_sample = produces_sample & not_finishing
    # <<< ILU MODIFICATION <<<
    return produces_sample if produces_sample.any() else None


def make_prepare_inputs_max_seq_len(
    original: Callable[..., InputBatch],
) -> Callable[..., InputBatch]:
    @wraps(original)
    def prepare_inputs(self, *args: Any, **kwargs: Any) -> InputBatch:
        input_batch = original(self, *args, **kwargs)
        if self.pp_handler is not None and not self.num_speculative_steps:
            setattr(
                input_batch,
                MAX_SEQ_LEN_ATTR,
                self.req_states.max_seq_len[input_batch.idx_mapping_np],
            )
        return input_batch

    return prepare_inputs


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
