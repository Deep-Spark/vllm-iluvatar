# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/model_runner.py

ILU wrapper implementation for V2 GPUModelRunner PP + MTP state handoff.
Runtime wiring and signature checks live in
``ilu_patch.general_patches.patch_v2_pp_mtp`` and are auto-discovered during
``register_general``.

Target   : vllm.v1.worker.gpu.model_runner.GPUModelRunner
Baseline : main_6e448d0
Why      : In PP+MTP, the draft model/proposer runs on the last PP rank, but
           non-last ranks build later verification inputs too. They therefore
           need the last rank's proposed draft tokens written into
           ``req_states.draft_tokens`` before the next decode step.
How      : wrap runner initialization to enable draft relay on PPHandler, wrap
           speculative propose() on the last rank to broadcast drafts, and wrap
           ``postprocess_sampled`` during community ``update_pp_decode_requests``
           so the relayed draft tensor is consumed without copying the upstream
           update method.
Scope    : V2 pipeline-parallel MTP/speculative decoding on Iluvatar.
Enable   : opt-in via ``VLLM_ILU_ENABLE_V2_PP_MTP=1``.
Remove   : when upstream carries equivalent V2 PP + MTP draft-token relay.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm.v1.worker.gpu.input_batch import InputBatch


def make_runner_init(original_init: Callable[..., None]) -> Callable[..., None]:
    @wraps(original_init)
    def _ilu_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if self.pp_handler is not None:
            # Diffusion LLMs may have speculative steps without a speculator; only
            # a real speculative config proposes draft tokens that need relaying.
            self.pp_handler.relay_draft_tokens = self.speculative_config is not None

    return _ilu_init


def make_update_pp_decode_requests(
    original_update_pp_decode_requests: Callable[..., None],
) -> Callable[..., None]:
    @wraps(original_update_pp_decode_requests)
    def _ilu_update_pp_decode_requests(self):
        original_postprocess_sampled = self.postprocess_sampled

        def _ilu_postprocess_sampled(**outputs: Any) -> None:
            # Upstream only applies sampled tokens. Under PP+MTP, the last rank is
            # also the drafter, so non-last ranks must consume the relayed drafts
            # before their next combine_sampled_and_draft_tokens() call.
            draft_tokens = outputs.pop("draft_tokens", None)
            idx_mapping = outputs["idx_mapping"]
            original_postprocess_sampled(**outputs)
            if draft_tokens is not None:
                # idx_mapping may contain -1 for requests freed after receive; keep
                # draft state aligned with the same filtered rows as sampled tokens.
                valid = idx_mapping >= 0
                self.req_states.draft_tokens[idx_mapping[valid]] = draft_tokens[valid]

        self.postprocess_sampled = _ilu_postprocess_sampled
        try:
            return original_update_pp_decode_requests(self)
        finally:
            self.postprocess_sampled = original_postprocess_sampled

    return _ilu_update_pp_decode_requests


def _wrap_speculator_propose(model_runner: Any) -> None:
    speculator = model_runner.speculator
    if (
        model_runner.pp_handler is None
        or speculator is None
        or getattr(speculator, "_ilu_v2_pp_mtp_propose_wrapped", False)
    ):
        return

    original_propose = speculator.propose

    @wraps(original_propose)
    def _ilu_propose(input_batch: InputBatch, *args: Any, **kwargs: Any):
        # TO DO:Isolation only: force DeepSeek MTP drafter propose() to eager while
        # leaving the target model CUDA graph controlled by server config.
        kwargs["is_profile"] = True
        draft_tokens = original_propose(input_batch, *args, **kwargs)
        # The proposed drafts are produced only on the last PP rank. Relay them on
        # the same side stream/group as sampled tokens so peer ranks verify the
        # same draft sequence instead of their zero-initialized local buffer.
        model_runner.pp_handler.broadcast_draft(draft_tokens, input_batch)
        return draft_tokens

    speculator.propose = _ilu_propose
    speculator._ilu_v2_pp_mtp_propose_wrapped = True


def make_sample_tokens(
    original_sample_tokens: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original_sample_tokens)
    def _ilu_sample_tokens(self, grammar_output):
        _wrap_speculator_propose(self)

        restore_supports_mm_inputs = None
        if (
            self.pp_handler is not None
            and self.speculator is not None
            and self.speculator.supports_mm_inputs
            and not self.model_state.supports_mm_inputs
        ):
            # The encoder runner exists only on PP rank 0. For later ranks, avoid
            # asking the model_state to gather embeddings it does not own.
            restore_supports_mm_inputs = self.speculator.supports_mm_inputs
            self.speculator.supports_mm_inputs = False
        try:
            return original_sample_tokens(self, grammar_output)
        finally:
            if restore_supports_mm_inputs is not None:
                self.speculator.supports_mm_inputs = restore_supports_mm_inputs

    return _ilu_sample_tokens
