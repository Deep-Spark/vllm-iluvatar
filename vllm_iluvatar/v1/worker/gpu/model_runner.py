# SPDX-License-Identifier: Apache-2.0
"""PP + speculative-decode helpers for V2 ``GPUModelRunner``.

Corresponds to: vllm/v1/worker/gpu/model_runner.py
Baseline : v0.30.0rc0
Why      : with PP batch_queue, ``DraftTokensHandler.get_draft_tokens`` returns
           ``[-1]*K`` placeholders, and a later in-flight step can overwrite
           ``req_states.draft_tokens`` and ``last_sampled_tokens`` before the
           prior batch is verified. V2 does not apply
           ``CachedRequestData.new_token_ids``; that payload is the catch-up
           token, so ranks otherwise verify from a stale ``last_sampled``.
How      : (1) D2H real draft ids onto ``ModelRunnerOutput``;
           (2) before ``prepare_inputs``, stage scheduled draft ids and the
           PP catch-up token (``new_token_ids[-1]``) into CUDA buffers.
Scope    : sync scheduling + PP>1 + speculative decode.
Remove   : when community stages both drafts and last_sampled for PP verify.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch
from vllm.v1.outputs import DraftTokenIds


def _runner_uses_async_scheduling(runner: Any) -> bool:
    sched = getattr(runner, "scheduler_config", None)
    if sched is None:
        vcfg = getattr(runner, "vllm_config", None)
        sched = getattr(vcfg, "scheduler_config", None) if vcfg is not None else None
    return bool(getattr(sched, "async_scheduling", False))


def _pp_size(runner: Any) -> int:
    vcfg = getattr(runner, "vllm_config", None)
    parallel = getattr(vcfg, "parallel_config", None) if vcfg is not None else None
    if parallel is not None:
        return max(int(getattr(parallel, "pipeline_parallel_size", 1) or 1), 1)
    return 1


def _needs_inline_drafts(runner: Any) -> bool:
    """PP batch-queue + sync spec needs drafts on the model-output contract."""
    if _runner_uses_async_scheduling(runner):
        return False
    return _pp_size(runner) > 1 and int(getattr(runner, "num_speculative_steps", 0)) > 0


def _pp_catchup_token_by_req(scheduler_output: Any) -> dict[str, int]:
    """Last non-draft token per req from PP ``CachedRequestData.new_token_ids``."""
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is None:
        return {}
    req_ids = list(getattr(cached, "req_ids", []) or [])
    new_toks = list(getattr(cached, "new_token_ids", []) or [])
    out: dict[str, int] = {}
    for i, req_id in enumerate(req_ids):
        if i >= len(new_toks) or not new_toks[i]:
            continue
        # Leading logits row uses the token immediately before drafts.
        out[req_id] = int(new_toks[i][-1])
    return out


def stage_scheduler_drafts(runner: Any, scheduler_output: Any) -> None:
    """Stage scheduled drafts and PP catch-up into CUDA verify buffers."""
    if _runner_uses_async_scheduling(runner):
        return
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    req_states = getattr(runner, "req_states", None)
    if not scheduled or req_states is None or not hasattr(req_states, "draft_tokens"):
        return
    draft_buf = req_states.draft_tokens
    width = int(draft_buf.shape[-1])
    catchup = _pp_catchup_token_by_req(scheduler_output)
    last_buf = getattr(req_states, "last_sampled_tokens", None)
    for req_id, tokens in scheduled.items():
        idx = req_states.req_id_to_index.get(req_id)
        if idx is None:
            continue
        ids = [int(tok) for tok in tokens if int(tok) >= 0]
        if not ids:
            # Keep local propose / PP-relayed drafts; never restage -1.
            continue
        row = np.full(width, -1, dtype=np.int64)
        row[: min(len(ids), width)] = ids[:width]
        dest = draft_buf[idx]
        dest.copy_(torch.as_tensor(row, dtype=dest.dtype, device=dest.device))
        # V2 ignores CachedRequestData.new_token_ids; without this, PP ranks
        # keep a later step's last_sampled and verify from the wrong token.
        if last_buf is not None and req_id in catchup:
            last_buf[idx] = catchup[req_id]


def real_draft_token_ids(runner: Any) -> DraftTokenIds | None:
    """D2H the just-proposed drafts for the scheduler (not ``[-1]*K``)."""
    handler = getattr(runner, "draft_tokens_handler", None)
    req_ids = list(getattr(handler, "req_ids", []) or [])
    if not req_ids:
        return runner.take_draft_token_ids()
    rows: list[list[int]] = []
    for req_id in req_ids:
        idx = runner.req_states.req_id_to_index[req_id]
        draft = runner.req_states.draft_tokens[idx]
        if hasattr(draft, "detach"):
            draft = draft.detach().to("cpu", non_blocking=False)
        rows.append([int(tok) for tok in np.asarray(draft).reshape(-1).tolist()])
    return DraftTokenIds(req_ids, rows)


def attach_draft_token_ids(runner: Any, output: Any) -> None:
    if not _needs_inline_drafts(runner):
        return
    drafts = real_draft_token_ids(runner)
    if drafts is None:
        return
    inner = getattr(output, "model_runner_output", None)
    if inner is not None:
        inner.draft_token_ids = drafts
        return
    if hasattr(output, "req_ids"):
        output.draft_token_ids = drafts


def make_prepare_inputs(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def prepare_inputs(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        stage_scheduler_drafts(self, scheduler_output)
        return original(self, scheduler_output, *args, **kwargs)

    return prepare_inputs


def make_sample_tokens(
    original_sample_tokens: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original_sample_tokens)
    def _ilu_sample_tokens(self, grammar_output):
        restore_supports_mm_inputs = None
        if (
            self.pp_handler is not None
            and self.speculator is not None
            and self.speculator.supports_mm_inputs
            and not self.model_state.supports_mm_inputs
        ):
            # Encoder runner exists only on PP rank 0.
            restore_supports_mm_inputs = self.speculator.supports_mm_inputs
            self.speculator.supports_mm_inputs = False
        try:
            output = original_sample_tokens(self, grammar_output)
            attach_draft_token_ids(self, output)
            return output
        finally:
            if restore_supports_mm_inputs is not None:
                self.speculator.supports_mm_inputs = restore_supports_mm_inputs

    return _ilu_sample_tokens
