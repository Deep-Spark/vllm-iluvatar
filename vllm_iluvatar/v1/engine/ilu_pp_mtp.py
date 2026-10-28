# SPDX-License-Identifier: Apache-2.0
"""Pair V4 V2 synchronous-PP drafts with the sampled batch they belong to.

The PP batch queue can return before sampled tokens are accounted. Publishing
drafts from post_step at that point schedules verification without the bonus
input row. Keep the actual per-batch proposals with its output instead; publish
them only after update_from_output has accounted samples and rejections.
"""

from __future__ import annotations

from functools import wraps

from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput

_DRAFT_ATTRIBUTE = "_ilu_v4_pp_draft_token_ids"


def needs_paired_drafts(config) -> bool:
    spec = config.speculative_config
    return (
        config.model_config.hf_config.model_type == "deepseek_v4"
        and spec is not None
        and spec.method == "mtp"
        and config.parallel_config.pipeline_parallel_size > 1
        and config.use_v2_model_runner
        and not config.scheduler_config.async_scheduling
    )


def _attach_drafts(output, drafts):
    if output.req_ids != drafts.req_ids:
        raise RuntimeError("V4 PP sampled output and draft request identities differ")
    setattr(output, _DRAFT_ATTRIBUTE, drafts)
    return output


class _PairedAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, output, drafts):
        self.output = output
        self.drafts = drafts

    def get_output(self) -> ModelRunnerOutput:
        return _attach_drafts(self.output.get_output(), self.drafts)


def wrap_sample_tokens(original):
    @wraps(original)
    def sample_tokens(self, grammar_output):
        output = original(self, grammar_output)
        if (
            output is None
            or not needs_paired_drafts(self.vllm_config)
            or not self.model_runner.is_last_pp_rank
        ):
            return output
        drafts = self.model_runner.take_draft_token_ids()
        if drafts is None:
            raise RuntimeError("V4 PP sample has no corresponding MTP proposals")
        # Snapshot before a later sample replaces the worker's draft handler.
        drafts = DraftTokenIds(
            list(drafts.req_ids), [list(tokens) for tokens in drafts.draft_token_ids]
        )
        if isinstance(output, AsyncModelRunnerOutput):
            return _PairedAsyncOutput(output, drafts)
        return _attach_drafts(output, drafts)

    return sample_tokens


def wrap_update_from_output(original):
    @wraps(original)
    def update_from_output(self, scheduler_output, model_runner_output):
        paired = needs_paired_drafts(self.vllm_config)
        drafts = getattr(model_runner_output, _DRAFT_ATTRIBUTE, None)
        if paired and model_runner_output.req_ids and drafts is None:
            raise RuntimeError("V4 PP output is missing its batch-bound MTP proposals")
        eligible = []
        if paired and drafts is not None:
            # A queued earlier prefill chunk may have no sample even though a
            # later chunk has already flipped shared is_prefill_chunk to false.
            # Stale results from before preemption must not publish old drafts.
            for req_id, tokens in zip(drafts.req_ids, drafts.draft_token_ids):
                request = self.requests.get(req_id)
                index = model_runner_output.req_id_to_index[req_id]
                if (
                    request is not None
                    and request.num_stale_output_tokens == 0
                    and model_runner_output.sampled_token_ids[index]
                ):
                    eligible.append(
                        (req_id, tokens, request, request.num_output_tokens)
                    )
        outputs = original(self, scheduler_output, model_runner_output)
        # The original can discard an invalid KV load or finish/remove a
        # request. Require actual output progress in the same request object.
        publish = [
            (req_id, tokens)
            for req_id, tokens, request, before in eligible
            if self.requests.get(req_id) is request
            and request.num_output_tokens > before
        ]
        if publish:
            self.update_draft_token_ids(
                DraftTokenIds(
                    [req_id for req_id, _ in publish], [tokens for _, tokens in publish]
                )
            )
        return outputs

    return update_from_output


def wrap_post_step(original):
    @wraps(original)
    def post_step(self, model_executed):
        if needs_paired_drafts(self.vllm_config):
            # Proposals are published with each completed batch, including a
            # queue drain whose current step executes no model tokens.
            return None
        return original(self, model_executed)

    return post_step
