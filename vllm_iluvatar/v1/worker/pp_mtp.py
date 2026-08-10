# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""PP + MTP GPU runner method implementations.

Corresponds to: vllm/v1/worker/gpu_model_runner.py
Baseline     : main_61ac3680210

Only methods whose required PP correction is in the middle of an upstream
method are copied here. Registration binds their code to the community module
globals, so the copied methods retain the community runtime namespace.
"""

from __future__ import annotations

_UPDATE_STATES_BASELINE = r'''def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        req_state = self.requests.pop(req_id, None)
        self._on_request_state_removed(req_id, req_state)
        self.num_prompt_logprobs.pop(req_id, None)
    self.late_interaction_runner.on_requests_finished(
        scheduler_output.finished_req_ids
    )
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Zero GPU memory for freshly allocated cache blocks to prevent
    # stale NaN/data from corrupting attention or SSM computation.
    if scheduler_output.new_block_ids_to_zero:
        self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
    if scheduler_output.kv_cache_block_copies:
        copy_kv_cache_blocks_inplace(
            self.kv_caches,
            self.kv_cache_config.num_blocks,
            scheduler_output.kv_cache_block_copies,
        )

    # Free the cached encoder outputs.
    self._process_encoder_cache_scheduler_output(scheduler_output)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
    # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
    # apart from the forced-preemption case in reset_prefix_cache. And in
    # that case we include the resumed_req_ids in the unscheduled set so
    # that they get cleared from the persistent batch before being re-scheduled
    # in the normal resumed request path.
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    is_ngram_gpu = (
        self.speculative_config is not None
        and self.speculative_config.use_ngram_gpu()
    )
    if is_ngram_gpu:
        ngram_gpu_new_reqs: list[CachedRequestState] = []

    reqs_to_add: list[CachedRequestState] = []
    deferred_spec_decode_corrections = []

    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        if req_id in self.requests:
            # For streaming case only.
            req_state = self._update_streaming_request(req_id, new_req_data)
            reqs_to_add.append(req_state)
            continue

        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if (
            sampling_params
            and sampling_params.sampling_type == SamplingType.RANDOM_SEED
        ):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            prompt_is_token_ids=new_req_data.prompt_is_token_ids,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state
        self.late_interaction_runner.register_request(req_id, pooling_params)

        if sampling_params and sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = (
                self.input_batch.vocab_size
                if sampling_params.prompt_logprobs == -1
                else sampling_params.prompt_logprobs
            )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._init_xdrope_positions(req_state)

        reqs_to_add.append(req_state)
        # Track new requests for ngram_gpu full tensor copy
        if is_ngram_gpu:
            ngram_gpu_new_reqs.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

    # Save scheduler-allocated spec lengths before trimming so
    # prev_num_draft_len keeps the optimistic count for rejection correction.
    original_num_spec_per_req: dict[str, int] = {}
    if (
        self.speculative_config is not None
        and self.speculative_config.use_ngram_gpu()
    ):
        for req_id, toks in scheduled_spec_tokens.items():
            original_num_spec_per_req[req_id] = len(toks)
        update_scheduler_for_invalid_drafts(
            self._num_valid_draft_tokens_event,
            self._num_valid_draft_tokens_cpu,
            scheduler_output,
            self.input_batch.req_id_to_index,
        )
    if self.use_async_spec_decode:
        self.prev_num_draft_tokens.np.fill(0)

    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_id in req_data.resumed_req_ids
        num_output_tokens = req_data.num_output_tokens[i]
        req_index = self.input_batch.req_id_to_index.get(req_id)

        if req_state.prev_num_draft_len and self.use_async_scheduling:
            # prev_num_draft_len is used in async scheduling mode with
            # spec decode. it indicates if need to update num_computed_tokens
            # of the request. for example:
            # first step: num_computed_tokens = 0, spec_tokens = [],
            # prev_num_draft_len = 0.
            # second step: num_computed_tokens = 100(prompt length),
            # spec_tokens = [a,b], prev_num_draft_len = 0.
            # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
            # prev_num_draft_len = 2.
            # num_computed_tokens in first step and second step doesn't contain
            # the spec tokens length, but in third step it contains the
            # spec tokens length. we only need to update num_computed_tokens
            # when prev_num_draft_len > 0.
            if req_index is None:
                req_state.prev_num_draft_len = 0
            else:
                # Optimistically assume all accepted; queue up a correction
                # to be called after the model forward to preserve async
                # scheduling. Corrected on GPU in _prepare_inputs.
                optimistic_num_accepted = req_state.prev_num_draft_len
                req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                deferred_spec_decode_corrections.append(
                    (req_id, optimistic_num_accepted, req_state)
                )

                prev_req_index = (
                    self.input_batch.prev_req_id_to_index.get(req_id)
                    if self.input_batch.prev_req_id_to_index
                    else None
                )
                if prev_req_index is not None:
                    self.prev_num_draft_tokens.np[prev_req_index] = (
                        optimistic_num_accepted
                    )

                if is_ngram_gpu and optimistic_num_accepted > 0:
                    self.input_batch.num_tokens_no_spec[req_index] += (
                        optimistic_num_accepted
                    )

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            if not req_data.new_token_ids:
                # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                new_token_ids: list[int] = []
            else:
                # Non-async scheduling with PP: The scheduler sends
                # sampled token ids back because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:]
                    )
        elif num_output_tokens < len(req_state.output_token_ids):
            # Some output tokens were discarded due to a sync-KV-load
            # failure, or output_token_ids was inflated by the optimistic
            # extend above (async spec decode). Align the cached state.
            del req_state.output_token_ids[num_output_tokens:]
            if req_index is not None:
                end_idx = (
                    self.input_batch.num_prompt_tokens[req_index]
                    + num_output_tokens
                )
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert req_index is None
            assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.

            if self.use_async_scheduling and num_output_tokens > 0:
                # We must recover the output token ids for resumed requests in the
                # async scheduling case, so that correct input_ids are obtained.
                resumed_token_ids = req_data.all_token_ids[req_id]
                req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

            reqs_to_add.append(req_state)
            # Track resumed requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            start_token_index = self.input_batch.num_tokens_no_spec[req_index]
            # For chunked prefill, num_computed_tokens may less
            # than num_tokens_no_spec.
            # Async scheduled PP: no new_token_ids, advance num_tokens_no_spec
            # according to num_computed_tokens.
            end_token_index = max(
                start_token_index,
                num_computed_tokens + len(new_token_ids),
            )
            if end_token_index > start_token_index:
                if new_token_ids:
                    # Add new_token_ids to token_ids_cpu.
                    num_new_tokens = end_token_index - start_token_index
                    tokens_to_append = new_token_ids[-num_new_tokens:]
                    self.input_batch.token_ids_cpu[
                        req_index, start_token_index:end_token_index
                    ] = tokens_to_append
                self.input_batch.is_token_ids[
                    req_index, start_token_index:end_token_index
                ] = True
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
        # Restore scheduler-side draft count after ngram trimming.
        if original_num_spec_per_req:
            orig = original_num_spec_per_req.get(req_id, 0)
            if orig != req_state.prev_num_draft_len:
                req_state.prev_num_draft_len = orig

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)
        self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()

    # Incrementally update ngram_gpu tensors after batch is stable
    if is_ngram_gpu:
        update_ngram_gpu_tensors_incremental(
            self.input_batch,
            self.token_ids_gpu_tensor,
            self.num_tokens_no_spec_gpu,
            ngram_gpu_new_reqs,
            self.device,
            _pinned_idx_buf=self._ngram_pinned_idx_buf,
            _pinned_val_buf=self._ngram_pinned_val_buf,
        )

    if deferred_spec_decode_corrections:

        def correct_spec_decode_token_counts():
            valid_sampled_token_count = self._get_valid_sampled_token_count()
            if not valid_sampled_token_count:
                return
            prev_req_id_to_index = self.input_batch.prev_req_id_to_index
            if not prev_req_id_to_index:
                return
            for (
                req_id,
                optimistic_num_accepted,
                req_state,
            ) in deferred_spec_decode_corrections:
                prev_req_index = prev_req_id_to_index.get(req_id)
                if prev_req_index is None:
                    continue
                num_accepted = valid_sampled_token_count[prev_req_index] - 1
                correction = optimistic_num_accepted - num_accepted
                req_state.num_computed_tokens -= correction
                cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                if cur_req_index is None:
                    continue
                self.input_batch.num_computed_tokens_cpu[cur_req_index] -= (
                    correction
                )
                if is_ngram_gpu and correction > 0:
                    self.input_batch.num_tokens_no_spec[cur_req_index] -= correction
                    self.num_tokens_no_spec_gpu[cur_req_index] -= correction

        return correct_spec_decode_token_counts
    else:
        return None
'''


def _ilu__update_states(self, scheduler_output: SchedulerOutput) -> Callable | None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        req_state = self.requests.pop(req_id, None)
        self._on_request_state_removed(req_id, req_state)
        self.num_prompt_logprobs.pop(req_id, None)
    self.late_interaction_runner.on_requests_finished(scheduler_output.finished_req_ids)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Zero GPU memory for freshly allocated cache blocks to prevent
    # stale NaN/data from corrupting attention or SSM computation.
    if scheduler_output.new_block_ids_to_zero:
        self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
    if scheduler_output.kv_cache_block_copies:
        copy_kv_cache_blocks_inplace(
            self.kv_caches,
            self.kv_cache_config.num_blocks,
            scheduler_output.kv_cache_block_copies,
        )

    # Free the cached encoder outputs.
    self._process_encoder_cache_scheduler_output(scheduler_output)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
    # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
    # apart from the forced-preemption case in reset_prefix_cache. And in
    # that case we include the resumed_req_ids in the unscheduled set so
    # that they get cleared from the persistent batch before being re-scheduled
    # in the normal resumed request path.
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    is_ngram_gpu = (
        self.speculative_config is not None and self.speculative_config.use_ngram_gpu()
    )
    if is_ngram_gpu:
        ngram_gpu_new_reqs: list[CachedRequestState] = []

    reqs_to_add: list[CachedRequestState] = []
    deferred_spec_decode_corrections = []

    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        if req_id in self.requests:
            # For streaming case only.
            req_state = self._update_streaming_request(req_id, new_req_data)
            reqs_to_add.append(req_state)
            continue

        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if (
            sampling_params
            and sampling_params.sampling_type == SamplingType.RANDOM_SEED
        ):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            prompt_is_token_ids=new_req_data.prompt_is_token_ids,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state
        self.late_interaction_runner.register_request(req_id, pooling_params)

        if sampling_params and sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = (
                self.input_batch.vocab_size
                if sampling_params.prompt_logprobs == -1
                else sampling_params.prompt_logprobs
            )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._init_xdrope_positions(req_state)

        reqs_to_add.append(req_state)
        # Track new requests for ngram_gpu full tensor copy
        if is_ngram_gpu:
            ngram_gpu_new_reqs.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

    # Save scheduler-allocated spec lengths before trimming so
    # prev_num_draft_len keeps the optimistic count for rejection correction.
    original_num_spec_per_req: dict[str, int] = {}
    if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
        for req_id, toks in scheduled_spec_tokens.items():
            original_num_spec_per_req[req_id] = len(toks)
        update_scheduler_for_invalid_drafts(
            self._num_valid_draft_tokens_event,
            self._num_valid_draft_tokens_cpu,
            scheduler_output,
            self.input_batch.req_id_to_index,
        )
    if self.use_async_spec_decode:
        self.prev_num_draft_tokens.np.fill(0)

    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_id in req_data.resumed_req_ids
        num_output_tokens = req_data.num_output_tokens[i]
        req_index = self.input_batch.req_id_to_index.get(req_id)

        if req_state.prev_num_draft_len and self.use_async_scheduling:
            # prev_num_draft_len is used in async scheduling mode with
            # spec decode. it indicates if need to update num_computed_tokens
            # of the request. for example:
            # first step: num_computed_tokens = 0, spec_tokens = [],
            # prev_num_draft_len = 0.
            # second step: num_computed_tokens = 100(prompt length),
            # spec_tokens = [a,b], prev_num_draft_len = 0.
            # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
            # prev_num_draft_len = 2.
            # num_computed_tokens in first step and second step doesn't contain
            # the spec tokens length, but in third step it contains the
            # spec tokens length. we only need to update num_computed_tokens
            # when prev_num_draft_len > 0.
            if req_index is None:
                req_state.prev_num_draft_len = 0
            else:
                # Optimistically assume all accepted; queue up a correction
                # to be called after the model forward to preserve async
                # scheduling. Corrected on GPU in _prepare_inputs.
                optimistic_num_accepted = req_state.prev_num_draft_len
                req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                deferred_spec_decode_corrections.append(
                    (req_id, optimistic_num_accepted, req_state)
                )

                prev_req_index = (
                    self.input_batch.prev_req_id_to_index.get(req_id)
                    if self.input_batch.prev_req_id_to_index
                    else None
                )
                if prev_req_index is not None:
                    self.prev_num_draft_tokens.np[prev_req_index] = (
                        optimistic_num_accepted
                    )

                if is_ngram_gpu and optimistic_num_accepted > 0:
                    self.input_batch.num_tokens_no_spec[req_index] += (
                        optimistic_num_accepted
                    )

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            if not req_data.new_token_ids:
                # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                new_token_ids: list[int] = []
            else:
                # Non-async scheduling with PP: The scheduler sends
                # sampled token ids back because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
        elif num_output_tokens < len(req_state.output_token_ids):
            # Some output tokens were discarded due to a sync-KV-load
            # failure, or output_token_ids was inflated by the optimistic
            # extend above (async spec decode). Align the cached state.
            del req_state.output_token_ids[num_output_tokens:]
            if req_index is not None:
                end_idx = (
                    self.input_batch.num_prompt_tokens[req_index] + num_output_tokens
                )
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert req_index is None
            assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.

            if self.use_async_scheduling and num_output_tokens > 0:
                # We must recover the output token ids for resumed requests in the
                # async scheduling case, so that correct input_ids are obtained.
                resumed_token_ids = req_data.all_token_ids[req_id]
                req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

            reqs_to_add.append(req_state)
            # Track resumed requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            start_token_index = self.input_batch.num_tokens_no_spec[req_index]
            # For chunked prefill, num_computed_tokens may less
            # than num_tokens_no_spec.
            # Async scheduled PP: no new_token_ids, advance num_tokens_no_spec
            # according to num_computed_tokens.
            end_token_index = max(
                start_token_index,
                num_computed_tokens + len(new_token_ids),
            )
            if end_token_index > start_token_index:
                if new_token_ids:
                    # Add new_token_ids to token_ids_cpu.
                    num_new_tokens = end_token_index - start_token_index
                    tokens_to_append = new_token_ids[-num_new_tokens:]
                    self.input_batch.token_ids_cpu[
                        req_index, start_token_index:end_token_index
                    ] = tokens_to_append
                self.input_batch.is_token_ids[
                    req_index, start_token_index:end_token_index
                ] = True
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
        # Restore scheduler-side draft count after ngram trimming.
        if original_num_spec_per_req:
            orig = original_num_spec_per_req.get(req_id, 0)
            if orig != req_state.prev_num_draft_len:
                req_state.prev_num_draft_len = orig

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)
        # PP non-last: requests re-added after unschedule/resume only get a dense
    # [prompt, prompt+len(output)) copy in add_request. With spec decode,
    # scheduler num_computed_tokens can extend past that dense tail (draft
    # slots in the prefix). Without realigning to req_data, drafts are
    # placed at num_tokens_no_spec from add_request and input_ids diverge
    # from KV → ~0% acceptance / garbage text.
    if not is_last_rank and req_data.new_token_ids and self.num_spec_tokens > 0:
        for req_state in reqs_to_add:
            req_id = req_state.req_id
            try:
                req_pos = req_data.req_ids.index(req_id)
            except ValueError:
                continue
            if req_pos >= len(req_data.new_token_ids):
                continue
            req_new = req_data.new_token_ids[req_pos]
            nct = req_data.num_computed_tokens[req_pos]
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            spec_toks = scheduled_spec_tokens.get(req_id, ())
            if req_new:
                end_main = nct + len(req_new)
                self.input_batch.token_ids_cpu[req_index, nct:end_main] = req_new
                self.input_batch.num_tokens_no_spec[req_index] = end_main
            elif spec_toks:
                self.input_batch.num_tokens_no_spec[req_index] = nct

    for request in reqs_to_add:
        self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()

    # Incrementally update ngram_gpu tensors after batch is stable
    if is_ngram_gpu:
        update_ngram_gpu_tensors_incremental(
            self.input_batch,
            self.token_ids_gpu_tensor,
            self.num_tokens_no_spec_gpu,
            ngram_gpu_new_reqs,
            self.device,
            _pinned_idx_buf=self._ngram_pinned_idx_buf,
            _pinned_val_buf=self._ngram_pinned_val_buf,
        )

    if deferred_spec_decode_corrections:

        def correct_spec_decode_token_counts():
            valid_sampled_token_count = self._get_valid_sampled_token_count()
            if not valid_sampled_token_count:
                return
            prev_req_id_to_index = self.input_batch.prev_req_id_to_index
            if not prev_req_id_to_index:
                return
            for (
                req_id,
                optimistic_num_accepted,
                req_state,
            ) in deferred_spec_decode_corrections:
                prev_req_index = prev_req_id_to_index.get(req_id)
                if prev_req_index is None:
                    continue
                num_accepted = valid_sampled_token_count[prev_req_index] - 1
                correction = optimistic_num_accepted - num_accepted
                req_state.num_computed_tokens -= correction
                cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                if cur_req_index is None:
                    continue
                self.input_batch.num_computed_tokens_cpu[cur_req_index] -= correction
                if is_ngram_gpu and correction > 0:
                    self.input_batch.num_tokens_no_spec[cur_req_index] -= correction
                    self.num_tokens_no_spec_gpu[cur_req_index] -= correction

        return correct_spec_decode_token_counts
    else:
        return None


_BUILD_ATTENTION_METADATA_BASELINE = r'''def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
    slot_mappings: dict[int, torch.Tensor] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    Returns:
        tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None

    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    assert num_reqs_padded is not None and num_tokens_padded is not None

    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    if for_cudagraph_capture:
        # For some attention backends (e.g. FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the sliding
        # window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_block_table(kv_cache_gid: int):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

        # Fill unused block table entries with NULL_BLOCK_ID (null block)
        # for CUDAGraph padding. Block 0 is reserved for padding.
        blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
        return blk_table_tensor

    assert slot_mappings is not None
    block_table_gid_0 = _get_block_table(0)
    slot_mapping_gid_0 = slot_mappings[0]

    if self.routed_experts_initialized:
        # Copy this step's attention slot_mapping into our private
        # device buffer. The shared ``slot_mappings[attn_gid]`` is
        # owned by the attention block table and will be overwritten
        # by the next ``_prepare_inputs``; we need a stable snapshot
        # because the async D2H may still be in flight on the copy
        # stream when the next step runs.
        attn_gid = self.routed_experts_attn_gid
        slot_mapping_attn = slot_mappings[attn_gid]
        self.routed_experts_slot_mapping_device[:num_tokens].copy_(
            slot_mapping_attn[:num_tokens]
        )

    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    seq_lens_cpu_upper_bound = seq_lens_cpu

    # is_prefilling: True if request is still in prefill phase.
    # Used by mamba backends to distinguish actual decodes from
    # short extends.
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    # Zero out padded rows so stale data from condense() doesn't
    # misclassify padding as prefill in CUDA graph mode.
    is_prefilling[num_reqs:] = False

    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    # Compute mm_prefix bidirectional ranges before building
    # attention metadata so builders handle them during build().
    # By default, ranges exceeding sliding_window are skipped to prevent
    # early tokens from attending across the entire image span. Models that
    # clamp mm_prefix to the sliding window *in-kernel* (e.g. Gemma4, which
    # needs HF's (causal OR blockwise) AND sliding_window on sliding layers)
    # opt out of the skip so the bidirectional range survives for images
    # larger than the window; the kernel then bounds it per-query.
    req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        hf_text_config = self.model_config.hf_text_config
        _bidi_sw = getattr(hf_text_config, "sliding_window", None)
        _clamps_in_kernel = getattr(
            self.model, "mm_prefix_clamp_sliding_window", False
        )
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                if mm_feature.modality == "audio":
                    continue
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                for r in img_doc_range:
                    if (
                        not _clamps_in_kernel
                        and _bidi_sw is not None
                        and (r[1] - r[0] + 1) > _bidi_sw
                    ):
                        continue
                    image_doc_ranges.append(r)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges

    # Reference Sliding Window Attention (R-SWA): pass per-request prompt
    # lengths so the attention backend can keep the prefix globally visible.
    # The backend owns the persistent CUDA-graph-safe GPU buffer.
    rswa_prefix_lens = None
    if self.model_config.rswa_window is not None:
        rswa_prefix_lens = num_prompt_tokens_cpu

    replayssm_decode_base_cpu = None
    if self.cache_config.use_replayssm:
        replayssm_decode_base_cpu = (
            self.input_batch.replayssm_decode_base_cpu_tensor[:num_reqs_padded]
        )

    cm_base = CommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        replayssm_decode_base_cpu=replayssm_decode_base_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        positions=self.positions[:num_tokens_padded],
        mm_req_doc_ranges=req_doc_ranges,
        rswa_prefix_lens=rswa_prefix_lens,
    )

    if self.dcp_world_size > 1:
        self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
            self.optimistic_seq_lens_cpu[:num_reqs],
            self.dcp_world_size,
            self.dcp_rank,
            self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
        self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

        cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
        cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
            :num_reqs_padded
        ]

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices
        )

    # Cache attention metadata builds across hybrid KV-cache groups
    # The only thing that changes between different hybrid KV-cache groups when the
    # same metadata builder and KVCacheSpec is the same is the block table, so we
    # can cache the attention metadata builds and just update the block table using
    # `builder.update_block_table` if the builder supports it.
    cached_attn_metadata: dict[
        tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
    ] = {}

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
        cache_key = (kv_cache_spec, type(builder))

        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
            if cascade_attn_prefix_lens
            else 0
        )

        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(
            builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                BailingLinearAttentionMetadataBuilder,
            ),
        ):
            assert ubid is None, (
                "UBatching not supported with GDN or linear attn yet"
            )
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                    :num_reqs_padded
                ],
            )
            if (
                isinstance(builder, Mamba2AttentionMetadataBuilder)
                and self.mamba_prev_last_scheduled_idx is not None
            ):
                extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                    self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                )

        if for_cudagraph_capture:
            attn_metadata_i = builder.build_for_cudagraph_capture(
                common_attn_metadata
            )
        elif (
            cache_key in cached_attn_metadata
            and builder.supports_update_block_table
        ):
            attn_metadata_i = builder.update_block_table(
                cached_attn_metadata[cache_key],
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.slot_mapping,
            )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            if builder.supports_update_block_table:
                cached_attn_metadata[cache_key] = attn_metadata_i

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm = copy(cm_base)  # shallow copy

        # Basically only the encoder seq_lens, block_table and slot_mapping change
        # for each kv_cache_group.
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
            for_cudagraph_capture=for_cudagraph_capture,
        )
        if kv_cache_gid > 0:
            cm.block_table_tensor = _get_block_table(kv_cache_gid)
            cm.slot_mapping = slot_mappings[kv_cache_gid]

        if self.speculative_config and spec_decode_common_attn_metadata is None:
            if isinstance(
                self.drafter,
                (
                    EagleProposer,
                    DFlashProposer,
                    Gemma4Proposer,
                    ExtractHiddenStatesProposer,
                ),
            ):
                if self.drafter.kv_cache_gid == kv_cache_gid:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        # Capture per-group block tables for multi-group proposers.
        if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
            )
        elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
            self.drafter.set_per_group_block_table(
                kv_cache_gid, cm.block_table_tensor
            )

        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            if ubatch_slices is not None:
                for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

            else:
                _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = (
            spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
        )

    return attn_metadata, spec_decode_common_attn_metadata
'''


def _ilu__build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
    slot_mappings: dict[int, torch.Tensor] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    Returns:
        tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None

    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    assert num_reqs_padded is not None and num_tokens_padded is not None

    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    if for_cudagraph_capture:
        # For some attention backends (e.g. FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the sliding
        # window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_block_table(kv_cache_gid: int):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

        # Fill unused block table entries with NULL_BLOCK_ID (null block)
        # for CUDAGraph padding. Block 0 is reserved for padding.
        blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
        return blk_table_tensor

    assert slot_mappings is not None
    block_table_gid_0 = _get_block_table(0)
    slot_mapping_gid_0 = slot_mappings[0]

    if self.routed_experts_initialized:
        # Copy this step's attention slot_mapping into our private
        # device buffer. The shared ``slot_mappings[attn_gid]`` is
        # owned by the attention block table and will be overwritten
        # by the next ``_prepare_inputs``; we need a stable snapshot
        # because the async D2H may still be in flight on the copy
        # stream when the next step runs.
        attn_gid = self.routed_experts_attn_gid
        slot_mapping_attn = slot_mappings[attn_gid]
        self.routed_experts_slot_mapping_device[:num_tokens].copy_(
            slot_mapping_attn[:num_tokens]
        )

    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    seq_lens_cpu_upper_bound = seq_lens_cpu

    # is_prefilling: True if request is still in prefill phase.
    # Used by mamba backends to distinguish actual decodes from
    # short extends.
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    # Zero out padded rows so stale data from condense() doesn't
    # misclassify padding as prefill in CUDA graph mode.
    is_prefilling[num_reqs:] = False

    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    # Compute mm_prefix bidirectional ranges before building
    # attention metadata so builders handle them during build().
    # By default, ranges exceeding sliding_window are skipped to prevent
    # early tokens from attending across the entire image span. Models that
    # clamp mm_prefix to the sliding window *in-kernel* (e.g. Gemma4, which
    # needs HF's (causal OR blockwise) AND sliding_window on sliding layers)
    # opt out of the skip so the bidirectional range survives for images
    # larger than the window; the kernel then bounds it per-query.
    req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        hf_text_config = self.model_config.hf_text_config
        _bidi_sw = getattr(hf_text_config, "sliding_window", None)
        _clamps_in_kernel = getattr(self.model, "mm_prefix_clamp_sliding_window", False)
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                if mm_feature.modality == "audio":
                    continue
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                for r in img_doc_range:
                    if (
                        not _clamps_in_kernel
                        and _bidi_sw is not None
                        and (r[1] - r[0] + 1) > _bidi_sw
                    ):
                        continue
                    image_doc_ranges.append(r)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges

    # Reference Sliding Window Attention (R-SWA): pass per-request prompt
    # lengths so the attention backend can keep the prefix globally visible.
    # The backend owns the persistent CUDA-graph-safe GPU buffer.
    rswa_prefix_lens = None
    if self.model_config.rswa_window is not None:
        rswa_prefix_lens = num_prompt_tokens_cpu

    replayssm_decode_base_cpu = None
    if self.cache_config.use_replayssm:
        replayssm_decode_base_cpu = self.input_batch.replayssm_decode_base_cpu_tensor[
            :num_reqs_padded
        ]

    cm_base = CommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        replayssm_decode_base_cpu=replayssm_decode_base_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        positions=self.positions[:num_tokens_padded],
        mm_req_doc_ranges=req_doc_ranges,
        rswa_prefix_lens=rswa_prefix_lens,
    )

    if self.dcp_world_size > 1:
        self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
            self.optimistic_seq_lens_cpu[:num_reqs],
            self.dcp_world_size,
            self.dcp_rank,
            self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
        self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

        cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
        cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[:num_reqs_padded]

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices
        )

    # Cache attention metadata builds across hybrid KV-cache groups
    # The only thing that changes between different hybrid KV-cache groups when the
    # same metadata builder and KVCacheSpec is the same is the block table, so we
    # can cache the attention metadata builds and just update the block table using
    # `builder.update_block_table` if the builder supports it.
    cached_attn_metadata: dict[
        tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
    ] = {}

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
        cache_key = (kv_cache_spec, type(builder))

        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
            if cascade_attn_prefix_lens
            else 0
        )

        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(
            builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                BailingLinearAttentionMetadataBuilder,
            ),
        ):
            assert ubid is None, "UBatching not supported with GDN or linear attn yet"
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                    :num_reqs_padded
                ],
            )
            if (
                isinstance(builder, Mamba2AttentionMetadataBuilder)
                and self.mamba_prev_last_scheduled_idx is not None
            ):
                extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                    self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                )

        if for_cudagraph_capture:
            attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
        elif cache_key in cached_attn_metadata and builder.supports_update_block_table:
            attn_metadata_i = builder.update_block_table(
                cached_attn_metadata[cache_key],
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.slot_mapping,
            )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            if builder.supports_update_block_table:
                cached_attn_metadata[cache_key] = attn_metadata_i

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm = copy(cm_base)  # shallow copy

        # Basically only the encoder seq_lens, block_table and slot_mapping change
        # for each kv_cache_group.
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
            for_cudagraph_capture=for_cudagraph_capture,
        )
        if kv_cache_gid > 0:
            cm.block_table_tensor = _get_block_table(kv_cache_gid)
            cm.slot_mapping = slot_mappings[kv_cache_gid]

        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and spec_decode_common_attn_metadata is None
        ):
            if isinstance(
                self.drafter,
                (
                    EagleProposer,
                    DFlashProposer,
                    Gemma4Proposer,
                    ExtractHiddenStatesProposer,
                ),
            ):
                if self.drafter.kv_cache_gid == kv_cache_gid:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        # Capture per-group block tables for multi-group proposers.
        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and isinstance(self.drafter, Step3p5MTPProposer)
        ):
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
            )
        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and isinstance(self.drafter, Gemma4Proposer)
        ):
            self.drafter.set_per_group_block_table(kv_cache_gid, cm.block_table_tensor)

        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            if ubatch_slices is not None:
                for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

            else:
                _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(
            num_tokens, num_reqs
        )

    return attn_metadata, spec_decode_common_attn_metadata


_DUMMY_RUN_BASELINE = r'''@torch.inference_mode()
def _dummy_run(
    self,
    num_tokens: int,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
    allow_microbatching: bool = True,
    skip_eplb: bool = False,
    is_profile: bool = False,
    create_mixed_batch: bool = False,
    remove_lora: bool = True,
    is_graph_capturing: bool = False,
    num_active_loras: int = 0,
    profile_seq_lens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a dummy forward pass to warm up/profile run or capture the
    CUDA graph for the model.

    Args:
        num_tokens: Number of tokens to run the dummy forward pass.
        cudagraph_runtime_mode: used to control the behavior.
            - if not set will determine the cudagraph mode based on using
                the self.cudagraph_dispatcher.
            - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
            - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
            - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                needed.
        force_attention: If True, always create attention metadata. Used to
            warm up attention backend when mode is NONE.
        uniform_decode: If True, the batch is a uniform decode batch.
        skip_eplb: If True, skip EPLB state update.
        is_profile: If True, this is a profile run.
        create_mixed_batch: If True, create a mixed batch with both decode
            (1 token) and prefill (multiple tokens) requests.
        remove_lora: If False, dummy LoRAs are not destroyed after the run
        num_active_loras: Number of distinct active LoRAs to capture for.
            LoRA is activated when num_active_loras > 0.
        profile_seq_lens: If provided, use this value for seq_lens instead
            of max_query_len. Used to profile attention workspace that
            scales with context length.
    """
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        # The current dummy run only covers LM execution, so we can skip it.
        # mm encoder dummy run may need to add in the future.
        return torch.tensor([]), torch.tensor([])

    assert (
        cudagraph_runtime_mode is None
        or cudagraph_runtime_mode.is_valid_runtime_mode()
    )

    # If cudagraph_mode.decode_mode() == FULL and
    # cudagraph_mode.separate_routine(). This means that we are using
    # different graphs and/or modes for mixed prefill-decode batches vs.
    # uniform decode batches. A uniform decode batch means that all
    # requests have identical query length, except a potential virtual
    # request (shorter) in the batch account for padding.
    # Uniform decode batch could either be common pure decode, where
    # max_query_len == 1, or speculative decode, where
    # max_query_len == 1 + num_spec_decode_tokens.

    # When setting max_query_len = 1, we switch to and capture the optimized
    # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
    # for GQA/MQA.
    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

    # Set num_scheduled_tokens based on num_tokens and max_num_seqs
    # for dummy run with LoRA so that the num_reqs collectively
    # has num_tokens in total.
    assert num_tokens <= self.max_num_tokens
    max_num_reqs = self.scheduler_config.max_num_seqs
    if create_mixed_batch:
        assert not uniform_decode
        # Create mixed batch:
        # first half decode tokens, second half one prefill
        num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
        num_prefill_tokens = num_tokens - num_decode_tokens
        num_reqs = num_decode_tokens + 1

        # Create decode requests (1 token each) followed by prefill request
        num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
        # Note: Overriding max_query_len to be the prefill tokens
        max_query_len = num_prefill_tokens
    elif uniform_decode:
        assert not create_mixed_batch
        num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

    assert sum(num_scheduled_tokens_list) == num_tokens
    assert len(num_scheduled_tokens_list) == num_reqs
    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())

    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile
            or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            # `force_num_active_loras` is used for cudagraph capture; because we
            # need to capture graphs for specific num_active_loras counts
            force_num_active_loras=num_active_loras,
        )
    )

    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = _cudagraph_mode
    else:
        assert cudagraph_runtime_mode == _cudagraph_mode, (
            f"Cudagraph runtime mode mismatch in dummy_run. "
            f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
        )

    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    dcp_dummy_context_len = get_dcp_dummy_context_len(
        self.dcp_world_size,
        self.parallel_config.cp_kv_cache_interleave_size,
        hasattr(self, "kv_cache_config"),
        create_mixed_batch,
        is_graph_capturing,
        uniform_decode,
    )
    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        self.vllm_config.parallel_config.num_ubatches,
    )
    logger.debug(
        "ubatch_slices: %s, ubatch_slices_padded: %s",
        ubatch_slices,
        ubatch_slices_padded,
    )

    attn_metadata: PerLayerAttnMetadata | None = None

    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens_unpadded,
        ubatch_slices=ubatch_slices_padded,
    )

    # Dummy runs have no real slot assignments — fill with -1 so
    # concat_and_cache kernels skip the KV write.
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
    # etc.) with execute_model.  It must participate in the same event
    # protocol so that back-to-back dummy/real steps don't overwrite
    # pinned memory while a prior non_blocking H2D DMA is still reading.
    with self.synchronize_input_prep():
        # If force_attention is True, we always capture attention.
        # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens  # type: ignore[assignment]
            elif create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                if dcp_dummy_context_len > 0:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1 + dcp_dummy_context_len] * num_decode_tokens
                        + [num_prefill_tokens + dcp_dummy_context_len],
                        dtype=torch.int,
                    )
                else:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=torch.int,
                    )
            elif dcp_dummy_context_len > 0:
                seq_lens = max_query_len + dcp_dummy_context_len  # type: ignore[assignment]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

            cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens, self.query_pos.np
            )
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                cum_num_tokens[-1]
            )
            self.query_start_loc.copy_to_gpu()

            prepare_dcp_dummy_context_metadata(
                input_batch=self.input_batch,
                kv_cache_config=getattr(self, "kv_cache_config", None),
                query_pos=self.query_pos,
                positions=self.positions,
                query_start_loc=self.query_start_loc,
                num_reqs=num_reqs,
                num_tokens_unpadded=num_tokens_unpadded,
                dcp_dummy_context_len=dcp_dummy_context_len,
            )

            # Sync block table CPU->GPU so cleared rows from
            # remove_request() are visible to the attention metadata
            # builder. Without this, stale block IDs from finished
            # requests can corrupt Mamba state.
            self.input_batch.block_table.commit_block_table(num_reqs_padded)

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                for_cudagraph_capture=is_graph_capturing,
                slot_mappings=slot_mappings_by_group,
                use_spec_decode=self.speculative_config is not None,
            )

    with self.maybe_dummy_run_with_lora(
        self.lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora,
        num_active_loras,
    ):
        # Make sure padding doesn't exceed max_num_tokens
        assert num_tokens_padded <= self.max_num_tokens
        model_kwargs = self._init_model_kwargs()
        if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

            model_kwargs = {
                **model_kwargs,
                **self._dummy_mm_kwargs(num_reqs),
            }
        elif self.enable_prompt_embeds:
            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            model_kwargs = self._init_model_kwargs()
        else:
            input_ids = self.input_ids.gpu[:num_tokens_padded]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_tokens_padded]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
        else:
            positions = self.positions[:num_tokens_padded]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )
                )

            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_tokens_padded, None, False
            )

        if ubatch_slices_padded is not None:
            # Adjust values to reflect a single ubatch.
            # TODO(sage,lucas): this is cruft that should be addressed in
            #  the padding refactor.
            num_tokens_padded = ubatch_slices_padded[0].num_tokens
            if num_tokens_across_dp is not None:
                num_tokens_across_dp[:] = num_tokens_padded

        with (
            self.maybe_randomize_inputs(input_ids, inputs_embeds),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
            ),
        ):
            outputs = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = outputs
        else:
            hidden_states = outputs

        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            assert self.speculative_config is not None
            # Eagle currently only supports PIECEWISE cudagraphs.
            # Therefore only use cudagraphs if the main model uses PIECEWISE
            # NOTE(lucas): this is a hack, need to clean up.
            use_cudagraphs = (
                (
                    is_graph_capturing
                    and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                )
                or (
                    not is_graph_capturing
                    and cudagraph_runtime_mode != CUDAGraphMode.NONE
                )
            ) and not self.speculative_config.enforce_eager

            # Note(gnovack) - We need to disable cudagraphs for one of the two
            # lora cases when cudagraph_specialize_lora is enabled. This is a
            # short term mitigation for issue mentioned in
            # https://github.com/vllm-project/vllm/issues/28334
            if (
                self.compilation_config.cudagraph_specialize_lora
                and num_active_loras > 0
            ):
                use_cudagraphs = False

            self.drafter.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )

    # We register layerwise NVTX hooks here after the first dynamo tracing is
    # done to avoid nvtx operations in hook functions being traced by
    # torch dynamo and causing graph breaks.
    # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
    # compiled model's dynamo tracing is only done once and the compiled model's
    # __call__ function is replaced by calling the compiled function.
    # So it's safe to register hooks here. Hooks will be registered to
    # both compiled and uncompiled models but they will never
    # be called on the compiled model execution path.
    self._register_layerwise_nvtx_hooks()

    # This is necessary to avoid blocking DP.
    # For dummy runs, we typically skip EPLB since we don't have any real
    # requests to process.
    # However, in DP settings, there may be cases when some DP ranks do
    # not have any requests to process, so they're executing dummy batches.
    # In such cases, we still have to trigger EPLB to make sure
    # ranks execute the rearrangement in synchronization.
    if not skip_eplb:
        self.eplb_step(is_dummy=True, is_profile=is_profile)

    logit_indices = np.cumsum(num_scheduled_tokens) - 1
    logit_indices_device = torch.from_numpy(logit_indices).to(
        self.device, non_blocking=True
    )
    return hidden_states, hidden_states[logit_indices_device]
'''


def _ilu__dummy_run(
    self,
    num_tokens: int,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
    allow_microbatching: bool = True,
    skip_eplb: bool = False,
    is_profile: bool = False,
    create_mixed_batch: bool = False,
    remove_lora: bool = True,
    is_graph_capturing: bool = False,
    num_active_loras: int = 0,
    profile_seq_lens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a dummy forward pass to warm up/profile run or capture the
    CUDA graph for the model.

    Args:
        num_tokens: Number of tokens to run the dummy forward pass.
        cudagraph_runtime_mode: used to control the behavior.
            - if not set will determine the cudagraph mode based on using
                the self.cudagraph_dispatcher.
            - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
            - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
            - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                needed.
        force_attention: If True, always create attention metadata. Used to
            warm up attention backend when mode is NONE.
        uniform_decode: If True, the batch is a uniform decode batch.
        skip_eplb: If True, skip EPLB state update.
        is_profile: If True, this is a profile run.
        create_mixed_batch: If True, create a mixed batch with both decode
            (1 token) and prefill (multiple tokens) requests.
        remove_lora: If False, dummy LoRAs are not destroyed after the run
        num_active_loras: Number of distinct active LoRAs to capture for.
            LoRA is activated when num_active_loras > 0.
        profile_seq_lens: If provided, use this value for seq_lens instead
            of max_query_len. Used to profile attention workspace that
            scales with context length.
    """
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        # The current dummy run only covers LM execution, so we can skip it.
        # mm encoder dummy run may need to add in the future.
        return torch.tensor([]), torch.tensor([])

    assert (
        cudagraph_runtime_mode is None or cudagraph_runtime_mode.is_valid_runtime_mode()
    )

    # If cudagraph_mode.decode_mode() == FULL and
    # cudagraph_mode.separate_routine(). This means that we are using
    # different graphs and/or modes for mixed prefill-decode batches vs.
    # uniform decode batches. A uniform decode batch means that all
    # requests have identical query length, except a potential virtual
    # request (shorter) in the batch account for padding.
    # Uniform decode batch could either be common pure decode, where
    # max_query_len == 1, or speculative decode, where
    # max_query_len == 1 + num_spec_decode_tokens.

    # When setting max_query_len = 1, we switch to and capture the optimized
    # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
    # for GQA/MQA.
    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

    # Set num_scheduled_tokens based on num_tokens and max_num_seqs
    # for dummy run with LoRA so that the num_reqs collectively
    # has num_tokens in total.
    assert num_tokens <= self.max_num_tokens
    max_num_reqs = self.scheduler_config.max_num_seqs
    if create_mixed_batch:
        assert not uniform_decode
        # Create mixed batch:
        # first half decode tokens, second half one prefill
        num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
        num_prefill_tokens = num_tokens - num_decode_tokens
        num_reqs = num_decode_tokens + 1

        # Create decode requests (1 token each) followed by prefill request
        num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
        # Note: Overriding max_query_len to be the prefill tokens
        max_query_len = num_prefill_tokens
    elif uniform_decode:
        assert not create_mixed_batch
        num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

    assert sum(num_scheduled_tokens_list) == num_tokens
    assert len(num_scheduled_tokens_list) == num_reqs
    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())

    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            # `force_num_active_loras` is used for cudagraph capture; because we
            # need to capture graphs for specific num_active_loras counts
            force_num_active_loras=num_active_loras,
        )
    )

    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = _cudagraph_mode
    else:
        assert cudagraph_runtime_mode == _cudagraph_mode, (
            f"Cudagraph runtime mode mismatch in dummy_run. "
            f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
        )

    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    dcp_dummy_context_len = get_dcp_dummy_context_len(
        self.dcp_world_size,
        self.parallel_config.cp_kv_cache_interleave_size,
        hasattr(self, "kv_cache_config"),
        create_mixed_batch,
        is_graph_capturing,
        uniform_decode,
    )
    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        self.vllm_config.parallel_config.num_ubatches,
    )
    logger.debug(
        "ubatch_slices: %s, ubatch_slices_padded: %s",
        ubatch_slices,
        ubatch_slices_padded,
    )

    attn_metadata: PerLayerAttnMetadata | None = None

    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens_unpadded,
        ubatch_slices=ubatch_slices_padded,
    )

    # Dummy runs have no real slot assignments — fill with -1 so
    # concat_and_cache kernels skip the KV write.
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
    # etc.) with execute_model.  It must participate in the same event
    # protocol so that back-to-back dummy/real steps don't overwrite
    # pinned memory while a prior non_blocking H2D DMA is still reading.
    with self.synchronize_input_prep():
        # If force_attention is True, we always capture attention.
        # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens  # type: ignore[assignment]
            elif create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                if dcp_dummy_context_len > 0:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1 + dcp_dummy_context_len] * num_decode_tokens
                        + [num_prefill_tokens + dcp_dummy_context_len],
                        dtype=torch.int,
                    )
                else:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=torch.int,
                    )
            elif dcp_dummy_context_len > 0:
                seq_lens = max_query_len + dcp_dummy_context_len  # type: ignore[assignment]
            else:
                seq_lens = self.max_model_len  # type: ignore[assignment]
            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

            # typos:ignore
            cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens, self.query_pos.np
            )
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                cum_num_tokens[-1]
            )
            self.query_start_loc.copy_to_gpu()

            prepare_dcp_dummy_context_metadata(
                input_batch=self.input_batch,
                kv_cache_config=getattr(self, "kv_cache_config", None),
                query_pos=self.query_pos,
                positions=self.positions,
                query_start_loc=self.query_start_loc,
                num_reqs=num_reqs,
                num_tokens_unpadded=num_tokens_unpadded,
                dcp_dummy_context_len=dcp_dummy_context_len,
            )

            # Sync block table CPU->GPU so cleared rows from
            # remove_request() are visible to the attention metadata
            # builder. Without this, stale block IDs from finished
            # requests can corrupt Mamba state.
            self.input_batch.block_table.commit_block_table(num_reqs_padded)

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                for_cudagraph_capture=is_graph_capturing,
                slot_mappings=slot_mappings_by_group,
                use_spec_decode=self.speculative_config is not None,
            )

    with self.maybe_dummy_run_with_lora(
        self.lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora,
        num_active_loras,
    ):
        # Make sure padding doesn't exceed max_num_tokens
        assert num_tokens_padded <= self.max_num_tokens
        model_kwargs = self._init_model_kwargs()
        if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

            model_kwargs = {
                **model_kwargs,
                **self._dummy_mm_kwargs(num_reqs),
            }
        elif self.enable_prompt_embeds:
            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            model_kwargs = self._init_model_kwargs()
        else:
            input_ids = self.input_ids.gpu[:num_tokens_padded]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_tokens_padded]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
        else:
            positions = self.positions[:num_tokens_padded]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=self.max_num_tokens,
                    dtype=self.model_config.dtype,
                    device=self.device,
                )

            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_tokens_padded, None, False
            )

        if ubatch_slices_padded is not None:
            # Adjust values to reflect a single ubatch.
            # TODO(sage,lucas): this is cruft that should be addressed in
            #  the padding refactor.
            num_tokens_padded = ubatch_slices_padded[0].num_tokens
            if num_tokens_across_dp is not None:
                num_tokens_across_dp[:] = num_tokens_padded

        with (
            self.maybe_randomize_inputs(input_ids, inputs_embeds),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
            ),
        ):
            outputs = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = outputs
        else:
            hidden_states = outputs

        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            )
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            assert self.speculative_config is not None
            # Eagle currently only supports PIECEWISE cudagraphs.
            # Therefore only use cudagraphs if the main model uses PIECEWISE
            # NOTE(lucas): this is a hack, need to clean up.
            use_cudagraphs = (
                (
                    is_graph_capturing
                    and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                )
                or (
                    not is_graph_capturing
                    and cudagraph_runtime_mode != CUDAGraphMode.NONE
                )
            ) and not self.speculative_config.enforce_eager

            # Note(gnovack) - We need to disable cudagraphs for one of the two
            # lora cases when cudagraph_specialize_lora is enabled. This is a
            # short term mitigation for issue mentioned in
            # https://github.com/vllm-project/vllm/issues/28334
            if (
                self.compilation_config.cudagraph_specialize_lora
                and num_active_loras > 0
            ):
                use_cudagraphs = False

            self.drafter.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )

    # We register layerwise NVTX hooks here after the first dynamo tracing is
    # done to avoid nvtx operations in hook functions being traced by
    # torch dynamo and causing graph breaks.
    # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
    # compiled model's dynamo tracing is only done once and the compiled model's
    # __call__ function is replaced by calling the compiled function.
    # So it's safe to register hooks here. Hooks will be registered to
    # both compiled and uncompiled models but they will never
    # be called on the compiled model execution path.
    self._register_layerwise_nvtx_hooks()

    # This is necessary to avoid blocking DP.
    # For dummy runs, we typically skip EPLB since we don't have any real
    # requests to process.
    # However, in DP settings, there may be cases when some DP ranks do
    # not have any requests to process, so they're executing dummy batches.
    # In such cases, we still have to trigger EPLB to make sure
    # ranks execute the rearrangement in synchronization.
    if not skip_eplb:
        self.eplb_step(is_dummy=True, is_profile=is_profile)

    logit_indices = np.cumsum(num_scheduled_tokens) - 1
    logit_indices_device = torch.from_numpy(logit_indices).to(
        self.device, non_blocking=True
    )
    if isinstance(hidden_states, IntermediateTensors):
        return hidden_states, None
    return hidden_states, hidden_states[logit_indices_device]


def make_runner_init(original_init):
    """Set the inline draft-output mode after normal runner initialization."""

    def _ilu_runner_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # [ILU override] GPUModelRunner.__init__ - delta: PP ranks must return
        # fresh drafts through ModelRunnerOutput rather than a later RPC.
        self.draft_in_model_output = (
            self.num_spec_tokens > 0 and self.parallel_config.pipeline_parallel_size > 1
        )

    return _ilu_runner_init


def make_sample_tokens(original_sample_tokens):
    """Attach fresh draft tokens to the synchronous PP model output.

    PP + MTP does not support async scheduling, so the wrapped
    ``sample_tokens`` always returns a plain ``ModelRunnerOutput`` (never an
    async wrapper) on the path this override handles.
    """

    def _ilu_sample_tokens(self, *args, **kwargs):
        output = original_sample_tokens(self, *args, **kwargs)
        # [ILU override] GPUModelRunner.sample_tokens - delta: the scheduler
        # consumes drafts from the output produced by the same PP batch.
        if getattr(self, "draft_in_model_output", False) and hasattr(output, "req_ids"):
            output.draft_token_ids = self.take_draft_token_ids()
        return output

    return _ilu_sample_tokens


def _call_without_spec_off_last_rank(original, self, args, kwargs):
    """Run ``original`` with ``speculative_config`` hidden on non-last PP ranks.

    Non-last PP ranks never build ``self.drafter`` (its creation in
    ``GPUModelRunner.__init__`` is guarded by ``get_pp_group().is_last_rank``).
    Community runner methods guard their drafter-only sub-steps with
    ``self.speculative_config`` alone, so clearing it off the last rank skips
    exactly those sub-steps while leaving every other branch untouched. This
    reproduces the ``and get_pp_group().is_last_rank`` guard without copying the
    whole method (and without a fingerprint baseline to keep in sync).
    """
    from vllm.distributed.parallel_state import get_pp_group

    if get_pp_group().is_last_rank:
        return original(self, *args, **kwargs)
    saved_spec_config = self.speculative_config
    self.speculative_config = None
    try:
        return original(self, *args, **kwargs)
    finally:
        self.speculative_config = saved_spec_config


def make_check_and_update_cudagraph_mode(original):
    """Skip the drafter cudagraph-keys sub-step on non-last PP ranks."""

    def _ilu_check_and_update_cudagraph_mode(self, *args, **kwargs):
        # [ILU override] GPUModelRunner._check_and_update_cudagraph_mode - delta:
        # off the last PP rank there is no drafter, so skip that sub-step.
        return _call_without_spec_off_last_rank(original, self, args, kwargs)

    return _ilu_check_and_update_cudagraph_mode


def make_initialize_metadata_builders(original):
    """Skip the drafter attn-backend init sub-step on non-last PP ranks."""

    def _ilu_initialize_metadata_builders(self, *args, **kwargs):
        # [ILU override] GPUModelRunner.initialize_metadata_builders - delta:
        # off the last PP rank there is no drafter, so skip that sub-step.
        return _call_without_spec_off_last_rank(original, self, args, kwargs)

    return _ilu_initialize_metadata_builders
