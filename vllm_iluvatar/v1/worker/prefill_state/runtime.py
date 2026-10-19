"""Explicit model-state lifecycle shared by producer and consumer contexts."""

from __future__ import annotations


class PrefixRuntime:
    def __init__(self, runner, model):
        self.model_runner = runner
        self.vllm_config = runner.vllm_config
        self.device = runner.device
        self._table_updates = {}

    @staticmethod
    def request_eligible(request):
        # Token IDs alone do not identify image, embedding, or adapter inputs.
        return (
            request.prompt_token_ids is not None
            and request.prompt_len >= 128
            and request.prefill_token_ids == request.prompt_token_ids
            and not request.mm_features
            and getattr(request, "prompt_embeds", None) is None
            and request.lora_request is None
        )

    def expand_blocks(self, blocks):
        return [
            [block * factor + k for block in ids for k in range(factor)]
            for ids, factor in zip(
                blocks, self.model_runner.block_tables.blocks_per_kv_block
            )
        ]

    def initial_blocks(self, request):
        return self.expand_blocks(
            self._table_updates.get(request.req_id, request.block_ids)
        )

    def observe_scheduler_output(self, output):
        """Mirror the runner's public block-update records, without replacing it."""
        self._table_updates = output.block_table_updates or {}
        invalidated = output.finished_req_ids | set(output.preempted_req_ids or ())
        with self.lock:
            for rid, blocks in self._table_updates.items():
                if rid in self.states and rid not in invalidated:
                    self.states[rid]["blocks"] = self.expand_blocks(blocks)
            cached = output.scheduled_cached_reqs
            for rid, blocks in zip(cached.req_ids, cached.new_block_ids):
                state = self.states.get(rid)
                if (
                    state is None
                    or rid in invalidated
                    or rid in self._table_updates
                    or blocks is None
                ):
                    continue
                expanded = self.expand_blocks(blocks)
                state["blocks"] = [
                    old + new for old, new in zip(state["blocks"], expanded)
                ]

    def prepare_attn(self, batch, tables, slots):
        pass

    def bind_metadata(self, metadata):
        pass

    def receive_results(self, result):
        pass
