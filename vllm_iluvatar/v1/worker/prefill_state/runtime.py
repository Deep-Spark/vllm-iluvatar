"""Explicit model-state lifecycle shared by producer and consumer contexts."""

from __future__ import annotations


def attach_prefix_state(connector, state):
    """Attach to the unique native NIXL child, including inside MultiConnector."""
    endpoints = []

    def visit(current):
        children = getattr(current, "sub_connectors", None)
        if children is not None:
            for child in children:
                visit(child)
        elif callable(getattr(current, "attach_prefix_state", None)):
            endpoints.append(current)

    visit(connector)
    if len(endpoints) != 1:
        raise RuntimeError("Prefix state requires exactly one native NIXL endpoint")
    endpoints[0].attach_prefix_state(state)


class PrefixRuntime:
    def __init__(self, runner, model):
        self.model_runner = runner
        self.vllm_config = runner.vllm_config
        self.device = runner.device
        self._table_updates = {}
        from vllm_iluvatar.v1.worker.mm_replay.configuration import ReplayOptions
        from vllm_iluvatar.v1.worker.mm_replay.identity import model_identity

        options = ReplayOptions.from_config(self.vllm_config)
        self._mm_prefix_reuse = options.prefix_reuse
        self._mm_model_identity = (
            model_identity(self.vllm_config.model_config)
            if options.prefix_reuse
            else None
        )

    @staticmethod
    def request_eligible(request, allow_mm=False):
        # Image reuse additionally requires a certified complete input identity.
        eligible = (
            request.prompt_token_ids is not None
            and request.prompt_len >= 128
            and request.prefill_token_ids == request.prompt_token_ids
            and getattr(request, "prompt_embeds", None) is None
            and request.lora_request is None
        )
        if not eligible or not request.mm_features:
            return eligible
        if not allow_mm:
            return False
        from vllm_iluvatar.v1.worker.mm_replay.identity import feature_descriptors

        try:
            feature_descriptors(request.mm_features)
        except (ValueError, AttributeError, TypeError):
            return False
        return True

    def mm_input_identity(self, request, token_sha256):
        if not request.mm_features or not self._mm_prefix_reuse:
            return None
        from vllm_iluvatar.v1.worker.mm_replay.identity import (
            feature_descriptors,
            request_identity,
        )

        return request_identity(
            self._mm_model_identity,
            token_sha256,
            request.prompt_len,
            feature_descriptors(request.mm_features),
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
