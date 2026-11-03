# SPDX-License-Identifier: Apache-2.0
"""Do not report a P/D KV transfer as a prefix-cache hit.

Corresponds to: vllm/v1/engine/output_processor.py
Target   : OutputProcessor.add_request, OutputProcessor.process_outputs
Baseline : v0.30.0rc0
Why      : Decode counts KV blocks pulled from Prefill as external cached
           tokens. A cold end-to-end request then reports cached_tokens close
           to prompt_tokens, and vllm:prompt_tokens_cached rises by the same
           amount. Those tokens were computed on Prefill.
How      : record whether a request explicitly asks for remote prefill, then
           exclude its transferred tokens before usage and iteration stats
           are read. CPU offload hits on ordinary or producer requests keep
           their original accounting. External, computed and cache-creation
           fields stay as recorded.
Scope    : requests carrying kv_transfer_params.do_remote_prefill.
Remove   : when serving stops folding KV transfer into cached_tokens.
"""

from __future__ import annotations

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def _report_local_prefix_hits(engine_core_outputs: object, request_states) -> None:
    if not isinstance(engine_core_outputs, list):
        return
    for output in engine_core_outputs:
        state = request_states.get(output.request_id)
        if not getattr(state, "_ilu_remote_prefill", False):
            continue
        stats = getattr(output, "prefill_stats", None)
        if stats is None:
            continue
        external = getattr(stats, "num_external_cached_tokens", 0) or 0
        if external <= 0:
            continue
        local = getattr(stats, "num_local_cached_tokens", 0) or 0
        stats.num_cached_tokens = local


def _apply() -> None:
    from vllm.v1.engine.output_processor import OutputProcessor

    if getattr(OutputProcessor.process_outputs, "_ilu_pd_transfer_usage", False):
        return

    original_add = OutputProcessor.add_request
    original = OutputProcessor.process_outputs

    @wraps(original_add)
    def add_request(self, request, *args, **kwargs):
        is_new = request.request_id not in self.request_states
        original_add(self, request, *args, **kwargs)
        # Streaming updates may omit the original transfer parameters.
        if not is_new:
            return
        state = self.request_states.get(request.request_id)
        if state is not None:
            extra = getattr(request.sampling_params, "extra_args", None) or {}
            transfer = extra.get("kv_transfer_params") or {}
            state._ilu_remote_prefill = bool(transfer.get("do_remote_prefill"))

    @wraps(original)
    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        _report_local_prefix_hits(engine_core_outputs, self.request_states)
        return original(self, engine_core_outputs, *args, **kwargs)

    process_outputs._ilu_pd_transfer_usage = True
    OutputProcessor.add_request = add_request
    OutputProcessor.process_outputs = process_outputs


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="KV transfer is not a prefix-cache hit",
    )
