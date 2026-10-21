# SPDX-License-Identifier: Apache-2.0
"""Do not report a P/D KV transfer as a prefix-cache hit.

Corresponds to: vllm/v1/engine/output_processor.py
Target   : OutputProcessor.process_outputs
Baseline : v0.30.0rc0
Why      : Decode counts KV blocks pulled from Prefill as external cached
           tokens. A cold end-to-end request then reports cached_tokens close
           to prompt_tokens, and vllm:prompt_tokens_cached rises by the same
           amount. Those tokens were computed on Prefill.
How      : before usage and iteration stats are read, rewrite
           prefill_stats.num_cached_tokens to the local prefix hit only.
           External, computed, and cache-creation fields stay as recorded,
           so vllm:external_prefix_cache_hits still counts the transfer.
Scope    : workers whose prefill stats include external KV-transfer tokens.
Remove   : when serving stops folding KV transfer into cached_tokens.
"""

from __future__ import annotations

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def _report_local_prefix_hits(engine_core_outputs: object) -> None:
    if not isinstance(engine_core_outputs, list):
        return
    for output in engine_core_outputs:
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

    original = OutputProcessor.process_outputs

    @wraps(original)
    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        _report_local_prefix_hits(engine_core_outputs)
        return original(self, engine_core_outputs, *args, **kwargs)

    OutputProcessor.process_outputs = process_outputs


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="KV transfer is not a prefix-cache hit",
    )
