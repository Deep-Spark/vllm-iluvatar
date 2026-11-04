# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py

Target   : OffloadingConnectorWorker.register_kv_caches
Baseline : v0.31.0 (341bf7d4d124)
Why      : Preserve paired KV pages and rank-local CPU offload storage on CoreX.
How      : Canonicalize paired pages; delegate ordinary registration unchanged.
Scope    : DeepSeek V4.1 paired / non-uniform PP adapters.
Remove   : When community provides equivalent geometry and storage ownership.
"""

from __future__ import annotations

_ORIGINAL_REGISTER_KV = None


# [ILU override] _ilu_register_kv_caches — delta: adapt paired pages / rank-local PP geometry.
def _ilu_register_kv_caches(self, kv_caches):
    from vllm_iluvatar.models.deepseek_v41.offloading import (
        canonical_paired_pages,
        has_paired_pages,
    )

    if not has_paired_pages(self.kv_cache_config):
        assert _ORIGINAL_REGISTER_KV is not None
        return _ORIGINAL_REGISTER_KV(self, kv_caches)
    context = self.vllm_config.compilation_config.static_forward_context
    self._init_worker(canonical_paired_pages(self.kv_cache_config, kv_caches, context))
