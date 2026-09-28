# SPDX-License-Identifier: Apache-2.0
"""Compile PP ``post_update`` during warmup without extra PP broadcasts.

Corresponds to: vllm/v1/worker/gpu/warmup.py
Target   : warmup_kernels
Baseline : v0.30.0
Why      : non-last PP ranks never hit ``post_update`` during community
           warmup; the first real batch JITs mid-flight, and flushing extra
           samples after CUDA-graph capture hits
           ``ixErrorStreamCaptureUnsupported``.
How      : after community warmup, launch ``post_update`` on live buffers with
           skipped ``idx_mapping``.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="PP post_update kernel compiled during warmup",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu import warmup as community

    from vllm_iluvatar.v1.worker.gpu.warmup import make_warmup_kernels

    community.warmup_kernels = make_warmup_kernels(community.warmup_kernels)
