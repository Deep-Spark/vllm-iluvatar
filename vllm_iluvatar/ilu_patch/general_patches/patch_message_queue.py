# SPDX-License-Identifier: Apache-2.0
"""Scale MessageQueue ring depth for deep PP batch queues.

Corresponds to: vllm/distributed/device_communicators/shm_broadcast.py
                vllm/v1/executor/multiproc_executor.py

ILU general-plugins patch, auto-discovered during ``register_general``.

Target   : MessageQueue.__init__ default max_chunks
Baseline : main_6e448d0
Why      : Community default max_chunks=10. With PP, EngineCore fills
           batch_queue to pp_size and enqueues execute_model + sample_tokens
           per batch (2 RPCs). For tp1pp16 that is ~32 in-flight broadcast
           messages while mid-ranks block in PP send_object and cannot drain
           the ring → "No available shared memory broadcast block" deadlock.
           queue=1 avoids the hang by keeping RPC depth ≤ ring size, but
           kills pipeline fill.
How      : When callers leave max_chunks at the community default (10), raise
           it for:
             - multi-reader (rpc_broadcast): max(10, 2*n_reader + 8)
             - single-reader (worker response): max(10, 48), and cap
               max_chunk_bytes to 4MiB so 16× response rings stay affordable
           Override with VLLM_ILU_MQ_MAX_CHUNKS (int) when set.
Scope    : Iluvatar multiproc PP (and any other default-chunk MQ).
Remove   : when upstream sizes max_chunks from max_concurrent_batches.
"""

from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p

_COMMUNITY_DEFAULT_CHUNKS = 10
_RESPONSE_MAX_CHUNK_BYTES = 4 * 1024 * 1024


def _scaled_mq_limits(
    n_reader: int,
    max_chunks: int,
    max_chunk_bytes: int,
    env_raw: str = "",
) -> tuple[int, int]:
    """Scale default-10 rings; leave explicit caller sizes unchanged."""
    if max_chunks != _COMMUNITY_DEFAULT_CHUNKS:
        return max_chunks, max_chunk_bytes
    env_raw = env_raw.strip()
    if env_raw.isdigit():
        return max(_COMMUNITY_DEFAULT_CHUNKS, int(env_raw)), max_chunk_bytes
    if n_reader > 1:
        return (
            max(_COMMUNITY_DEFAULT_CHUNKS, 2 * int(n_reader) + 8),
            max_chunk_bytes,
        )
    chunks = max(_COMMUNITY_DEFAULT_CHUNKS, 48)
    if max_chunk_bytes > _RESPONSE_MAX_CHUNK_BYTES:
        max_chunk_bytes = _RESPONSE_MAX_CHUNK_BYTES
    return chunks, max_chunk_bytes


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="MessageQueue max_chunks scale for deep PP",
    )


def _apply() -> None:
    import vllm.envs as envs
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    orig_init = MessageQueue.__init__

    # [ILU override] MessageQueue.__init__ — delta: enlarge default max_chunks
    # for deep PP rpc/response rings; optional VLLM_ILU_MQ_MAX_CHUNKS override.
    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        bound = inspect.signature(orig_init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        n_reader = bound.arguments["n_reader"]
        max_chunks = bound.arguments["max_chunks"]
        max_chunk_bytes = bound.arguments["max_chunk_bytes"]
        env_raw = str(getattr(envs, "VLLM_ILU_MQ_MAX_CHUNKS", "") or "")
        new_chunks, new_bytes = _scaled_mq_limits(
            n_reader, max_chunks, max_chunk_bytes, env_raw
        )
        if (new_chunks, new_bytes) != (max_chunks, max_chunk_bytes):
            bound.arguments["max_chunks"] = new_chunks
            bound.arguments["max_chunk_bytes"] = new_bytes
            logger.debug(
                "ILU: MessageQueue max_chunks=%s n_reader=%s "
                "max_chunk_bytes=%s (deep PP rpc ring)",
                new_chunks,
                n_reader,
                new_bytes,
            )
        return orig_init(*bound.args, **bound.kwargs)

    p.check_signatures(
        MessageQueue,
        SimpleNamespace(__init__=__init__),
        ["__init__"],
        strict=True,
    )
    MessageQueue.__init__ = __init__  # type: ignore[method-assign]
