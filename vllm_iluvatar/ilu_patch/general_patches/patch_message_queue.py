# SPDX-License-Identifier: Apache-2.0
"""Scale MessageQueue ring depth for deep PP batch queues.

Corresponds to: vllm/distributed/device_communicators/shm_broadcast.py
                vllm/v1/executor/multiproc_executor.py

ILU general-plugins patch, auto-discovered during ``register_general``.

Target   : MessageQueue.__init__ default max_chunks
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

import os

from vllm_iluvatar.ilu_patch import general_patch as p

_COMMUNITY_DEFAULT_CHUNKS = 10
_RESPONSE_MAX_CHUNK_BYTES = 4 * 1024 * 1024


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="MessageQueue max_chunks scale for deep PP",
    )


def _apply() -> None:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    orig_init = MessageQueue.__init__

    def __init__(
        self,
        n_reader,
        n_local_reader,
        local_reader_ranks=None,
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = _COMMUNITY_DEFAULT_CHUNKS,
        connect_ip: str | None = None,
    ):
        if max_chunks == _COMMUNITY_DEFAULT_CHUNKS:
            env_raw = os.environ.get("VLLM_ILU_MQ_MAX_CHUNKS", "").strip()
            if env_raw.isdigit():
                max_chunks = max(_COMMUNITY_DEFAULT_CHUNKS, int(env_raw))
            elif n_reader > 1:
                # rpc_broadcast_mq: readers == world_size (e.g. pp*tp).
                max_chunks = max(
                    _COMMUNITY_DEFAULT_CHUNKS, 2 * int(n_reader) + 8
                )
            else:
                # worker_response_mq (only output_rank actually enqueues).
                max_chunks = max(_COMMUNITY_DEFAULT_CHUNKS, 48)
                if max_chunk_bytes > _RESPONSE_MAX_CHUNK_BYTES:
                    max_chunk_bytes = _RESPONSE_MAX_CHUNK_BYTES

            logger.info(
                "ILU: MessageQueue max_chunks=%s n_reader=%s "
                "max_chunk_bytes=%s (deep PP rpc ring)",
                max_chunks,
                n_reader,
                max_chunk_bytes,
            )

        return orig_init(
            self,
            n_reader,
            n_local_reader,
            local_reader_ranks,
            max_chunk_bytes,
            max_chunks,
            connect_ip,
        )

    MessageQueue.__init__ = __init__  # type: ignore[method-assign]
