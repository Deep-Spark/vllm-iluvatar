# SPDX-License-Identifier: Apache-2.0
"""Treat Kpool tail pages as ring buffers in DecodeBenchConnector."""

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="DecodeBenchConnector Kpool tail rings")


def _apply() -> None:
    from vllm.v1 import kv_cache_interface

    kpool_tail_spec = getattr(kv_cache_interface, "KpoolTailSpec", None)
    if kpool_tail_spec is None:
        return

    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        decode_bench_connector as community,
    )

    if hasattr(community, "_CIRCULAR_CACHE_SPECS"):
        return

    circular_buffer_spec = kv_cache_interface.CircularBufferSpec
    if getattr(community, "CircularBufferSpec", None) is not circular_buffer_spec:
        return

    # The v0.31.0 connector uses this module-local name only as the type
    # argument to isinstance in scheduler allocation, startup fill, and
    # worker zero-fill. The upstream cache spec classes stay intact.
    community.__dict__["CircularBufferSpec"] = (
        circular_buffer_spec,
        kpool_tail_spec,
    )
