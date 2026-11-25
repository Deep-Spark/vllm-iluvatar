# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure wire-format tag validation helpers for Ilu NIXL handshakes."""

from __future__ import annotations


def validate_main_kv_format(
    local_format: str,
    remote_format: str,
    *,
    remote_engine_id: str,
) -> bool:
    """Check explicit format identity and return whether hash checks are mandatory."""
    supported = ("none", "i8g64", "i4g32")
    for side, value in (("local", local_format), ("remote", remote_format)):
        if value not in supported:
            raise RuntimeError(f"Unsupported {side} NIXL main KV format: {value!r}")
    quantized = local_format != "none" or remote_format != "none"
    if quantized and local_format != remote_format:
        raise RuntimeError(
            f"NIXL main KV format mismatch for remote engine {remote_engine_id}: "
            f"local={local_format}, remote={remote_format}"
        )
    return quantized


def validate_indexer_kv_layout(
    local_layout: str,
    remote_layout: str,
    *,
    remote_engine_id: str,
) -> None:
    """Reject split vs legacy indexer planes even when main KV is unquantized."""
    supported = ("legacy", "split")
    for side, value in (("local", local_layout), ("remote", remote_layout)):
        if value not in supported:
            raise RuntimeError(f"Unsupported {side} NIXL indexer KV layout: {value!r}")
    if local_layout != remote_layout:
        raise RuntimeError(
            f"NIXL indexer KV layout mismatch for remote engine "
            f"{remote_engine_id}: local={local_layout}, remote={remote_layout}"
        )
