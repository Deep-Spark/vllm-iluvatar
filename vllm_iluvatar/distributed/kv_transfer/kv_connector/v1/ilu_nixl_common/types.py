# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared region/pairing types and Run-Length Encoding (RLE) block-ID wire codecs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegionPairing:
    """Explicit 4-tuple pairing between local and remote regions and groups."""

    local_region_id: int
    remote_region_id: int
    local_group_id: int
    remote_group_id: int
    identity: str = ""


@dataclass(frozen=True)
class HmaTransferRoute:
    """Per-(local_group, remote_group) descriptor routing for hybrid MLA+Mamba HMA."""

    local_group_idx: int
    remote_group_idx: int
    is_ssm: bool
    local_desc_region_ids: np.ndarray
    remote_desc_region_ids: np.ndarray
    raw_local_region_ids: tuple[int, ...]
    raw_remote_region_ids: tuple[int, ...]


@dataclass
class SemanticTransferRegion:
    """Semantic NIXL transfer region descriptor shared across Ilu NIXL connectors."""

    region_id: int
    block_len: int
    shared_by: list[str]
    layer_indices: list[int]
    group_ids: list[int]
    cache_kinds: list[str]
    identities: list[str]
    physical_backing_id: int = 0
    byte_offset: int = 0
    block_stride: int = 0
    kernel_blocks: int = 1
    block_tokens: int = 0
    num_blocks: int = 0
    identity_groups: dict[str, list[int]] | None = None


def encode_block_ids_rle(
    block_ids: Sequence[Sequence[int]] | Sequence[int],
) -> list[list[list[int]]]:
    """Compress grouped block IDs into JSON-friendly ``[[start_block_id, run_len], ...]``."""
    if not block_ids:
        return []
    if not isinstance(block_ids[0], (list, tuple)):
        groups: Sequence[Sequence[int]] = (block_ids,)  # type: ignore[assignment]
    else:
        groups = block_ids  # type: ignore[assignment]

    encoded: list[list[list[int]]] = []
    for group in groups:
        if not group:
            encoded.append([])
            continue
        runs: list[list[int]] = []
        run_start = int(group[0])
        run_len = 1
        prev = run_start
        for raw_blk in group[1:]:
            blk = int(raw_blk)
            if blk == prev + 1:
                run_len += 1
            else:
                runs.append([run_start, run_len])
                run_start = blk
                run_len = 1
            prev = blk
        runs.append([run_start, run_len])
        encoded.append(runs)
    return encoded


def decode_block_ids_rle(
    rle_groups: Sequence[Sequence[Sequence[int]]],
) -> tuple[list[int], ...]:
    """Restore ``tuple[list[int], ...]`` from ``encode_block_ids_rle`` output."""
    decoded: list[list[int]] = []
    for runs in rle_groups:
        group: list[int] = []
        for start, count in runs:
            s = int(start)
            c = int(count)
            if c == 1:
                group.append(s)
            elif c > 1:
                group.extend(range(s, s + c))
        decoded.append(group)
    return tuple(decoded)


def resolve_remote_block_ids(params: dict[str, Any]) -> Any:
    """Transparently expand ``remote_block_ids_rle`` in ``kv_transfer_params`` if present."""
    if "remote_block_ids_rle" in params and not params.get("remote_block_ids"):
        expanded = decode_block_ids_rle(params.pop("remote_block_ids_rle"))
        params["remote_block_ids"] = expanded
        return expanded
    return params.get("remote_block_ids")
