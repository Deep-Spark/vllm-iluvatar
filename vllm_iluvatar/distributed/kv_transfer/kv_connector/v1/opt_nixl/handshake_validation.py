"""Pure validation helpers for NIXL handshakes and transfer inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_compatibility_hash(
    local_hash: str | None,
    remote_hash: str,
    *,
    remote_engine_id: str,
    enforce: bool = True,
) -> None:
    """Fail before decoding/using incompatible transfer geometry."""

    if not enforce:
        return
    if local_hash is None:
        raise RuntimeError("local NIXL compatibility hash is not initialized")
    if local_hash != remote_hash:
        raise RuntimeError(
            "NIXL compatibility hash mismatch for remote engine "
            f"{remote_engine_id}: local={local_hash}, remote={remote_hash}"
        )


def validate_parallel_topology(
    *,
    local_tp_size: int,
    local_pp_size: int,
    remote_tp_size: int,
    remote_pp_size: int,
) -> None:
    for name, value in (
        ("local_tp_size", local_tp_size),
        ("local_pp_size", local_pp_size),
        ("remote_tp_size", remote_tp_size),
        ("remote_pp_size", remote_pp_size),
    ):
        require_positive(name, value)
    if max(local_tp_size, remote_tp_size) % min(local_tp_size, remote_tp_size):
        raise ValueError(
            "P/D tensor parallel sizes must have an integral ratio, got "
            f"local={local_tp_size}, remote={remote_tp_size}"
        )


def validate_same_length(name: str, *values: Sequence[object]) -> int:
    """Validate parallel vectors and return their common length."""

    lengths = [len(value) for value in values]
    if not lengths or lengths[0] == 0:
        raise ValueError(f"{name}: input vectors must not be empty")
    if len(set(lengths)) != 1:
        raise ValueError(f"{name}: input length mismatch: {lengths}")
    return lengths[0]


def validate_descriptor_geometry(
    *,
    local_ptr: Sequence[int],
    remote_ptr: Sequence[int],
    local_desc: Sequence[Sequence[tuple[int, int]]],
    remote_desc: Sequence[Sequence[tuple[int, int]]],
    local_block_len: Sequence[int],
    remote_block_len: Sequence[int],
    layer_desc: Sequence[object],
) -> None:
    """Validate all per-region vectors before TP slicing mutates them."""

    count = validate_same_length(
        "descriptor geometry",
        local_ptr,
        remote_ptr,
        local_desc,
        remote_desc,
        local_block_len,
        remote_block_len,
        layer_desc,
    )
    for index in range(count):
        require_positive(f"local_block_len[{index}]", local_block_len[index])
        require_positive(f"remote_block_len[{index}]", remote_block_len[index])
        local_segs = local_desc[index]
        remote_segs = remote_desc[index]
        if len(local_segs) != len(remote_segs):
            raise ValueError(
                f"descriptor geometry[{index}]: local/remote segment count "
                f"differs ({len(local_segs)} != {len(remote_segs)})"
            )
        if not local_desc[index]:
            raise ValueError(
                f"descriptor geometry[{index}]: no payload segments configured"
            )
        for side, segments, block_stride in (
            ("local", local_desc[index], local_block_len[index]),
            ("remote", remote_desc[index], remote_block_len[index]),
        ):
            for offset, length in segments:
                if offset < 0 or length <= 0:
                    raise ValueError(
                        f"{side} descriptor[{index}] has invalid segment "
                        f"({offset}, {length})"
                    )
                if offset + length > block_stride:
                    raise ValueError(
                        f"{side} descriptor[{index}] segment ({offset}, {length}) "
                        f"exceeds block stride {block_stride}"
                    )


def validate_block_id_groups(
    *,
    local_block_ids: Sequence[object],
    remote_block_ids: Sequence[object],
    expected_groups: int,
) -> None:
    """Validate group shape while allowing an individual group to be empty.

    Only the local side has a fixed group count.  The remote tuple follows the
    peer's own grouping, which may be smaller (D has no PP forwarder groups);
    each transfer indexes it with the peer group id recorded at handshake.
    """

    if len(local_block_ids) != expected_groups:
        raise ValueError(
            f"local block group count {len(local_block_ids)} != {expected_groups}"
        )
    if not remote_block_ids:
        raise ValueError("remote block-id tuple is empty")


def validate_pp_splits(pp_split: Sequence[int], num_hidden_layers: int) -> None:
    if not pp_split:
        raise ValueError("PP split must not be empty")
    if pp_split[0] != 0:
        raise ValueError(f"PP split must start at layer 0, got {pp_split[0]}")
    if any(left >= right for left, right in zip(pp_split, pp_split[1:])):
        raise ValueError(f"PP split must be strictly increasing: {list(pp_split)}")
    if pp_split[-1] >= num_hidden_layers:
        raise ValueError(
            f"last PP split {pp_split[-1]} must be below layer count "
            f"{num_hidden_layers}"
        )
