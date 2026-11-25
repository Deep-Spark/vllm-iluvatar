# SPDX-License-Identifier: Apache-2.0
"""Wire protocol and IPC socket naming for Mooncake Store LookupKey RPC."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Frame-0 message tags for REQ/REP ZMQ channel between Scheduler and Worker rank 0.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"

RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"

TAIL_KEY_BOUNDARY_ENTRY_SIZE: int = 8


@dataclass(frozen=True)
class TailKeyBoundary:
    """Hash boundary used to key a cache group's tail block in the store."""

    group_id: int
    num_tokens: int


@dataclass
class MooncakeLookupResult:
    """Lookup result returned by worker rank 0 to the scheduler."""

    hit_length: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


def encode_lookup_response(result: MooncakeLookupResult) -> bytes:
    hit_length = max(0, int(result.hit_length)).to_bytes(4, "big")
    if not result.tail_key_boundaries:
        return hit_length

    payload = bytearray(hit_length)
    for boundary in result.tail_key_boundaries:
        payload.extend(int(boundary.group_id).to_bytes(4, "big"))
        payload.extend(int(boundary.num_tokens).to_bytes(4, "big"))
    return bytes(payload)


def decode_lookup_response(payload: bytes) -> MooncakeLookupResult:
    if len(payload) < 4 or (len(payload) - 4) % TAIL_KEY_BOUNDARY_ENTRY_SIZE != 0:
        raise ValueError(
            f"Invalid Mooncake lookup response length: {len(payload)} bytes"
        )

    hit_length = int.from_bytes(payload[:4], "big")
    if len(payload) == 4:
        return MooncakeLookupResult(hit_length)

    boundaries: list[TailKeyBoundary] = []
    for offset in range(4, len(payload), TAIL_KEY_BOUNDARY_ENTRY_SIZE):
        boundaries.append(
            TailKeyBoundary(
                group_id=int.from_bytes(payload[offset : offset + 4], "big"),
                num_tokens=int.from_bytes(payload[offset + 4 : offset + 8], "big"),
            )
        )
    return MooncakeLookupResult(hit_length, tuple(boundaries))


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Build a short, collision-free IPC socket path per engine & DP rank.

    Incorporating a digest of ``engine_id`` (shared by Scheduler and Worker
    subprocesses of one ``vllm serve`` instance, distinct across P1/P2
    instances on the same host) guarantees single-node 2P/xP isolation even
    when ``lookup_rpc_port`` is not explicitly set, while keeping the Unix
    domain socket path well within Linux's 108-byte ``sun_path`` limit.
    """
    assert vllm_config.kv_transfer_config is not None
    dp_rank = getattr(vllm_config.parallel_config, "data_parallel_index", 0)
    base_url = getattr(envs, "VLLM_RPC_BASE_PATH", "/tmp")
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    rpc_port = extra_config.get(
        "lookup_rpc_port", extra_config.get("mooncake_rpc_port", "auto")
    )
    engine_id = vllm_config.kv_transfer_config.engine_id or "default"
    token = hashlib.sha256(f"{engine_id}:{rpc_port}".encode()).hexdigest()[:12]
    path = f"ipc://{base_url}/ilu_mc_lk_{token}_dp{dp_rank}.sock"
    logger.debug("Mooncake lookup IPC socket: %s", path)
    return path
