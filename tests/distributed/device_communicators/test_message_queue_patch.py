# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for MessageQueue max_chunks scaling."""

from __future__ import annotations

from vllm_iluvatar.ilu_patch.general_patches.patch_message_queue import (
    _COMMUNITY_DEFAULT_CHUNKS,
    _RESPONSE_MAX_CHUNK_BYTES,
    _scaled_mq_limits,
    register,
)


def test_scaled_mq_limits_leaves_explicit_chunks():
    assert _scaled_mq_limits(16, 32, 1024) == (32, 1024)


def test_scaled_mq_limits_broadcast_uses_n_reader():
    chunks, nbytes = _scaled_mq_limits(16, _COMMUNITY_DEFAULT_CHUNKS, 24 * 1024 * 1024)
    assert chunks == 2 * 16 + 8
    assert nbytes == 24 * 1024 * 1024


def test_scaled_mq_limits_response_caps_chunk_bytes():
    chunks, nbytes = _scaled_mq_limits(1, _COMMUNITY_DEFAULT_CHUNKS, 24 * 1024 * 1024)
    assert chunks == 48
    assert nbytes == _RESPONSE_MAX_CHUNK_BYTES


def test_scaled_mq_limits_env_override():
    chunks, nbytes = _scaled_mq_limits(
        16, _COMMUNITY_DEFAULT_CHUNKS, 24 * 1024 * 1024, env_raw="64"
    )
    assert chunks == 64
    assert nbytes == 24 * 1024 * 1024


def test_message_queue_patch_register_is_idempotent_and_wraps():
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    register()
    first = MessageQueue.__init__
    register()
    register()
    assert MessageQueue.__init__ is first
    assert hasattr(MessageQueue.__init__, "__wrapped__")
