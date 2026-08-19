# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/utils.py

ILU override implementation for create_tcp_store. This module holds ONLY the
override implementation; the platform-stage wiring lives in
``vllm_iluvatar.ilu_patch.patch_platform_import_hooks`` so the target community
module can be patched right after import without being eagerly imported during
platform resolution.

Target   : vllm.distributed.utils.create_tcp_store
Baseline : main_6e448d0
Why      : the libuv TCPStore backend does not support init from a pre-bound
           listen fd; force the legacy backend when taking ownership of a
           listen socket.
How      : wrapper delegating to the community function; only sets
           ``use_libuv=False`` (default) on the listen_socket path. No body copy,
           so the rest tracks upstream automatically.
Scope    : create_tcp_store callers (elastic-EP coordinator / stateless PG).
Remove   : when upstream handles libuv + pre-bound listen fd.
"""

from __future__ import annotations


# [ILU override] create_tcp_store — delta: setdefault use_libuv=False on the
# listen_socket path, then delegate to the community function (**kwargs keeps it
# forward-compatible with upstream signature changes; delegation-safe, no
# explicit signature check needed).
def _make_create_tcp_store(orig):
    def create_tcp_store(host, port, listen_socket=None, **kwargs):
        if listen_socket is not None:
            kwargs.setdefault("use_libuv", False)
        return orig(host, port, listen_socket, **kwargs)

    return create_tcp_store
