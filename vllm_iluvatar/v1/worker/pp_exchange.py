# SPDX-License-Identifier: Apache-2.0
"""Schema-driven eager pipeline exchange with optional 1-step send deferral."""

from __future__ import annotations

import math
from collections import deque
from contextlib import contextmanager

import torch
import torch.distributed as dist

Schema = tuple[tuple[str, tuple[int, ...], torch.dtype], ...]


class RetainedSend:
    """Keep a send tensor alive until its async P2P request completes."""

    def __init__(self, work, tensor):
        self.work = work
        self.tensor = tensor

    def wait(self):
        self.work.wait()

    def is_completed(self):
        return self.work.is_completed()


def schema_bytes(schema: Schema, rows: int) -> int:
    return sum(rows * math.prod(shape) * dtype.itemsize for _, shape, dtype in schema)


def send_schema_tensors(
    pp,
    tensors: dict[str, torch.Tensor],
    rows: int,
    schema: Schema,
    dst: int | None = None,
    *,
    retain_send: bool = True,
):
    if dst is None:
        dst = (pp.rank_in_group + 1) % pp.world_size
    expected_keys = {key for key, _, _ in schema}
    if set(tensors) != expected_keys:
        raise ValueError(
            f"Unexpected PP intermediate keys: got {sorted(tensors)}, expected {sorted(expected_keys)}"
        )
    for key, shape, dtype in schema:
        tensor = tensors[key]
        if (
            tensor.shape != (rows, *shape)
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Invalid PP state {key}: shape={tensor.shape}, dtype={tensor.dtype}"
            )

    # Worker.execute_model reserves index 0 for metadata; device handles follow.
    handles: list = [None]
    for key, _, _ in schema:
        tensor = tensors[key]
        if not retain_send:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
        work = dist.isend(tensor, dst=pp.ranks[dst], group=pp.device_group)
        handles.append(RetainedSend(work, tensor) if retain_send else work)
    return handles


def recv_schema_tensors(
    pp,
    rows: int,
    schema: Schema,
    device: torch.device,
    src: int | None = None,
):
    if src is None:
        src = (pp.rank_in_group - 1) % pp.world_size
    tensors = {
        key: torch.empty((rows, *shape), dtype=dtype, device=device)
        for key, shape, dtype in schema
    }
    handles = [
        dist.irecv(tensor, src=pp.ranks[src], group=pp.device_group)
        for tensor in tensors.values()
    ]
    return tensors, handles, []


@contextmanager
def schema_pp_exchange(
    pp,
    rows: int,
    device: torch.device,
    *,
    send_schema: Schema | None = None,
    recv_schema: Schema | None = None,
    pending=(),
    retain_send: bool = True,
):
    previous_send = pp.isend_tensor_dict
    previous_recv = pp.irecv_tensor_dict

    def send(tensors, dst=None, all_gather_group=None, all_gather_tensors=None):
        assert send_schema is not None
        assert not all_gather_tensors
        assert all_gather_group is None or all_gather_group.world_size == 1
        if pending:
            old_storage = {
                handle.tensor.untyped_storage().data_ptr()
                for handles in pending
                for handle in handles
            }
            if any(
                tensor.untyped_storage().data_ptr() in old_storage
                for tensor in tensors.values()
            ):
                raise ValueError("PP forward reused a still-pending output buffer")
        return send_schema_tensors(
            pp, tensors, rows, send_schema, dst, retain_send=retain_send
        )

    def recv(src=None, all_gather_group=None, all_gather_tensors=None):
        assert recv_schema is not None
        assert not all_gather_tensors
        assert all_gather_group is None or all_gather_group.world_size == 1
        return recv_schema_tensors(pp, rows, recv_schema, device, src)

    if send_schema is not None:
        pp.isend_tensor_dict = send
    if recv_schema is not None:
        pp.irecv_tensor_dict = recv
    try:
        yield
    finally:
        pp.isend_tensor_dict = previous_send
        pp.irecv_tensor_dict = previous_recv


def defer_pp_send_wait(worker):
    """Retain one older send while the next step starts computing."""
    pending = getattr(worker, "_pp_deferred_sends", None)
    if pending is None:
        worker._pp_deferred_sends = pending = deque()
    while pending and all(handle.is_completed() for handle in pending[0]):
        pending.popleft()
    pp_work = getattr(worker, "_pp_send_work", None)
    if pp_work:
        if not all(isinstance(handle, RetainedSend) for handle in pp_work):
            return pending
        pending.append(pp_work)
        worker._pp_send_work = []
    if len(pending) > 1:
        for handle in pending.popleft():
            handle.wait()
    return pending


def drain_pp_send_wait(worker):
    """Wait and clear any deferred PP send handles and retained tensors."""
    pending = getattr(worker, "_pp_deferred_sends", None)
    if pending:
        while pending:
            for handle in pending.popleft():
                handle.wait()
    pp_work = getattr(worker, "_pp_send_work", None)
    if pp_work and all(isinstance(handle, RetainedSend) for handle in pp_work):
        for handle in pp_work:
            handle.wait()
        worker._pp_send_work = []
