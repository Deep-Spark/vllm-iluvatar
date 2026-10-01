# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import zmq
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())
TransferHandle = int
Transfer = list[tuple[TransferHandle, bool, float]]  # (xfer_handle, done, start_time)


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


class NodeError(Exception):
    """a enum class"""


def zmq_comm2proxy(socket: zmq.Socket, send_msg: bytes, need_recv: bool, logger: Any):
    try:
        socket.send(send_msg)
        msg = socket.recv()
        if msg == b"error: node not found":
            raise NodeError()
        if msg != b"success":
            logger.error(
                "Proxy failed to handle message: %s, please check the "
                "message format or proxy status",
                msg,
            )
        if need_recv:
            msg = socket.recv()
            return msg
    except zmq.ZMQError as e:
        logger.error(
            "Failed to communicate with proxy: %s, please check the proxy status",
            e,
        )


class NotifyPayload(NamedTuple):
    """Producer-side fields of one completion notification.

    Wire form (between ``<req_id>:`` and ``:<status>``)::

        <pp>-<pp_start>-<pp_end>:<pp_layer>-<pp_layer_start>-<pp_layer_end>:<tp>-<tp_start>-<tp_end>-<count>

    ``pp`` is the writer's PP rank and ``[pp_start, pp_end)`` all PP ranks
    that write this consumer rank.  ``pp_layer`` is what this notification
    counts for: per-layer it is the written layer and
    ``[pp_layer_start, pp_layer_end)`` the consumer stage's layers; per-rank
    it equals the ``pp`` field.  ``count`` is how many notifications the
    writer ``(tp, pp)`` sends for this request.

    per-layer::

        0-0-3:1-0-30:0-0-1-10    pp 0 of 3 writers, layer 1, 10 in total
        1-0-3:11-0-30:0-0-1-5    pp 1, layer 11, 5 in total

    per-rank (one notification per writer)::

        0-0-3:0-0-3:0-0-1-1
    """

    src_pp: int
    pp_start: int
    pp_end: int
    pp_layer: int
    pp_layer_start: int
    pp_layer_end: int
    src_tp: int
    tp_start: int
    tp_end: int
    count: int

    def encode(self) -> str:
        return (
            f"{self.src_pp}-{self.pp_start}-{self.pp_end}:"
            f"{self.pp_layer}-{self.pp_layer_start}-{self.pp_layer_end}:"
            f"{self.src_tp}-{self.tp_start}-{self.tp_end}-{self.count}"
        )

    @classmethod
    def decode(cls, pp_text: str, layer_text: str, tp_text: str) -> "NotifyPayload":
        return cls(
            *(int(value) for value in pp_text.split("-")),
            *(int(value) for value in layer_text.split("-")),
            *(int(value) for value in tp_text.split("-")),
        )


class TransferStatusChecker:
    """聚合一个 request 的 NIXL completion notifications。

    两张表：

    * 总数表 ``[tp_end - tp_start, pp_end - pp_start]``：每个写入端
      ``(tp, pp)`` 一格，第一条 notify 到达时登记它的 count，之后必须一致；
    * 写入表 ``[tp_end - tp_start, pp_layer_end - pp_layer_start]``：记录
      某层（per-layer）或某 rank（per-rank）是否已收到，防止重复。

    每格收到 count 条互不相同的 pp_layer 即算收齐，所有格都登记且收齐时
    request 完成。per-layer 下多个 group 一次发出，所以 count ≤ 层数。

    不写 KV 的 rank（例如 MLA 下 local TP > remote TP 的副本）和本 overlap
    里没有可传层的 rank 都只发 1 条 notify，count=1。例：producer TP4 →
    consumer TP2 的 consumer rank 0，由 source tp 0（写）和 1（MLA 副本，
    只发 notify）喂数据；producer PP4，per-layer，PP rank 2 没有可传的层::

        pp \\ tp   0   1
        0         10   1
        1          5   1
        2          1   1
        3          5   1
    """

    def __init__(
        self,
        tp_start: int = 0,
        tp_end: int = 0,
        pp_start: int = 0,
        pp_end: int = 0,
        pp_layer_start: int = 0,
        pp_layer_end: int = 0,
    ):
        # notification 中携带的均为左闭右开范围。
        self.tp_start: int = tp_start
        self.tp_end: int = tp_end
        self.pp_start: int = pp_start
        self.pp_end: int = pp_end
        self.pp_layer_start: int = pp_layer_start
        self.pp_layer_end: int = pp_layer_end
        shape = (tp_end - tp_start, pp_end - pp_start)
        self.declared = np.full(shape, -1, dtype=np.int32)
        self.received_per_cell = np.zeros(shape, dtype=np.int32)
        self.received = np.zeros(
            (tp_end - tp_start, pp_layer_end - pp_layer_start), dtype=np.bool_
        )

    @classmethod
    def for_payload(cls, payload: NotifyPayload) -> "TransferStatusChecker":
        return cls(
            payload.tp_start,
            payload.tp_end,
            payload.pp_start,
            payload.pp_end,
            payload.pp_layer_start,
            payload.pp_layer_end,
        )

    def matches(self, payload: NotifyPayload) -> bool:
        return (
            payload.tp_start == self.tp_start
            and payload.tp_end == self.tp_end
            and payload.pp_start == self.pp_start
            and payload.pp_end == self.pp_end
            and payload.pp_layer_start == self.pp_layer_start
            and payload.pp_layer_end == self.pp_layer_end
        )

    def update(self, payload: NotifyPayload, logger) -> None:
        if not (
            self.tp_start <= payload.src_tp < self.tp_end
            and self.pp_start <= payload.src_pp < self.pp_end
            and self.pp_layer_start <= payload.pp_layer < self.pp_layer_end
            and payload.count > 0
        ):
            logger.error(
                "recv a notification outside this consumer's producer table "
                "tp [%s, %s) x pp [%s, %s) x pp_layer [%s, %s): %s",
                self.tp_start,
                self.tp_end,
                self.pp_start,
                self.pp_end,
                self.pp_layer_start,
                self.pp_layer_end,
                payload,
            )
            return
        tp_index = payload.src_tp - self.tp_start
        cell = (tp_index, payload.src_pp - self.pp_start)
        declared = int(self.declared[cell])
        if declared < 0:
            self.declared[cell] = payload.count
        elif declared != payload.count:
            logger.error(
                "recv inconsistent notify count for source tp %s pp %s: old %s, new %s",
                payload.src_tp,
                payload.src_pp,
                declared,
                payload.count,
            )
            return
        key = (tp_index, payload.pp_layer - self.pp_layer_start)
        if self.received[key]:
            logger.error("recv a double kv write: %s", payload)
            return
        if self.received_per_cell[cell] >= payload.count:
            logger.error(
                "recv more notifications than declared for source tp %s pp %s: %s",
                payload.src_tp,
                payload.src_pp,
                payload,
            )
            return
        self.received[key] = True
        self.received_per_cell[cell] += 1
        logger.debug("notify accepted: %s", payload)

    def is_done(self) -> bool:
        return bool(
            np.all(self.declared >= 0)
            and np.array_equal(self.received_per_cell, self.declared)
        )


def find_number_on_layer_name(text: str) -> list[str]:
    pattern = r"\.(0|100|[1-9]\d?)\."
    result = re.findall(pattern, text)
    return result


@dataclass
class TransferWrapper:
    ready_for_check: bool = False
    transfer_list: list[Transfer] = None
