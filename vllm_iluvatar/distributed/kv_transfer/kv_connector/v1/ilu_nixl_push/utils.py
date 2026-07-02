# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
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


class TransferStatusChecker:
    init_pp_start = None
    init_pp_end = None

    def __init__(
        self,
        tp_start: int,
        tp_end: int,
        pp_start: int,
        pp_end: int,
        num_groups: int,
        engine_id: str = None,
    ):
        self.tp_start = tp_start
        self.tp_end = tp_end
        self.pp_start = pp_start
        self.pp_end = pp_end
        self.num_groups = num_groups
        self.pp_recv_record = np.ones([pp_end - pp_start], dtype=np.int32)
        self.pp_num_record = np.zeros([pp_end - pp_start], dtype=np.int32)
        self.record = np.ones(
            [tp_end - tp_start, pp_end - pp_start, num_groups], dtype=np.int32
        )
        self.record_num = 0

        self.total_num_init = tp_end - tp_start
        self.total_num = -1
        self.need_calculateed = True

    def is_all_pp_recv(self):
        return np.sum(self.pp_recv_record) == 0

    def update(self, tp, pp, transfer_index, transfer_nums, logger):
        if (
            self.pp_start <= pp < self.pp_end
            and self.tp_start <= tp < self.tp_end
            and transfer_index < self.num_groups
            and transfer_nums <= self.num_groups
        ):
            if (tp_index := tp - self.tp_start) == 0 and self.pp_recv_record[
                pp_index := pp - self.pp_start
            ]:
                self.pp_recv_record[pp_index] = 0
                self.pp_num_record[pp_index] = transfer_nums

            if self.need_calculateed and self.is_all_pp_recv():
                self.need_calculateed = False
                self.total_num = self.total_num_init * np.sum(self.pp_num_record)
                logger.debug(
                    "recv infos: %s, nums: %s", self.pp_num_record, self.total_num
                )

            self.record_num += 1
            if self.record[tp_index, pp_index, transfer_index]:
                self.record[tp_index, pp_index, transfer_index] = 0
                logger.debug(
                    "update with tp %s pp %s transfer_index %s transfer_nums %s ",
                    tp,
                    pp,
                    transfer_index,
                    transfer_nums,
                )
            else:
                logger.error(
                    "recv a double kv write for tp: %s, pp: %s, transfer_index: %s",
                    tp_index,
                    pp_index,
                    transfer_index,
                )
        else:
            logger.error(
                "recv a tp, pp layer(rank), should not write to this pp rank, "
                "get tp: %s, should satisfy %s <= tp < %s, layer/rank: %s, "
                "should satisfy %s <= pp < %s, transfer_index: %s, "
                "transfer_nums: %s, should both satisfy <= %s",
                tp,
                self.tp_start,
                self.tp_end,
                pp,
                self.pp_start,
                self.pp_end,
                transfer_index,
                transfer_nums,
                self.num_groups,
            )

    def is_done(self):
        return self.record_num == self.total_num


class D2HManger:
    """
    /**
    * work flow:
    * check_prestep_status()   first return True, pass
    * bind_slotmapping()
    * transfer_d2h_once * n   (n means the layer nums need to be processed on one step), the copy and result flag write running on other thread
    * transfer_h2h_once * n
    * reset()                 reset the flag from main thread after done(the transfer thread are waiting data, so no write conflict)
    * ⬇ next step
    * check_prestep_status    this fun only read result flag from main thread, so will not generate data conflict
    * bind_slotmapping()
    * transfer_d2h_once * n   (next step transfer beginning)
    * transfer_h2h_once * n
    * reset()
    **/
    """

    def __init__(
        self,
        buffers: list[list[torch.Tensor]],
        kv_caches: list[list[torch.Tensor]],
    ) -> None:
        import ixformer._C._functions as CF

        total_layer_nums = len(buffers)
        assert total_layer_nums > 0
        assert len(buffers) == len(kv_caches)

        buffer_size_per_layer = set(len(any_cache) for any_cache in kv_caches)
        assert len(buffer_size_per_layer) == 1
        self.elm_per_layer = buffer_size_per_layer.pop()

        for index in range(self.elm_per_layer):
            any_cache = kv_caches[0][index]
            assert any_cache.dim() == 3, print(
                "d2h only support mla kv cache shape: [num_blocks, block_size, head_dim] for now"
            )
            for cache in kv_caches:
                assert cache[index].is_cpu
                assert cache[index].is_contiguous()
                assert cache[index].shape == any_cache.shape

        buffer_size_per_layer = set(len(any_buffer) for any_buffer in buffers)
        assert len(buffer_size_per_layer) == 1
        assert buffer_size_per_layer.pop() == self.elm_per_layer

        for index in range(self.elm_per_layer):
            any_buffer = buffers[0][index]
            for buffer in buffers:
                assert buffer[index].is_cpu
                assert buffer[index].is_contiguous()
                assert buffer[index].shape == any_buffer.shape

        self.d2h_manager = CF.D2HManger(kv_caches, buffers)
        self.kv_caches = kv_caches
        self.buffers = buffers
        self.slot_mapping = None
        self.last_layer_index = len(kv_caches) - 1
        self.layer_nums = len(kv_caches)
        self.transfer_d2h_count = 0
        self.transfer_h2h_count = 0
        self.elm_per_layer_count = 0
        self.bind_slot_count = 0

    def check_status(self, index):
        assert 0 <= index <= self.last_layer_index

        return self.d2h_manager.query_status(index) == 1

    def check_prestep_status(self):
        return self.d2h_manager.query_status(self.last_layer_index) == 0

    def bind_slotmapping(self, slotmapping: torch.Tensor):
        assert self.transfer_d2h_count == 0 and self.transfer_h2h_count == 0
        assert slotmapping.is_cpu

        # NOTE(gyf): for now, the slotmapping here is a buffer create on gpu_model_runner
        # we need a clone in case the data changes when we using it.
        if self.slot_mapping is None:
            self.slot_mapping = slotmapping.clone()
        self.bind_slot_count += 1
        assert self.bind_slot_count <= self.elm_per_layer
        self.d2h_manager.set_slotmapping(
            self.slot_mapping,
            self.bind_slot_count == 1,
            self.bind_slot_count == self.elm_per_layer,
        )

    def reset(self):
        self.bind_slot_count = 0
        self.transfer_d2h_count = 0
        self.transfer_h2h_count = 0
        self.slot_mapping = None
        self.d2h_manager.reset()

    def transfer_d2h_once(self, index, inputs):
        assert 0 <= index <= self.last_layer_index
        assert self.transfer_d2h_count < self.layer_nums
        assert self.slot_mapping is not None

        B = inputs.shape[0]
        self.buffers[index][self.elm_per_layer_count][:B].copy_(
            inputs, non_blocking=True
        )
        self.elm_per_layer_count = (self.elm_per_layer_count + 1) % self.elm_per_layer
        if self.elm_per_layer_count == 0:
            self.transfer_d2h_count += 1
            return torch.cuda.current_stream().record_event()
        return None

    def transfer_h2h_once(self, index):
        assert 0 <= index <= self.last_layer_index
        assert self.transfer_h2h_count < self.layer_nums
        assert self.slot_mapping is not None

        if self.elm_per_layer_count == 0:
            self.transfer_h2h_count += 1
            self.d2h_manager.transfer(
                index
            )  # this c++ pushes tasks will work on other thread


def find_number_on_layer_name(text: str) -> list[str]:
    pattern = r"\.(0|100|[1-9]\d?)\."
    result = re.findall(pattern, text)
    return result


@dataclass
class NixlTpKVTopology(TransferTopology):
    def tp_ratio(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Calculate the tensor parallel ratio between local and remote TP.
        We can think of it as the number of local TP workers-per-remote TP
        workers. Local workers will read from the same remote TP worker in
        groups of size `tp_ratio`.
        """
        return max(self.tp_size, remote_tp_size) // min(self.tp_size, remote_tp_size)

    def block_size_ratio(self, remote_block_size: int) -> int:
        if self.block_size >= remote_block_size:
            return True, self.block_size // remote_block_size
        return False, remote_block_size // self.block_size

    def get_target_remote_rank(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        if self.tp_size < remote_tp_size:
            target_tp_rank = self.tp_rank * tp_ratio
        else:
            target_tp_rank = self.tp_rank // tp_ratio

        return target_tp_rank


@dataclass
class TransferWrapper:
    ready_for_check: bool = False
    transfer_list: list[Transfer] = None
