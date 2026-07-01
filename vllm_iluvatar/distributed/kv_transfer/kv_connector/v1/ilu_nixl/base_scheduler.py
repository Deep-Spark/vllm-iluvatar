# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side PP extensions for the Ilu NIXL connector."""

import time

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)

from .metadata import HeartbeatInfo, IluNixlConnectorMetadata


class IluNixlBaseConnectorScheduler(NixlBaseConnectorScheduler):
    """Base scheduler that propagates PP topology through connector metadata."""

    def on_new_request(self, request) -> None:
        # Keep this method in sync with vLLM
        # vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        # at community commit 63e161f29. Ilu delta: carry pp_size/pp_split in
        # heartbeat metadata for heterogeneous PP handshakes.
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return

        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        pp_size = params.get("pp_size", 1)
        pp_split = params.get("pp_split", "")
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return

        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                pp_size=pp_size,
                pp_split=pp_split,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def build_connector_meta(self, scheduler_output):
        # Keep this method in sync with vLLM
        # vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        # at community commit 63e161f29. Ilu delta: use IluNixlConnectorMetadata
        # so ReqMeta carries remote PP topology to the worker.
        meta = IluNixlConnectorMetadata()

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta
