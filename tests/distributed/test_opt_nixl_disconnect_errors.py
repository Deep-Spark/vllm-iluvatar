import asyncio
import contextlib
import time
import queue
import threading
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import zmq
import zmq.asyncio

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl import (
    scheduler as scheduler_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl import (
    worker as worker_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.lifecycle import (
    RequestOwnershipState,
    RequestTombstoneStore,
    RemoteEngineSession,
    RemoteSessionState,
    SendRequestState,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
    ENGINE_REACHABILITY_MSG,
    OWNERSHIP_CANCEL_MSG,
    OWNERSHIP_RESULT_ACCEPTED,
    OWNERSHIP_RESULT_ALIVE,
    OWNERSHIP_RESULT_CANCELED,
    OWNERSHIP_RESULT_UNKNOWN,
    OWNERSHIP_PROBE_MSG,
    OWNERSHIP_RESULT_TERMINAL,
    PeerMeta,
    ReqMeta,
    TEARDOWN_COMPLETE_MSG,
    TransferIntent,
    TransferNotificationStatus,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.utils import (
    NodeError,
    NotifyPayload,
    TransferStatusChecker,
    zmq_comm2proxy,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    NixlConnectorWorker,
    _is_terminal_xfer_query_error,
)


class nixlRemoteDisconnectError(RuntimeError):
    pass


class nixlNotFoundError(RuntimeError):
    pass


class nixlCancelledError(RuntimeError):
    pass


class nixlBackendError(RuntimeError):
    pass


def _init_scheduler_ownership_state(
    scheduler: NixlConnectorScheduler,
) -> None:
    scheduler._ownership_generations = {}
    scheduler._ownership_remote_engines = {}
    scheduler._ownership_states = {}
    scheduler._ownership_deadlines = {}
    scheduler._ownership_cancel_pending = set()
    scheduler._ownership_probe_pending = set()
    scheduler._lease_renewals = {}
    scheduler._probe_unsupported = set()
    scheduler._lease_renew_interval = 120.0
    scheduler._max_lease_renewals = 10
    scheduler._recv_release_proofs = set()
    scheduler._ownership_safe_releases = queue.SimpleQueue()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        RuntimeError("backend returned NIXL_ERR_NOT_FOUND"),
        RuntimeError("NIXL_ERR_CANCELED"),
        RuntimeError("NIXL_ERR_BACKEND"),
        RuntimeError("NIXL_ERR_INVALID_PARAM"),
        RuntimeError("NIXL_ERR_MISMATCH"),
        RuntimeError("NIXL_ERR_NOT_ALLOWED"),
        RuntimeError("NIXL_ERR_NOT_POSTED"),
        RuntimeError("NIXL_ERR_NOT_SUPPORTED"),
        RuntimeError("NIXL_ERR_REPOST_ACTIVE"),
        RuntimeError("NIXL_ERR_UNKNOWN"),
        nixlRemoteDisconnectError(),
        nixlNotFoundError(),
        nixlCancelledError(),
        nixlBackendError(),
    ],
)
def test_terminal_xfer_query_error(error: BaseException) -> None:
    assert _is_terminal_xfer_query_error(error)


def test_nonterminal_xfer_query_error() -> None:
    assert not _is_terminal_xfer_query_error(RuntimeError("temporary query failure"))


def test_scheduler_shutdown_cancels_blocked_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler._stop_event = threading.Event()
    scheduler._async_loop = asyncio.new_event_loop()
    started: list[str] = []
    cancelled: list[str] = []
    all_started = threading.Event()

    async def blocked(name: str) -> None:
        started.append(name)
        if len(started) == 3:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(name)

    monkeypatch.setattr(
        scheduler,
        "_proxy_connector",
        lambda _metadata: blocked("proxy"),
    )
    monkeypatch.setattr(
        scheduler,
        "_worker_data_forward",
        lambda: blocked("worker"),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_send_infos",
        lambda: blocked("send"),
    )

    scheduler_thread = threading.Thread(
        target=scheduler.scheduler_loop,
        args=({},),
        daemon=True,
    )
    scheduler._scheduler_loop_t = scheduler_thread
    scheduler_thread.start()
    assert all_started.wait(timeout=2)

    scheduler.shutdown()

    assert not scheduler_thread.is_alive()
    assert scheduler._scheduler_loop_t is None
    assert sorted(cancelled) == ["proxy", "send", "worker"]
    scheduler._async_loop.close()


def test_missing_remote_after_alloc_fails_only_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = True
    scheduler.kv_group_nums = 1
    scheduler._reqs_not_send_to_worker = set()
    scheduler._scheduler_failed_recvs = set()
    scheduler._pending_worker_sends = set()
    scheduler._live_recv_reqs = set()
    _init_scheduler_ownership_state(scheduler)

    def missing_peer(_remote_engine_id: str) -> None:
        raise NodeError()

    monkeypatch.setattr(scheduler, "_get_or_connect_peer", missing_peer)
    request = SimpleNamespace(
        request_id="request-selected-before-prefill-disconnect",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_engine_id": "departed-prefill",
        },
    )

    scheduler.on_new_request(request)
    scheduler.update_state_after_alloc(request, object(), num_external_tokens=64)

    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert scheduler._reqs_not_send_to_worker == {request.request_id}
    assert scheduler._scheduler_failed_recvs == {request.request_id}

    output = SimpleNamespace(
        finished_sending=None,
        finished_recving=None,
        failed_recving=None,
    )
    scheduler.update_connector_output(output)

    assert output.finished_recving == {request.request_id}
    assert output.failed_recving == {request.request_id}
    assert not scheduler._scheduler_failed_recvs


def test_teardown_result_is_reported_once_per_generation() -> None:
    worker = object.__new__(NixlConnectorWorker)
    worker.global_rank = 3
    worker._teardown_outbox = asyncio.Queue()
    session = RemoteEngineSession(
        engine_id="decode-generation-2",
        generation=2,
        role="producer",
        teardown_id=99,
    )

    worker._report_teardown(session, "closed")
    worker._report_teardown(session, "stuck")

    assert worker._teardown_outbox.qsize() == 1
    assert worker._teardown_outbox.get_nowait() == (
        "decode-generation-2",
        2,
        99,
        3,
        "closed",
    )


def test_teardown_without_router_token_is_not_acknowledged() -> None:
    worker = object.__new__(NixlConnectorWorker)
    worker.global_rank = 0
    worker._teardown_outbox = asyncio.Queue()
    session = RemoteEngineSession(
        engine_id="local-shutdown",
        generation=1,
        role="consumer",
    )

    worker._report_teardown(session, "closed")

    assert worker._teardown_outbox.empty()
    assert session.reported_teardown_id is None


def test_closed_session_acknowledges_new_teardown_token() -> None:
    worker = object.__new__(NixlConnectorWorker)
    worker.global_rank = 1
    worker._teardown_outbox = asyncio.Queue()
    session = RemoteEngineSession(
        engine_id="already-closed",
        generation=1,
        role="consumer",
        state=RemoteSessionState.CLOSED,
        teardown_id=10,
        reported_teardown_id=10,
    )
    worker.remote_sessions = {session.engine_id: session}
    worker._engine_generations = {session.engine_id: 1}

    worker._mark_engine_disconnected(session.engine_id, teardown_id=11)

    assert worker._teardown_outbox.get_nowait() == (
        session.engine_id,
        1,
        11,
        1,
        "closed",
    )


def test_teardown_report_reaches_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exercise() -> None:
        ctx = zmq.asyncio.Context()
        router = ctx.socket(zmq.ROUTER)
        endpoint = f"inproc://teardown-report-{id(router)}"
        router.bind(endpoint)
        monkeypatch.setattr(
            worker_module,
            "make_zmq_path",
            lambda _transport, _address: endpoint,
        )

        worker = object.__new__(NixlConnectorWorker)
        worker.ctx = ctx
        worker.side_channel_addr = "unused"
        worker._teardown_outbox = asyncio.Queue()
        task = asyncio.create_task(worker._run_teardown_reports())
        try:
            await worker._teardown_outbox.put(
                ("decode-generation-3", 3, 101, 4, "closed")
            )
            identity, empty, encoded = await asyncio.wait_for(
                router.recv_multipart(), timeout=2
            )
            op_type, payload = msgspec.msgpack.decode(encoded)
            assert empty == b""
            assert op_type == TEARDOWN_COMPLETE_MSG
            assert tuple(payload) == (
                "decode-generation-3",
                3,
                101,
                4,
                "closed",
            )
            await router.send_multipart((identity, b"", b""))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            router.close(linger=0)
            ctx.term()

    asyncio.run(exercise())


def test_cancelled_request_drops_stale_scheduler_receive_failure() -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler._scheduler_failed_recvs = {"cancelled-request"}
    scheduler._pending_worker_sends = set()
    scheduler._live_recv_reqs = set()
    _init_scheduler_ownership_state(scheduler)
    output = SimpleNamespace(
        finished_recving=None,
        failed_recving=None,
        finished_sending=None,
    )

    scheduler.update_connector_output(output)

    assert output.finished_recving is None
    assert output.failed_recving is None
    assert not scheduler._scheduler_failed_recvs


def test_live_receive_tracking_stops_on_failure() -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = True
    scheduler._live_recv_reqs = set()
    scheduler._scheduler_failed_recvs = set()
    scheduler._pending_worker_sends = set()
    _init_scheduler_ownership_state(scheduler)

    request = SimpleNamespace(
        request_id="waiting-for-long-prefill",
        kv_transfer_params={"do_remote_prefill": True},
    )
    scheduler.on_new_request(request)
    assert scheduler._live_recv_reqs == {request.request_id}

    output = SimpleNamespace(
        finished_recving={request.request_id},
        failed_recving={request.request_id},
        finished_sending=None,
    )
    scheduler.update_connector_output(output)
    assert not scheduler._live_recv_reqs


def test_proxy_transport_error_is_request_scoped() -> None:
    class FailingSocket:
        def send(self, _payload: bytes) -> None:
            raise zmq.Again()

    with pytest.raises(NodeError, match="timed out or disconnected"):
        zmq_comm2proxy(FailingSocket(), b"request", True, None)


def test_proxy_rejection_is_request_scoped() -> None:
    class RejectedSocket:
        replies = iter((b"malformed request",))

        def send(self, _payload: bytes) -> None:
            return None

        def recv(self) -> bytes:
            return next(self.replies)

    with pytest.raises(NodeError, match="proxy rejected"):
        zmq_comm2proxy(RejectedSocket(), b"request", True, None)


def test_transfer_status_failure_diagnostic_is_precise() -> None:
    payload = NotifyPayload(0, 0, 2, 0, 0, 4, 0, 0, 1, 1)
    checker = TransferStatusChecker.for_payload(payload)
    test_logger = SimpleNamespace(
        debug=lambda *_args: None,
        error=lambda *_args: None,
    )

    checker.update(payload, test_logger)

    assert not checker.is_done()
    summary = checker.progress_summary()
    assert "undeclared_sources=[(0, 1)]" in summary
    assert "incomplete_sources=[]" in summary
    assert "received_layers={0: [0]}" in summary


class _ProxySocket:
    def __init__(self, replies, more_frames=(1, 0)):
        self._replies = iter(replies)
        self._more_frames = iter(more_frames)

    def send(self, _payload: bytes) -> None:
        return None

    def recv(self):
        reply = next(self._replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def getsockopt(self, _option: int) -> int:
        return next(self._more_frames)


def test_proxy_body_timeout_is_request_scoped() -> None:
    socket = _ProxySocket((b"success", zmq.Again()))

    with pytest.raises(NodeError, match="timed out or disconnected"):
        zmq_comm2proxy(socket, b"request", True, None)


def test_proxy_non_bytes_frame_is_request_scoped() -> None:
    socket = _ProxySocket(("success",), more_frames=(0,))

    with pytest.raises(NodeError, match="proxy rejected"):
        zmq_comm2proxy(socket, b"request", False, None)


def test_proxy_success_without_body_frame_is_request_scoped() -> None:
    socket = _ProxySocket((b"success",), more_frames=(0,))

    with pytest.raises(NodeError, match="missing its metadata body frame"):
        zmq_comm2proxy(socket, b"request", True, None)


def test_proxy_extra_body_frame_is_request_scoped() -> None:
    socket = _ProxySocket((b"success", b"metadata"), more_frames=(1, 1))

    with pytest.raises(NodeError, match="unexpected extra reply frames"):
        zmq_comm2proxy(socket, b"request", True, None)


def _metadata_scheduler() -> NixlConnectorScheduler:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler._has_multi_size = False
    scheduler.block_size = 128
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 19001
    return scheduler


def _valid_peer_infos() -> dict:
    return {
        "physical_block_size": 128,
        "side_channel_addr": "192.0.2.10:19002",
        "pp_size": 4,
        "tp_size": 2,
        "pp_split": [0, 6, 12, 18, 24],
        "dp_rank": 0,
    }


@pytest.mark.parametrize(
    "encoded",
    [
        b"\xc1",
        msgspec.msgpack.encode(["not", "a", "mapping"]),
        msgspec.msgpack.encode(
            {key: value for key, value in _valid_peer_infos().items() if key != "tp_size"}
        ),
        msgspec.msgpack.encode({**_valid_peer_infos(), "physical_block_size": 64}),
        msgspec.msgpack.encode({**_valid_peer_infos(), "side_channel_addr": "bad"}),
        msgspec.msgpack.encode({**_valid_peer_infos(), "pp_size": True}),
        msgspec.msgpack.encode({**_valid_peer_infos(), "pp_split": "0,24"}),
    ],
)
def test_invalid_peer_metadata_is_typed(encoded: bytes) -> None:
    scheduler = _metadata_scheduler()

    with pytest.raises(NodeError, match="invalid metadata for remote engine peer"):
        scheduler._decode_peer_infos("peer", encoded)


def test_valid_peer_metadata_is_accepted() -> None:
    scheduler = _metadata_scheduler()
    peer_infos = _valid_peer_infos()

    assert scheduler._decode_peer_infos(
        "peer", msgspec.msgpack.encode(peer_infos)
    ) == peer_infos


def test_missing_remote_engine_id_fails_only_the_request() -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = True
    scheduler.kv_group_nums = 1
    scheduler._reqs_not_send_to_worker = set()
    scheduler._scheduler_failed_recvs = set()
    scheduler._live_recv_reqs = set()
    _init_scheduler_ownership_state(scheduler)
    request = SimpleNamespace(
        request_id="missing-remote-engine",
        kv_transfer_params={"do_remote_prefill": True},
    )

    scheduler.on_new_request(request)
    scheduler.update_state_after_alloc(request, object(), num_external_tokens=64)

    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert scheduler._reqs_not_send_to_worker == {request.request_id}
    assert scheduler._scheduler_failed_recvs == {request.request_id}


def test_request_tombstone_is_generation_scoped_and_idempotent() -> None:
    tombstones = RequestTombstoneStore(ttl=60.0, limit=4)

    assert (
        tombstones.record(
            "request", "generation-1", RequestOwnershipState.CANCELED, now=1.0
        )
        == RequestOwnershipState.CANCELED
    )
    assert (
        tombstones.record(
            "request", "generation-1", RequestOwnershipState.TERMINAL, now=2.0
        )
        == RequestOwnershipState.CANCELED
    )
    assert (
        tombstones.get("request", "generation-1", now=3.0)
        == RequestOwnershipState.CANCELED
    )
    assert tombstones.get("request", "generation-2", now=3.0) is None


def test_request_tombstone_capacity_fails_closed() -> None:
    tombstones = RequestTombstoneStore(ttl=60.0, limit=1)
    tombstones.record(
        "request-1", "generation-1", RequestOwnershipState.TERMINAL, now=1.0
    )

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        tombstones.record(
            "request-2",
            "generation-2",
            RequestOwnershipState.CANCELED,
            now=2.0,
        )


def test_router_request_id_generation_is_notification_wire_safe() -> None:
    request_id = (
        "___prefill_addr_192.168.77.1:15583"
        "___decode_addr_192.168.77.15:18000_0123456789abcdef"
    )
    params: dict[str, str] = {}
    generation = NixlConnectorScheduler._request_generation(request_id, params)

    assert generation == NixlConnectorScheduler._request_generation(
        request_id, params
    )
    assert generation == NixlConnectorScheduler._request_generation(
        request_id, {}
    )
    assert ":" not in generation
    notification = NixlConnectorWorker._build_notification(
        request_id,
        generation,
        "0-1:0-1:0-1-1",
        TransferNotificationStatus.SUCCESS,
    )
    parsed_request, parsed_generation, *_ = notification.decode().rsplit(":", 5)
    assert parsed_request == request_id
    assert parsed_generation == generation


def _ownership_worker() -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker._send_requests = {}
    worker._sends_still_progressing = set()
    worker._send_transfers = {}
    worker._send_failures = {}
    worker._send_tombstones = RequestTombstoneStore(ttl=60.0, limit=16)
    worker._completion_outbox = queue.SimpleQueue()
    worker._accepting_work = True
    worker.remote_sessions = {}
    worker.xfer_stats = SimpleNamespace(
        record_failed_transfer=lambda: None,
        record_transfer=lambda _telemetry: None,
    )
    return worker


def test_cancel_before_write_tombstones_late_prepare() -> None:
    worker = _ownership_worker()

    worker._cancel_send_request(
        "request", "generation-1", "consumer canceled"
    )

    assert "request" not in worker._send_requests
    assert (
        worker._send_tombstones.get("request", "generation-1")
        == RequestOwnershipState.CANCELED
    )
    assert (
        worker._send_state_for_generation("request", "generation-1")
        is None
    )
    assert worker._send_state_for_generation(
        "request", "generation-2"
    ) is not None


def _joined_send_state(worker: NixlConnectorWorker) -> SendRequestState:
    state = SendRequestState(
        req_id="request",
        generation="generation-1",
        deadline=1.0,
        ownership_state=RequestOwnershipState.PREPARED,
    )
    state.metadata = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )
    state.remote_block_ids = [np.array([[2]], dtype=np.int32)]
    worker._send_requests[state.req_id] = state
    return state


def test_generation_send_lease_waits_for_consumer_cancel() -> None:
    worker = _ownership_worker()
    worker._early_recv_deadlines = {}
    worker._early_recv_notifications = {}
    # Both halves joined, so a WRITE still follows: ownership is retained for
    # as long as it takes.
    state = _joined_send_state(worker)

    worker._expire_requests(now=2.0)

    assert worker._send_requests[state.req_id] is state
    assert state.deadline is None
    assert worker._completion_outbox.empty()
    assert (
        worker._send_tombstones.get(state.req_id, state.generation)
        is None
    )

    worker._cancel_send_request(
        state.req_id, state.generation, "consumer canceled while queued"
    )

    assert state.req_id not in worker._send_requests
    assert (
        worker._send_tombstones.get(state.req_id, state.generation)
        == RequestOwnershipState.CANCELED
    )


def test_unjoined_send_survives_while_still_queued_on_producer() -> None:
    # A deep prefill queue holds a request far longer than the orphan window,
    # and the producer's own scheduler keeps answering the consumer's ownership
    # probes with ALIVE for exactly that reason.  Reclaiming the send behind
    # that answer drops a healthy request, so the worker has to consult the
    # same truth before calling a send ownerless.
    worker = _ownership_worker()
    worker._early_recv_deadlines = {}
    worker._early_recv_notifications = {}
    worker.producer_orphan_timeout = 600.0
    worker.xfer_stats = SimpleNamespace(
        record_failed_transfer=lambda: None,
        record_transfer=lambda _telemetry: None,
        record_kv_expired_req=lambda: None,
    )
    state = SendRequestState(
        req_id="request",
        generation="generation-1",
        deadline=1.0,
        ownership_state=RequestOwnershipState.PREPARED,
    )
    state.remote_block_ids = [np.array([[2]], dtype=np.int32)]
    worker._send_requests[state.req_id] = state
    worker._sends_still_progressing = {"request"}

    # First expiry still only arms the grace window.
    worker._expire_requests(now=2.0)
    assert state.orphan_grace_armed is True
    assert state.deadline == pytest.approx(602.0)

    # The grace window elapses, but this engine is still going to serve it.
    worker._expire_requests(now=700.0)
    assert worker._send_requests[state.req_id] is state
    assert state.deadline == pytest.approx(1300.0)
    assert worker._completion_outbox.empty()

    # Once the scheduler stops vouching, the reclaim proceeds as before.
    worker._sends_still_progressing = set()
    worker._expire_requests(now=1400.0)
    assert state.req_id not in worker._send_requests
    batch = worker._completion_outbox.get_nowait()
    assert "ownerless send reclaimed" in batch.failures["request"]


@pytest.mark.parametrize(
    "with_metadata,with_remote_blocks",
    [(False, False), (True, False), (False, True)],
)
def test_ownerless_send_is_reclaimed_after_one_grace_window(
    with_metadata: bool, with_remote_blocks: bool
) -> None:
    # A client that kills the request mid-PD leaves one half missing forever.
    worker = _ownership_worker()
    worker._early_recv_deadlines = {}
    worker._early_recv_notifications = {}
    worker.producer_orphan_timeout = 600.0
    worker.xfer_stats = SimpleNamespace(
        record_failed_transfer=lambda: None,
        record_transfer=lambda _t: None,
        record_kv_expired_req=lambda: None,
    )
    state = SendRequestState(
        req_id="request",
        generation="generation-1",
        deadline=1.0,
        ownership_state=RequestOwnershipState.PREPARED,
    )
    if with_metadata:
        state.metadata = ReqMeta(
            local_block_ids=[[1]],
            engine_id="decode-0",
            generation="generation-1",
        )
    if with_remote_blocks:
        state.remote_block_ids = [np.array([[2]], dtype=np.int32)]
    worker._send_requests[state.req_id] = state

    # First expiry grants one more window instead of waiting forever.
    worker._expire_requests(now=2.0)
    assert worker._send_requests[state.req_id] is state
    assert state.orphan_grace_armed is True
    assert state.deadline == pytest.approx(602.0)
    assert worker._completion_outbox.empty()

    # Second expiry reclaims the source blocks and fences a late PREPARED.
    worker._expire_requests(now=700.0)

    assert state.req_id not in worker._send_requests
    batch = worker._completion_outbox.get_nowait()
    assert batch.done_sending == {"request"}
    assert "ownerless send reclaimed" in batch.failures["request"]
    assert (
        worker._send_tombstones.get("request", "generation-1")
        == RequestOwnershipState.CANCELED
    )
    assert worker._send_state_for_generation("request", "generation-1") is None


def _consumer_disconnect_worker() -> tuple[
    NixlConnectorWorker, RemoteEngineSession
]:
    worker = _ownership_worker()
    worker.is_consumer = True
    worker.num_groups = 1
    worker._recv_requests = {
        "request": ReqMeta(
            local_block_ids=[[7]],
            engine_id="prefill",
            generation="generation-1",
        )
    }
    worker._recv_generations = {"request": "generation-1"}
    worker._recv_tombstones = RequestTombstoneStore(ttl=60.0, limit=16)
    worker._recv_notification_status = {}
    worker._aborted_recv_requests = set()
    session = RemoteEngineSession(
        engine_id="prefill",
        generation=1,
        role="consumer",
        state=RemoteSessionState.DRAINING,
        pending_recv={"request"},
    )
    worker.remote_sessions = {"prefill": session}
    worker._remote_agents_per_layer = {"prefill": [["mapping"]]}
    worker._remote_agents_per_rank = {"prefill": [["mapping"]]}
    worker.dst_xfer_side_handles = {"prefill": [11]}
    worker._remote_agents = {"prefill": ["agent"]}
    worker._tp_size = {"prefill": 1}
    worker._pp_size = {"prefill": 1}
    worker._block_size = {"prefill": 128}
    return worker, session


def test_consumer_disconnect_releases_after_native_fence() -> None:
    worker, session = _consumer_disconnect_worker()
    cleanup_order = []
    worker._release_dlist_handle = (
        lambda handle: cleanup_order.append(("dlist", handle)) or True
    )
    worker._remove_remote_agent = (
        lambda agent: cleanup_order.append(("agent", agent)) or True
    )

    worker._cleanup_session_if_drained(
        session, fence_pending_recv=True
    )

    assert cleanup_order == [("dlist", 11), ("agent", "agent")]
    assert session.state == RemoteSessionState.CLOSED
    assert not session.pending_recv
    result = worker._completion_outbox.get_nowait()
    assert result.done_recving == {"request"}
    assert result.failures == {
        "request": "remote producer disconnected after native teardown"
    }


def test_consumer_disconnect_cleanup_failure_keeps_ownership() -> None:
    worker, session = _consumer_disconnect_worker()
    worker._release_dlist_handle = lambda _handle: False
    worker._remove_remote_agent = lambda _agent: True

    worker._cleanup_session_if_drained(
        session, fence_pending_recv=True
    )

    assert session.state == RemoteSessionState.STUCK
    assert session.pending_recv == {"request"}
    assert "request" in worker._recv_requests
    assert worker._completion_outbox.empty()


def test_cancel_during_write_waits_for_native_terminal() -> None:
    class FakeNixl:
        state = "PROC"

        def check_xfer_state(self, _handle):
            return self.state

        def get_xfer_telemetry(self, _handle):
            return None

        def release_xfer_handle(self, _handle):
            return None

    worker = _ownership_worker()
    worker.nixl_wrapper = FakeNixl()
    state = SendRequestState(
        req_id="request",
        generation="generation-1",
        started=True,
        ownership_state=RequestOwnershipState.WRITING,
    )
    worker._send_requests[state.req_id] = state
    worker._send_transfers[state.req_id] = SimpleNamespace(
        transfer_list=[[7, False, 0.0, "decode"]],
        ready_for_check=False,
    )

    worker._cancel_send_request(
        state.req_id, state.generation, "consumer canceled"
    )

    assert state.req_id in worker._send_requests
    assert worker._completion_outbox.empty()

    worker.nixl_wrapper.state = "DONE"
    worker._poll_send_transfers(include_unready=True)

    assert state.req_id not in worker._send_requests
    assert (
        worker._send_tombstones.get(state.req_id, state.generation)
        == RequestOwnershipState.TERMINAL
    )
    assert worker._completion_outbox.get_nowait().done_sending == {
        state.req_id
    }


def test_failed_notification_does_not_release_without_terminal_proof() -> None:
    worker = _ownership_worker()
    worker._recv_requests = {
        "request": ReqMeta(
            local_block_ids=[[1]],
            engine_id="prefill",
            generation="generation-1",
        )
    }
    worker._recv_generations = {"request": "generation-1"}
    worker._recv_tombstones = RequestTombstoneStore(ttl=60.0, limit=16)
    worker._aborted_recv_requests = set()
    worker._recv_notification_status = {}
    worker._early_recv_notifications = {}
    worker._early_recv_deadlines = {}
    payload = NotifyPayload(0, 0, 1, 0, 0, 1, 0, 0, 1, 1)

    worker._apply_recv_notification(
        "request",
        "generation-1",
        TransferNotificationStatus.FAILED,
        payload,
    )

    assert "request" in worker._recv_requests
    assert worker._completion_outbox.empty()


def _lease_scheduler() -> tuple[NixlConnectorScheduler, list]:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.engine_id = "decode"
    scheduler._live_recv_reqs = {"request"}
    scheduler._ownership_generations = {"request": "generation-1"}
    scheduler._ownership_remote_engines = {"request": "prefill"}
    scheduler._ownership_states = {
        "request": RequestOwnershipState.ACCEPTED
    }
    scheduler._ownership_deadlines = {"request": 1.0}
    scheduler._ownership_cancel_pending = set()
    scheduler._ownership_probe_pending = set()
    scheduler._lease_renewals = {}
    scheduler._probe_unsupported = set()
    scheduler._lease_renew_interval = 120.0
    scheduler._max_lease_renewals = 10
    scheduler._ownership_safe_releases = queue.SimpleQueue()
    commands: list = []
    scheduler._queue_peer_command = (
        lambda remote, command: commands.append((remote, command))
    )
    return scheduler, commands


def test_receive_lease_probes_then_cancels_without_releasing_blocks() -> None:
    scheduler, commands = _lease_scheduler()

    scheduler._expire_recv_ownership(now=2.0)

    # A lease expiry asks before it executes.
    assert scheduler._ownership_probe_pending == {"request"}
    assert scheduler._ownership_cancel_pending == set()
    assert commands == [
        (
            "prefill",
            (
                OWNERSHIP_PROBE_MSG,
                ("request", "generation-1", "decode", "receive lease expired"),
            ),
        )
    ]

    # The producer will not vouch for it, so the cancel follows -- and neither
    # step released a destination block.
    scheduler._apply_probe_reply(
        "request",
        "generation-1",
        "prefill",
        msgspec.msgpack.encode((OWNERSHIP_RESULT_UNKNOWN, "generation-1")),
    )
    scheduler._fail_open_probe("prefill", "request", "generation-1")

    assert scheduler._ownership_cancel_pending == {"request"}
    assert scheduler._ownership_safe_releases.empty()
    assert commands[-1] == (
        "prefill",
        (
            OWNERSHIP_CANCEL_MSG,
            ("request", "generation-1", "decode", "receive lease expired"),
        ),
    )


def test_alive_answer_renews_the_lease_up_to_the_bound() -> None:
    scheduler, commands = _lease_scheduler()
    scheduler._max_lease_renewals = 2
    alive = msgspec.msgpack.encode((OWNERSHIP_RESULT_ALIVE, "generation-1"))

    for expected in (1, 2):
        scheduler._ownership_deadlines["request"] = 1.0
        scheduler._expire_recv_ownership(now=2.0)
        assert scheduler._ownership_probe_pending == {"request"}
        scheduler._apply_probe_reply(
            "request", "generation-1", "prefill", alive
        )
        assert scheduler._lease_renewals["request"] == expected
        assert scheduler._ownership_probe_pending == set()
        assert scheduler._ownership_deadlines["request"] > 2.0

    # Renewals exhausted: the probe may extend the wait, never remove its bound.
    scheduler._ownership_deadlines["request"] = 1.0
    scheduler._expire_recv_ownership(now=2.0)

    assert scheduler._ownership_probe_pending == set()
    assert scheduler._ownership_cancel_pending == {"request"}
    assert commands[-1][1][0] == OWNERSHIP_CANCEL_MSG


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"not-msgpack",
        msgspec.msgpack.encode((OWNERSHIP_RESULT_ALIVE, "generation-9")),
        msgspec.msgpack.encode(("nonsense", "generation-1")),
    ],
)
def test_every_unusable_probe_answer_falls_back_to_cancel(
    response: bytes,
) -> None:
    scheduler, commands = _lease_scheduler()
    scheduler._expire_recv_ownership(now=2.0)

    scheduler._apply_probe_reply(
        "request", "generation-1", "prefill", response
    )
    # What the send loop's finally clause does for anything still pending.
    scheduler._fail_open_probe("prefill", "request", "generation-1")

    assert scheduler._ownership_cancel_pending == {"request"}
    assert scheduler._ownership_safe_releases.empty()
    assert commands[-1][1][0] == OWNERSHIP_CANCEL_MSG


def test_empty_probe_answer_stops_probing_that_peer() -> None:
    scheduler, commands = _lease_scheduler()
    scheduler._expire_recv_ownership(now=2.0)
    scheduler._apply_probe_reply("request", "generation-1", "prefill", b"")
    scheduler._fail_open_probe("prefill", "request", "generation-1")

    assert "prefill" in scheduler._probe_unsupported
    # An old producer must cost one failed probe, not one per request.
    scheduler._ownership_cancel_pending.clear()
    scheduler._ownership_deadlines["request"] = 1.0
    scheduler._expire_recv_ownership(now=2.0)

    assert scheduler._ownership_probe_pending == set()
    assert scheduler._ownership_cancel_pending == {"request"}


def test_probe_answer_for_a_superseded_generation_is_ignored() -> None:
    scheduler, commands = _lease_scheduler()
    scheduler._expire_recv_ownership(now=2.0)
    # Preemption re-registered the request under a new generation.
    scheduler._ownership_generations["request"] = "generation-2"

    scheduler._apply_probe_reply(
        "request",
        "generation-1",
        "prefill",
        msgspec.msgpack.encode((OWNERSHIP_RESULT_ALIVE, "generation-1")),
    )
    scheduler._fail_open_probe("prefill", "request", "generation-1")

    assert scheduler._lease_renewals == {}
    assert scheduler._ownership_cancel_pending == set()


def test_terminal_probe_answer_releases_through_the_proof_path() -> None:
    scheduler, _commands = _lease_scheduler()
    scheduler._expire_recv_ownership(now=2.0)

    scheduler._apply_probe_reply(
        "request",
        "generation-1",
        "prefill",
        msgspec.msgpack.encode((OWNERSHIP_RESULT_TERMINAL, "generation-1")),
    )

    assert scheduler._ownership_probe_pending == set()
    assert scheduler._ownership_safe_releases.get_nowait() == (
        "request",
        "generation-1",
        "producer ownership terminal",
    )


def test_ack_loss_retries_without_releasing_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LostAckSocket:
        def __init__(self):
            self.failed = asyncio.Event()

        async def send(self, _payload):
            return None

        async def recv(self):
            self.failed.set()
            raise zmq.Again()

        def close(self, **_kwargs):
            return None

        def setsockopt(self, *_args):
            return None

    class TerminalSocket:
        sends = 0

        async def send(self, _payload):
            self.sends += 1

        async def recv(self):
            return msgspec.msgpack.encode(
                (OWNERSHIP_RESULT_TERMINAL, "generation-1")
            )

        def close(self, **_kwargs):
            return None

        def setsockopt(self, *_args):
            return None

    async def exercise() -> None:
        scheduler = object.__new__(NixlConnectorScheduler)
        scheduler._stop_event = threading.Event()
        scheduler.send_to_peer_infos = asyncio.Queue()
        scheduler._ownership_generations = {
            "request": "generation-1"
        }
        scheduler._ownership_cancel_pending = {"request"}
        scheduler._ownership_states = {
            "request": RequestOwnershipState.ACCEPTED
        }
        scheduler._ownership_safe_releases = queue.SimpleQueue()
        scheduler.ctx = object()
        lost = LostAckSocket()
        terminal = TerminalSocket()
        scheduler.peer_node_metas = {
            "prefill": PeerMeta(
                addr="192.0.2.1:19001",
                socket=lost,
            )
        }
        monkeypatch.setattr(
            scheduler_module,
            "make_zmq_socket",
            lambda **_kwargs: terminal,
        )
        command = (
            OWNERSHIP_CANCEL_MSG,
            (
                "request",
                "generation-1",
                "decode",
                "receive lease expired",
            ),
        )
        task = asyncio.create_task(scheduler._run_send_infos())
        await scheduler.send_to_peer_infos.put(("prefill", command, 0))
        await asyncio.wait_for(lost.failed.wait(), timeout=1)
        assert scheduler._ownership_safe_releases.empty()

        for _ in range(100):
            if not scheduler._ownership_safe_releases.empty():
                break
            await asyncio.sleep(0.01)
        proof = scheduler._ownership_safe_releases.get_nowait()
        assert proof[:2] == ("request", "generation-1")
        assert terminal.sends == 1
        scheduler._stop_event.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_stale_generation_notification_cannot_complete_new_receive() -> None:
    worker = _ownership_worker()
    worker._recv_requests = {
        "request": ReqMeta(
            local_block_ids=[[1]],
            engine_id="prefill",
            generation="generation-2",
        )
    }
    worker._recv_generations = {"request": "generation-2"}
    worker._recv_tombstones = RequestTombstoneStore(ttl=60.0, limit=16)
    worker._aborted_recv_requests = set()
    worker._recv_notification_status = {}
    worker._early_recv_notifications = {}
    worker._early_recv_deadlines = {}
    payload = NotifyPayload(0, 0, 1, 0, 0, 1, 0, 0, 1, 1)

    worker._apply_recv_notification(
        "request",
        "generation-1",
        TransferNotificationStatus.SUCCESS,
        payload,
    )

    assert "request" in worker._recv_requests
    assert worker._completion_outbox.empty()


def _transport_failure_worker() -> NixlConnectorWorker:
    worker = _ownership_worker()
    worker.engine_id = "prefill-0"
    worker.is_consumer = False
    worker._session_role = "producer"
    worker._engine_generations = {}
    worker._remote_node_metas = {}
    worker._engines_needing_rehandshake = set()
    worker._rehandshake_retry_at = {}
    worker._rehandshake_failures = {}
    worker._reachability_outbox = asyncio.Queue()
    worker._reported_reachability = {}
    worker._pending_transport_failures = {}
    worker._remote_agents_per_layer = {}
    worker._remote_agents_per_rank = {}
    worker._remote_agents = {}
    worker.dst_xfer_side_handles = {}
    worker._tp_size = {}
    worker._pp_size = {}
    worker._block_size = {}
    worker.rehandshake_retry_interval = 5.0
    worker.producer_disconnect_drain_timeout = 45.0
    worker.consumer_disconnect_grace = 0.0
    worker._teardown_outbox = asyncio.Queue()
    worker.global_rank = 0
    worker._early_recv_deadlines = {}
    worker._early_recv_notifications = {}
    session = RemoteEngineSession(
        engine_id="decode-0", generation=1, role="producer"
    )
    session.activate()
    worker.remote_sessions["decode-0"] = session
    worker._remote_agents_per_layer["decode-0"] = [[None]]
    worker._remote_agents_per_rank["decode-0"] = [[None]]
    return worker


def test_terminal_transport_error_invalidates_session() -> None:
    worker = _transport_failure_worker()

    worker._note_transport_failure(
        "decode-0",
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        "transfer setup failed",
    )
    # Deferred so a running _write_blocks keeps its dlists.
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.ACTIVE

    worker._apply_transport_failures()

    assert "decode-0" in worker._engines_needing_rehandshake
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.CLOSED
    assert "decode-0" not in worker._remote_agents_per_layer


def test_request_scoped_error_keeps_session_active() -> None:
    worker = _transport_failure_worker()

    worker._note_transport_failure(
        "decode-0",
        ValueError("local group 0 has 3 blocks, fewer than the 4 blocks"),
        "transfer start failed",
    )
    worker._apply_transport_failures()

    assert worker._engines_needing_rehandshake == set()
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.ACTIVE


def test_invalidated_engine_rejects_instead_of_hanging() -> None:
    worker = _transport_failure_worker()
    worker._note_transport_failure(
        "decode-0",
        RuntimeError("NIXL_ERR_NOT_FOUND"),
        "failure notification failed",
        agent_scoped=True,
    )
    worker._apply_transport_failures()
    meta = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )

    # No cached connect metadata, so the rebuild cannot run and the request
    # must fail fast rather than wait for an ACTIVE session that never comes.
    accepted = worker._session_accepts_request("request", meta)

    assert accepted is False
    assert "request" in worker._send_failures or not worker._completion_outbox.empty()
    assert "decode-0" in worker._engines_needing_rehandshake


def test_invalidated_engine_is_rebuilt_from_cached_connect_metadata() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    worker._note_transport_failure(
        "decode-0",
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        "transfer setup failed",
    )
    worker._apply_transport_failures()
    handshakes: list[str] = []

    def fake_handshake(node_meta):
        handshakes.append(node_meta.engine_id)
        return [["layer-cell"]], [["rank-cell"]]

    worker._nixl_handshake = fake_handshake
    meta = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )

    accepted = worker._session_accepts_request("request", meta)

    assert accepted is True
    assert handshakes == ["decode-0"]
    assert worker._engines_needing_rehandshake == set()
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.ACTIVE
    assert worker._remote_agents_per_layer["decode-0"] == [["layer-cell"]]


def test_failed_rebuild_is_rate_limited() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    worker._note_transport_failure(
        "decode-0",
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        "transfer setup failed",
    )
    worker._apply_transport_failures()
    attempts: list[str] = []

    def failing_handshake(node_meta):
        attempts.append(node_meta.engine_id)
        raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")

    worker._nixl_handshake = failing_handshake
    meta = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )

    for req_id in ("request-1", "request-2", "request-3"):
        assert worker._session_accepts_request(req_id, meta) is False

    # One proxy round trip per interval, not one per request.
    assert attempts == ["decode-0"]


def _aborted_recv_scheduler() -> NixlConnectorScheduler:
    from vllm.v1.request import RequestStatus

    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = True
    scheduler.mm_replay = None
    scheduler.prefix_state = None
    scheduler.engine_id = "decode-0"
    scheduler._reqs_not_send_to_worker = set()
    scheduler._reqs_not_processed_recv = set()
    scheduler._live_recv_reqs = set()
    scheduler._async_loop = SimpleNamespace(
        call_soon_threadsafe=lambda fn, *args: fn(*args)
    )
    scheduler.send_to_peer_infos = SimpleNamespace(put_nowait=lambda _item: None)
    _init_scheduler_ownership_state(scheduler)
    scheduler._request_status = RequestStatus
    return scheduler


def _failed_recv_request() -> SimpleNamespace:
    from vllm.v1.request import RequestStatus

    return SimpleNamespace(
        request_id="long-prompt",
        status=RequestStatus.FINISHED_ERROR,
        kv_transfer_params={
            "do_remote_prefill": False,
            "remote_engine_id": "prefill-0",
            "_opt_nixl_generation": "generation-1",
        },
    )


def test_live_receive_abort_still_delays_release_until_producer_proof() -> None:
    scheduler = _aborted_recv_scheduler()
    request = _failed_recv_request()
    scheduler._live_recv_reqs.add(request.request_id)
    scheduler._ownership_generations[request.request_id] = "generation-1"
    scheduler._ownership_remote_engines[request.request_id] = "prefill-0"

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is True
    assert request.request_id in scheduler._ownership_cancel_pending
    assert scheduler._reqs_not_processed_recv == {request.request_id}


def test_proven_terminal_receive_releases_blocks_instead_of_stranding_them() -> None:
    scheduler = _aborted_recv_scheduler()
    request = _failed_recv_request()
    # The 480s lease cancel got a producer terminal proof, which
    # _drain_safe_releases recorded before the completion dropped the
    # ownership generation.
    scheduler._recv_release_proofs.add(request.request_id)

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is False
    assert scheduler._reqs_not_processed_recv == set()
    assert scheduler._recv_release_proofs == set()


def test_unproven_receive_keeps_delaying_release() -> None:
    scheduler = _aborted_recv_scheduler()
    request = _failed_recv_request()
    # A receive the worker rejected while its session was draining reports a
    # completion without any teardown proof, so P may still issue a late
    # WRITE: the blocks must not be released here.

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is True
    assert scheduler._reqs_not_processed_recv == {request.request_id}


def test_safe_release_records_the_terminal_proof() -> None:
    scheduler = _aborted_recv_scheduler()
    scheduler._ownership_generations["req"] = "generation-1"
    scheduler._ownership_safe_releases.put(
        ("req", "generation-1", "producer ownership terminal")
    )
    meta = SimpleNamespace(reqs_to_release={})

    scheduler._drain_safe_releases(meta)

    assert scheduler._recv_release_proofs == {"req"}
    assert meta.reqs_to_release == {
        "req": ("generation-1", "producer ownership terminal")
    }


def _receive_completed(
    scheduler: NixlConnectorScheduler, req_id: str, *, failed: bool
) -> None:
    scheduler._scheduler_failed_recvs = set()
    scheduler._live_recv_reqs.add(req_id)
    scheduler._ownership_generations[req_id] = "generation-1"
    scheduler._ownership_remote_engines[req_id] = "prefill-0"
    scheduler.update_connector_output(
        SimpleNamespace(
            finished_recving={req_id},
            failed_recving={req_id} if failed else None,
            finished_sending=None,
        )
    )


@pytest.mark.parametrize(
    "status", ["FINISHED_ABORTED", "FINISHED_ERROR", "FINISHED_STOPPED"]
)
def test_completed_receive_releases_blocks_on_any_finish(status: str) -> None:
    from vllm.v1.request import RequestStatus

    scheduler = _aborted_recv_scheduler()
    request = _failed_recv_request()
    request.status = getattr(RequestStatus, status)
    # The router cut a request D was already decoding.  Its completion
    # dropped the generation, so no producer proof can follow, and only the
    # completion itself shows that every WRITE landed.
    _receive_completed(scheduler, request.request_id, failed=False)

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is False
    assert scheduler._reqs_not_processed_recv == set()
    assert scheduler._recv_release_proofs == set()


def test_failed_receive_completion_is_not_a_proof() -> None:
    scheduler = _aborted_recv_scheduler()
    request = _failed_recv_request()
    # Rejected while its session was draining, after P had already accepted
    # the destination blocks: P may still issue a late WRITE.
    _receive_completed(scheduler, request.request_id, failed=True)

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is True
    assert scheduler._reqs_not_processed_recv == {request.request_id}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("NIXL_ERR_CANCELED"),
        nixlCancelledError("cancelled by consumer"),
        RuntimeError("NIXL_ERR_INVALID_PARAM"),
        RuntimeError("backend returned NIXL_ERR_MISMATCH"),
        nixlBackendError("NIXL_ERR_BACKEND"),
        ValueError("local group 0 has 3 blocks, fewer than the 4 blocks"),
    ],
)
def test_request_scoped_errors_never_tear_down_the_session(
    error: BaseException,
) -> None:
    worker = _transport_failure_worker()

    # Both scopes: even a notification failure must not blame the session for
    # a status that does not prove the peer is gone.
    worker._note_transport_failure("decode-0", error, "ctx")
    worker._note_transport_failure("decode-0", error, "ctx", agent_scoped=True)
    worker._apply_transport_failures()

    assert worker._engines_needing_rehandshake == set()
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.ACTIVE


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        nixlRemoteDisconnectError("peer went away"),
    ],
)
def test_disconnect_status_tears_down_the_session(error: BaseException) -> None:
    worker = _transport_failure_worker()

    worker._note_transport_failure("decode-0", error, "ctx")
    worker._apply_transport_failures()

    assert "decode-0" in worker._engines_needing_rehandshake


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("NIXL_ERR_NOT_FOUND"),
        nixlNotFoundError("unknown remote agent"),
    ],
)
def test_not_found_only_blames_the_session_for_agent_scoped_calls(
    error: BaseException,
) -> None:
    descriptor_scoped = _transport_failure_worker()
    descriptor_scoped._note_transport_failure("decode-0", error, "transfer setup")
    descriptor_scoped._apply_transport_failures()
    assert descriptor_scoped._engines_needing_rehandshake == set()

    agent_scoped = _transport_failure_worker()
    agent_scoped._note_transport_failure(
        "decode-0", error, "failure notification", agent_scoped=True
    )
    agent_scoped._apply_transport_failures()
    assert "decode-0" in agent_scoped._engines_needing_rehandshake


def test_draining_session_does_not_burn_the_rebuild_cooldown() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    # A session still draining cannot be rebuilt, and no handshake runs, so
    # the next attempt must not be delayed by the cooldown.
    worker.remote_sessions["decode-0"].start_draining(0.0, 45.0)
    worker._engines_needing_rehandshake.add("decode-0")

    assert worker._rehandshake_remote_engine("decode-0") is False
    assert "decode-0" not in worker._rehandshake_retry_at

    worker.remote_sessions["decode-0"].state = RemoteSessionState.CLOSED
    worker._nixl_handshake = lambda _meta: ([["cell"]], [["cell"]])

    assert worker._rehandshake_remote_engine("decode-0") is True


def test_repeated_rebuild_failures_back_off() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    worker._engines_needing_rehandshake.add("decode-0")
    worker.remote_sessions["decode-0"].state = RemoteSessionState.CLOSED

    def failing_handshake(_meta):
        raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")

    worker._nixl_handshake = failing_handshake
    waits = []
    for _ in range(4):
        before = time.perf_counter()
        worker._rehandshake_remote_engine("decode-0")
        waits.append(worker._rehandshake_retry_at["decode-0"] - before)
        worker._rehandshake_retry_at.pop("decode-0")
        worker.remote_sessions["decode-0"].state = RemoteSessionState.CLOSED

    assert waits == sorted(waits)
    assert waits[-1] > waits[0]
    # Bounded, so a permanently dead peer cannot push the retry out forever.
    assert waits[-1] <= worker.rehandshake_retry_interval * 8 + 1.0


def test_router_teardown_fences_automatic_rebuild() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    worker._note_transport_failure(
        "decode-0", RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"), "ctx"
    )
    worker._apply_transport_failures()
    assert "decode-0" in worker._engines_needing_rehandshake

    # What the disconnect branch of _scheduler_listener does for a router
    # teardown; after it, no cached metadata may resurrect the session.
    worker._remote_node_metas.pop("decode-0", None)
    worker._engines_needing_rehandshake.discard("decode-0")
    worker._rehandshake_retry_at.pop("decode-0", None)
    worker._rehandshake_failures.pop("decode-0", None)
    worker._mark_engine_disconnected("decode-0", 7)

    worker._nixl_handshake = lambda _meta: pytest.fail(
        "rebuilt a session the router tore down"
    )

    assert worker._rehandshake_remote_engine("decode-0") is False


def test_connect_disconnect_stale_connect_retracts_the_teardown() -> None:
    worker = _transport_failure_worker()
    node_meta = SimpleNamespace(engine_id="decode-0")

    # CONNECT: already ACTIVE from the fixture, with cached connect metadata.
    worker._remote_node_metas["decode-0"] = node_meta

    # DISCONNECT: router teardown 7, reported closed by this rank while the
    # scheduler is still collecting the other ranks' acks.  The fence lives in
    # the disconnect branch of _scheduler_listener.
    worker._remote_node_metas.pop("decode-0", None)
    worker._engines_needing_rehandshake.discard("decode-0")
    worker._rehandshake_retry_at.pop("decode-0", None)
    worker._rehandshake_failures.pop("decode-0", None)
    worker._mark_engine_disconnected("decode-0", 7)
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.CLOSED
    assert worker._teardown_outbox.get_nowait() == (
        "decode-0",
        1,
        7,
        0,
        "closed",
    )
    # The fence must also have dropped the cached metadata, so nothing can
    # rebuild the session on its own.
    assert "decode-0" not in worker._remote_node_metas
    assert worker._rehandshake_remote_engine("decode-0") is False

    # Stale CONNECT: still reactivates the engine, but this rank now retracts
    # its ack so the pending teardown resolves as failed instead of ok.
    previous = worker.remote_sessions["decode-0"]
    session = worker._new_connect_session("decode-0")
    assert session is not previous
    session.activate()
    worker._retract_teardown_if_unsettled(previous)

    assert worker._teardown_outbox.get_nowait() == (
        "decode-0",
        1,
        7,
        0,
        "reconnected",
    )
    # Any ack other than "closed" makes the scheduler report the teardown as
    # failed, so the Router retries instead of believing the pair is gone.
    assert worker._teardown_outbox.empty()


def test_teardown_retraction_is_scoped_and_idempotent() -> None:
    worker = _transport_failure_worker()

    # Never torn down: nothing to retract.
    worker._retract_teardown_if_unsettled(worker.remote_sessions["decode-0"])
    assert worker._teardown_outbox.empty()

    # Torn down locally without a router token: never reported, so there is no
    # ack to overwrite.
    worker._mark_engine_disconnected("decode-0")
    worker._retract_teardown_if_unsettled(worker.remote_sessions["decode-0"])
    assert worker._teardown_outbox.empty()


def test_handshake_failure_fails_next_request_instead_of_hanging() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")

    # What the connect branch does when _nixl_handshake raises, e.g. on the
    # remote lease assertion.
    worker._note_handshake_failure(
        "decode-0", AssertionError("expire time must >= local expire time")
    )

    assert "decode-0" in worker._engines_needing_rehandshake
    # Cooldown is armed, so the next request neither retries the handshake it
    # just saw fail nor waits for a session that cannot activate.
    attempts: list[str] = []
    worker._nixl_handshake = lambda meta: attempts.append(meta.engine_id)
    meta = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )

    assert worker._session_accepts_request("request", meta) is False
    assert attempts == []
    batch = worker._completion_outbox.get_nowait()
    assert batch.done_sending == {"request"}
    assert batch.failures["request"].endswith("new handshake")
    # No CONNECTING placeholder was left behind to accept later work.
    assert not worker.remote_sessions["decode-0"].accepts_work


def test_handshake_failure_recovers_once_the_cooldown_elapses() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")
    worker._note_handshake_failure("decode-0", RuntimeError("proxy timeout"))
    worker._rehandshake_retry_at["decode-0"] = time.perf_counter() - 1.0
    worker._nixl_handshake = lambda _meta: ([["cell"]], [["cell"]])
    meta = ReqMeta(
        local_block_ids=[[1]], engine_id="decode-0", generation="generation-1"
    )

    assert worker._session_accepts_request("request", meta) is True
    assert worker._engines_needing_rehandshake == set()
    assert worker.remote_sessions["decode-0"].state == RemoteSessionState.ACTIVE


def test_producer_reports_unreachable_then_reachable_once() -> None:
    worker = _transport_failure_worker()
    worker._remote_node_metas["decode-0"] = SimpleNamespace(engine_id="decode-0")

    worker._note_transport_failure(
        "decode-0", RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"), "ctx"
    )
    worker._apply_transport_failures()

    assert worker._reachability_outbox.get_nowait() == ("decode-0", False)
    # Repeated failures must not re-report the same view.
    worker._note_transport_failure(
        "decode-0", RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"), "ctx"
    )
    worker._apply_transport_failures()
    assert worker._reachability_outbox.empty()

    worker._nixl_handshake = lambda _meta: ([["cell"]], [["cell"]])
    assert worker._rehandshake_remote_engine("decode-0") is True
    assert worker._reachability_outbox.get_nowait() == ("decode-0", True)


def test_consumer_never_reports_reachability() -> None:
    worker = _transport_failure_worker()
    worker.is_consumer = True

    worker._report_reachability("prefill-0", reachable=False)

    assert worker._reachability_outbox.empty()


def _endpoint_scheduler(endpoint: str, pub_endpoint: str) -> NixlConnectorScheduler:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = False
    scheduler.mm_replay = None
    scheduler.prefix_state = None
    scheduler.engine_id = "prefill-0"
    scheduler.local_addr = endpoint
    scheduler.scheduler_addr = pub_endpoint
    scheduler.ctx = zmq.asyncio.Context()
    scheduler.node_info = {"tp_size": 1, "pp_size": 1}
    scheduler._stop_event = threading.Event()
    scheduler._teardown_timeout = 30.0
    scheduler._ownership_tombstone_ttl = 60.0
    scheduler._ownership_tombstone_limit = 64
    scheduler._ownership_terminal_events = asyncio.Queue()
    scheduler.send_to_peer_infos = asyncio.Queue()
    scheduler._unreachable_ranks = {}
    scheduler._live_send_reqs = set()
    scheduler._reqs_to_send = {}
    scheduler._pending_worker_sends = set()
    _init_scheduler_ownership_state(scheduler)
    return scheduler


def test_scheduler_refuses_block_ids_while_a_rank_cannot_reach_the_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        endpoint = f"inproc://endpoint-{id(exercise)}"
        pub_endpoint = f"inproc://pub-{id(exercise)}"
        monkeypatch.setattr(
            scheduler_module, "make_zmq_path", lambda _t, _a: endpoint
        )
        scheduler = _endpoint_scheduler(endpoint, pub_endpoint)
        task = asyncio.create_task(scheduler._worker_data_forward())
        subscriber = scheduler.ctx.socket(zmq.XSUB)
        peer = scheduler.ctx.socket(zmq.REQ)
        peer.setsockopt(zmq.RCVTIMEO, 3000)
        try:
            subscriber.connect(pub_endpoint)
            await subscriber.send(b"\x01")
            assert await asyncio.wait_for(subscriber.recv(), timeout=3) == b"READY"
            peer.connect(endpoint)

            async def request(message: object) -> bytes:
                await peer.send(msgspec.msgpack.encode(message))
                return await asyncio.wait_for(peer.recv(), timeout=3)

            block_ids = (
                "block_ids",
                (
                    "req-1",
                    "generation-1",
                    TransferIntent.RECEIVE_KV.value,
                    [[7]],
                    "decode-0",
                    [-1],
                ),
            )

            # Healthy: accepted and forwarded to the local workers.
            reply = await request(block_ids)
            assert msgspec.msgpack.decode(reply) == [
                OWNERSHIP_RESULT_ACCEPTED,
                "generation-1",
            ]
            assert await asyncio.wait_for(subscriber.recv(), timeout=3)

            # A rank loses the consumer; the producer cannot notify it, so the
            # scheduler refuses the next generation instead of leaving it to
            # the consumer's lease.
            assert await request(
                (ENGINE_REACHABILITY_MSG, ("decode-0", 0, False))
            ) == b""
            assert scheduler._unreachable_ranks["decode-0"] == {0}

            reply = await request(
                ("block_ids", ("req-2", "generation-2", "receive_kv", [[8]], "decode-0", [-1]))
            )
            assert msgspec.msgpack.decode(reply) == [
                OWNERSHIP_RESULT_CANCELED,
                "generation-2",
            ]

            # Recovery re-opens the pair.
            assert await request(
                (ENGINE_REACHABILITY_MSG, ("decode-0", 0, True))
            ) == b""
            reply = await request(
                ("block_ids", ("req-3", "generation-3", "receive_kv", [[9]], "decode-0", [-1]))
            )
            assert msgspec.msgpack.decode(reply) == [
                OWNERSHIP_RESULT_ACCEPTED,
                "generation-3",
            ]
        finally:
            scheduler._stop_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            subscriber.close(linger=0)
            peer.close(linger=0)
            scheduler.ctx.term()

    asyncio.run(exercise())


def test_recv_lease_scales_with_prompt_length() -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler._kv_lease_duration = 480
    scheduler._kv_lease_per_1k_tokens = 1.0

    short = scheduler._recv_lease_for(SimpleNamespace(num_prompt_tokens=1024))
    long = scheduler._recv_lease_for(SimpleNamespace(num_prompt_tokens=713 * 1024))

    assert short == pytest.approx(481.0)
    assert long == pytest.approx(1193.0)
    # A 700K-token remote prefill must not be killed on the same clock as a 1K
    # one; the lease still has a bound.
    assert long > short


def test_unreachable_consumer_refusal_is_a_cancel_not_a_terminal_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        endpoint = f"inproc://endpoint-cancel-{id(exercise)}"
        pub_endpoint = f"inproc://pub-cancel-{id(exercise)}"
        monkeypatch.setattr(
            scheduler_module, "make_zmq_path", lambda _t, _a: endpoint
        )
        scheduler = _endpoint_scheduler(endpoint, pub_endpoint)
        task = asyncio.create_task(scheduler._worker_data_forward())
        subscriber = scheduler.ctx.socket(zmq.XSUB)
        peer = scheduler.ctx.socket(zmq.REQ)
        peer.setsockopt(zmq.RCVTIMEO, 3000)
        try:
            subscriber.connect(pub_endpoint)
            await subscriber.send(b"\x01")
            assert await asyncio.wait_for(subscriber.recv(), timeout=3) == b"READY"
            peer.connect(endpoint)

            async def request(message: object) -> bytes:
                await peer.send(msgspec.msgpack.encode(message))
                return await asyncio.wait_for(peer.recv(), timeout=3)

            def block_ids(req_id: str, generation: str) -> tuple:
                return (
                    "block_ids",
                    (req_id, generation, "receive_kv", [[7]], "decode-0", [-1]),
                )

            # A generation this loop already published to its workers: a WRITE
            # may be in flight against those blocks.
            assert msgspec.msgpack.decode(
                await request(block_ids("req-live", "generation-1"))
            ) == [OWNERSHIP_RESULT_ACCEPTED, "generation-1"]
            assert await asyncio.wait_for(subscriber.recv(), timeout=3)

            assert await request(
                (ENGINE_REACHABILITY_MSG, ("decode-0", 0, False))
            ) == b""

            # D retries block_ids whenever an ACK is lost.  The retry must not
            # be answered from the unreachable shortcut, or D would release
            # blocks an in-flight WRITE still targets.
            assert msgspec.msgpack.decode(
                await request(block_ids("req-live", "generation-1"))
            ) == [OWNERSHIP_RESULT_ACCEPTED, "generation-1"]
            assert await asyncio.wait_for(subscriber.recv(), timeout=3)

            # A generation that was never published is safe to refuse, but the
            # answer is a cancel: refusing proves nothing about other WRITEs.
            assert msgspec.msgpack.decode(
                await request(block_ids("req-new", "generation-2"))
            ) == [OWNERSHIP_RESULT_CANCELED, "generation-2"]
        finally:
            scheduler._stop_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            subscriber.close(linger=0)
            peer.close(linger=0)
            scheduler.ctx.term()

    asyncio.run(exercise())


def test_expired_receive_lease_acts_and_logs_once() -> None:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.engine_id = "decode-0"
    scheduler._live_recv_reqs = {"req"}
    scheduler._async_loop = SimpleNamespace(
        call_soon_threadsafe=lambda fn, *args: fn(*args)
    )
    queued: list[object] = []
    scheduler.send_to_peer_infos = SimpleNamespace(put_nowait=queued.append)
    _init_scheduler_ownership_state(scheduler)
    scheduler._ownership_generations["req"] = "generation-1"
    scheduler._ownership_remote_engines["req"] = "prefill-0"
    scheduler._ownership_deadlines["req"] = 1.0
    scheduler._lease_renew_interval = 120.0
    scheduler._max_lease_renewals = 10

    # build_connector_meta calls this every scheduler step.
    for _ in range(5):
        scheduler._expire_recv_ownership(now=2.0)

    assert len(queued) == 1
    assert queued[0][1][0] == OWNERSHIP_PROBE_MSG
    assert scheduler._ownership_probe_pending == {"req"}


def _probe_producer() -> NixlConnectorScheduler:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = False
    scheduler._unreachable_ranks = {}
    scheduler._live_send_reqs = set()
    scheduler._reqs_to_send = {}
    scheduler._pending_worker_sends = set()
    _init_scheduler_ownership_state(scheduler)
    return scheduler


def _probe(scheduler: NixlConnectorScheduler, generation: str = "generation-1") -> str:
    return scheduler._probe_ownership(
        "req", generation, "decode-0", RequestTombstoneStore(ttl=60.0, limit=16)
    )


def test_probe_vouches_for_a_request_still_queued_on_the_producer() -> None:
    # The 700K-token case: admitted but not scheduled for a long time.  A
    # fixed lease cannot tell this apart from a dead peer; the probe can.
    queued = _probe_producer()
    queued._live_send_reqs.add("req")
    assert _probe(queued) == OWNERSHIP_RESULT_ALIVE

    prefilling = _probe_producer()
    prefilling._reqs_to_send["req"] = (1, True, "decode-0", "generation-1")
    assert _probe(prefilling) == OWNERSHIP_RESULT_ALIVE

    writing = _probe_producer()
    writing._pending_worker_sends.add("req")
    assert _probe(writing) == OWNERSHIP_RESULT_ALIVE


def test_probe_never_vouches_for_an_unknown_or_superseded_generation() -> None:
    unknown = _probe_producer()
    assert _probe(unknown) == OWNERSHIP_RESULT_UNKNOWN

    superseded = _probe_producer()
    superseded._live_send_reqs.add("req")
    superseded._ownership_generations["req"] = "generation-2"
    assert _probe(superseded) == OWNERSHIP_RESULT_CANCELED


def test_probe_answer_for_an_unreachable_consumer_depends_on_disclosure() -> None:
    # Never published: no worker learned the destination blocks, so reclaiming
    # them is safe.
    undisclosed = _probe_producer()
    undisclosed._live_send_reqs.add("req")
    undisclosed._unreachable_ranks["decode-0"] = {0}
    assert _probe(undisclosed) == OWNERSHIP_RESULT_CANCELED

    # Published: a WRITE may be in flight, so the probe must not vouch for it
    # either -- only worker completion can prove it terminal.
    disclosed = _probe_producer()
    disclosed._live_send_reqs.add("req")
    disclosed._pending_worker_sends.add("req")
    disclosed._unreachable_ranks["decode-0"] = {0}
    disclosed._ownership_generations["req"] = "generation-1"
    disclosed._ownership_states["req"] = RequestOwnershipState.ACCEPTED
    assert _probe(disclosed) == OWNERSHIP_RESULT_UNKNOWN


def test_probe_honours_an_existing_terminal_tombstone() -> None:
    scheduler = _probe_producer()
    scheduler._live_send_reqs.add("req")
    tombstones = RequestTombstoneStore(ttl=60.0, limit=16)
    tombstones.record("req", "generation-1", RequestOwnershipState.TERMINAL)

    assert (
        scheduler._probe_ownership(
            "req", "generation-1", "decode-0", tombstones
        )
        == OWNERSHIP_RESULT_TERMINAL
    )


def test_producer_stops_vouching_once_the_request_finishes() -> None:
    from vllm.v1.request import RequestStatus

    scheduler = _probe_producer()
    scheduler.mm_replay = None
    scheduler.prefix_state = None
    scheduler._reqs_blocks = {}
    scheduler._sends_completed_early = set()
    scheduler._sends_awaiting_release = set()
    request = SimpleNamespace(
        request_id="req",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={"do_remote_decode": True},
    )
    scheduler.on_new_request(request)
    assert scheduler._live_send_reqs == {"req"}
    assert _probe(scheduler) == OWNERSHIP_RESULT_ALIVE

    scheduler.request_finished_all_groups(request, ([1],))

    assert scheduler._live_send_reqs == set()
    assert _probe(scheduler) == OWNERSHIP_RESULT_UNKNOWN


def _finishing_producer() -> NixlConnectorScheduler:
    scheduler = _probe_producer()
    scheduler.mm_replay = None
    scheduler._reqs_blocks = {}
    scheduler._sends_completed_early = set()
    scheduler._sends_awaiting_release = set()
    return scheduler


def _aborted_send_request() -> SimpleNamespace:
    from vllm.v1.request import RequestStatus

    # update_state_after_alloc cleared do_remote_decode at the first chunk.
    return SimpleNamespace(
        request_id="req",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={"do_remote_decode": False},
    )


def test_producer_abort_before_handoff_frees_blocks_now() -> None:
    # The router cut a 1M-token prefill mid-way: no worker was ever handed the
    # send, so no completion it reports is accepted later, and a delayed free
    # pinned these blocks until the engine restarted.
    scheduler = _finishing_producer()
    request = _aborted_send_request()
    scheduler._live_send_reqs.add("req")
    scheduler._reqs_to_send["req"] = (1024, False, "decode-0", "generation-1")
    scheduler._reqs_blocks["req"] = [[1, 2]]

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is False
    assert "req" not in scheduler._reqs_to_send
    assert "req" not in scheduler._reqs_blocks
    assert _probe(scheduler) == OWNERSHIP_RESULT_UNKNOWN


def test_producer_abort_after_handoff_still_waits_for_the_worker() -> None:
    # The final chunk was scheduled, so a WRITE may still read these blocks.
    scheduler = _finishing_producer()
    request = _aborted_send_request()
    scheduler._pending_worker_sends.add("req")

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    assert delay_free is True
    assert scheduler._sends_awaiting_release == {"req"}


def test_worker_may_reclaim_a_handed_off_send_after_its_request_finished() -> None:
    scheduler = _probe_producer()
    scheduler.prefix_state = None
    scheduler._reqs_to_send_time_record = {}
    scheduler._live_send_reqs.add("queued")
    scheduler._reqs_to_send["prefilling"] = (1, False, "decode-0", "generation-1")
    scheduler._pending_worker_sends.add("handed-off")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], resumed_req_ids=set()),
        num_scheduled_tokens={},
    )

    meta = scheduler.build_connector_meta(scheduler_output)

    # Only the consumer can still join "handed-off"; producer_orphan_timeout
    # bounds that wait instead of extending it forever.
    assert meta.sends_still_progressing == {"queued", "prefilling"}


def test_probe_round_trip_over_the_side_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        endpoint = f"inproc://endpoint-probe-{id(exercise)}"
        pub_endpoint = f"inproc://pub-probe-{id(exercise)}"
        monkeypatch.setattr(
            scheduler_module, "make_zmq_path", lambda _t, _a: endpoint
        )
        scheduler = _endpoint_scheduler(endpoint, pub_endpoint)
        task = asyncio.create_task(scheduler._worker_data_forward())
        subscriber = scheduler.ctx.socket(zmq.XSUB)
        peer = scheduler.ctx.socket(zmq.REQ)
        peer.setsockopt(zmq.RCVTIMEO, 3000)
        try:
            subscriber.connect(pub_endpoint)
            await subscriber.send(b"\x01")
            assert await asyncio.wait_for(subscriber.recv(), timeout=3) == b"READY"
            peer.connect(endpoint)

            async def probe(generation: str) -> list:
                await peer.send(
                    msgspec.msgpack.encode(
                        (
                            OWNERSHIP_PROBE_MSG,
                            ("req-1", generation, "decode-0", "lease expired"),
                        )
                    )
                )
                return msgspec.msgpack.decode(
                    await asyncio.wait_for(peer.recv(), timeout=3)
                )

            assert await probe("generation-1") == [
                OWNERSHIP_RESULT_UNKNOWN,
                "generation-1",
            ]

            scheduler._live_send_reqs.add("req-1")
            assert await probe("generation-1") == [
                OWNERSHIP_RESULT_ALIVE,
                "generation-1",
            ]
            # Read-only: no tombstone, no cancel, nothing published to workers.
            assert scheduler._ownership_cancel_pending == set()
            assert scheduler._ownership_states == {}
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(subscriber.recv(), timeout=0.3)
        finally:
            scheduler._stop_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            subscriber.close(linger=0)
            peer.close(linger=0)
            scheduler.ctx.term()

    asyncio.run(exercise())


def _abort_scheduler() -> tuple[NixlConnectorScheduler, list]:
    scheduler, commands = _lease_scheduler()
    scheduler.mm_replay = None
    scheduler.prefix_state = None
    scheduler.is_consumer = True
    scheduler.kv_group_nums = 1
    scheduler._reqs_not_send_to_worker = set()
    scheduler._reqs_not_processed_recv = set()
    scheduler._recv_release_proofs = set()
    scheduler._scheduler_failed_recvs = set()
    scheduler._live_send_reqs = set()
    return scheduler, commands


def test_client_abort_after_disclosure_cancels_the_producer() -> None:
    from vllm.v1.request import RequestStatus

    scheduler, commands = _abort_scheduler()
    request = SimpleNamespace(
        request_id="request",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={
            "do_remote_prefill": False,
            "remote_engine_id": "prefill",
            "_opt_nixl_generation": "generation-1",
        },
    )

    delay_free, _params = scheduler.request_finished_all_groups(request, ([1, 2],))

    # The producer may still have a WRITE in flight against these blocks, so
    # they are held until it proves otherwise -- but it is told right away.
    assert delay_free is True
    assert scheduler._ownership_cancel_pending == {"request"}
    assert commands[-1] == (
        "prefill",
        (
            OWNERSHIP_CANCEL_MSG,
            ("request", "generation-1", "decode", "consumer request aborted"),
        ),
    )


def test_client_abort_before_disclosure_cancels_without_holding_blocks() -> None:
    from vllm.v1.request import RequestStatus

    scheduler, commands = _abort_scheduler()
    scheduler._get_or_connect_peer = lambda _engine: None
    request = SimpleNamespace(
        request_id="request",
        status=RequestStatus.FINISHED_ABORTED,
        # Killed before update_state_after_alloc ran, so no destination was
        # ever disclosed.
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_engine_id": "prefill",
            "_opt_nixl_generation": "generation-1",
        },
    )

    delay_free, _params = scheduler.request_finished_all_groups(request, ((),))

    assert delay_free is False
    assert commands[-1][1][0] == OWNERSHIP_CANCEL_MSG
    # Nothing was disclosed, so local release is already safe.
    assert "request" not in scheduler._live_recv_reqs
