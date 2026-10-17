import asyncio
import contextlib
import queue
import threading
from types import SimpleNamespace

import msgspec
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
    OWNERSHIP_CANCEL_MSG,
    OWNERSHIP_RESULT_TERMINAL,
    PeerMeta,
    ReqMeta,
    TEARDOWN_COMPLETE_MSG,
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


def test_generation_send_lease_waits_for_consumer_cancel() -> None:
    worker = _ownership_worker()
    worker._early_recv_deadlines = {}
    worker._early_recv_notifications = {}
    state = SendRequestState(
        req_id="request",
        generation="generation-1",
        deadline=1.0,
        ownership_state=RequestOwnershipState.PREPARED,
    )
    worker._send_requests[state.req_id] = state

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


def test_receive_lease_requests_cancel_without_releasing_blocks() -> None:
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
    scheduler._ownership_safe_releases = queue.SimpleQueue()
    commands = []
    scheduler._queue_peer_command = (
        lambda remote, command: commands.append((remote, command))
    )

    scheduler._expire_recv_ownership(now=2.0)

    assert scheduler._ownership_cancel_pending == {"request"}
    assert scheduler._ownership_safe_releases.empty()
    assert commands == [
        (
            "prefill",
            (
                OWNERSHIP_CANCEL_MSG,
                (
                    "request",
                    "generation-1",
                    "decode",
                    "receive lease expired",
                ),
            ),
        )
    ]


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
