"""Real CPU/TCP receiver readiness, generic chunk sizes and terminal cleanup."""

import threading
import time

import msgspec
import pytest
import zmq

from vllm_iluvatar.v1.worker.mm_replay.archive import SourceArchive
from vllm_iluvatar.v1.worker.mm_replay.receiver import RangeReceiver, read_window


@pytest.fixture
def endpoint(tmp_path):
    archive = SourceArchive(
        tmp_path / "producer-private",
        "P",
        "run1",
        budget_bytes=4096,
        hidden_size=8,
        max_transfer_rows=7,
    )
    feature = dict(key="image", start=3, rows=20, input=b"patches", fields=[])
    archive.admit("request", "gen", "a" * 64, 129, [feature])
    raw = bytes(range(160)) * 2
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    messages = []

    def run():
        while not stop.is_set():
            if not socket.poll(10):
                continue
            message = msgspec.msgpack.decode(socket.recv())
            messages.append(message)
            reply, payload = archive.handle(message)
            socket.send_multipart([msgspec.msgpack.encode(reply), payload])

    thread = threading.Thread(target=run)
    thread.start()
    yield (
        archive,
        raw,
        feature,
        dict(
            side_channel_addr=f"127.0.0.1:{port}",
            mm_replay=dict(version=1, hidden_size=8, max_transfer_rows=7),
        ),
        messages,
    )
    stop.set()
    thread.join(3)
    socket.close(0)
    context.term()


def test_receiver_does_not_block_scheduler_and_combines_general_chunks(
    tmp_path, endpoint
):
    archive, raw, feature, peer, messages = endpoint
    receiver = RangeReceiver(
        tmp_path / "decoder-private",
        lambda _: peer,
        hidden_size=8,
        max_window_rows=129,
        max_num_seqs=3,
    )
    spec = dict(
        request_id="request",
        generation="gen",
        input_digest="a" * 64,
        prompt_len=129,
        features=[{k: feature[k] for k in ("key", "start", "rows")}],
        engine_id="P",
    )
    try:
        receiver.register(spec)
        before = time.monotonic()
        assert not receiver.ensure("request", 0, 129)
        assert time.monotonic() - before < 0.05
        time.sleep(0.05)
        assert not receiver.ensure("request", 0, 129)
        archive.publish("request", "gen", "image", raw)
        deadline = time.monotonic() + 3
        while not receiver.ensure("request", 0, 129):
            assert not receiver.failures(), receiver.failures()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        meta, payload = read_window(
            receiver.root, "request", "gen", "a" * 64, 0, 129, 8, [feature]
        )
        assert payload == raw
        assert (
            len(meta["records"]) == 1
            and meta["records"][0]["start"] == 3
            and meta["records"][0]["end"] == 23
        )
        assert all(
            m[-1]["end"] - m[-1]["start"] <= 7 for m in messages if m[2] == "GET"
        )
        receiver.finish("request")
        deadline = time.monotonic() + 3
        while archive.stats()["requests"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert archive.stats()["leased_images"] == 0
    finally:
        receiver.shutdown()


def test_unadvertised_legacy_peer_is_rejected_without_sending_opcode(
    tmp_path, endpoint
):
    archive, raw, feature, peer, messages = endpoint
    receiver = RangeReceiver(
        tmp_path / "decoder-private",
        lambda _: dict(side_channel_addr=peer["side_channel_addr"]),
        hidden_size=8,
        max_window_rows=129,
        max_num_seqs=3,
    )
    try:
        receiver.register(
            dict(
                request_id="request",
                generation="gen",
                input_digest="a" * 64,
                prompt_len=129,
                features=[{k: feature[k] for k in ("key", "start", "rows")}],
                engine_id="P",
            )
        )
        deadline = time.monotonic() + 2
        while not receiver.failures():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not messages
    finally:
        receiver.shutdown()


def test_native_registry_multipart_and_legacy_safe_capability_query(
    tmp_path, monkeypatch
):
    """The real router drops unknown fields: capabilities come from P directly."""
    from vllm_iluvatar.v1.worker.mm_replay.archive import is_capability_request
    from vllm_iluvatar.v1.worker.mm_replay.scheduler import ReplayScheduler

    context = zmq.Context()
    proxy = context.socket(zmq.REP)
    producer = context.socket(zmq.REP)
    proxy_port = proxy.bind_to_random_port("tcp://127.0.0.1")
    p_port = producer.bind_to_random_port("tcp://127.0.0.1")
    observed = []

    def registry():
        observed.append(msgspec.msgpack.decode(proxy.recv()))
        proxy.send_multipart(
            [
                b"success",
                msgspec.msgpack.encode(dict(side_channel_addr=f"127.0.0.1:{p_port}")),
            ]
        )

    def capability():
        message = msgspec.msgpack.decode(producer.recv())
        observed.append(message)
        assert is_capability_request(message)
        producer.send_multipart(
            [
                msgspec.msgpack.encode(
                    dict(
                        version=3,
                        status="MM_CAPABILITIES",
                        engine_id="P",
                        mm_replay=dict(
                            version=1,
                            hidden_size=8,
                            model_digest="model",
                            max_transfer_rows=7,
                        ),
                    )
                ),
                b"",
            ]
        )

    threads = [threading.Thread(target=registry), threading.Thread(target=capability)]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("VLLM_ILU_NIXL_PROXY_ADDR", f"tcp://127.0.0.1:{proxy_port}")
    replay = ReplayScheduler.__new__(ReplayScheduler)
    replay.model_digest = "model"
    try:
        info = replay.resolve("P")
        assert info["mm_replay"]["version"] == 1
        assert len(observed) == 2
    finally:
        for thread in threads:
            thread.join(3)
        proxy.close(0)
        producer.close(0)
        context.term()
