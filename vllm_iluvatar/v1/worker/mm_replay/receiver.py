"""Background immutable image pages; no socket work in scheduling/forward."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path

from .archive import (
    MM_OP,
    VERSION,
    atomic_bytes,
    atomic_json,
    sha256,
    storage_key,
    validate_reply,
)
from .identity import overlapping


def required_pages(features, start, end, page_rows):
    """Page only intersecting image spans, independently of model/context length."""
    result = []
    for feature in features:
        lo = max(start, feature["start"])
        hi = min(end, feature["start"] + feature["rows"])
        if lo >= hi:
            continue
        first = (lo - feature["start"]) // page_rows * page_rows + feature["start"]
        for a in range(first, hi, page_rows):
            result.append((a, min(a + page_rows, feature["start"] + feature["rows"])))
    return result


class RangeReceiver:
    """Bounded CPU pages survive preemption and asynchronous in-flight batches.

    Only requested image pages are fetched. Retaining immutable received pages
    until worker terminal acknowledgement avoids reclaiming a file still used
    by a queued batch. GPU storage remains one actual-batch-sized window; CPU
    pages have a separate hard budget and never allocate a full context tensor.
    """

    def __init__(
        self,
        root: Path,
        resolve,
        *,
        hidden_size: int,
        max_window_rows: int,
        max_num_seqs: int,
        lease_seconds: float = 480,
        host_budget_bytes=4 * 1024**3,
        page_rows=64,
    ):
        if (
            min(
                hidden_size, max_window_rows, max_num_seqs, host_budget_bytes, page_rows
            )
            <= 0
        ):
            raise ValueError("Invalid decoder image page capacity")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.resolve = resolve
        self.hidden_size = hidden_size
        self.max_window_rows = max_window_rows
        self.host_budget_bytes = host_budget_bytes
        self.page_rows = min(page_rows, max_window_rows)
        self.lease_seconds = lease_seconds
        self.lock = threading.Lock()
        self.requests = {}
        self.retiring = {}
        self.release_queue = deque()
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="pd-image-range-receiver", daemon=True
        )
        self.thread.start()

    def register(self, spec):
        rid = spec["request_id"]
        with self.lock:
            old = self.requests.get(rid)
            if old:
                if (
                    old["generation"] != spec["generation"]
                    or old["input_digest"] != spec["input_digest"]
                ):
                    raise ValueError("Duplicate live multimodal identity")
                return
            directory = self.root / storage_key(rid)
            directory.mkdir(exist_ok=True)
            atomic_json(directory / "request.json", spec)
            self.requests[rid] = {
                **spec,
                "pending": deque(),
                "ready": set(),
                "serial": 0,
                "registered": time.monotonic(),
                "heartbeat": 0.0,
                "error": None,
                "session": None,
            }
        self.wakeup.set()

    def ensure(self, rid, start, end):
        with self.lock:
            state = self.requests[rid]
            end = min(end, state["prompt_len"])
            pages = required_pages(state["features"], start, end, self.page_rows)
            if not pages:
                return True
            if end - start > self.max_window_rows:
                raise ValueError("Replay exceeds actual batch capacity")
            missing = [p for p in pages if p not in state["ready"]]
            if not missing:
                return True
            queued = set(state["pending"])
            new = [p for p in missing if p not in queued]
            reserved = sum(
                sum(
                    (b - a) * self.hidden_size * 2
                    for a, b in (s["ready"] | set(s["pending"]))
                )
                for s in [*self.requests.values(), *self.retiring.values()]
            )
            needed = sum((b - a) * self.hidden_size * 2 for a, b in new)
            if reserved + needed <= self.host_budget_bytes:
                state["pending"].extend(new)
                self.wakeup.set()
            return False

    def failures(self):
        with self.lock:
            return {rid: s["error"] for rid, s in self.requests.items() if s["error"]}

    def finish(self, rid):
        with self.lock:
            state = self.requests.pop(rid, None)
            if state:
                state["pending"].clear()
                state["retired"] = time.monotonic()
                self.retiring[rid] = state
                self.release_queue.append(state)
                directory = self.root / storage_key(rid)
                (directory / "finished").touch()
        self.wakeup.set()

    def _publish(self, state, start, end, reply, raw):
        with self.lock:
            if self.requests.get(state["request_id"]) is not state:
                return
            pair = (start, end)
            if pair not in state["pending"]:
                return
            directory = self.root / storage_key(state["request_id"])
            serial = state["serial"] + 1
            state["serial"] = serial
            atomic_bytes(directory / f"{serial}.bin", raw)
            atomic_json(directory / f"{serial}.json", dict(reply, serial=serial))
            state["session"] = reply["session"]
            state["ready"].add(pair)
            state["pending"].remove(pair)

    def _run(self):
        import msgspec
        import zmq

        context = zmq.Context()
        sockets = {}
        peers = {}

        def peer_info(state):
            engine = state["engine_id"]
            if engine not in peers:
                peer = self.resolve(engine)
                cap = peer.get("mm_replay", {})
                if (
                    cap.get("version") != VERSION
                    or cap.get("hidden_size") != self.hidden_size
                ):
                    raise ValueError("Producer image replay capability mismatch")
                peers[engine] = peer
            return peers[engine]

        def exchange(state, action, start=0, end=0):
            engine = state["engine_id"]
            peer = peer_info(state)
            socket = sockets.get(engine)
            if socket is None:
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, 250)
                socket.setsockopt(zmq.RCVTIMEO, 250)
                socket.connect("tcp://" + peer["side_channel_addr"])
                sockets[engine] = socket
            socket.send(
                msgspec.msgpack.encode(
                    (
                        MM_OP,
                        VERSION,
                        action,
                        engine,
                        state["request_id"],
                        state["generation"],
                        state["input_digest"],
                        dict(start=start, end=end),
                    )
                )
            )
            frames = socket.recv_multipart()
            if len(frames) != 2:
                raise ValueError("Expected image metadata and payload frames")
            reply = msgspec.msgpack.decode(frames[0])
            raw = frames[1]
            if reply.get("status") in (
                "WRONG_ENGINE",
                "IDENTITY_OR_PAYLOAD_ERROR",
                "BAD_REQUEST",
                "UNSUPPORTED",
                "CANCELLED",
            ):
                raise ValueError(
                    "Producer rejected image replay: " + str(reply.get("status"))
                )
            if action == "GET":
                validate_reply(
                    reply,
                    raw,
                    engine=engine,
                    rid=state["request_id"],
                    generation=state["generation"],
                    digest=state["input_digest"],
                    start=start,
                    end=end,
                    hidden_size=self.hidden_size,
                    features=state["features"],
                    session=state["session"],
                )
            elif (
                reply.get("version") != VERSION
                or reply.get("engine_id") != engine
                or reply.get("generation") != state["generation"]
                or raw
            ):
                raise ValueError("Image lease acknowledgement mismatch")
            return reply, raw

        def fetch_page(state, start, end):
            limit = peer_info(state)["mm_replay"].get("max_transfer_rows")
            if type(limit) is not int or limit <= 0:
                raise ValueError("Invalid P transfer capacity")
            parts = []
            records = []
            offset = 0
            first = None
            for a in range(start, end, limit):
                b = min(a + limit, end)
                reply, raw = exchange(state, "GET", a, b)
                if reply["status"] != "READY":
                    return
                if first is not None and reply["session"] != first["session"]:
                    raise ValueError("Producer image session changed across chunks")
                first = reply
                for item in reply["records"]:
                    record = dict(item, offset_bytes=item["offset_bytes"] + offset)
                    if (
                        records
                        and records[-1]["key"] == record["key"]
                        and records[-1]["end"] == record["start"]
                    ):
                        previous = records[-1]
                        previous["end"] = record["end"]
                        previous["shape"][0] += record["shape"][0]
                        previous["byte_count"] += record["byte_count"]
                    else:
                        records.append(record)
                parts.append(raw)
                offset += len(raw)
            raw = b"".join(parts)
            reply = dict(
                first,
                start=start,
                end=end,
                records=records,
                byte_count=len(raw),
                sha256=sha256(raw),
            )
            validate_reply(
                reply,
                raw,
                engine=state["engine_id"],
                rid=state["request_id"],
                generation=state["generation"],
                digest=state["input_digest"],
                start=start,
                end=end,
                hidden_size=self.hidden_size,
                features=state["features"],
                session=state["session"],
            )
            self._publish(state, start, end, reply, raw)

        try:
            while not self.stop_event.is_set():
                with self.lock:
                    states = list(self.requests.values())
                    releases = list(self.release_queue)
                    self.release_queue.clear()
                    retired = list(self.retiring.items())
                for state in releases:
                    with suppress(Exception):
                        exchange(state, "RELEASE")
                now = time.monotonic()
                for rid, state in retired:
                    directory = self.root / storage_key(rid)
                    if (directory / "worker_done").exists() or now - state[
                        "retired"
                    ] > self.lease_seconds:
                        shutil.rmtree(directory, ignore_errors=True)
                        with self.lock:
                            self.retiring.pop(rid, None)
                for state in states:
                    if self.stop_event.is_set():
                        break
                    if state["error"]:
                        continue
                    try:
                        with self.lock:
                            pending = state["pending"][0] if state["pending"] else None
                        if pending:
                            fetch_page(state, *pending)
                        elif now - state["heartbeat"] >= min(
                            30, self.lease_seconds / 3
                        ):
                            exchange(state, "RENEW")
                        else:
                            continue
                        state["heartbeat"] = time.monotonic()
                    except zmq.ZMQError:
                        socket = sockets.pop(state["engine_id"], None)
                        if socket is not None:
                            socket.close(0)
                        if (
                            now - max(state["heartbeat"], state["registered"])
                            > self.lease_seconds
                        ):
                            state["error"] = (
                                "Producer image source exceeded the native lease"
                            )
                    except Exception as error:
                        state["error"] = repr(error)
                delay = (
                    0.005
                    if any(s["pending"] for s in states)
                    else (1.0 if states or retired else None)
                )
                self.wakeup.wait(delay)
                self.wakeup.clear()
        finally:
            with self.lock:
                remaining = list(self.requests.values()) + list(self.release_queue)
            for state in remaining:
                with suppress(Exception):
                    exchange(state, "RELEASE")
            for socket in sockets.values():
                socket.close(0)
            context.term()

    def shutdown(self):
        self.stop_event.set()
        self.wakeup.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Image receiver thread did not stop")
        shutil.rmtree(self.root, ignore_errors=True)


def read_window(root, rid, generation, digest, start, end, hidden_size, features):
    """Assemble only the scheduled rows from immutable D-private CPU pages."""
    directory = Path(root) / storage_key(rid)
    frames = []
    for path in directory.glob("*.json"):
        if not path.stem.isdigit():
            continue
        meta = json.loads(path.read_text())
        if not overlapping(features, max(start, meta["start"]), min(end, meta["end"])):
            continue
        raw = path.with_suffix(".bin").read_bytes()
        validate_reply(
            meta,
            raw,
            engine=meta["engine_id"],
            rid=rid,
            generation=generation,
            digest=digest,
            start=meta["start"],
            end=meta["end"],
            hidden_size=hidden_size,
            features=features,
        )
        frames.append((meta, raw))
    frames.sort(key=lambda pair: pair[0]["start"])
    records = []
    parts = []
    offset = 0
    first = None
    for feature in features:
        a, b = (
            max(start, feature["start"]),
            min(end, feature["start"] + feature["rows"]),
        )
        if a >= b:
            continue
        cursor = a
        pieces = []
        for meta, raw in frames:
            for item in meta["records"]:
                if (
                    item["key"] != feature["key"]
                    or not item["start"] <= cursor < item["end"]
                ):
                    continue
                if first is not None and meta["session"] != first["session"]:
                    raise ValueError(
                        "Decoder image pages have inconsistent source sessions"
                    )
                first = meta
                last = min(b, item["end"])
                byte0 = (
                    item["offset_bytes"] + (cursor - item["start"]) * hidden_size * 2
                )
                pieces.append(raw[byte0 : byte0 + (last - cursor) * hidden_size * 2])
                cursor = last
                if cursor == b:
                    break
            if cursor == b:
                break
        if cursor != b:
            raise RuntimeError("Scheduled image has incomplete decoder-local coverage")
        part = b"".join(pieces)
        parts.append(part)
        records.append(
            dict(
                key=feature["key"],
                start=a,
                end=b,
                shape=[b - a, hidden_size],
                dtype="BF16",
                offset_bytes=offset,
                byte_count=len(part),
            )
        )
        offset += len(part)
    if first is None:
        raise RuntimeError("Scheduled image has no decoder-local pages")
    raw = b"".join(parts)
    reply = dict(
        first,
        start=start,
        end=end,
        records=records,
        byte_count=len(raw),
        sha256=sha256(raw),
    )
    validate_reply(
        reply,
        raw,
        engine=first["engine_id"],
        rid=rid,
        generation=generation,
        digest=digest,
        start=start,
        end=end,
        hidden_size=hidden_size,
        features=features,
    )
    return reply, raw
