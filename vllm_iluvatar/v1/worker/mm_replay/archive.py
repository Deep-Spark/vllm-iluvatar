"""Bounded producer-private image archives and versioned range transport.

The network endpoint reads host bytes only. GPU encoding and publication belong
on the normal prefill worker. No producer filename is part of the wire format.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

MM_OP = b"pd_mm_replay_v1"
VERSION = 1
CAPABILITY_REQUEST = "__ilu_mm_replay_capabilities_v1__"
CAPABILITY_DIGEST = hashlib.sha256(b"pd-mm-replay-capabilities/v1").hexdigest()


def capability_request(engine_id):
    """Use a legacy-safe E25 query; old producers return PENDING/UNSUPPORTED."""
    from vllm_iluvatar.v1.worker.prefill_state.source_state import GET_STATE

    return (GET_STATE, engine_id, CAPABILITY_REQUEST, 0, 128, CAPABILITY_DIGEST)


def is_capability_request(message):
    return (
        isinstance(message, (tuple, list))
        and len(message) == 6
        and tuple(message) == capability_request(message[1])
    )


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def storage_key(value: str) -> str:
    return sha256(value.encode())


def exact_int(value, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


class SourceArchive:
    """A process-safe host cache with request leases and capacity backpressure.

    Input blobs and the maximum encoded output are reserved together at
    admission. Eviction can remove unleased images only. A matching terminal
    acknowledgement frees a lease; it never frees another request generation.
    """

    def __init__(
        self,
        root: Path,
        engine_id: str,
        session: str,
        *,
        budget_bytes: int,
        hidden_size: int,
        lease_seconds: float = 480,
        max_transfer_rows: int = 4096,
    ):
        if not exact_int(budget_bytes, 1, 1 << 50) or not exact_int(
            hidden_size, 1, 1 << 20
        ):
            raise ValueError("Invalid archive dimensions")
        if lease_seconds <= 0 or not exact_int(max_transfer_rows, 1, 1 << 20):
            raise ValueError("Invalid archive lease/transfer limit")
        self.root = Path(root)
        self.engine_id = engine_id
        self.session = session
        self.budget_bytes = budget_bytes
        self.hidden_size = hidden_size
        self.lease_seconds = float(lease_seconds)
        self.max_transfer_rows = max_transfer_rows
        for name in ("images", "requests", "cancelled"):
            (self.root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path = self.root / "archive.lock"

    @contextmanager
    def locked(self):
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def request_path(self, rid: str) -> Path:
        return self.root / "requests" / (storage_key(rid) + ".json")

    def cancelled_path(self, rid, generation):
        return self.root / "cancelled" / (storage_key(rid + ":" + generation) + ".json")

    def _load_request(self, rid, generation, digest=None):
        try:
            request = json.loads(self.request_path(rid).read_text())
        except FileNotFoundError:
            return None
        if request["engine_id"] != self.engine_id or request["session"] != self.session:
            raise ValueError("Source session mismatch")
        if request["request_id"] != rid or request["generation"] != generation:
            raise ValueError("Source generation mismatch")
        if digest is not None and request["input_digest"] != digest:
            raise ValueError("Source input identity mismatch")
        return request

    def _gc(self, now):
        live = set()
        for path in (self.root / "requests").glob("*.json"):
            request = json.loads(path.read_text())
            if request["expires"] <= now:
                path.unlink()
                continue
            live.update(f["key"] for f in request["features"])
        for path in (self.root / "cancelled").glob("*.json"):
            if json.loads(path.read_text())["expires"] <= now:
                path.unlink()
        images = []
        for directory in (self.root / "images").iterdir():
            meta_path = directory / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                images.append((directory, meta))
        return live, images

    def admit(
        self,
        rid: str,
        generation: str,
        digest: str,
        prompt_len: int,
        features: list[dict],
    ) -> bool:
        """Bind canonical image descriptors and CPU inputs, or apply backpressure.

        Each feature supplies key/start/rows and a typed ``input`` binary blob
        plus ``fields`` descriptors. Duplicate images share one content record.
        """
        if (
            not exact_int(prompt_len, 1, 1 << 31)
            or not rid
            or not generation
            or len(digest) != 64
        ):
            raise ValueError("Invalid source request")
        keys = {}
        ranges = []
        for f in features:
            if not isinstance(f["key"], str) or not exact_int(f["rows"], 1, prompt_len):
                raise ValueError("Invalid image descriptor")
            if not exact_int(f["start"], 0, prompt_len - f["rows"]):
                raise ValueError("Invalid image range")
            if ranges and f["start"] < ranges[-1][1]:
                raise ValueError("Overlapping image spans")
            ranges.append((f["start"], f["start"] + f["rows"]))
            if f["key"] in keys and (
                keys[f["key"]]["rows"] != f["rows"]
                or keys[f["key"]]["input"] != f["input"]
            ):
                raise ValueError("Conflicting content identity")
            keys[f["key"]] = f
        with self.locked():
            now = time.time()
            live, images = self._gc(now)
            cancelled = self.cancelled_path(rid, generation)
            if (
                cancelled.exists()
                and json.loads(cancelled.read_text())["generation"] == generation
            ):
                raise ValueError("Source request was cancelled")
            old = self._load_request(rid, generation, digest)
            if old is not None:
                old["expires"] = now + self.lease_seconds
                atomic_json(self.request_path(rid), old)
                return True
            known = {m["key"]: (p, m) for p, m in images}
            needed = sum(
                len(f["input"]) + f["rows"] * self.hidden_size * 2
                for k, f in keys.items()
                if k not in known
            )
            used = sum(m["reserved_bytes"] for _, m in images)
            if needed > self.budget_bytes:
                raise ValueError("One request exceeds mm_host_cache_bytes")
            evictable = sorted(
                (
                    (p, m)
                    for p, m in images
                    if m["key"] not in live and m["key"] not in keys
                ),
                key=lambda pair: pair[1]["touched"],
            )
            while used + needed > self.budget_bytes and evictable:
                path, meta = evictable.pop(0)
                used -= meta["reserved_bytes"]
                shutil.rmtree(path)
            if used + needed > self.budget_bytes:
                return False
            for key, f in keys.items():
                directory = self.root / "images" / storage_key(key)
                if key not in known:
                    directory.mkdir(exist_ok=True)
                    atomic_bytes(directory / "input.bin", f["input"])
                    meta = dict(
                        key=key,
                        rows=f["rows"],
                        hidden_size=self.hidden_size,
                        fields=f["fields"],
                        input_bytes=len(f["input"]),
                        input_sha256=sha256(f["input"]),
                        output_bytes=f["rows"] * self.hidden_size * 2,
                        reserved_bytes=len(f["input"])
                        + f["rows"] * self.hidden_size * 2,
                        touched=now,
                        output_sha256=None,
                    )
                else:
                    meta = known[key][1]
                    if (
                        meta["rows"] != f["rows"]
                        or meta["hidden_size"] != self.hidden_size
                        or meta["input_sha256"] != sha256(f["input"])
                    ):
                        raise ValueError("Canonical image input mismatch")
                    meta["touched"] = now
                atomic_json(directory / "meta.json", meta)
            request = dict(
                engine_id=self.engine_id,
                session=self.session,
                request_id=rid,
                generation=generation,
                input_digest=digest,
                prompt_len=prompt_len,
                features=[
                    {k: f[k] for k in ("key", "start", "rows")} for f in features
                ],
                expires=now + self.lease_seconds,
            )
            atomic_json(self.request_path(rid), request)
            return True

    def request(self, rid, generation, digest=None):
        with self.locked():
            return self._load_request(rid, generation, digest)

    def image_input(self, key):
        with self.locked():
            directory = self.root / "images" / storage_key(key)
            meta = json.loads((directory / "meta.json").read_text())
            raw = (directory / "input.bin").read_bytes()
            if len(raw) != meta["input_bytes"] or sha256(raw) != meta["input_sha256"]:
                raise ValueError("Corrupt archived encoder inputs")
            return meta, raw

    def image_output(self, key):
        with self.locked():
            directory = self.root / "images" / storage_key(key)
            meta = json.loads((directory / "meta.json").read_text())
            if meta["output_sha256"] is None:
                return None
            raw = (directory / "output.bin").read_bytes()
            if len(raw) != meta["output_bytes"] or sha256(raw) != meta["output_sha256"]:
                raise ValueError("Corrupt archived encoder output")
            return raw

    def publish(self, rid, generation, key, raw):
        with self.locked():
            request = self._load_request(rid, generation)
            if request is None or not any(f["key"] == key for f in request["features"]):
                return False
            directory = self.root / "images" / storage_key(key)
            meta = json.loads((directory / "meta.json").read_text())
            if len(raw) != meta["output_bytes"]:
                raise ValueError("Wrong encoded output shape/dtype")
            checksum = sha256(raw)
            if meta["output_sha256"] is not None:
                # Preserve the first valid encoding when floating reduction
                # order changes. Existing leases must keep immutable bytes.
                return True
            atomic_bytes(directory / "output.bin", raw)
            meta["output_sha256"] = checksum
            meta["touched"] = time.time()
            atomic_json(directory / "meta.json", meta)
            return True

    def renew(self, rid, generation, digest):
        with self.locked():
            self._gc(time.time())
            request = self._load_request(rid, generation, digest)
            if request is None:
                return False
            request["expires"] = time.time() + self.lease_seconds
            atomic_json(self.request_path(rid), request)
            return True

    def release(self, rid, generation, digest):
        with self.locked():
            request = self._load_request(rid, generation, digest)
            if request is not None:
                self.request_path(rid).unlink()
            atomic_json(
                self.cancelled_path(rid, generation),
                dict(generation=generation, expires=time.time() + self.lease_seconds),
            )

    def stats(self):
        with self.locked():
            live, images = self._gc(time.time())
            return dict(
                reserved_bytes=sum(m["reserved_bytes"] for _, m in images),
                budget_bytes=self.budget_bytes,
                images=len(images),
                leased_images=len(live),
                requests=len(list((self.root / "requests").glob("*.json"))),
            )

    def handle(self, message) -> tuple[dict, bytes]:
        reply = dict(
            version=VERSION,
            engine_id=self.engine_id,
            session=self.session,
            status="BAD_REQUEST",
        )
        if (
            not isinstance(message, (list, tuple))
            or len(message) != 8
            or message[:2] not in ((MM_OP, VERSION), [MM_OP, VERSION])
        ):
            return reply, b""
        _, _, action, engine, rid, generation, digest, params = message
        if engine != self.engine_id:
            return {**reply, "status": "WRONG_ENGINE"}, b""
        if (
            not isinstance(rid, str)
            or not 0 < len(rid) <= 512
            or not isinstance(generation, str)
            or not 0 < len(generation) <= 128
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(params, dict)
        ):
            return reply, b""
        reply.update(request_id=rid, generation=generation, input_digest=digest)
        try:
            if action != "RELEASE" and self.cancelled_path(rid, generation).exists():
                return {**reply, "status": "CANCELLED"}, b""
            if action == "RELEASE":
                self.release(rid, generation, digest)
                return {**reply, "status": "RELEASED"}, b""
            if action == "RENEW":
                ready = self.renew(rid, generation, digest)
                return {**reply, "status": "LIVE" if ready else "PENDING"}, b""
            if action != "GET":
                return reply, b""
            start, end = params.get("start"), params.get("end")
            if not exact_int(start, 0, 1 << 31) or not exact_int(
                end, start + 1, start + self.max_transfer_rows
            ):
                return reply, b""
            with self.locked():
                now = time.time()
                self._gc(now)
                request = self._load_request(rid, generation, digest)
                if request is None:
                    return {**reply, "status": "PENDING"}, b""
                if end > request["prompt_len"]:
                    return reply, b""
                request["expires"] = now + self.lease_seconds
                atomic_json(self.request_path(rid), request)
                records = []
                parts = []
                offset = 0
                for f in request["features"]:
                    a, b = max(start, f["start"]), min(end, f["start"] + f["rows"])
                    if a >= b:
                        continue
                    directory = self.root / "images" / storage_key(f["key"])
                    meta = json.loads((directory / "meta.json").read_text())
                    if meta["output_sha256"] is None:
                        return {**reply, "status": "PENDING"}, b""
                    row_bytes = self.hidden_size * 2
                    encoded = (directory / "output.bin").read_bytes()
                    if (
                        len(encoded) != meta["output_bytes"]
                        or sha256(encoded) != meta["output_sha256"]
                    ):
                        raise ValueError("Corrupt producer image storage")
                    part = encoded[
                        (a - f["start"]) * row_bytes : (b - f["start"]) * row_bytes
                    ]
                    if len(part) != (b - a) * row_bytes:
                        raise ValueError("Truncated encoded image")
                    records.append(
                        dict(
                            key=f["key"],
                            start=a,
                            end=b,
                            shape=[b - a, self.hidden_size],
                            dtype="BF16",
                            offset_bytes=offset,
                            byte_count=len(part),
                        )
                    )
                    parts.append(part)
                    offset += len(part)
                raw = b"".join(parts)
                return {
                    **reply,
                    "status": "READY",
                    "start": start,
                    "end": end,
                    "records": records,
                    "byte_count": len(raw),
                    "sha256": sha256(raw),
                }, raw
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            return {**reply, "status": "IDENTITY_OR_PAYLOAD_ERROR"}, b""


def validate_reply(
    reply,
    raw,
    *,
    engine,
    rid,
    generation,
    digest,
    start,
    end,
    hidden_size,
    features,
    session=None,
):
    """Validate complete absolute-range coverage before publishing D readiness."""
    if (
        not isinstance(reply, dict)
        or reply.get("version") != VERSION
        or reply.get("engine_id") != engine
    ):
        raise ValueError("Multimodal endpoint/version mismatch")
    if (
        reply.get("request_id") != rid
        or reply.get("generation") != generation
        or reply.get("input_digest") != digest
    ):
        raise ValueError("Multimodal request identity mismatch")
    if (
        not isinstance(reply.get("session"), str)
        or not reply["session"]
        or session is not None
        and reply["session"] != session
    ):
        raise ValueError("Multimodal producer session changed")
    if reply.get("status") != "READY":
        if raw:
            raise ValueError("Unexpected non-ready embedding payload")
        return False
    wanted = [
        (f["key"], max(start, f["start"]), min(end, f["start"] + f["rows"]))
        for f in features
        if max(start, f["start"]) < min(end, f["start"] + f["rows"])
    ]
    records = reply.get("records")
    if not isinstance(records, list) or len(records) != len(wanted):
        raise ValueError("Incomplete image coverage")
    offset = 0
    for record, (key, a, b) in zip(records, wanted):
        size = (b - a) * hidden_size * 2
        if (
            record.get("key") != key
            or record.get("start") != a
            or record.get("end") != b
            or record.get("shape") != [b - a, hidden_size]
            or record.get("dtype") != "BF16"
            or type(record.get("offset_bytes")) is not int
            or record["offset_bytes"] != offset
            or type(record.get("byte_count")) is not int
            or record["byte_count"] != size
        ):
            raise ValueError("Invalid multimodal range layout")
        offset += size
    if (
        reply.get("start") != start
        or reply.get("end") != end
        or reply.get("byte_count") != offset
        or len(raw) != offset
        or reply.get("sha256") != sha256(raw)
    ):
        raise ValueError("Multimodal payload checksum/size mismatch")
    return True
