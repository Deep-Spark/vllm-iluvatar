"""Producer-local prefix states and CPU transport over the NIXL side channel.

Only the producer reads its shard directory. Consumers fetch immutable packets
from the existing producer scheduler socket; no shared P/D filesystem or extra
service is required. This module never allocates device memory.
"""

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from .configuration import PrefixGeometry, ProducerTopology

PROTOCOL_VERSION = 3
GET_STATE = b"p21_state_v1"
DTYPES = {
    "torch.bfloat16": ("BF16", 2),
    "torch.float32": ("FP32", 4),
    "torch.int32": ("I32", 4),
    "BF16": ("BF16", 2),
    "FP32": ("FP32", 4),
    "I32": ("I32", 4),
    "torch.uint8": ("U8", 1),
    "U8": ("U8", 1),
}


def replay_layout(n: int) -> tuple[int, int, int]:
    """Even received prefix H, replay start S, and complete query128/129."""
    if type(n) is not int or n < 128:
        raise ValueError("Prefix-state prompt length must be an integer >=128")
    hit = n - n % 2
    return hit, hit - 128, 128 + n % 2


def packet_bytes(
    n: int, indexer_bytes: int = 132, *, geometry=None, ring_rows=None
) -> int:
    replay_layout(n)
    geometry = geometry or PrefixGeometry(indexer_width=indexer_bytes)
    return sum(
        math.prod(shape) * DTYPES[dtype][1]
        for shape, dtype in geometry.fields(n, ring_rows).values()
    )


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def request_dir(root: Path, request_id: str) -> Path:
    """Keep arbitrary router request IDs out of filesystem path components."""
    return root / hashlib.sha256(request_id.encode()).hexdigest()


def validate_fields(
    fields: list[dict],
    size: int,
    n: int = 128,
    *,
    geometry: PrefixGeometry | None = None,
    ring_rows=None,
) -> None:
    """Reject overlapping, incomplete or incorrectly typed destination records."""
    replay_layout(n)
    if not isinstance(fields, list):
        raise ValueError("Expected a complete producer-prefix packet")
    if geometry is None:
        widths = [
            f.get("shape", [0])[0] for f in fields if f.get("kind") == "indexer_tail"
        ]
        geometry = PrefixGeometry(indexer_width=widths[0] if len(widths) == 1 else 132)
    wanted = geometry.fields(n, ring_rows)
    if size != packet_bytes(n, geometry=geometry, ring_rows=ring_rows):
        raise ValueError("Prefix-state packet byte count mismatch")
    seen, names, offset = set(), set(), 0
    for field in sorted(fields, key=lambda f: f["offset_bytes"]):
        if (
            not isinstance(field["name"], str)
            or not field["name"]
            or type(field["offset_bytes"]) is not int
            or type(field["byte_count"]) is not int
            or not isinstance(field["shape"], (tuple, list))
            or any(type(dim) is not int or dim <= 0 for dim in field["shape"])
        ):
            raise ValueError("Prefix-state field types must be exact integers")
        kind = field["kind"]
        if kind != "boundary21" and type(field["layer"]) is not int:
            raise ValueError("Prefix-state layer must be an integer")
        key = (kind, field["name"] if kind == "boundary21" else field["layer"])
        dtype, element_size = DTYPES[field["dtype"]]
        shape = tuple(field["shape"])
        if (
            key in seen
            or field["name"] in names
            or key not in wanted
            or wanted[key] != (shape, dtype)
            or field["offset_bytes"] != offset
            or offset % element_size
            or field["byte_count"] != math.prod(shape) * element_size
        ):
            raise ValueError("Invalid prefix-state field layout")
        seen.add(key)
        names.add(field["name"])
        offset += field["byte_count"]
    if seen != wanted.keys() or offset != size:
        raise ValueError("Incomplete prefix-state fields")


def join_packets(
    wire: Path,
    source: Path,
    engine_id: str,
    nonce: str,
    request_id: str,
    epoch: int,
    rows: list[dict],
    topology: ProducerTopology | None = None,
    shard_directory: Path | None = None,
) -> None:
    if not rows:
        raise ValueError("Missing producer shards")
    topology = topology or ProducerTopology.from_dict(rows[0]["topology"])
    if len(rows) != topology.world_size or {r["rank"] for r in rows} != set(
        range(topology.world_size)
    ):
        raise ValueError("Expected all producer ranks")
    geometry = PrefixGeometry.from_dict(rows[0]["geometry"])
    if geometry.prefix_depth != topology.prefix_depth:
        raise ValueError("Producer topology and geometry disagree")
    n, digest = rows[0]["N"], rows[0]["token_sha256"]
    hit, start, _ = replay_layout(n)
    if not all(
        r["engine_id"] == engine_id
        and r["run_uuid"] == nonce
        and r["id"] == request_id
        and r["N"] == n
        and r["H"] == hit
        and r["S"] == start
        and r["token_sha256"] == digest
        and r["rank"] == r["pp"] * topology.tp_size + r["tp"]
        and 0 <= r["pp"] < topology.pp_size
        and 0 <= r["tp"] < topology.tp_size
        and r["topology"] == topology.to_dict()
        and r["geometry"] == geometry.to_dict()
        and r["admission"] == rows[0]["admission"]
        and r["admission"] > 0
        and r["generation"] > 0
        and r["fresh"] is True
        and r["abort"] is False
        for r in rows
    ):
        raise ValueError("Producer identity or capture mismatch")
    shard_directory = shard_directory or request_dir(source, request_id)
    for row in rows:
        observed = json.loads(
            (shard_directory / f"observed_{row['rank']}.json").read_text()
        )
        if observed["admission"] != row["admission"]:
            raise ValueError("Producer admission changed during publication")
    target = request_dir(wire, request_id)
    target.mkdir(exist_ok=True)
    for tp in range(topology.tp_size):
        stages = sorted((r for r in rows if r["tp"] == tp), key=lambda r: r["pp"])
        fields, parts, offset = [], [], 0
        for row in stages:
            raw = (shard_directory / f"rank{row['rank']}.bin").read_bytes()
            if (
                len(raw) != row["byte_count"]
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise ValueError("Invalid producer shard payload")
            for field in row["records"]:
                owner = (
                    topology.boundary_pp
                    if field["kind"] == "boundary21"
                    else topology.owner(field["layer"])
                )
                if row["pp"] != owner:
                    raise ValueError("Producer field belongs to another pipeline stage")
                if (
                    field["offset_bytes"] < 0
                    or field["byte_count"] < 0
                    or field["offset_bytes"] + field["byte_count"] > len(raw)
                ):
                    raise ValueError("Producer field extends beyond its shard")
                fields.append({**field, "offset_bytes": offset + field["offset_bytes"]})
            parts.append(raw)
            offset += len(raw)
        raw = b"".join(parts)
        ring_rows = {f["layer"]: f["shape"][0] for f in fields if f["kind"] == "ring"}
        validate_fields(fields, len(raw), n, geometry=geometry, ring_rows=ring_rows)
        temporary = target / f"packet_tp{tp}.bin.tmp"
        temporary.write_bytes(raw)
        temporary.replace(target / f"packet_tp{tp}.bin")
        save(
            target / f"packet_tp{tp}.json",
            dict(
                version=PROTOCOL_VERSION,
                topology=topology.to_dict(),
                geometry=geometry.to_dict(),
                ring_rows=ring_rows,
                admission=rows[0]["admission"],
                engine_id=engine_id,
                request_id=request_id,
                run_uuid=nonce,
                epoch=epoch,
                tp=tp,
                N=n,
                H=hit,
                S=start,
                token_sha256=digest,
                byte_count=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                fields=fields,
                source_complete=True,
                source_generation=stages[-1]["generation"],
                published_mono_ns=time.monotonic_ns(),
            ),
        )
    # The marker becomes visible only after every producer TP packet is complete.
    save(
        target / "current_source.json",
        dict(
            engine_id=engine_id,
            request_id=request_id,
            run_uuid=nonce,
            epoch=epoch,
            topology=topology.to_dict(),
            admission=rows[0]["admission"],
            N=n,
            token_sha256=digest,
            fresh=True,
            ready_mono_ns=time.monotonic_ns(),
        ),
    )


class StatePublisher:
    """Called by the producer scheduler's host export loop, never a separate process."""

    def __init__(
        self,
        wire: Path,
        engine_id: str,
        nonce: str,
        shard_timeout_s: float = 3600,
        packet_ttl_s: float = 60,
        *,
        topology: ProducerTopology,
    ):
        if not all(math.isfinite(t) and t > 0 for t in (shard_timeout_s, packet_ttl_s)):
            raise ValueError("State timeouts must be positive and finite")
        self.topology = topology
        self.wire, self.source = wire, wire / "source"
        self.engine_id, self.nonce = engine_id, nonce
        self.shard_timeout_s, self.packet_ttl_s = shard_timeout_s, packet_ttl_s
        self.pending, self.retired = {}, {}
        self.epoch = 0
        self.counts = dict(requests=0, fresh=0, unavailable=0, expired=0, errors=0)
        self.last_report = float("-inf")

    def poll(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        present = {
            p.name
            for p in self.source.iterdir()
            if not p.name.startswith(".") and p.is_dir() and not p.is_symlink()
        }
        # A completed shard directory was removed after publication. Its
        # reappearance is a new admission of the same request ID, not a lease
        # extension for the old packet.
        for name in present & self.retired.keys():
            if self.retired[name]["published"]:
                self.retired.pop(name)
                shutil.rmtree(self.wire / name, ignore_errors=True)
        for name in sorted(present - self.pending.keys() - self.retired.keys()):
            shutil.rmtree(self.wire / name, ignore_errors=True)
            self.epoch += 1
            self.pending[name] = dict(epoch=self.epoch, first_seen=now)
            self.counts["requests"] += 1
        for name, entry in list(self.pending.items()):
            directory = self.source / name
            rows = []
            for rank in range(self.topology.world_size):
                try:
                    rows.append(
                        json.loads((directory / f"auto_{rank}.json").read_text())
                    )
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
            reason = None
            if len(rows) == self.topology.world_size:
                # Rename the completed admission before publishing. A new
                # admission of the same id creates a new directory and is never
                # removed by cleanup of the old one. Readers see either source
                # or the hidden publication directory until commit is complete.
                try:
                    directory.rename(self.source / f".{name}")
                except FileNotFoundError:
                    continue
                directory = self.source / f".{name}"
                try:
                    rid = rows[0]["id"]
                    if request_dir(self.source, rid).name != name:
                        raise ValueError("Request directory identity mismatch")
                    if all(
                        r.get("fresh") is True and r.get("abort") is False for r in rows
                    ):
                        join_packets(
                            self.wire,
                            self.source,
                            self.engine_id,
                            self.nonce,
                            rid,
                            entry["epoch"],
                            rows,
                            self.topology,
                            directory,
                        )
                        self.counts["fresh"] += 1
                    else:
                        reason = "producer_capture_incomplete"
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    OSError,
                ) as error:
                    self.counts["errors"] += 1
                    reason = f"invalid_producer_shards: {error}"
                # Every auto_N.json is terminal; no writer still owns these files.
                shutil.rmtree(directory, ignore_errors=True)
            elif now - entry["first_seen"] > self.shard_timeout_s:
                reason = "producer_shard_timeout"
                # Late writers may still finish: retain a tombstone until TTL.
            else:
                continue
            if reason is not None:
                self.counts["unavailable"] += 1
                target = self.wire / name
                target.mkdir(exist_ok=True)
                save(
                    target / "unavailable.json",
                    dict(reason=reason, ready_ranks=len(rows)),
                )
            self.pending.pop(name)
            self.retired[name] = dict(at=now, published=reason is None)
        for name, entry in list(self.retired.items()):
            if now - entry["at"] > self.packet_ttl_s:
                self.retired.pop(name)
                self.counts["expired"] += 1
                shutil.rmtree(self.wire / name, ignore_errors=True)
                shutil.rmtree(self.source / name, ignore_errors=True)
        if now - self.last_report >= 1:
            save(
                self.wire / "source_counters.json",
                {
                    **self.counts,
                    "pending": len(self.pending),
                    "published_live": len(self.retired),
                    "run_uuid": self.nonce,
                    "engine_id": self.engine_id,
                },
            )
            self.last_report = now


def serve_state(
    wire: Path | None, engine_id: str, message: tuple
) -> tuple[dict, bytes]:
    """Serve a versioned pull on the already registered P side-channel socket."""
    reply = dict(version=PROTOCOL_VERSION, status="BAD_REQUEST", engine_id=engine_id)
    if not isinstance(message, (tuple, list)) or len(message) != 6:
        return reply, b""
    _, remote_id, rid, tp, n, digest = message
    if remote_id != engine_id:
        return {**reply, "status": "WRONG_ENGINE"}, b""
    if wire is None:
        return {**reply, "status": "UNSUPPORTED"}, b""
    if (
        not isinstance(rid, str)
        or not 0 < len(rid) <= 512
        or type(tp) is not int
        or not 0 <= tp < 256
        or type(n) is not int
        or n < 128
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        return reply, b""
    directory = request_dir(wire, rid)
    if (
        request_dir(wire / "source", rid).exists()
        or (wire / "source" / f".{directory.name}").exists()
    ):
        return {**reply, "status": "PENDING"}, b""
    try:
        source = json.loads((directory / "current_source.json").read_text())
        topology = ProducerTopology.from_dict(source["topology"])
        source_tp = tp % topology.tp_size
        meta = json.loads((directory / f"packet_tp{source_tp}.json").read_text())
        if not (
            meta["engine_id"] == source["engine_id"] == engine_id
            and meta["request_id"] == source["request_id"] == rid
            and meta["N"] == source["N"] == n
            and meta["tp"] == source_tp
            and meta["run_uuid"] == source["run_uuid"]
            and meta["epoch"] == source["epoch"]
            and meta["admission"] == source["admission"]
            and meta["topology"] == source["topology"]
            and meta["token_sha256"] == source["token_sha256"] == digest
            and source["fresh"]
            and meta["source_complete"]
        ):
            return {**reply, "status": "IDENTITY_MISMATCH"}, b""
        raw = (directory / f"packet_tp{source_tp}.bin").read_bytes()
        if len(raw) != meta["byte_count"]:
            return {**reply, "status": "INVALID_PAYLOAD"}, b""
        validate_fields(
            meta["fields"],
            len(raw),
            n,
            geometry=PrefixGeometry.from_dict(meta["geometry"]),
            ring_rows={int(k): v for k, v in meta["ring_rows"].items()},
        )
        return {**reply, "status": "READY", "metadata": meta, "source": source}, raw
    except FileNotFoundError:
        status = (
            "UNAVAILABLE" if (directory / "unavailable.json").exists() else "PENDING"
        )
        return {**reply, "status": status}, b""
    except (ValueError, KeyError, TypeError, AttributeError):
        return {**reply, "status": "INVALID_PAYLOAD"}, b""


def fetch_state(
    socket,
    engine_id: str,
    request_id: str,
    tp: int,
    n: int,
    digest: str,
    *,
    geometry=None,
    ring_rows=None,
) -> tuple[dict, bytes]:
    """The D receiver owns this socket; inference/Graph code never uses it."""
    import msgspec

    socket.send(
        msgspec.msgpack.encode((GET_STATE, engine_id, request_id, tp, n, digest))
    )
    frames = socket.recv_multipart()
    if len(frames) != 2:
        raise ValueError("Expected prefix-state metadata and CPU payload frames")
    reply = msgspec.msgpack.decode(frames[0])
    raw = frames[1]
    if reply.get("version") != PROTOCOL_VERSION or reply.get("engine_id") != engine_id:
        raise ValueError("Prefix-state endpoint identity mismatch")
    if reply.get("status") != "READY":
        if raw:
            raise ValueError("Unexpected payload for unavailable state")
        return reply, raw
    meta, source = reply["metadata"], reply["source"]
    topology = ProducerTopology.from_dict(meta["topology"])
    hit, start, _ = replay_layout(n)
    if not (
        meta["request_id"] == source["request_id"] == request_id
        and meta["engine_id"] == source["engine_id"] == engine_id
        and meta["run_uuid"] == source["run_uuid"]
        and meta["run_uuid"]
        and meta["version"] == PROTOCOL_VERSION
        and meta["N"] == source["N"] == n
        and meta["H"] == hit
        and meta["S"] == start
        and meta["tp"] == tp % topology.tp_size
        and meta["token_sha256"] == source["token_sha256"] == digest
        and meta["epoch"] == source["epoch"]
        and meta["epoch"] > 0
        and meta["admission"] == source["admission"]
        and meta["admission"] > 0
        and meta["topology"] == source["topology"]
        and meta["source_generation"] > 0
        and meta["source_complete"]
        and source["fresh"]
        and len(raw) == meta["byte_count"]
        and hashlib.sha256(raw).hexdigest() == meta["sha256"]
    ):
        raise ValueError("Prefix-state payload identity or checksum mismatch")
    packet_geometry = PrefixGeometry.from_dict(meta["geometry"])
    packet_rings = {int(k): v for k, v in meta["ring_rows"].items()}
    if (
        geometry is not None
        and packet_geometry != geometry
        or ring_rows is not None
        and packet_rings != ring_rows
    ):
        raise ValueError("Producer and consumer cache geometry differ")
    validate_fields(
        meta["fields"], len(raw), n, geometry=packet_geometry, ring_rows=packet_rings
    )
    return reply, raw
