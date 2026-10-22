"""Native connector-owned source leases and nonblocking MM scheduling."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading

from vllm.logger import init_logger

from vllm_iluvatar.v1.worker.prefill_state.configuration import prefix_transfer_config

from .archive import VERSION, SourceArchive, atomic_json, capability_request
from .configuration import ReplayOptions, paths
from .identity import (
    certified_image_id,
    feature_descriptors,
    model_identity,
    request_identity,
    token_digest,
)
from .receiver import RangeReceiver
from .tensors import pack_tensors

logger = init_logger("vllm." + __name__)


def generation(request):
    params = request.kv_transfer_params or {}
    if params.get("_opt_nixl_generation"):
        return str(params["_opt_nixl_generation"])
    value = str(
        params.get("transfer_id")
        or params.get("remote_request_id")
        or request.request_id
    )
    result = hashlib.sha256(value.encode()).hexdigest()
    params["_opt_nixl_generation"] = result
    return result


def descriptor(config, request):
    features = feature_descriptors(request.mm_features)
    digest = request_identity(
        model_identity(config.model_config),
        token_digest(request.prompt_token_ids),
        request.num_prompt_tokens,
        features,
    )
    return dict(
        request_id=request.request_id,
        generation=generation(request),
        input_digest=digest,
        prompt_len=request.num_prompt_tokens,
        features=features,
    )


class ReplayScheduler:
    """A scheduler component attached to the existing native NIXL endpoint."""

    def __init__(self, config, engine_id):
        self.config = config
        self.options = ReplayOptions.from_config(config)
        self.transfer = prefix_transfer_config(config)
        self.consumer = self.transfer.kv_role == "kv_consumer"
        self.root, self.session = paths(config)
        self.engine_id = engine_id
        self.hidden_size = config.model_config.get_inputs_embeds_size()
        self.model_digest = model_identity(config.model_config)
        self.pending = {}
        self.admitted = set()
        self.errors = {}
        self.records = {}
        self.counts = dict(admitted=0, backpressure=0, completed=0, failed=0)
        lease = float(self.transfer.get_from_extra_config("kv_lease_duration", 480))
        from .lifecycle import claim_session

        claim_session(self.root)
        self._stop = threading.Event()
        self._maintenance_thread = None
        self.archive = None
        self.receiver = None
        if self.consumer:
            self.receiver = RangeReceiver(
                self.root / "decode",
                self.resolve,
                hidden_size=self.hidden_size,
                max_window_rows=config.scheduler_config.max_num_batched_tokens + 1,
                max_num_seqs=config.scheduler_config.max_num_seqs,
                lease_seconds=lease,
                host_budget_bytes=self.options.host_cache_bytes,
            )
        else:
            self.archive = SourceArchive(
                self.root / "source",
                engine_id,
                self.session,
                budget_bytes=self.options.host_cache_bytes,
                hidden_size=self.hidden_size,
                lease_seconds=lease,
                max_transfer_rows=config.scheduler_config.max_num_batched_tokens,
            )

        if self.archive is not None:
            self._maintenance_thread = threading.Thread(
                target=self._maintain, name="pd-image-source-leases", daemon=True
            )
            self._maintenance_thread.start()

    def _maintain(self):
        while not self._stop.wait(1):
            # Expire leases even while no requests arrive. Cached image data is
            # intentionally retained within the fixed host LRU budget.
            atomic_json(self.root / "source_cache.json", self.archive.stats())

    @classmethod
    def create(cls, config, engine_id):
        return (
            cls(config, engine_id)
            if ReplayOptions.from_config(config).enabled
            else None
        )

    def capabilities(self):
        return dict(
            version=VERSION,
            hidden_size=self.hidden_size,
            model_digest=self.model_digest,
            max_transfer_rows=self.config.scheduler_config.max_num_batched_tokens,
        )

    def resolve(self, engine_id):
        """The receiver thread alone resolves producer capabilities via SDK registry."""
        import msgspec
        import zmq

        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
            GET_NODE_META_MSG,
        )

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        try:
            socket.connect(os.environ["VLLM_ILU_NIXL_PROXY_ADDR"])
            from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.utils import (
                zmq_comm2proxy,
            )

            # The SDK router intentionally returns a fixed metadata subset.
            # Negotiate on P's existing endpoint using an opcode old P accepts.
            # No router change, shared filesystem, or speculative new opcode.
            encoded = zmq_comm2proxy(
                socket,
                msgspec.msgpack.encode((GET_NODE_META_MSG, engine_id)),
                need_recv=True,
                logger=logger,
            )
            peer = msgspec.msgpack.decode(encoded)
            if not isinstance(peer, dict):
                raise ValueError("Producer image metadata unavailable")
            with context.socket(zmq.REQ) as capability_socket:
                capability_socket.setsockopt(zmq.LINGER, 0)
                capability_socket.setsockopt(zmq.SNDTIMEO, 1000)
                capability_socket.setsockopt(zmq.RCVTIMEO, 1000)
                capability_socket.connect("tcp://" + peer["side_channel_addr"])
                capability_socket.send(
                    msgspec.msgpack.encode(capability_request(engine_id))
                )
                frames = capability_socket.recv_multipart()
            if len(frames) != 2 or frames[1]:
                raise ValueError("Invalid producer image capability frames")
            capability = msgspec.msgpack.decode(frames[0])
            if (
                capability.get("engine_id") != engine_id
                or capability.get("status") != "MM_CAPABILITIES"
            ):
                raise ValueError("Producer does not support remote image replay")
            peer["mm_replay"] = capability["mm_replay"]
            if peer.get("mm_replay", {}).get("model_digest") != self.model_digest:
                raise ValueError("Producer/decoder multimodal model identities differ")
            return peer
        finally:
            socket.close(0)
            context.term()

    def on_new_request(self, request):
        if not any(f.modality == "image" for f in request.mm_features):
            return
        params = request.kv_transfer_params or {}
        if not self.consumer and not params.get("do_remote_decode"):
            return
        try:
            record = descriptor(self.config, request)
            self.records[request.request_id] = record
            if self.consumer:
                engine = params.get("remote_engine_id")
                if not engine:
                    raise ValueError(
                        "P-only D image requests require a remote prefill engine"
                    )
                self.receiver.register(dict(record, engine_id=engine))
            else:
                self.pending[request.request_id] = request
        except Exception as error:
            self.errors[request.request_id] = repr(error)

    def can_schedule(self, request, start, count, lookahead=0):
        rid = request.request_id
        if rid in self.errors:
            return False
        if self.consumer:
            if rid in self.receiver.failures():
                return False
            return self.receiver.ensure(
                rid, start, min(request.num_prompt_tokens, start + count + lookahead)
            )
        if rid not in self.pending:
            return True
        try:
            record = self.records[rid]
            features = []
            for feature, meta in zip(
                request.mm_features, record["features"], strict=True
            ):
                if feature.data is None:
                    raise ValueError(
                        "Producer lost CPU image inputs before archive admission"
                    )
                fields, raw = pack_tensors(feature.data.get_data())
                if certified_image_id(self.model_digest, fields, raw) != meta["key"]:
                    raise ValueError(
                        "Processed image certification does not match producer inputs"
                    )
                features.append(dict(meta, input=raw, fields=fields))
            admitted = self.archive.admit(
                rid,
                record["generation"],
                record["input_digest"],
                record["prompt_len"],
                features,
            )
            if admitted:
                self.pending.pop(rid, None)
                self.admitted.add(rid)
                self.counts["admitted"] += 1
            else:
                self.counts["backpressure"] += 1
            self.report()
            return admitted
        except Exception as error:
            self.errors[rid] = repr(error)
            return False

    def failures(self):
        result = dict(self.errors)
        if self.receiver is not None:
            result.update(self.receiver.failures())
        return result

    def finished(self, request):
        rid = request.request_id
        if rid not in self.records:
            return
        if self.receiver is not None:
            self.receiver.finish(rid)
        else:
            from vllm.v1.request import RequestStatus

            if request.status not in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            ):
                record = self.records[rid]
                self.archive.release(rid, record["generation"], record["input_digest"])
                self.counts["failed"] += 1
        self.pending.pop(rid, None)
        self.admitted.discard(rid)
        self.records.pop(rid, None)
        self.errors.pop(rid, None)
        self.counts["completed"] += 1
        self.report()
        # A P completion is deliberately not a source release. D owns the
        # generation lease until its terminal acknowledgement or lease expiry.

    def report(self):
        value = dict(
            self.counts,
            engine_id=self.engine_id,
            session=self.session,
            requests=len(self.records),
        )
        if self.archive is not None:
            value["cache"] = self.archive.stats()
        atomic_json(self.root / "counters.json", value)

    def shutdown(self):
        self._stop.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=5)
        if self.receiver is not None:
            self.receiver.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)


def find_replay(connector):
    if connector is None:
        return None
    children = getattr(connector, "sub_connectors", None)
    if children is not None:
        matches = [r for child in children if (r := find_replay(child)) is not None]
        if len(matches) > 1:
            raise ValueError("Remote image replay requires one native endpoint")
        return matches[0] if matches else None
    scheduler = getattr(connector, "connector_scheduler", None)
    return getattr(scheduler, "mm_replay", None)
