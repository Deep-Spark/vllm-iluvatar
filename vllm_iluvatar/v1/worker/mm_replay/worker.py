"""Normal-stream P encoding and bounded stable-address D embedding injection."""

from __future__ import annotations

import json

import torch
from vllm.distributed import get_pp_group, get_tp_group

from vllm_iluvatar.v1.worker.prefill_state.configuration import prefix_transfer_config

from .archive import SourceArchive, atomic_json, storage_key
from .configuration import ReplayOptions, paths
from .identity import (
    feature_descriptors,
    model_identity,
    overlapping,
    request_identity,
    token_digest,
)
from .receiver import read_window
from .tensors import unpack_tensors


class ProducerEmbeddings:
    """Complete CPU archives on the normal first-stage worker, including hits.

    At most one extra image is encoded at once. Scheduled P encoder-cache
    entries reuse archived outputs; unscheduled images remain on the host.
    """

    consumer = False

    def __init__(self, state):
        self.state = state
        config = state.vllm_config
        transfer = prefix_transfer_config(config)
        self.root, self.session = paths(config)
        self.pending = set()
        self.options = ReplayOptions.from_config(config)
        self.archive = SourceArchive(
            self.root / "source",
            transfer.engine_id,
            self.session,
            budget_bytes=self.options.host_cache_bytes,
            hidden_size=state.inputs_embeds_size,
            lease_seconds=float(
                transfer.get_from_extra_config("kv_lease_duration", 480)
            ),
            max_transfer_rows=state.max_num_tokens,
        )
        self.counts = dict(encoded=0, reused=0, archived_bytes=0)
        self.last_report = dict(self.counts)

    def add_request(self, request):
        if any(f.modality == "image" for f in request.mm_features):
            self.pending.add(request.req_id)

    def remove_request(self, rid):
        self.pending.discard(rid)

    @torch.inference_mode()
    def prepare(self, scheduled):
        for rid in list(self.pending):
            path = self.archive.request_path(rid)
            if not path.exists():
                # A direct P request has no remote decoder lease.
                self.pending.discard(rid)
                continue
            header = json.loads(path.read_text())
            request = self.archive.request(
                rid, header["generation"], header["input_digest"]
            )
            if request is None:
                raise RuntimeError("Producer image lease disappeared before encoding")
            for feature in request["features"]:
                key = feature["key"]
                existing = self.archive.image_output(key)
                # Every TP rank must make the same decision before a sharded
                # vision encoder can enter collectives. TP1 does no collective.
                need_encode = existing is None
                tp = get_tp_group()
                if tp.world_size > 1:
                    need_encode = tp.broadcast_object(
                        need_encode if tp.rank_in_group == 0 else None, src=0
                    )
                if need_encode:
                    output = self.state.encoder_cache.encoder_outputs.get(key)
                    if output is None:
                        meta, raw = self.archive.image_input(key)
                        inputs = unpack_tensors(meta["fields"], raw)
                        inputs["patches"] = inputs["patches"].to(self.state.device)
                        inputs["vit_grid"] = inputs["vit_grid"].reshape(-1, 2)
                        inputs["llm_grid"] = inputs["llm_grid"].reshape(-1, 2)
                        output = self.state.model.embed_multimodal(**inputs)[0]
                        self.counts["encoded"] += 1
                    host = (
                        output.detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
                    self.archive.publish(rid, request["generation"], key, raw)
                    self.counts["archived_bytes"] += len(raw)
                    del host, output
                else:
                    self.counts["reused"] += 1
            self.pending.discard(rid)
        # Keep ordinary encoder cache accounting and lifetime. Only entries
        # actually admitted by the P scheduler are materialized on the GPU.
        for rid, indices in scheduled.items():
            for index in indices:
                feature = self.state.encoder_cache.mm_features[rid][index]
                if feature.identifier in self.state.encoder_cache.encoder_outputs:
                    continue
                try:
                    raw = self.archive.image_output(feature.identifier)
                except FileNotFoundError:
                    continue  # local/profile-only P input
                if raw is None:
                    raise RuntimeError("Scheduled producer image was not encoded")
                value = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
                    -1, self.state.inputs_embeds_size
                )
                self.state.encoder_cache.encoder_outputs[feature.identifier] = value.to(
                    self.state.device
                )
        if self.counts != self.last_report:
            atomic_json(self.root / "worker_encode.json", self.counts)
            self.last_report = dict(self.counts)


class DecoderEmbeddings:
    """One device receive window and two pinned host windows per DP worker.

    Buffers are allocated before memory profiling. They scale with the real
    token budget, not model length or number/size of prompt images. The normal
    runner already owns the final inputs_embeds window.
    """

    consumer = True

    def __init__(self, state):
        self.state = state
        self.root, self.session = paths(state.vllm_config)
        self.directory = self.root / "decode"
        self.active = {}
        self.cached = {}
        self.sequence = 0
        shape = (state.max_num_tokens, state.inputs_embeds_size)
        self.receive = torch.empty(shape, dtype=state.dtype, device=state.device)
        self.host = [
            torch.empty(shape, dtype=state.dtype, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self.events = [None, None]
        self.model_digest = model_identity(state.model_config)
        self.counts = dict(
            vision_calls=0,
            replay_batches=0,
            replay_rows=0,
            device_receive_bytes=self.receive.numel() * self.receive.element_size(),
        )
        atomic_json(self.root / "worker_decode.json", self.counts)

    def add_request(self, request):
        if not any(f.modality == "image" for f in request.mm_features):
            return
        path = self.directory / storage_key(request.req_id) / "request.json"
        record = json.loads(path.read_text())
        features = feature_descriptors(request.mm_features)
        identity = request_identity(
            self.model_digest,
            token_digest(request.prompt_token_ids),
            request.prompt_len,
            features,
        )
        if identity != record["input_digest"] or features != record["features"]:
            raise ValueError(
                "Decoder worker multimodal request identity differs from scheduler"
            )
        self.active[request.req_id] = record
        self.cached.pop(request.req_id, None)

    def remove_request(self, rid):
        self.active.pop(rid, None)
        self.cached.pop(rid, None)
        directory = self.directory / storage_key(rid)
        if (directory / "finished").exists():
            (directory / "worker_done").touch()

    def window(self, rid, start, end):
        record = self.active[rid]
        for meta, raw in self.cached.get(rid, ()):
            if meta["start"] <= start and meta["end"] >= end:
                return meta, raw
        meta, raw = read_window(
            self.directory,
            rid,
            record["generation"],
            record["input_digest"],
            start,
            end,
            self.state.inputs_embeds_size,
            record["features"],
        )
        values = self.cached.setdefault(rid, [])
        values.append((meta, raw))
        del values[:-2]
        return meta, raw

    @torch.inference_mode()
    def prepare(self, batch):
        if not self.active:
            return None
        ranges = []
        for i, rid in enumerate(batch.req_ids):
            if rid not in self.active:
                continue
            start = int(batch.num_computed_tokens_np[i])
            end = min(
                int(batch.prefill_len_np[i]), start + int(batch.num_scheduled_tokens[i])
            )
            if not overlapping(self.active[rid]["features"], start, end):
                continue
            meta, raw = self.window(rid, start, end)
            for item in meta["records"]:
                a, b = max(start, item["start"]), min(end, item["end"])
                if a >= b:
                    continue
                byte0 = (
                    item["offset_bytes"]
                    + (a - item["start"]) * self.state.inputs_embeds_size * 2
                )
                rows = b - a
                value = torch.frombuffer(
                    bytearray(
                        raw[byte0 : byte0 + rows * self.state.inputs_embeds_size * 2]
                    ),
                    dtype=torch.bfloat16,
                ).reshape(rows, -1)
                q = int(batch.query_start_loc_np[i]) + a - start
                ranges.append((q, value))
        if not ranges:
            return None
        slot = self.sequence % 2
        self.sequence += 1
        event = self.events[slot]
        if event is not None and not event.query():
            event.synchronize()
        host = self.host[slot]
        offset = 0
        copies = []
        for q, value in ranges:
            count = value.shape[0]
            if offset + count > self.receive.shape[0]:
                raise RuntimeError("Image replay exceeds the configured device window")
            host[offset : offset + count].copy_(value)
            copies.append((q, offset, count))
            offset += count
        self.receive[:offset].copy_(host[:offset], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.events[slot] = event
        ids = batch.input_ids[: batch.num_tokens]
        output = self.state.encoder_runner.inputs_embeds
        output[: batch.num_tokens].copy_(self.state.model.embed_input_ids(ids))
        for q, source, count in copies:
            output[q : q + count].copy_(self.receive[source : source + count])
        self.counts["replay_batches"] += 1
        self.counts["replay_rows"] += offset
        atomic_json(self.root / "worker_decode.json", self.counts)
        return output[: batch.num_tokens_after_padding]


def configure(state):
    options = ReplayOptions.from_config(state.vllm_config)
    if not options.enabled or not get_pp_group().is_first_rank:
        return None
    transfer = prefix_transfer_config(state.vllm_config)
    return (
        DecoderEmbeddings(state)
        if transfer.kv_role == "kv_consumer"
        else ProducerEmbeddings(state)
    )
