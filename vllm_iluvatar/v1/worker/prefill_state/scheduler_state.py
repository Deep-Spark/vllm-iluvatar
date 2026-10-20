"""Prefix-state publication and replay receipts owned by the native connector.

Receipts use successful connector completions and the scheduler's public block
snapshot. They neither replace scheduler methods nor change KV publication.
"""

from __future__ import annotations

import hashlib
import threading
import time

import numpy as np

from .configuration import (
    PrefixGeometry, ProducerTopology, enabled, prefix_transfer_config, session_paths,
)
from .source_state import StatePublisher, replay_layout, save


def selected_groups(cache_config, geometry):
    sources = {*geometry.ring_layers, geometry.prefix_depth - 1}
    return [
        g
        for g, group in enumerate(cache_config.kv_cache_groups)
        if any(
            ".layers." in name
            and int(name.split(".layers.")[1].split(".")[0]) in sources
            and (name.endswith(".attn") or name.endswith(".attn.indexer.k_cache"))
            for name in group.layer_names
        )
    ]


class PrefixSchedulerState:
    def __init__(self, config, cache_config):
        self.wire, self.nonce = session_paths(config)
        self.engine_id = prefix_transfer_config(config).engine_id
        self.consumer = prefix_transfer_config(config).kv_role == "kv_consumer"
        self.cache_config = cache_config
        self.geometry = PrefixGeometry.from_config(config)
        self.selected = selected_groups(cache_config, self.geometry)
        self.requests = {}
        self.serial = 0
        self.stop = threading.Event()
        self.thread = None
        if not self.consumer:
            self.publisher = StatePublisher(
                self.wire,
                self.engine_id,
                self.nonce,
                topology=ProducerTopology.from_config(config),
            )
            self.thread = threading.Thread(
                target=self._publish, daemon=True, name="producer-prefix-publisher"
            )
            self.thread.start()

    @classmethod
    def create(cls, config, cache_config):
        return cls(config, cache_config) if enabled(config) else None

    def _publish(self):
        while not self.stop.is_set():
            try:
                self.publisher.poll()
            except Exception as error:
                # Optional state failure must not terminate the KV connector.
                save(self.wire / "publisher_error.json", dict(error=repr(error)))
            self.stop.wait(0.002)

    def allocated(self, request, blocks, external):
        if not self.consumer or external <= 0:
            return
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill") or not params.get("remote_engine_id"):
            return
        self.serial += 1
        self.requests[request.request_id] = dict(
            request=request,
            serial=self.serial,
            external=external,
            local=request.num_computed_tokens,
            remote_engine_id=params["remote_engine_id"],
            received=False,
            failed=False,
            received_positions={
                g: [
                    dict(logical=j, block_id=b.block_id)
                    for j, b in enumerate(blocks.blocks[g])
                    if b.block_hash is None and not b.is_null
                ]
                for g in self.selected
            },
        )

    def completed(self, output):
        for rid in output.failed_recving or ():
            if rid in self.requests:
                self.requests[rid]["failed"] = True
        # A group-level failure without a request id cannot certify any replay.
        if output.invalid_block_ids:
            for record in self.requests.values():
                record["failed"] = True
        for rid in output.finished_recving or ():
            if rid in self.requests:
                self.requests[rid]["received"] = True

    def add_metadata(self, metadata, output, peers):
        receipts, endpoints = {}, {}
        snapshot = output.kv_connector_block_state
        for rid, record in list(self.requests.items()):
            if rid in output.finished_req_ids or rid in (
                output.preempted_req_ids or ()
            ):
                self.requests.pop(rid)
                continue
            peer = peers.get(record["remote_engine_id"])
            if rid in metadata.reqs_to_recv and peer is not None:
                endpoints[rid] = dict(
                    engine_id=record["remote_engine_id"], addr=peer.addr
                )
            request = record["request"]
            n = request.num_prompt_tokens
            if (
                not record["received"]
                or record["failed"]
                or n < self.geometry.window
                or rid not in output.num_scheduled_tokens
                or snapshot is None
            ):
                continue
            hit, start, rows = replay_layout(n)
            if request.num_computed_tokens != start or request.replay_start != start:
                continue
            current = snapshot.get_block_ids(rid)
            if current is None:
                continue
            receipts[rid] = dict(
                request_id=rid,
                run_uuid=self.nonce,
                engine_id=self.engine_id,
                N=n,
                H=hit,
                S=start,
                query_rows=rows,
                fullkv_valid_before_replay=hit,
                original_receive_success=True,
                alloc_serial=record["serial"],
                token_sha256=hashlib.sha256(
                    np.asarray(request.prompt_token_ids, dtype=np.int32).tobytes()
                ).hexdigest(),
                mark_mono_ns=time.monotonic_ns(),
                external=record["external"],
                local=record["local"],
                groups={g: list(current[g]) for g in self.selected},
                received_positions=record["received_positions"],
                remote_engine_id=record["remote_engine_id"],
                group_specs={
                    g: dict(
                        block_size=self.cache_config.kv_cache_groups[
                            g
                        ].kv_cache_spec.block_size,
                        layer_names=list(
                            self.cache_config.kv_cache_groups[g].layer_names
                        ),
                        dtype=str(
                            self.cache_config.kv_cache_groups[g].kv_cache_spec.dtype
                        ),
                    )
                    for g in self.selected
                },
            )
        metadata.source_full_ready_receipts = receipts
        metadata.source_state_endpoints = endpoints

    def shutdown(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
