"""Capture complete current-request prefix states on the producer stream."""

import hashlib
import json
import threading
import time
from collections import deque

import numpy as np
import torch

from .cache_rows import copy_indexer_row_to_host, main_boundary_sources
from .configuration import (
    PrefixGeometry,
    ProducerTopology,
    prefix_transfer_config,
    session_paths,
)
from .runtime import PrefixRuntime
from .source_state import replay_layout, request_dir


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value) + "\n")
    tmp.replace(path)


class ProducerPrefixState(PrefixRuntime):
    def __init__(self, runner, model):
        super().__init__(runner, model)
        from vllm.distributed import get_pp_group, get_tp_group

        self._src_pp = get_pp_group().rank_in_group
        self._src_tp = get_tp_group().rank_in_group
        self.topology = ProducerTopology.from_config(self.vllm_config)
        self.geometry = PrefixGeometry.from_config(self.vllm_config)
        self._src_rank = self.topology.rank(self._src_pp, self._src_tp)
        self._src_engine = prefix_transfer_config(self.vllm_config).engine_id
        self._src_wire, self._src_nonce = session_paths(self.vllm_config)
        self._src_result = self._src_wire / "source"
        self._src_cap = self.vllm_config.scheduler_config.max_num_seqs
        self._src_stop = threading.Event()
        self._src_thread = None
        self._src_model = model
        self._admission = 0
        self._src_ready = False
        self._src_lock = threading.Lock()
        self._src_active = {}
        self._src_export = deque()
        self._src_free = deque()
        self._src_generations = {}
        self._src_batch = None
        self._src_counts = dict(admitted=0, full=0, partial=0, capacity=0, aborted=0)

    @property
    def states(self):
        return self._src_active

    @property
    def lock(self):
        return self._src_lock

    def add_request(self, idx, request):
        if not self._src_ready:
            return
        req_id, prompt_len = request.req_id, request.prompt_len
        tokens = request.prefill_token_ids
        self._admission += 1
        gen = self._src_generations.get(idx, 0) + 1
        self._src_generations[idx] = gen
        key = hashlib.sha256(
            np.asarray(tokens[:prompt_len], dtype=np.int32).tobytes()
        ).hexdigest()
        eligible = self.request_eligible(request, allow_mm=self._mm_prefix_reuse)
        hit, start, rows = replay_layout(prompt_len) if eligible else (prompt_len, 0, 0)
        with self._src_lock:
            pool = self._src_free.popleft() if eligible and self._src_free else None
            state = dict(
                id=req_id,
                idx=idx,
                generation=gen,
                admission=self._admission,
                N=prompt_len,
                H=hit,
                S=start,
                token_sha256=key,
                mm_input_digest=self.mm_input_identity(request, key)
                if eligible
                else None,
                initial_computed=int(request.num_computed_tokens),
                run_uuid=self._src_nonce,
                engine_id=self._src_engine,
                blocks=self.initial_blocks(request),
                pool=pool,
                coverage=[False] * rows,
                spans=[],
                events=[],
                records=[],
                phase="ACTIVE",
                abort=False,
                fresh=False,
                add_mono_ns=time.monotonic_ns(),
            )
            self._src_active[req_id] = state
            self._src_counts["admitted"] += 1
            if eligible and pool is None:
                self._src_counts["capacity"] += 1
        directory = request_dir(self._src_result, req_id)
        directory.mkdir(exist_ok=True)
        save(directory / f"observed_{self._src_rank}.json", self._scalar(state))

    def remove_request(self, req_id):
        with self._src_lock:
            state = self._src_active.pop(req_id, None)
            if state and state["phase"] == "ACTIVE":
                state["abort"] = True
                state["phase"] = "EXPORT_PENDING"
                self._src_export.append(state)

    def prepare_inputs(self, batch):
        if self._src_ready:
            q = batch.query_start_loc_np[: batch.num_reqs + 1]
            self._src_batch = [
                dict(
                    id=rid,
                    idx=int(batch.idx_mapping_np[j]),
                    q0=int(q[j]),
                    q=int(q[j + 1] - q[j]),
                    c=int(batch.num_computed_tokens_np[j]),
                    N=int(batch.prefill_len_np[j]),
                    prefill=bool(batch.is_prefilling_np[j]),
                )
                for j, rid in enumerate(batch.req_ids)
            ]

    def end_step(self):
        self._src_batch = None

    def _scalar(self, state):
        result = {
            k: state[k]
            for k in (
                "id",
                "idx",
                "generation",
                "admission",
                "N",
                "H",
                "S",
                "token_sha256",
                "initial_computed",
                "run_uuid",
                "engine_id",
                "add_mono_ns",
            )
        }
        if state.get("mm_input_digest") is not None:
            result["mm_input_digest"] = state["mm_input_digest"]
        return result

    def initialize_cache(self):
        runner = self.model_runner
        static = self.vllm_config.compilation_config.static_forward_context
        self._src_specs = []
        for g, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if ".layers." not in name:
                    continue
                layer_id = int(name.split(".layers.")[1].split(".")[0])
                if (
                    not self._src_model.start_layer
                    <= layer_id
                    < self._src_model.end_layer
                ):
                    continue
                cache = getattr(static.get(name), "kv_cache", None)
                if not isinstance(cache, torch.Tensor) or not cache.numel():
                    continue
                kind = (
                    "SWA"
                    if name.endswith(".attn.swa_cache")
                    else "ring"
                    if name.endswith(".attn.compressor.state_cache")
                    and layer_id in self.geometry.ring_layers
                    else None
                )
                if kind:
                    self._src_specs.append(
                        dict(
                            name=name,
                            kind=kind,
                            layer=layer_id,
                            group=g,
                            cache=cache,
                            R=int(cache.shape[1]),
                            ratio=1
                            if layer_id == self.geometry.prefix_depth - 1
                            else 2,
                        )
                    )
                elif layer_id == self.geometry.prefix_depth - 1 and (
                    name.endswith(".attn") or name.endswith(".attn.indexer.k_cache")
                ):
                    if name.endswith(".attn"):
                        self._src_specs.extend(
                            main_boundary_sources(static[name], name, g, self.geometry)
                        )
                        continue
                    kind = "indexer_tail"
                    self._src_specs.append(
                        dict(
                            name=name,
                            kind=kind,
                            layer=layer_id,
                            group=g,
                            cache=cache,
                            R=int(cache.shape[1]),
                            ratio=1,
                            scales=getattr(static[name], "kv_cache_scales", None),
                        )
                    )
        assert (
            sum(s["kind"] == "SWA" for s in self._src_specs)
            == self._src_model.end_layer - self._src_model.start_layer
        )
        assert sum(s["kind"] == "ring" for s in self._src_specs) == sum(
            self.topology.owner(i) == self._src_pp for i in self.geometry.ring_layers
        )
        before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        host_bytes = 0
        for _ in range(self._src_cap):
            pool = {}
            for s in self._src_specs:
                if s["kind"] == "SWA":
                    shape, dtype = (129, self.geometry.head_dim), torch.bfloat16
                elif s["kind"] == "ring":
                    shape, dtype = tuple(s["cache"].shape[1:]), torch.float32
                elif s["kind"] in ("main_tail", "main_tail_scale"):
                    shape, dtype = (1, s["cache"].shape[-1]), s["cache"].dtype
                else:
                    assert s["kind"] == "indexer_tail"
                    width = self.geometry.indexer_width
                    shape, dtype = (width,), torch.uint8
                pool[s["name"]] = torch.empty(shape, dtype=dtype, pin_memory=True)
            if self._src_pp == self.topology.boundary_pp:
                pool.update(
                    HC=torch.empty(
                        (129, self.geometry.hc_mult, self.geometry.hidden_size),
                        dtype=torch.bfloat16,
                        pin_memory=True,
                    ),
                    pre_mix=torch.empty(
                        (129, self.geometry.hc_mult),
                        dtype=torch.float32,
                        pin_memory=True,
                    ),
                    topk20=torch.empty(
                        (129, self.geometry.topk), dtype=torch.int32, pin_memory=True
                    ),
                    candidates20=torch.empty(
                        (129, self.geometry.candidates),
                        dtype=torch.int32,
                        pin_memory=True,
                    ),
                )
            host_bytes += sum(t.numel() * t.element_size() for t in pool.values())
            self._src_free.append(pool)
        assert before == (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        ), "source GPU owner changed"
        save(
            self._src_result / f"installed_{self._src_rank}.json",
            dict(
                rank=self._src_rank,
                pp=self._src_pp,
                tp=self._src_tp,
                active_cap=self._src_cap,
                pinned_host_bytes=host_bytes,
                new_GPU_owner_bytes=0,
                max_model_len=self.vllm_config.model_config.max_model_len,
                specs=[
                    {k: v for k, v in s.items() if k not in ("cache", "scales")}
                    for s in self._src_specs
                ],
            ),
        )

    def start(self):
        self._src_ready = True
        self._src_thread = threading.Thread(
            target=self._export_loop, daemon=True, name="P21-host-export"
        )
        self._src_thread.start()

    def capture(self, model, hc, pre_mix, positions):
        if not self._src_ready or self._src_batch is None:
            return
        for row in self._src_batch:
            if not row["prefill"]:
                continue
            with self._src_lock:
                state = self._src_active.get(row["id"])
            if state is None or state["phase"] != "ACTIVE":
                continue
            assert (
                state["idx"] == row["idx"]
                and state["generation"] == self._src_generations[row["idx"]]
            )
            assert state["N"] == row["N"]
            pool = state["pool"]
            a, b = max(row["c"], state["S"]), min(row["c"] + row["q"], state["N"])
            if (
                pool is not None
                and not state["abort"]
                and a < b
                and self._src_pp == self.topology.boundary_pp
            ):
                src = slice(row["q0"] + a - row["c"], row["q0"] + b - row["c"])
                dst = slice(a - state["S"], b - state["S"])
                assert not any(state["coverage"][dst]), "duplicate HC span"
                for name, source in (
                    ("HC", hc),
                    ("pre_mix", pre_mix),
                    ("topk20", model.topk_indices_buffer),
                    ("candidates20", model.candidate_idxs_buffer),
                ):
                    pool[name][dst].copy_(source[src], non_blocking=True)
                state["coverage"][dst] = [True] * (b - a)
            state["spans"].append(dict(computed=row["c"], query=row["q"]))
            event = torch.cuda.Event()
            event.record()
            state["events"].append(event)
            if row["c"] + row["q"] == state["N"]:
                state["fresh"] = (
                    pool is not None
                    and not state["abort"]
                    and min(span["computed"] for span in state["spans"]) <= state["S"]
                    and (
                        self._src_pp != self.topology.boundary_pp
                        or all(state["coverage"])
                    )
                )
                if state["fresh"]:
                    state["records"] = self._copy_caches(state)
                    event = torch.cuda.Event()
                    event.record()
                    state["events"].append(event)
                state["phase"] = "EXPORT_PENDING"
                with self._src_lock:
                    self._src_export.append(state)

    def _copy_caches(self, state):
        records = []
        for spec in self._src_specs:
            cache = spec["cache"]
            host = state["pool"][spec["name"]]
            blocks = state["blocks"][spec["group"]]
            segments = []
            if spec["kind"] in ("main_tail", "main_tail_scale", "indexer_tail"):
                if state["N"] == state["H"]:
                    continue
                start, stop = state["H"], state["N"]
                logical, row = divmod(start, spec["R"])
                assert (
                    logical < len(blocks) and 0 < int(blocks[logical]) < cache.shape[0]
                )
                block = int(blocks[logical])
                if spec["kind"] in ("main_tail", "main_tail_scale"):
                    host[0].copy_(cache[block, row], non_blocking=True)
                else:
                    copy_indexer_row_to_host(
                        host, cache, block, row, spec.get("scales")
                    )
                segments.append(
                    dict(
                        logical_rows=[start, stop],
                        physical_block=block,
                        physical_rows=[row, row + 1],
                    )
                )
            elif spec["kind"] == "ring":
                assert len(blocks) == 1 and 0 < int(blocks[0]) < cache.shape[0]
                host.copy_(cache[int(blocks[0])], non_blocking=True)
                start, stop = state["N"] - spec["R"], state["N"]
            else:
                start, stop = state["S"], state["N"]
                p = start
                while p < stop:
                    logical, lo = divmod(p, spec["R"])
                    count = min(stop - p, spec["R"] - lo)
                    assert (
                        logical < len(blocks)
                        and 0 < int(blocks[logical]) < cache.shape[0]
                    )
                    block = int(blocks[logical])
                    off = p - start
                    host[off : off + count].copy_(
                        cache[block, lo : lo + count], non_blocking=True
                    )
                    segments.append(
                        dict(
                            logical_rows=[p, p + count],
                            physical_block=block,
                            physical_rows=[lo, lo + count],
                        )
                    )
                    p += count
            records.append(
                dict(
                    name=spec["name"],
                    kind=spec["kind"],
                    layer=spec["layer"],
                    group=spec["group"],
                    dtype=str(host.dtype),
                    shape=(
                        [state["N"] - state["S"], self.geometry.head_dim]
                        if spec["kind"] == "SWA"
                        else list(host.shape)
                    ),
                    absolute_interval=[start, stop],
                    segments=segments,
                )
            )
        return records

    def _export_loop(self):
        torch.cuda.set_device(self.device)
        report_time = 0
        while not self._src_stop.is_set():
            with self._src_lock:
                state = self._src_export[0] if self._src_export else None
            if state is not None and all(event.query() for event in state["events"]):
                try:
                    meta = self._scalar(state)
                    meta.update(
                        rank=self._src_rank,
                        pp=self._src_pp,
                        tp=self._src_tp,
                        topology=self.topology.to_dict(),
                        geometry=self.geometry.to_dict(),
                        fresh=state["fresh"],
                        spans=state["spans"],
                        abort=state["abort"],
                        export_mono_ns=time.monotonic_ns(),
                        records=[],
                    )
                    directory = request_dir(self._src_result, state["id"])
                    directory.mkdir(exist_ok=True)
                    if state["fresh"]:
                        # CPU-only serialization after all writer-stream D2H events.
                        pieces = []
                        offset = 0
                        records = list(state["records"])
                        if self._src_pp == self.topology.boundary_pp:
                            records += [
                                dict(
                                    name=k,
                                    kind="boundary21",
                                    dtype=str(state["pool"][k].dtype),
                                    shape=[
                                        state["N"] - state["S"],
                                        *state["pool"][k].shape[1:],
                                    ],
                                )
                                for k in ("HC", "pre_mix", "topk20", "candidates20")
                            ]
                        for record in records:
                            host = state["pool"][record["name"]]
                            if tuple(host.shape) != tuple(record["shape"]):
                                host = host[: record["shape"][0]]
                            raw = host.contiguous().view(torch.uint8).numpy().tobytes()
                            if host.dtype == torch.bfloat16:
                                # CPU bit-level finite check avoids a large
                                # FP32 conversion/CPU-thread-pool in exporters.
                                words = np.frombuffer(raw, dtype=np.uint16)
                                assert bool(np.all((words & 0x7F80) != 0x7F80))
                            elif host.dtype == torch.float32:
                                assert bool(
                                    np.isfinite(
                                        np.frombuffer(raw, dtype=np.float32)
                                    ).all()
                                )
                            meta["records"].append(
                                {
                                    **record,
                                    "offset_bytes": offset,
                                    "byte_count": len(raw),
                                }
                            )
                            pieces.append(raw)
                            offset += len(raw)
                        raw = b"".join(pieces)
                        (directory / f"rank{self._src_rank}.bin").write_bytes(raw)
                        meta.update(
                            byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest()
                        )
                        self._src_counts["full"] += 1
                    elif state["abort"]:
                        self._src_counts["aborted"] += 1
                    else:
                        self._src_counts["partial"] += 1
                    save(directory / f"auto_{self._src_rank}.json", meta)
                except Exception as error:
                    save(
                        self._src_result / f"error_{self._src_rank}.json",
                        dict(error=repr(error), request_id=state["id"]),
                    )
                finally:
                    with self._src_lock:
                        assert self._src_export.popleft() is state
                        state["phase"] = "EXPORTED"
                        if state["pool"] is not None:
                            self._src_free.append(state.pop("pool"))
                        state["events"] = []
                        state["blocks"] = []
            if time.monotonic() - report_time >= 1:
                save(
                    self._src_result / f"counters_{self._src_rank}.json",
                    self._src_counts,
                )
                report_time = time.monotonic()
            self._src_stop.wait(0.002)

    def shutdown(self):
        self._src_ready = False
        if hasattr(self, "_src_stop"):
            self._src_stop.set()
        if getattr(self, "_src_thread", None) is not None:
            self._src_thread.join()
