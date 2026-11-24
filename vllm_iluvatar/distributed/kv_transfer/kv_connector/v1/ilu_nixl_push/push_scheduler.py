# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push scheduler with PP topology, Write-Once chunked-bulk, and hybrid Mamba split-cadence push."""

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_sliding_window,
)

from ..ilu_nixl_common.fp8_hetero.wire_layout import (
    HeteroDtypeConfig,
    _unwrap_push_connector,
    group_page_ratios,
    use_host_staging_wire,
)
from ..ilu_nixl_common.pool_directory import PoolDirectory
from .base_scheduler import IluNixlPushBaseConnectorScheduler
from .metadata import ChunkPushReqMeta
from .proxy_registration import ProxyRegistrationMixin

logger = init_logger(__name__)


def _encode_wire_page_ids(
    compute_block_ids: list[int],
    page_ratio: int,
    start_page: int,
    end_page: int,
) -> list[int]:
    """Encode ``(P block id, subpage)`` as a region-local descriptor page id."""
    return [
        compute_block_ids[target_page // page_ratio] * page_ratio
        + target_page % page_ratio
        for target_page in range(start_page, end_page)
    ]


def _retained_d_pages(num_tokens: int, sliding_window: int, block_size: int) -> int:
    """D manager remaining pages: skip the prefix whose tokens are all outside the window."""
    total = (num_tokens + block_size - 1) // block_size
    skipped = max(0, num_tokens - sliding_window + 1) // block_size
    return total - skipped


def _union_block_groups(
    previous: tuple[tuple[int, ...], ...] | None,
    incoming: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Per-group union, keeping ``previous`` order and dropping duplicates."""
    if not previous:
        return incoming
    merged: list[tuple[int, ...]] = []
    for idx in range(max(len(previous), len(incoming))):
        old = previous[idx] if idx < len(previous) else ()
        new = incoming[idx] if idx < len(incoming) else ()
        if not old:
            merged.append(new)
            continue
        seen = set(old)
        merged.append(old + tuple(b for b in new if b not in seen))
    return tuple(merged)


def _drop_blocks_in_fresh_range(
    requantize: tuple[tuple[int, ...], ...],
    quant_block_ids: list[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    """Per group: drop the re-quantizations the fresh range already rebuilds."""
    if not requantize:
        return ()
    out: list[tuple[int, ...]] = []
    for group_idx, stale in enumerate(requantize):
        if not stale or group_idx >= len(quant_block_ids):
            out.append(stale)
            continue
        fresh = frozenset(quant_block_ids[group_idx])
        if not fresh:
            out.append(stale)
            continue
        out.append(tuple(block for block in stale if block not in fresh))
    return tuple(out)


def _infer_pp_split(vllm_config) -> str:
    """Return comma-separated layer counts for every PP stage."""
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    model_config = vllm_config.model_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "num_hidden_layers"):
        num_hidden_layers = hf_text_config.num_hidden_layers
    else:
        num_hidden_layers = model_config.hf_config.num_hidden_layers

    parts: list[str] = []
    for pp_rank in range(pp_size):
        start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
        parts.append(str(end - start))
    return ",".join(parts)


def _group_contains_kpool_tail(spec) -> bool:
    if isinstance(spec, KpoolTailSpec):
        return True
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return any(
            isinstance(s, KpoolTailSpec) for s in spec.kv_cache_specs.values()
        )
    return False


class IluNixlPushConnectorScheduler(
    ProxyRegistrationMixin,
    NixlPushConnectorScheduler,
    IluNixlPushBaseConnectorScheduler,
):
    """Push scheduler with Write-Once chunked-bulk and hybrid Mamba split-cadence push."""

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._hetero_cfg = HeteroDtypeConfig.from_vllm_config(
            vllm_config, kv_cache_config
        )
        self._pp_split = envs.VLLM_PP_LAYER_PARTITION or _infer_pp_split(vllm_config)
        self._host_staging_producer = use_host_staging_wire(vllm_config)

        # P side: producer requests sent chunk-by-chunk, keyed by request id.
        self._chunk_candidates: dict[str, dict] = {}
        self._chunk_blocks: dict[str, list[list[int]]] = {}
        self._chunk_block_sizes = [
            int(group.kv_cache_spec.block_size)
            for group in kv_cache_config.kv_cache_groups
        ]
        self._transfer_group_ids: tuple[int, ...] = tuple(
            getattr(
                kv_cache_config,
                "transfer_group_ids",
                range(len(kv_cache_config.kv_cache_groups)),
            )
        )
        self._group_idx_to_xfer_idx: dict[int, int] = {
            gid: xfer_idx
            for xfer_idx, gid in enumerate(self._transfer_group_ids)
        }
        self._kpool_tail_xfer_groups: tuple[bool, ...] = tuple(
            _group_contains_kpool_tail(group.kv_cache_spec)
            for group in getattr(
                kv_cache_config,
                "transfer_groups",
                kv_cache_config.kv_cache_groups,
            )
        )
        self._mamba_xfer_groups: tuple[bool, ...] = tuple(
            isinstance(group.kv_cache_spec, MambaSpec)
            for group in getattr(
                kv_cache_config,
                "transfer_groups",
                kv_cache_config.kv_cache_groups,
            )
        )
        is_consumer = (
            getattr(
                _unwrap_push_connector(
                    getattr(vllm_config, "kv_transfer_config", None)
                ),
                "kv_role",
                None,
            )
            == "kv_consumer"
        )
        target_config = getattr(kv_cache_config, "wire_kv_cache_config", None)
        if self._hetero_cfg.enabled and not is_consumer and target_config is None:
            raise ValueError(
                "hetero_kv_dtype scheduler requires wire_kv_cache_config"
            )
        self._wire_page_ratios = (
            group_page_ratios(
                kv_cache_config.kv_cache_groups,
                target_config.kv_cache_groups,
            )
            if self._hetero_cfg.enabled and not is_consumer
            else (1,) * len(self._chunk_block_sizes)
        )
        self._wire_sliding_windows = tuple(
            int(sw)
            if (sw := get_kv_cache_spec_sliding_window(group.kv_cache_spec))
            else 0
            for group in kv_cache_config.kv_cache_groups
        )
        self._chunk_committed: set[str] = set()

        if self._hetero_cfg.enabled and not is_consumer and not self.use_host_buffer:
            raise ValueError(
                "hetero_kv_dtype requires kv_buffer_device=cpu: "
                "PD hardware heterogeneity must transfer through a host buffer"
            )
        self._pool_dir = PoolDirectory() if self._host_staging_producer else None
        self._requantize_pending: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._hit_prefix: dict[str, tuple[int, ...]] = {}
        self._init_proxy_registration()

    def on_new_request(self, request) -> None:
        # Fill the producer topology *before* the upstream hooks inspect the
        # request: the base scheduler keys heartbeat/lease tracking off
        # ``remote_request_id`` and the push scheduler indexes ``tp_size``.
        self.on_proxy_new_request(request)
        super().on_new_request(request)

    def set_xfer_handshake_metadata(self, metadata) -> None:
        # The keepalive publishes ``physical_block_size``, which only becomes
        # known once the workers have registered their KV caches, so start (or
        # restart) the registration here rather than in ``__init__``.
        super().set_xfer_handshake_metadata(metadata)
        self.start_proxy_registration()

    def shutdown(self) -> None:
        self._stop_proxy_registration()
        super().shutdown()

    def _stage_release_only_registration(
        self, request_id: str, params: dict | None
    ) -> None:
        """Stage a release-only PUSH_REG when D needs 0 external blocks (full hit or abort)."""
        if not params:
            return
        remote_engine_id = params.get("remote_engine_id")
        remote_host = params.get("remote_host")
        remote_port = params.get("remote_port")
        remote_tp_size = params.get("tp_size")
        if (
            remote_engine_id is None
            or remote_host is None
            or remote_port is None
            or remote_tp_size is None
        ):
            return
        decode_pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self._push_pending_registrations[request_id] = {
            "request_id": request_id,
            "decode_engine_id": self.engine_id,
            "decode_host": self.side_channel_host,
            "decode_port": self.side_channel_port,
            "decode_tp_size": (
                self.vllm_config.parallel_config.tensor_parallel_size
            ),
            "decode_dcp_size": (
                self.vllm_config.parallel_config.decode_context_parallel_size
            ),
            "decode_pp_size": params.get("decode_pp_size", decode_pp_size),
            "decode_pp_split": params.get("decode_pp_split") or self._pp_split,
            "local_block_ids": (),
            "notify_release_only": True,
            "remote_engine_id": remote_engine_id,
            "remote_host": remote_host,
            "remote_port": int(remote_port),
            "remote_tp_size": int(remote_tp_size),
            "remote_dcp_size": int(params.get("dcp_size", 1)),
            "remote_pp_size": int(params.get("pp_size", 1)),
            "remote_pp_split": params.get("pp_split", "") or "",
        }

    def update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ):
        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            prefix_state.allocated(request, blocks, num_external_tokens)
        params = request.kv_transfer_params
        is_decode_registration = params is not None and params.get("do_remote_prefill")
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if (
            self._host_staging_producer
            and params is not None
            and params.get("do_remote_decode")
        ):
            self._reqs_need_save.pop(request.request_id, None)

        self._maybe_note_chunk_candidate(request)
        self._maybe_note_stale_pool_blocks(request, blocks)
        if not is_decode_registration:
            return
        pending = self._push_pending_registrations.get(request.request_id)
        if pending is None:
            if num_external_tokens <= 0 and params is not None:
                self._stage_release_only_registration(
                    request.request_id, params
                )
                self._stop_heartbeat(request.request_id)
                params["do_remote_prefill"] = False
                logger.info(
                    "[push] D full prefix-cache hit for %s: staged release-only PUSH_REG",
                    request.request_id,
                )
                return
            if (
                prefix_state is not None
                and request.request_id not in self._reqs_need_recv
            ):
                prefix_state.requests.pop(request.request_id, None)
            logger.warning(
                "[push] D skip PUSH_REG %s: external_tokens=%d "
                "(stock did not stage registration)",
                request.request_id,
                num_external_tokens,
            )
            return
        block_starts = self.compute_remote_block_starts(blocks)
        self._remote_block_starts[request.request_id] = block_starts
        pending["remote_block_starts"] = block_starts
        logger.info(
            "[push] D staged PUSH_REG %s: external_tokens=%d "
            "local_block_groups=%s remote_block_starts=%s",
            request.request_id,
            num_external_tokens,
            [len(group) for group in pending.get("local_block_ids", ())],
            block_starts,
        )
        pending["remote_dcp_size"] = int(params.get("dcp_size", 1))
        pending["remote_pp_size"] = int(params.get("pp_size", 1))
        pending["remote_pp_split"] = params.get("pp_split", "") or ""
        decode_pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        pending["decode_dcp_size"] = (
            self.vllm_config.parallel_config.decode_context_parallel_size
        )
        pending["decode_pp_size"] = decode_pp_size
        pending["decode_pp_split"] = self._pp_split
        params_decode_pp_size = params.get("decode_pp_size")
        params_decode_pp_split = params.get("decode_pp_split")
        if params_decode_pp_size is not None:
            pending["decode_pp_size"] = params_decode_pp_size
        if params_decode_pp_split:
            pending["decode_pp_split"] = params_decode_pp_split

    # --- Chunked-bulk (P side, including Hybrid Split-Cadence Mamba) --- #

    def _maybe_note_chunk_candidate(self, request) -> None:
        """Record a P-side chunked-bulk candidate (both pure-Attention and Hybrid Mamba)."""
        if request.request_id in self._chunk_candidates:
            return
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_decode"):
            return
        if self.use_host_buffer and not self._host_staging_producer:
            return
        token_ids = request.prompt_token_ids or []
        if params.get("_p_side_truncated"):
            prefill_tokens = len(token_ids)
        else:
            prefill_tokens = self._get_remote_prefill_token_count(len(token_ids))
        num_compute_groups = len(self._chunk_block_sizes)
        self._chunk_candidates[request.request_id] = {
            "prefill_tokens": prefill_tokens,
            "remote_engine_id": params.get("remote_engine_id"),
            "next_chunk_index": 0,
            "emitted_tokens": 0,
            "emitted_target_blocks": [0] * num_compute_groups,
            "emitted_quant_blocks": [0] * num_compute_groups,
        }

    def _maybe_note_stale_pool_blocks(self, request, blocks) -> None:
        if self._pool_dir is None:
            return
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_decode"):
            return
        request_id = request.request_id
        self._hit_prefix.setdefault(request_id, self._pool_dir.hit_prefix(blocks))
        stale = self._pool_dir.stale_hits(blocks)
        if any(stale):
            self._requantize_pending[request_id] = _union_block_groups(
                self._requantize_pending.get(request_id), stale
            )
        self._pool_dir.record(blocks)

    def _accumulate_chunk_blocks(self, request_id: str, new_block_ids) -> None:
        if new_block_ids is None:
            return
        acc = self._chunk_blocks.get(request_id)
        if acc is None:
            acc = [list(grp) for grp in new_block_ids]
            self._chunk_blocks[request_id] = acc
            return
        for group_idx, group_new in enumerate(new_block_ids):
            if group_idx < len(acc):
                acc[group_idx].extend(group_new)

    def _is_tail_only_xfer_group(self, xfer_idx: int) -> bool:
        """Return True for running-state / circular scratch groups piggybacked on final chunk."""
        ssm_blocks = getattr(self, "_ssm_spec_blocks", ())
        if xfer_idx < len(ssm_blocks) and ssm_blocks[xfer_idx] is not None:
            return True
        mamba_groups = getattr(self, "_mamba_xfer_groups", ())
        if xfer_idx < len(mamba_groups) and mamba_groups[xfer_idx]:
            return True
        kpool_groups = getattr(self, "_kpool_tail_xfer_groups", ())
        if xfer_idx < len(kpool_groups) and kpool_groups[xfer_idx]:
            return True
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        if xfer_idx < len(ilu_clips) and ilu_clips[xfer_idx] == slice(0, 1):
            return True
        return False

    def _clip_tail_only_group_blocks(
        self, xfer_idx: int, group_blocks: list[int], page_ratio: int
    ) -> list[int]:
        """Clip running-state Mamba / KpoolTail / CircularBuffer blocks for final-chunk emission."""
        if not group_blocks:
            return []
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        if xfer_idx < len(ilu_clips) and ilu_clips[xfer_idx] == slice(0, 1):
            return _encode_wire_page_ids(group_blocks, page_ratio, 0, page_ratio)

        kpool_groups = getattr(self, "_kpool_tail_xfer_groups", ())
        blocks_per_sw = getattr(self, "blocks_per_sw", ())
        if xfer_idx < len(kpool_groups) and kpool_groups[xfer_idx]:
            n_sw = (
                blocks_per_sw[xfer_idx]
                if xfer_idx < len(blocks_per_sw) and blocks_per_sw[xfer_idx] > 0
                else 1
            )
            return list(group_blocks[-n_sw:])

        ssm_blocks = getattr(self, "_ssm_spec_blocks", ())
        clipped = list(group_blocks)
        if xfer_idx < len(ssm_blocks) and ssm_blocks[xfer_idx] is not None:
            n_spec_cfg = ssm_blocks[xfer_idx]
            if n_spec_cfg:
                n_spec = min(n_spec_cfg, len(clipped) - 1)
                if n_spec > 0:
                    clipped = clipped[:-n_spec]
            if not getattr(self, "_ssm_state_slots_are_positional", False):
                clipped = clipped[-1:]
            return clipped

        return list(group_blocks[-1:])

    def _xfer_group_allows_empty_chunk(
        self, xfer_idx: int, *, is_last_chunk: bool
    ) -> bool:
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        if xfer_idx < len(ilu_clips) and ilu_clips[xfer_idx] == slice(0, 0):
            return True
        if not is_last_chunk and self._is_tail_only_xfer_group(xfer_idx):
            return True
        if xfer_idx < len(self._transfer_group_ids):
            gid = self._transfer_group_ids[xfer_idx]
            if (
                gid < len(self._wire_sliding_windows)
                and self._wire_sliding_windows[gid] > 0
            ):
                return True
        # Under Write-Once sealed blocks, a group with larger block_size may
        # seal 0 blocks on an intermediate chunk while a smaller-block group seals >=1.
        if not is_last_chunk:
            return True
        return False

    def _maybe_emit_chunk(
        self,
        meta,
        request_id: str,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        new_block_ids,
    ) -> None:
        candidate = self._chunk_candidates.get(request_id)
        if candidate is None or num_scheduled_tokens <= 0:
            return

        chunk_end = min(
            num_computed_tokens + num_scheduled_tokens,
            candidate["prefill_tokens"],
        )
        chunk_start = candidate.get("emitted_tokens", 0)
        if chunk_end <= chunk_start:
            return
        is_last_chunk = chunk_end >= candidate["prefill_tokens"]

        # P0-3 Short-request single-step fast path: skip _chunk_blocks dict
        # allocation & list copying when the entire prompt finishes in one step.
        single_step_fast_path = (
            is_last_chunk
            and candidate.get("next_chunk_index", 0) == 0
            and request_id not in self._chunk_blocks
            and new_block_ids is not None
        )
        if single_step_fast_path:
            full_block_ids = [list(grp) for grp in new_block_ids]
        else:
            self._accumulate_chunk_blocks(request_id, new_block_ids)
            full_block_ids = self._chunk_blocks.get(request_id)
        if not full_block_ids:
            return

        block_sizes = getattr(self, "_chunk_block_sizes", None)
        if not block_sizes:
            block_sizes = [getattr(self, "block_size", 16)] * len(full_block_ids)
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        group_to_xfer = getattr(self, "_group_idx_to_xfer_idx", None)
        if group_to_xfer is None:
            group_to_xfer = {
                gid: idx for idx, gid in enumerate(self._transfer_group_ids)
            }

        prev_target_blocks: list[int] = candidate.get("emitted_target_blocks") or (
            [0] * len(full_block_ids)
        )
        prev_quant_blocks: list[int] = candidate.get("emitted_quant_blocks") or (
            [0] * len(full_block_ids)
        )
        if len(prev_target_blocks) < len(full_block_ids):
            prev_target_blocks.extend(
                [0] * (len(full_block_ids) - len(prev_target_blocks))
            )
        if len(prev_quant_blocks) < len(full_block_ids):
            prev_quant_blocks.extend(
                [0] * (len(full_block_ids) - len(prev_quant_blocks))
            )

        next_target_blocks = list(prev_target_blocks)
        next_quant_blocks = list(prev_quant_blocks)
        block_offsets: list[int] = []
        chunk_block_ids: list[list[int]] = []
        quant_block_ids: list[tuple[int, ...]] = []
        quant_first_pages: list[int] = []
        hit_run = self._hit_prefix.get(request_id, ())

        for group_idx, group_blocks in enumerate(full_block_ids):
            block_size = block_sizes[min(group_idx, len(block_sizes) - 1)]
            page_ratio = self._wire_page_ratios[
                min(group_idx, len(self._wire_page_ratios) - 1)
            ]
            target_block_size = block_size // page_ratio

            # P0-3 Write-Once: intermediate chunks only quantize/emit fully sealed blocks;
            # partial blocks wait until sealed by a subsequent chunk or is_last_chunk.
            start_quant_block = prev_quant_blocks[group_idx]
            if is_last_chunk:
                end_quant_block = (chunk_end + block_size - 1) // block_size
            else:
                end_quant_block = chunk_end // block_size
            end_quant_block = max(start_quant_block, end_quant_block)

            computed_block = num_computed_tokens // block_size
            quant_start_block = start_quant_block
            if group_idx < len(hit_run):
                quant_start_block = min(
                    max(
                        start_quant_block,
                        min(hit_run[group_idx], computed_block),
                    ),
                    end_quant_block,
                )
            quant_block_ids.append(tuple(group_blocks[quant_start_block:end_quant_block]))
            quant_first_pages.append(quant_start_block)
            next_quant_blocks[group_idx] = end_quant_block

            xfer_idx = group_to_xfer.get(group_idx)
            if xfer_idx is None:
                continue

            ilu_clip = ilu_clips[xfer_idx] if xfer_idx < len(ilu_clips) else None
            if ilu_clip == slice(0, 0):
                block_offsets.append(0)
                chunk_block_ids.append([])
                continue

            if self._is_tail_only_xfer_group(xfer_idx):
                # P1-1 Hybrid Split-Cadence: running-state Mamba / KpoolTail /
                # single-slot ring groups stay empty on intermediate chunks and
                # piggyback their final state slot on is_last_chunk (offset = -1).
                block_offsets.append(-1)
                if is_last_chunk:
                    chunk_block_ids.append(
                        self._clip_tail_only_group_blocks(
                            xfer_idx, group_blocks, page_ratio
                        )
                    )
                else:
                    chunk_block_ids.append([])
                continue

            sw = (
                self._wire_sliding_windows[group_idx]
                if group_idx < len(self._wire_sliding_windows)
                else 0
            )
            target_start_block = prev_target_blocks[group_idx]
            if is_last_chunk:
                target_end_block = (
                    chunk_end + target_block_size - 1
                ) // target_block_size
            else:
                target_end_block = chunk_end // target_block_size
            target_end_block = max(target_start_block, target_end_block)
            next_target_blocks[group_idx] = target_end_block

            total_blocks = (
                candidate["prefill_tokens"] + target_block_size - 1
            ) // target_block_size
            retained = (
                _retained_d_pages(
                    candidate["prefill_tokens"], sw, target_block_size
                )
                if sw
                else 0
            )
            remote_base = max(0, total_blocks - retained) if retained > 0 else 0
            selected_start = max(target_start_block, remote_base)
            selected_end = max(selected_start, target_end_block)
            block_offsets.append(selected_start - remote_base)
            chunk_block_ids.append(
                _encode_wire_page_ids(
                    group_blocks,
                    page_ratio,
                    selected_start,
                    selected_end,
                )
            )

        if not chunk_block_ids or not any(chunk_block_ids):
            if is_last_chunk and (request_id in self._chunk_committed):
                # Check if every non-clipped group already emitted all its blocks earlier.
                all_complete = all(
                    self._xfer_group_allows_empty_chunk(
                        xfer_idx, is_last_chunk=True
                    )
                    or next_target_blocks[self._transfer_group_ids[xfer_idx]] > 0
                    for xfer_idx in range(len(chunk_block_ids))
                )
                if not all_complete:
                    raise RuntimeError(
                        "final prefill chunk has incomplete block mapping for "
                        f"{request_id}: {[len(g) for g in chunk_block_ids]}"
                    )
            else:
                return

        if is_last_chunk and any(
            not group
            and next_target_blocks[self._transfer_group_ids[xfer_idx]] == 0
            and not self._xfer_group_allows_empty_chunk(
                xfer_idx, is_last_chunk=True
            )
            for xfer_idx, group in enumerate(chunk_block_ids)
        ):
            if request_id in self._chunk_committed:
                raise RuntimeError(
                    "final prefill chunk has incomplete block mapping for "
                    f"{request_id}: {[len(g) for g in chunk_block_ids]}"
                )
            return

        chunk_index = candidate.get("next_chunk_index", 0)
        candidate["next_chunk_index"] = chunk_index + 1
        candidate["emitted_tokens"] = chunk_end
        candidate["emitted_target_blocks"] = next_target_blocks
        candidate["emitted_quant_blocks"] = next_quant_blocks
        requantize_block_ids = _drop_blocks_in_fresh_range(
            self._requantize_pending.pop(request_id, ()),
            quant_block_ids,
        )
        meta.add_chunk_req(
            request_id,
            ChunkPushReqMeta(
                local_block_ids=tuple(chunk_block_ids),
                decode_engine_id=candidate["remote_engine_id"],
                prefill_tokens=candidate["prefill_tokens"],
                block_offsets=tuple(block_offsets),
                chunk_index=chunk_index,
                is_last_chunk=is_last_chunk,
                quant_block_ids=tuple(quant_block_ids),
                quant_first_pages=tuple(quant_first_pages),
                num_valid_tokens=chunk_end,
                requantize_block_ids=requantize_block_ids,
                protocol_version=1,
            ),
        )
        self._chunk_committed.add(request_id)
        if is_last_chunk:
            self._chunk_candidates.pop(request_id, None)
            self._chunk_blocks.pop(request_id, None)

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        if not self._chunk_candidates:
            return meta

        for new_req in scheduler_output.scheduled_new_reqs:
            self._maybe_emit_chunk(
                meta,
                new_req.req_id,
                scheduler_output.num_scheduled_tokens.get(new_req.req_id, 0),
                new_req.num_computed_tokens,
                new_req.block_ids,
            )

        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", set()) or set()
        block_table_updates = (
            getattr(scheduler_output, "block_table_updates", None) or {}
        )
        for idx, request_id in enumerate(cached.req_ids):
            new_blocks = cached.new_block_ids[idx]
            if request_id in resumed:
                self._chunk_blocks.pop(request_id, None)
                if cand := self._chunk_candidates.get(request_id):
                    cand["emitted_tokens"] = 0
                    cand["emitted_target_blocks"] = [0] * len(
                        self._chunk_block_sizes
                    )
                    cand["emitted_quant_blocks"] = [0] * len(
                        self._chunk_block_sizes
                    )
            if request_id in block_table_updates:
                self._chunk_blocks.pop(request_id, None)
                new_blocks = block_table_updates[request_id]
            self._maybe_emit_chunk(
                meta,
                request_id,
                scheduler_output.num_scheduled_tokens.get(request_id, 0),
                cached.num_computed_tokens[idx],
                new_blocks,
            )

        return meta

    def request_finished(self, request, block_ids):
        req_params = request.kv_transfer_params
        was_unallocated_remote_prefill = bool(
            req_params and req_params.get("do_remote_prefill")
        )
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if was_unallocated_remote_prefill:
            self._reqs_need_recv.pop(request.request_id, None)
            self._stage_release_only_registration(
                request.request_id, req_params
            )
        if params is not None:
            params["pp_size"] = (
                self.vllm_config.parallel_config.pipeline_parallel_size
            )
            params["pp_split"] = self._pp_split

        if request.request_id in self._chunk_committed:
            self._newly_finished_push_blocks.pop(request.request_id, None)
            self._chunk_committed.discard(request.request_id)

        self._chunk_candidates.pop(request.request_id, None)
        self._chunk_blocks.pop(request.request_id, None)
        self._requantize_pending.pop(request.request_id, None)
        self._hit_prefix.pop(request.request_id, None)
        self._remote_block_starts.pop(request.request_id, None)
        return delay_free_blocks, params
