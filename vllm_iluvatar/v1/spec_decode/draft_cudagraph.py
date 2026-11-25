# SPDX-License-Identifier: Apache-2.0
"""Independent GLM MTP forward graphs, with exact uniform decode layouts.

The proposer still owns its loop, metadata builders, sampling and two DP
coordination calls. Graphs are captured at startup, never on a live request.
Each (request count, query length, index reuse) variant owns stable metadata;
the native vLLM wrapper owns capture/replay and participates in clear_all_graphs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import wraps

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_iluvatar.v1.worker.dp_utils import coordinate_batch_across_dp

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class DraftGraphKey:
    num_reqs: int
    query_len: int
    reuse_index: bool

    @property
    def num_tokens(self) -> int:
        return self.num_reqs * self.query_len


def _capture_keys(sizes, max_reqs, first_query_len):
    keys = {
        DraftGraphKey(n // q, q, reuse)
        for n in sizes
        for q, reuse in [(1, False), (first_query_len, False), (1, True)]
        if n % q == 0 and 0 < n // q <= max_reqs
    }
    # Target sizes are rounded to num_speculative_tokens + 1. Every captured
    # first pass also needs a one-token reuse graph for the same requests;
    # e.g. MTP2's 12-token first pass needs a 4-token reuse graph.
    keys.update(
        DraftGraphKey(key.num_reqs, 1, True) for key in tuple(keys) if key.query_len > 1
    )
    return keys


def uniform_key(cad, reuse_index: bool, query_lengths: tuple[int, ...]):
    """Read only CPU layout metadata; never synchronize a GPU length tensor."""
    if cad.num_reqs <= 0 or cad.max_query_len not in query_lengths:
        return None
    lengths = torch.diff(cad.query_start_loc_cpu)
    q = cad.max_query_len
    if (
        lengths.numel() != cad.num_reqs
        or cad.num_actual_tokens != cad.num_reqs * q
        or not bool(torch.all(lengths == q))
    ):
        return None
    return DraftGraphKey(cad.num_reqs, q, reuse_index)


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                f.name: _clone_tree(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init
            },
        )
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported draft graph metadata: {type(value)}")


def _stage_tree(dst, src):
    """Update captured storage and fail on an unrepresented structural change."""
    if isinstance(dst, torch.Tensor):
        if (
            not isinstance(src, torch.Tensor)
            or dst.shape != src.shape
            or dst.dtype != src.dtype
            or dst.device != src.device
        ):
            raise ValueError("Draft graph tensor metadata changed layout")
        dst.copy_(src, non_blocking=True)
    elif dataclasses.is_dataclass(dst):
        if type(dst) is not type(src):
            raise ValueError("Draft graph metadata backend changed")
        for f in dataclasses.fields(dst):
            if f.init:
                _stage_tree(getattr(dst, f.name), getattr(src, f.name))
    elif isinstance(dst, dict):
        if not isinstance(src, dict) or dst.keys() != src.keys():
            raise ValueError("Draft graph metadata layer set changed")
        for k in dst:
            _stage_tree(dst[k], src[k])
    elif dst != src:
        raise ValueError(f"Draft graph static metadata changed: {dst!r} -> {src!r}")


def supported(proposer) -> bool:
    import vllm.envs as envs

    pc = proposer.vllm_config.parallel_config
    sc = proposer.speculative_config
    arch = proposer.draft_model_config.hf_config.architectures or []
    return bool(
        envs.VLLM_ILU_GLM_DRAFT_CUDAGRAPH
        and proposer.method == "mtp"
        and arch == ["IluGlmMoeDsaMTPModel"]
        and not sc.enforce_eager
        and not proposer.vllm_config.model_config.enforce_eager
        and proposer.vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
        and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        and pc.pipeline_parallel_size == 1
        and pc.prefill_context_parallel_size == 1
        and pc.decode_context_parallel_size == 1
        and pc.all2all_backend == "allgather_reducescatter"
        and not pc.enable_dbo
        and not pc.enable_eplb
        and proposer.vllm_config.lora_config is None
        and not proposer.parallel_drafting
        and not proposer.supports_mm_inputs
        and proposer.draft_model_config.hf_config.num_nextn_predict_layers == 1
        and proposer.draft_model_config.hf_config.index_share_for_mtp_iteration
    )


@dataclasses.dataclass
class DraftGraphEntry:
    metadata: dict
    wrapper: CUDAGraphWrapper


class DraftGraphManager:
    def __init__(self, proposer):
        self.proposer = proposer
        self.runnable = proposer.model
        self.entries: dict[DraftGraphKey, DraftGraphEntry] = {}
        self.query_lengths = (1, proposer.num_speculative_tokens + 1)
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.capturing = False
        self.ready = False
        self.in_propose = False
        self.key = None
        self.first_key = None
        self.coordination_count = 0
        self.first_replay = False

    def __getattr__(self, key):
        # Preserve the proposer's model.model / compute_logits / embeddings API.
        return getattr(self.runnable, key)

    def reset(self):
        self.key = self.first_key = None
        self.coordination_count = 0
        self.first_replay = False

    def calibrated(self):
        return all(
            not getattr(m, "_fp8_kv_scale_calibration_enabled", False)
            or getattr(m, "_fp8_kv_scale_frozen", False)
            for m in self.runnable.modules()
        )

    def build_metadata(self, orig, cad, draft_index=0):
        key = uniform_key(
            cad, draft_index > 0, (1,) if draft_index else self.query_lengths
        )
        # max_seq_len is an upper bound. Keep Python scalars fixed while real
        # lengths, page tables and causal masks remain device-driven.
        if key is not None:
            cad = dataclasses.replace(cad, max_seq_len=self.proposer.max_model_len)
        result = orig(self.proposer, cad, draft_index=draft_index)
        if draft_index == 0:
            self.first_key = key
        self.key = key
        return result

    def coordinate(self, num_tokens, use_cudagraphs=True):
        if self.coordination_count == 0:
            key = self.first_key
        else:
            key = DraftGraphKey(num_tokens, 1, True) if self.first_replay else None
        self.coordination_count += 1
        entry = self.entries.get(key)
        eligible = bool(
            use_cudagraphs
            and self.ready
            and entry is not None
            and key.num_tokens == num_tokens
            and entry.wrapper.concrete_cudagraph_entries
            and self.calibrated()
        )
        mode = CUDAGraphMode.FULL if eligible else CUDAGraphMode.NONE
        # uniform_decode intentionally omitted: retain the original draft MoE
        # stage, and do not switch gate placement/communication with graph mode.
        _, counts, synced_mode = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens,
            num_tokens_padded=num_tokens,
            allow_microbatching=False,
            parallel_config=self.proposer.vllm_config.parallel_config,
            cudagraph_mode=mode.value,
            require_uniform_graph_tokens=True,
        )
        mode = CUDAGraphMode(synced_mode)
        if self.coordination_count == 1:
            self.first_replay = mode == CUDAGraphMode.FULL
        return mode, num_tokens, counts

    def __call__(self, **kwargs):
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL or self.capturing:
            return self.runnable(**kwargs)
        entry = self.entries[self.key]
        if not entry.wrapper.concrete_cudagraph_entries:
            raise RuntimeError(
                "Draft graphs were cleared; restart capture before replay"
            )
        _stage_tree(entry.metadata, context.attn_metadata)
        old_metadata, old_desc = context.attn_metadata, context.batch_descriptor
        context.attn_metadata = entry.metadata
        context.batch_descriptor = BatchDescriptor(num_tokens=self.key.num_tokens)
        try:
            result = entry.wrapper(**kwargs)
        finally:
            context.attn_metadata, context.batch_descriptor = old_metadata, old_desc
        logger.info_once("GLM independent draft Graph replay: %s", self.key)
        return result

    @torch.inference_mode()
    def capture(self):
        if self.ready and all(
            entry.wrapper.concrete_cudagraph_entries for entry in self.entries.values()
        ):
            return
        self.ready = False
        self.entries.clear()
        p = self.proposer
        cfg = p.vllm_config
        max_reqs = cfg.scheduler_config.max_num_seqs
        sizes = cfg.compilation_config.cudagraph_capture_sizes
        keys = _capture_keys(sizes, max_reqs, self.query_lengths[-1])
        self.capturing = True
        try:
            for key in sorted(
                keys, key=lambda k: (-k.num_tokens, k.reuse_index, k.query_len)
            ):
                n, r, q = key.num_tokens, key.num_reqs, key.query_len
                starts_cpu = torch.arange(r + 1, dtype=torch.int32) * q
                # Page zero is reserved by vLLM; slot -1 prevents dummy KV writes.
                block_table = torch.zeros(
                    (r, (p.max_model_len + p.block_size - 1) // p.block_size),
                    device=p.device,
                    dtype=torch.int32,
                )
                cad = CommonAttentionMetadata(
                    query_start_loc=starts_cpu.to(p.device),
                    query_start_loc_cpu=starts_cpu,
                    seq_lens=torch.full((r,), q, dtype=torch.int32, device=p.device),
                    seq_lens_cpu_upper_bound=torch.full((r,), q, dtype=torch.int32),
                    num_reqs=r,
                    num_actual_tokens=n,
                    max_query_len=q,
                    max_seq_len=p.max_model_len,
                    block_table_tensor=block_table,
                    slot_mapping=torch.full(
                        (n,), -1, device=p.device, dtype=torch.int64
                    ),
                )
                _, metadata = p.build_per_group_and_layer_attn_metadata(
                    cad, draft_index=int(key.reuse_index)
                )
                if any(md.num_prefills for md in metadata.values()):
                    continue
                metadata = _clone_tree(metadata)
                p.input_ids[:n].zero_()
                p.positions[:n].zero_()
                p.hidden_states[:n].zero_()
                self.runnable.model.set_skip_topk(key.reuse_index)
                self.runnable.model.topk_indices_buffer.fill_(-1)
                kwargs, _ = p.build_model_inputs_first_pass(n, n, None)
                wrapper = CUDAGraphWrapper(
                    self.runnable,
                    cfg,
                    CUDAGraphMode.FULL,
                    CUDAGraphOptions(weak_ref_output=False),
                )
                wrapper.graph_pool = self.graph_pool
                # Reuse the ordinary coordinator to establish the draft phase
                # and identical collective sizes on every rank before capture.
                _, counts, _ = coordinate_batch_across_dp(
                    n, False, cfg.parallel_config, cudagraph_mode=0
                )
                for mode in (
                    CUDAGraphMode.NONE,
                    CUDAGraphMode.NONE,
                    CUDAGraphMode.FULL,
                ):
                    with set_forward_context(
                        metadata,
                        cfg,
                        num_tokens=n,
                        num_tokens_across_dp=counts,
                        cudagraph_runtime_mode=mode,
                        batch_descriptor=BatchDescriptor(num_tokens=n),
                        slot_mapping=p._get_slot_mapping(n, cad.slot_mapping),
                    ):
                        wrapper(**kwargs)
                    torch.cuda.synchronize()
                self.entries[key] = DraftGraphEntry(metadata, wrapper)
            self.ready = True
            logger.info(
                "Captured %d independent GLM draft forward graphs", len(self.entries)
            )
        finally:
            self.runnable.model.set_skip_topk(False)
            self.capturing = False
            self.reset()


def make_load_model(orig: Callable):
    @wraps(orig)
    def load_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if supported(self):
            self._ilu_draft_graph = DraftGraphManager(self)
            self.model = self._ilu_draft_graph
        return result

    return load_model


def make_propose(orig: Callable):
    @wraps(orig)
    def propose(self, *args, **kwargs):
        manager = getattr(self, "_ilu_draft_graph", None)
        if manager is None:
            return orig(self, *args, **kwargs)
        manager.reset()
        manager.in_propose = True
        try:
            return orig(self, *args, **kwargs)
        finally:
            manager.in_propose = False

    return propose


def make_build_metadata(orig: Callable):
    @wraps(orig)
    def build(self, common_attn_metadata, draft_index=0):
        manager = getattr(self, "_ilu_draft_graph", None)
        if manager is not None:
            return manager.build_metadata(orig, common_attn_metadata, draft_index)
        return orig(self, common_attn_metadata, draft_index=draft_index)

    return build


def make_determine_batch(orig: Callable):
    @wraps(orig)
    def determine(self, num_tokens, use_cudagraphs=True):
        manager = getattr(self, "_ilu_draft_graph", None)
        if manager is not None and manager.ready and manager.in_propose:
            return manager.coordinate(num_tokens, use_cudagraphs)
        return orig(self, num_tokens, use_cudagraphs=use_cudagraphs)

    return determine


def make_capture_cudagraphs(orig: Callable):
    @wraps(orig)
    def capture(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        manager = getattr(getattr(self, "drafter", None), "_ilu_draft_graph", None)
        if manager is not None:
            # Already inside worker graph_capture and memory accounting, before
            # workspace locking. All ranks visit the same startup sequence.
            manager.capture()
        return result

    return capture
