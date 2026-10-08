# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``AutoRegressiveSpeculator`` Option B draft-decode override.

Corresponds to: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py ::
``AutoRegressiveSpeculator.__init__`` / ``capture`` / ``_multi_step_decode``

ixfd's TP ``all_gather`` (used inside ``sample_draft``'s ``compute_logits``) is
not CUDA-graph-safe: it allocates a fresh output buffer and issues an async
collective, and ixfd has no NCCL-style graph-window registration. Capturing it
into the draft decode graph corrupts the shared ixfd communicator and yields
garbage draft/target logits (seen as repetitive-loop output). ``all_reduce`` is
graph-safe (in-place, fixed buffer), so the target decode graph — which only
captures the forward — is unaffected; only the draft, which originally captured
the whole ``_generate_draft`` (including the all_gather), broke.

Option B splits ``_generate_draft`` into a graph-captured part
(``_generate_draft_graph_part``: ``_prepare_eplb_forward`` + ``_run_model`` +
copy into persistent buffers) and an eager tail (``_generate_draft_tail``:
``sample_draft`` + ``update_draft_inputs``). The persistent buffers are allocated
in ``__init__`` (regular pool, before any graph pool), so their addresses are
stable across capture and replay; no graph-pool tensor is held past capture.

Baseline : v0.30.0 (ced6857afa)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    update_draft_inputs,
)


def make_init(orig_init: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``__init__`` to allocate the two persistent graph-output buffers."""

    @functools.wraps(orig_init)
    def __init__(self, vllm_config, device):  # type: ignore[no-redef]
        orig_init(self, vllm_config, device)
        # Persistent buffers for the captured _run_model outputs. Allocated
        # here (regular pool, before any graph pool) so the address is stable
        # across capture and replay; sized for the decode batch (num_reqs).
        self._cg_out_last_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self._cg_out_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )

    return __init__


def make_capture(orig_capture: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``capture`` so the non-fused decode graph captures only the
    graph-safe part (``_generate_draft_graph_part``) instead of the full
    ``_generate_draft``.

    Done by temporarily shadowing ``self._generate_draft`` with
    ``self._generate_draft_graph_part`` for the duration of the community
    capture call, then restoring it so the eager fallback path keeps using the
    real ``_generate_draft``. Avoids re-implementing the long capture method.
    """

    @functools.wraps(orig_capture)
    def capture(self):  # type: ignore[no-redef]
        # Shadow the instance attribute so community capture's
        # `decode_fn = ... else self._generate_draft` resolves to the
        # graph-safe part. Only affects the non-fused branch; the fused
        # branch uses self._generate_fused_drafts (unaffected).
        self.__dict__["_generate_draft"] = self._generate_draft_graph_part
        try:
            orig_capture(self)
        finally:
            # Drop the shadow so self._generate_draft falls back to the real
            # class method (used by the eager fallback path).
            self.__dict__.pop("_generate_draft", None)

    return capture


def make_multi_step_decode(
    orig_msd: Callable[..., None],
) -> Callable[..., None]:
    """Wrap ``_multi_step_decode`` so each FULL-graph replay is followed by the
    eager tail (``_generate_draft_tail``), which runs the non-graph-safe
    all_gather + argmax + update.

    Done by temporarily wrapping ``run_fullgraph`` on this speculator's decode
    cudagraph manager for the duration of the community method, so the tail
    interleaves per step without re-implementing the loop.
    """

    @functools.wraps(orig_msd)
    def _multi_step_decode(  # type: ignore[no-redef]
        self,
        num_reqs,
        skip_attn,
        batch_desc,
        num_tokens_across_dp,
        seq_lens_cpu_upper_bound,
    ):
        mgr = self.decode_cudagraph_manager
        orig_rfg = None if mgr is None else mgr.run_fullgraph

        def run_fullgraph_with_tail(desc):
            assert orig_rfg is not None
            orig_rfg(desc)
            self._generate_draft_tail(num_reqs)

        if mgr is not None and batch_desc.cg_mode == CUDAGraphMode.FULL:
            mgr.run_fullgraph = run_fullgraph_with_tail  # type: ignore[method-assign]
        try:
            orig_msd(
                self,
                num_reqs,
                skip_attn,
                batch_desc,
                num_tokens_across_dp,
                seq_lens_cpu_upper_bound,
            )
        finally:
            if mgr is not None and orig_rfg is not None:
                mgr.run_fullgraph = orig_rfg  # type: ignore[method-assign]

    return _multi_step_decode


def _generate_draft_graph_part(
    self,
    num_reqs: int,
    num_tokens_padded: int,
    attn_metadata: dict[str, Any] | None,
    slot_mappings: dict[str, torch.Tensor] | None,
    num_tokens_across_dp: torch.Tensor | None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> None:
    """Graph-captured part of ``_generate_draft``: ``_prepare_eplb_forward`` +
    ``_run_model`` only (collectives here are in-place all_reduce, graph-safe)
    and a copy of the two hidden-state outputs into persistent buffers. The TP
    all_gather (in ``compute_logits``) is deliberately excluded — it is not
    graph-safe on ixfd and runs eager in ``_generate_draft_tail``.
    """
    self._prepare_eplb_forward(num_reqs)
    last_hidden_states, hidden_states = self._run_model(
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    )
    # Copy into pre-allocated persistent buffers (NOT graph-pool tensors). The
    # copy is captured, so on replay it writes the replayed _run_model outputs
    # here; the eager tail reads these. No graph-pool tensor is kept past
    # capture (holding those corrupts the shared graph pool's address reuse).
    self._cg_out_last_hidden[:num_reqs].copy_(last_hidden_states[:num_reqs])
    self._cg_out_hidden[:num_reqs].copy_(hidden_states[:num_reqs])


def _generate_draft_tail(self, num_reqs: int) -> None:
    """Eager tail of ``_generate_draft``, run after the decode graph replay.
    Reads the persistent buffers filled by the captured graph and runs
    ``sample_draft`` (TP all_gather + argmax) + ``update_draft_inputs``, which
    are not captured because the all_gather is not graph-safe on ixfd.
    """
    idx_mapping = self.idx_mapping[:num_reqs]
    sample_hidden_states = self._cg_out_last_hidden[:num_reqs]
    sample_src_positions = self.sample_src_positions[:num_reqs]
    draft_tokens = self.sample_draft(
        sample_hidden_states,
        sample_src_positions,
        idx_mapping,
        self.temperature,
        self.seeds,
        self.current_draft_step,
        self.draft_logits,
    )

    update_draft_inputs(
        draft_tokens,
        self.current_draft_step,
        self._cg_out_hidden,
        self.draft_tokens,
        self.hidden_states,
        self.input_buffers,
        self.sample_src_positions,
        num_reqs,
        self.max_model_len,
        self.num_speculative_steps,
        advance_draft_positions=self.advance_draft_positions,
    )
