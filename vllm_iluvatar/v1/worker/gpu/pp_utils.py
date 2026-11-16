# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/pp_utils.py

ILU implementation for the V2 PP sampled-token transport. Runtime wiring and
drift guards live in ``ilu_patch.general_patches.patch_v2_pp_token_p2p`` and are
auto-discovered during ``register_general``.

Target   : vllm.v1.worker.gpu.pp_utils.PPHandler
Baseline : v0.31.0 (db9527a468)
Why      : IXCCL has no copy-engine implementation for broadcast, so the
           community collective launches a resident
           ``ixcclDevKernel_Broadcast_RING_LL``. On non-last PP ranks that
           kernel stays pending for the whole rank skew (measured 437-512 ms on
           DeepSeek-R1 int4 PP=4/TP=4) while it waits for the last rank's
           sampled tokens. Per SWDRV-5168 the MR event table is a linear
           256-slot SRAM, so the 256th compute kernel dispatched after the
           broadcast starts collides with the slot the resident kernel holds
           and the CP firmware spins until it retires. One compute kernel per
           iteration is therefore pinned to the end of the broadcast, which
           serialises the pipeline: every engine step pays a full traversal
           regardless of batch size.
How      : express the replication as send/recv over the PP group's ixfd
           communicator instead of a broadcast. IXCCL only implements its
           copy-engine path for send/recv, so no resident kernel is left on a
           compute engine. ixfd send/recv are enqueued on the current stream,
           so the copy stays ordered on the broadcast stream exactly like the
           collective it replaces.
Scope    : V2 pipeline parallelism on Iluvatar.
Enable   : opt-in via ``VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P=1``.
Remove   : on CQ silicon, where the event table allocates dynamically and the
           resident broadcast kernel no longer blocks compute.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


def _init_token_p2p(handler: Any) -> None:
    """Cache the PP group's ixfd comm so the per-step copy can use send/recv.

    Stores ``None`` when the comm is unavailable (pynccl-backed deployments,
    ixfd disabled), which makes ``ilu_token_broadcast`` fall back to the
    community collective.

    Deliberately reuses the PP group's comm instead of building one on the
    community ``broadcast_group``: relaying over an ixfd comm built on
    ``broadcast_group`` corrupts results under load (DeepSeek-R1-int4
    PP=4/TP=4 on MR, MTP=3, concurrency 128: GSM8K 94.5% -> 81.5%, throughput
    -37%) even though the token payloads arrive bit-exact. Root cause is not
    yet known; do not switch comms without re-running that check.
    """
    device_communicator = get_pp_group().device_communicator
    group = getattr(device_communicator, "ixfd_group", None)
    ranks = getattr(device_communicator, "ranks", None)
    if group is None or not ranks:
        logger.warning(
            "ILU: V2 PP token send/recv requested but the PP device "
            "communicator exposes no ixfd group; keeping the community "
            "broadcast transport."
        )
        handler._ilu_token_p2p = None
        return
    handler._ilu_token_p2p = (group, list(ranks))


def ilu_token_broadcast(handler: Any, tensor: torch.Tensor) -> None:
    """Replicate ``tensor`` from the last PP rank to every other PP rank.

    Uses ixfd send/recv when ``_ilu_token_p2p`` was populated, so the
    copy takes IXCCL's copy-engine path rather than a resident RING_LL kernel.
    The ranks recorded in ``_ilu_token_p2p`` are global, matching the
    convention ``IluCommunicator.send``/``recv`` already use.
    """
    state = getattr(handler, "_ilu_token_p2p", None)
    if state is None:
        torch.distributed.broadcast(
            tensor, src=handler.last_rank, group=handler.broadcast_group
        )
        return

    import ixformer.distributed as ixfd

    group, ranks = state
    if not handler.is_last_rank:
        ixfd.recv(tensor, handler.last_rank, group)
        return
    for dst in ranks:
        if dst != handler.last_rank:
            ixfd.send(tensor, dst, group)


def make_pphandler_init(original_init: Callable[..., None]) -> Callable[..., None]:
    @wraps(original_init)
    def _ilu_init(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ) -> None:
        original_init(self, max_num_reqs, num_speculative_steps, device)
        # Resolve the comm once, outside any graph capture or hot path.
        _init_token_p2p(self)

    return _ilu_init


def make_receive(pp_utils_mod: Any) -> Callable[..., bool]:
    """Community ``PPHandler.receive`` routed through ``ilu_token_broadcast``.

    Copied rather than wrapped because the collectives are issued inside the
    method body; ``RECEIVE_BASELINE`` in ``patch_v2_pp_token_p2p`` guards the
    copy against upstream drift.
    """

    def _ilu_receive(self, input_batch: InputBatch) -> bool:
        assert not self.is_last_rank
        need_sampled_mask = pp_utils_mod.compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            ilu_token_broadcast(self, sampled_tokens)
            ilu_token_broadcast(self, combined)
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                ilu_token_broadcast(self, draft_tokens)
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream
            # but later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = pp_utils_mod.PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    return _ilu_receive


def make_broadcast(pp_utils_mod: Any) -> Callable[..., None]:
    """Community ``PPHandler.broadcast`` routed through ``ilu_token_broadcast``."""

    def _ilu_broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if pp_utils_mod.compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            ilu_token_broadcast(self, send_tokens.contiguous())
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            ilu_token_broadcast(self, combined)
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)

    return _ilu_broadcast


def make_broadcast_drafts(pp_utils_mod: Any) -> Callable[..., None]:
    """Community ``PPHandler.broadcast_drafts`` routed the same way."""

    def _ilu_broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        assert self.is_last_rank
        if pp_utils_mod.compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            ilu_token_broadcast(self, send)

    return _ilu_broadcast_drafts
