# SPDX-License-Identifier: Apache-2.0
"""Replicated dense vocabulary projection distributed over fixed DP subgroups.

Weights keep their checkpoint precision and original ownership. Each member
computes a vocabulary slice for the subgroup's inputs, then returns BF16/FP16
logits to their request owner. One worker stream serializes the shared staging
buffers. Process-group teardown belongs to the worker's distributed lifecycle.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import timedelta
from functools import wraps
from inspect import Parameter, signature
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed import get_dp_group, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)
_active_projection: ContextVar[DPVocabController | None] = ContextVar(
    "ilu_dp_vocab_projection", default=None
)


class DPVocabProjection:
    """Bounded collective buffers for a fixed, contiguous DP subgroup.

    All DP ranks must invoke each projection in the same phase order, including
    idle ranks. Local row counts may differ or be zero. All participants must
    select the same capacity from their shared CPU DP agreement; local tensor
    shapes never select a collective's extent. Maximum buffers remain allocated
    before KV profiling, including when a smaller packed prefix is used.
    """

    def __init__(
        self, weight: torch.Tensor, capacity: int, ranks: list[int], group_size: int
    ) -> None:
        if (
            weight.ndim != 2
            or not weight.is_cuda
            or not weight.is_contiguous()
            or weight.dtype not in (torch.bfloat16, torch.float16)
        ):
            raise ValueError(
                "DP vocabulary projection requires a dense CUDA FP16/BF16 weight"
            )
        if capacity <= 0 or group_size <= 1 or len(ranks) % group_size:
            raise ValueError("Vocabulary group size must divide the DP world size")
        if weight.shape[0] % group_size:
            raise ValueError("Vocabulary rows must divide evenly across the subgroup")
        self.capacity = capacity
        self.group_size = group_size
        self.vocab_size, self.hidden_size = weight.shape
        self.dtype, self.device = weight.dtype, weight.device
        self.shard_rows = self.vocab_size // group_size
        self.group = None
        rank = dist.get_rank()
        for begin in range(0, len(ranks), group_size):
            members = ranks[begin : begin + group_size]
            group = dist.new_group(
                members, backend="nccl", timeout=timedelta(seconds=90)
            )
            if rank in members:
                self.group, self.group_rank = group, members.index(rank)
        if self.group is None:
            raise ValueError("Worker is not a member of the configured DP ranks")
        options = dict(device=self.device, dtype=self.dtype)
        self.input = torch.zeros((capacity, self.hidden_size), **options)
        self.gathered = torch.empty(
            (group_size * capacity, self.hidden_size), **options
        )
        self.partial = torch.empty((group_size * capacity, self.shard_rows), **options)
        self.received = torch.empty_like(self.partial)
        self.dummy = torch.zeros((1, self.hidden_size), **options)

    def project(
        self, hidden: torch.Tensor, weight: torch.Tensor, capacity: int | None = None
    ) -> torch.Tensor:
        capacity = self.capacity if capacity is None else capacity
        if type(capacity) is not int or not 0 < capacity <= self.capacity:
            raise ValueError("Vocabulary capacity exceeds its allocated extent")
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.hidden_size
            or hidden.shape[0] > capacity
            or hidden.dtype != self.dtype
            or hidden.device != self.device
        ):
            raise ValueError(
                "Hidden states exceed the configured vocabulary projection extent"
            )
        if (
            tuple(weight.shape) != (self.vocab_size, self.hidden_size)
            or weight.dtype != self.dtype
            or weight.device != self.device
            or not weight.is_contiguous()
        ):
            raise ValueError("Vocabulary weight geometry or precision changed")
        # Resolve the current parameter, rather than caching a weight pointer.
        shard = weight.narrow(0, self.group_rank * self.shard_rows, self.shard_rows)
        inputs, gathered = self.input, self.gathered
        partial, received = self.partial, self.received
        if capacity != self.capacity:
            # Pack the peer blocks with stride `capacity`, not the maximum
            # stride. Both collectives and GEMM consume this same prefix.
            inputs = inputs[:capacity]
            gathered = gathered[: self.group_size * capacity]
            partial = partial[: self.group_size * capacity]
            received = received[: self.group_size * capacity]
        inputs.zero_()
        inputs[: hidden.shape[0]].copy_(hidden)
        dist.all_gather_into_tensor(gathered, inputs, group=self.group)
        torch.mm(gathered, shard.t(), out=partial)
        dist.all_to_all_single(received, partial, group=self.group)
        local = received.view(self.group_size, capacity, self.shard_rows)
        return local.transpose(0, 1).reshape(capacity, self.vocab_size)[
            : hidden.shape[0]
        ]


class DPVocabController:
    """Keep target and draft collective phases aligned with dummy execution."""

    def __init__(
        self,
        projector: DPVocabProjection,
        target_head: Any,
        draft_head: Any,
        *,
        per_request_bound: int = 1,
        dynamic_capacity_supported: bool = False,
    ) -> None:
        if dynamic_capacity_supported and (
            type(per_request_bound) is not int
            or per_request_bound < 2
            or projector.capacity % per_request_bound
        ):
            raise ValueError("Dynamic capacity requires maxseq*(K+1), K>=1")
        self.projector = projector
        self.heads = (target_head, draft_head)
        self.pending_target = False
        self.per_request_bound = per_request_bound
        self.dynamic_capacity_supported = dynamic_capacity_supported
        self.dynamic_capacity = False
        self._pair_capacity: int | None = None
        self._pair_projections = 0

    def enable_dynamic_capacity(self) -> None:
        """Enable supported worker RPCs only after profiling and graph capture."""
        self.dynamic_capacity = self.dynamic_capacity_supported

    def capacity_for(self, dp_sync) -> int:
        """Use the all-DP request bound, never a local request/tensor count.

        Unknown or invalid common metadata keeps maximum capacity on all
        participants. A local geometry violation later raises; it must never
        change only that rank's collective size.
        """
        full = self.projector.capacity
        requests = getattr(dp_sync, "num_reqs", None)
        if (
            self.dynamic_capacity
            and type(requests) is int
            and 0 < requests <= full // self.per_request_bound
        ):
            return requests * self.per_request_bound
        return full

    @contextmanager
    def pair(self, dp_sync):
        """Pin one capacity across target, draft and their idle substitutes."""
        if self._pair_capacity is not None or self.pending_target:
            raise RuntimeError("Vocabulary target/draft pairs must not overlap")
        self._pair_capacity = self.capacity_for(dp_sync)
        self._pair_projections = 0
        try:
            yield
            if self._pair_projections != 2 or self.pending_target:
                raise RuntimeError(
                    "Vocabulary pair requires target and draft projections"
                )
        finally:
            self.pending_target = False
            self._pair_capacity = None
            self._pair_projections = 0

    def project(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self._pair_capacity is None:
            return self.projector.project(hidden, weight)
        if self._pair_projections >= 2:
            raise RuntimeError("Unexpected extra vocabulary projection in one pair")
        self._pair_projections += 1
        return self.projector.project(hidden, weight, capacity=self._pair_capacity)

    @contextmanager
    def scope(self):
        token = _active_projection.set(self)
        try:
            yield
        finally:
            _active_projection.reset(token)

    def wrap_sample(self, original):
        @wraps(original)
        def sample(*args, **kwargs):
            if self.pending_target:
                raise RuntimeError(
                    "Previous target projection has not reached its draft phase"
                )
            with self.scope():
                result = original(*args, **kwargs)
            self.pending_target = True
            return result

        return sample

    def wrap_sequential(self, original):
        @wraps(original)
        def sequential(*args, **kwargs):
            with self.scope():
                return original(*args, **kwargs)

        return sequential

    def wrap_propose(self, original):
        # The public proposer accepts positional arguments too. Resolve their
        # positions once at installation, outside the inference loop.
        positional = [
            p.name
            for p in signature(original).parameters.values()
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ]
        sync_pos = positional.index("dp_sync") if "dp_sync" in positional else None
        dummy_pos = positional.index("dummy_run") if "dummy_run" in positional else None

        def argument(args, kwargs, name, position, default=None):
            if name in kwargs:
                return kwargs[name]
            return (
                args[position]
                if position is not None and position < len(args)
                else default
            )

        @wraps(original)
        def propose(*args, **kwargs):
            dp_sync = argument(args, kwargs, "dp_sync", sync_pos)
            dummy_run = argument(args, kwargs, "dummy_run", dummy_pos, False)
            if type(dummy_run) is not bool:
                raise TypeError("dummy_run must be a host bool")
            if dummy_run and self.pending_target:
                raise RuntimeError(
                    "Dummy target projection would duplicate an active phase"
                )
            context = (
                self.pair(dp_sync)
                if dummy_run and self.dynamic_capacity
                else nullcontext()
            )
            with context:
                if dummy_run:
                    # Idle ranks skip target sampling but must still join its
                    # collective before the common draft phase.
                    self.project(self.projector.dummy, self.heads[0].weight)
                elif not self.pending_target:
                    raise RuntimeError(
                        "Draft proposal requires a preceding target projection"
                    )
                if self.dynamic_capacity and (
                    self._pair_capacity is None
                    or self.capacity_for(dp_sync) != self._pair_capacity
                ):
                    raise RuntimeError(
                        "Target and draft require the same DP capacity agreement"
                    )
                try:
                    return original(*args, **kwargs)
                finally:
                    self.pending_target = False

        return propose


def _enable_scoped_head_dispatch() -> None:
    original = LogitsProcessor._apply_head
    if getattr(original, "_ilu_dp_vocab_projection", False):
        return

    @wraps(original)
    def apply_head(processor, lm_head, hidden, bias):
        owner = _active_projection.get()
        if owner is not None and any(lm_head is head for head in owner.heads):
            if bias is not None or processor.head_dtype not in (None, hidden.dtype):
                raise ValueError(
                    "DP vocabulary projection does not change head precision or bias"
                )
            return owner.project(hidden, lm_head.weight)
        return original(processor, lm_head, hidden, bias)

    apply_head._ilu_dp_vocab_projection = True
    LogitsProcessor._apply_head = apply_head


def install_dp_vocab_projection(runner, group_size: int) -> DPVocabController:
    """Install the opt-in policy after all target and draft weights are loaded.

    The supported contract is V2 DSpark, TP1/PP1, independent draft eager.
    Prompt-logprob projections outside the sampler scope retain the original
    implementation. The whole fixed-size subgroup is initialized collectively
    before memory profiling so its buffers are included in the KV budget.
    """
    if hasattr(runner, "_ilu_dp_vocab_projection"):
        raise RuntimeError("DP vocabulary projection is already installed")
    if get_tp_group().world_size != 1 or get_pp_group().world_size != 1:
        raise ValueError("DP vocabulary projection requires TP1 and PP1")
    draft = runner.speculator
    if (
        draft is None
        or draft.__class__.__name__ != "DSparkSpeculator"
        or not draft.speculative_config.enforce_eager
        or draft._draft_topk is not None
    ):
        raise ValueError(
            "DP vocabulary projection requires an eager DSpark draft without top-k heads"
        )
    language = getattr(runner.model, "language_model", runner.model)
    target_head, draft_head = language.lm_head, draft.model.lm_head
    for head in (target_head, draft_head):
        if not isinstance(
            head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            raise ValueError(
                "DP vocabulary projection preserves unquantized head weights"
            )
    if target_head.weight.data_ptr() != draft_head.weight.data_ptr():
        raise ValueError("Target and draft must share the replicated vocabulary weight")
    if language.logits_processor.head_dtype not in (None, target_head.weight.dtype):
        raise ValueError("DP vocabulary projection preserves the checkpoint head dtype")
    capacity = runner.scheduler_config.max_num_seqs * (draft.num_speculative_steps + 1)
    owner = DPVocabProjection(
        target_head.weight, capacity, list(get_dp_group().ranks), group_size
    )
    owner.project(owner.dummy, target_head.weight)
    torch.cuda.synchronize()
    # Ray's compiled DAG can bypass Worker.sample_tokens. Keep one common
    # maximum-capacity policy there and for unknown/custom executors, rather
    # than allowing real and idle ranks to select different buffer extents.
    dynamic_supported = (
        runner.parallel_config.distributed_executor_backend in ("uni", "mp")
        and draft.num_speculative_steps >= 1
        and runner.model_state.num_new_sampled_tokens_per_step == 1
        and owner.shard_rows > 1
    )
    controller = DPVocabController(
        owner,
        target_head,
        draft_head,
        per_request_bound=draft.num_speculative_steps + 1,
        dynamic_capacity_supported=dynamic_supported,
    )
    _enable_scoped_head_dispatch()
    runner.sample = controller.wrap_sample(runner.sample)
    draft._sample_sequential = controller.wrap_sequential(draft._sample_sequential)
    draft.propose = controller.wrap_propose(draft.propose)
    runner._ilu_dp_vocab_projection = controller
    logger.info(
        "DP vocabulary projection: group=%d capacity=%d dtype=%s dynamic=%s",
        group_size,
        capacity,
        owner.dtype,
        dynamic_supported,
    )
    return controller
