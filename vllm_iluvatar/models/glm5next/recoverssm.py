# SPDX-License-Identifier: Apache-2.0
"""GLM KDA record ownership and resolved GPU commit plans."""

from __future__ import annotations

import weakref

import torch
from ixformer.inference.functions.ops.linear_attn.float.kda_recoverssm import (
    KDARecoverSSMFoldContext,
    kda_recoverssm_fold,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_PLAN_FIELDS = (
    "source",
    "record_rows",
    "request_rows",
    "verified_lens",
    "computed",
    "final",
    "commit_lens",
    "boundary",
    "boundary_lens",
    "final_cols",
    "request_slots",
    "computed_by_request",
)


@triton.jit
def _stage_record_metadata(
    source_in,
    source_out,
    record_rows,
    request_rows,
    cu,
    seq_lens,
    verified_lens,
    computed,
    computed_by_request,
    B: tl.constexpr,
    R: tl.constexpr,
    T: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < B
    request = tl.load(request_rows + b, mask=mask, other=0)
    begin = tl.load(cu + b, mask=mask, other=0)
    end = tl.load(cu + b + 1, mask=mask, other=0)
    length = end - begin
    tl.device_assert(
        (~mask) | ((length >= 0) & (length <= T)), "KDA verify length exceeds T"
    )
    source = tl.load(source_in + b * SOURCE_STRIDE, mask=mask, other=0)
    valid = mask & (b < R) & (source > 0) & (length > 0)
    sequence = tl.load(seq_lens + request, mask=mask, other=0)
    tl.store(source_out + b, tl.where(valid, source, 0), mask)
    tl.store(record_rows + b, tl.where(b < R, b, 0), mask)
    tl.store(verified_lens + b, tl.where(valid, length, 0), mask)
    tl.store(computed + b, sequence - length, mask)
    tl.store(computed_by_request + request, sequence - length, mask)


@triton.jit
def _prepare_commit_plan(
    source,
    request_rows,
    verified_lens,
    computed,
    num_sampled,
    idx_mapping,
    block_table,
    final,
    commit_lens,
    boundary,
    boundary_lens,
    final_cols,
    request_slots,
    B: tl.constexpr,
    ALIGN: tl.constexpr,
    HAS_IDX: tl.constexpr,
    NUM_SAMPLED: tl.constexpr,
    NUM_IDX: tl.constexpr,
    S: tl.constexpr,
    BT_ROW_STRIDE: tl.constexpr,
    BT_COL_STRIDE: tl.constexpr,
    BT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < B
    request = tl.load(request_rows + b, mask=mask, other=0)
    src = tl.load(source + b, mask=mask, other=0)
    length = tl.load(verified_lens + b, mask=mask, other=0)
    # Graph metadata can include padding after the sampler's actual batch.
    # Mask before either gather; ignoring the result later does not make an
    # out-of-range read valid.
    candidate = mask & (src > 0) & (length > 0) & (request >= 0)
    candidate = candidate & (request < NUM_SAMPLED)
    slot = request
    if HAS_IDX:
        candidate = candidate & (request < NUM_IDX)
        slot = tl.load(idx_mapping + request, mask=candidate, other=-1)
    sampled = tl.load(num_sampled + request, mask=candidate & (slot >= 0), other=0)
    m = tl.minimum(tl.maximum(sampled, 0), length)
    valid = mask & (slot >= 0) & (src > 0) & (m > 0)
    dest = src
    column = tl.full((BLOCK,), 0, tl.int32)
    bound = tl.full((BLOCK,), 0, tl.int32)
    bound_len = tl.full((BLOCK,), 0, tl.int32)
    if ALIGN:
        previous = tl.load(computed + b, mask=mask, other=0)
        end = previous + m
        column = tl.maximum(end - 1, 0) // S
        tl.device_assert(
            (~valid) | (column < BT_WIDTH), "KDA final block is not allocated"
        )
        dest = tl.load(
            block_table + request.to(tl.int64) * BT_ROW_STRIDE + column * BT_COL_STRIDE,
            mask=valid & (column < BT_WIDTH),
            other=0,
        )
        next_boundary = (previous // S + 1) * S
        crosses = valid & (end >= next_boundary)
        boundary_column = next_boundary // S - 1
        bound = tl.load(
            block_table
            + request.to(tl.int64) * BT_ROW_STRIDE
            + boundary_column * BT_COL_STRIDE,
            mask=crosses & (boundary_column < BT_WIDTH),
            other=0,
        )
        bound_len = tl.where(crosses, next_boundary - previous, 0)
    tl.store(final + b, tl.where(valid, dest, 0), mask)
    tl.store(commit_lens + b, tl.where(valid, m, 0), mask)
    tl.store(boundary + b, tl.where(valid, bound, 0), mask)
    tl.store(boundary_lens + b, tl.where(valid, bound_len, 0), mask)
    tl.store(final_cols + b, column, mask)
    tl.store(request_slots + b, tl.where(valid, slot, -1), mask)


@triton.jit
def _finish_commit(
    request_slots,
    commit_lens,
    final_cols,
    running_cols,
    accepted_out,
    B: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < B
    slot = tl.load(request_slots + b, mask=mask, other=-1)
    length = tl.load(commit_lens + b, mask=mask, other=0)
    valid = mask & (slot >= 0) & (length > 0)
    if ALIGN:
        column = tl.load(final_cols + b, mask=mask, other=0)
        tl.store(running_cols + slot, column, valid)
    tl.store(accepted_out + slot, 1, valid)


class KDARecoverSSMGroup:
    """One attention group's plan, sharing an early memory reservation."""

    def __init__(self, resources, layer_names, group_index, layer_start):
        self.resources = resources
        self.layer_names = tuple(layer_names)
        self.layers = tuple(resources.layers_by_name[name] for name in layer_names)
        self.context = resources.context.subcontext(
            layer_start, layer_start + len(self.layers)
        )
        for i, name in enumerate(_PLAN_FIELDS):
            setattr(self, name, resources.plans[group_index, i])
        self.bound_generation = -1

    def bind(self):
        states, convs = [], []
        for layer in self.layers:
            if not layer._recoverssm_weights_ready:
                raise RuntimeError(
                    "GLM KDA weights must be prepared before cache binding"
                )
            conv, state = layer.kv_cache
            expected_dtypes = layer.get_state_dtype()
            if (conv.dtype, state.dtype) != expected_dtypes:
                raise ValueError(
                    f"GLM KDA cache dtypes for {layer.prefix} are "
                    f"{(conv.dtype, state.dtype)}, expected {expected_dtypes}"
                )
            if not layer._conv_state_dim_first:
                conv = conv.transpose(-1, -2)
            convs.append(conv)
            states.append(state)
        self.context.bind(
            states,
            [layer._recoverssm_correction for layer in self.layers],
            [layer._recoverssm_key_gate for layer in self.layers],
            [layer.A_log_view for layer in self.layers],
            [layer.dt_bias_view for layer in self.layers],
            conv_state_pools=convs,
        )
        self.bound_generation = self.resources.generation

    def ensure_bound(self):
        if not self.resources.ready:
            raise RuntimeError("GLM RecoverSSM cache binding has not completed")
        if self.resources.groups.get(self.layer_names) is not self:
            raise RuntimeError(
                "Stale GLM RecoverSSM attention metadata after cache rebind"
            )
        if self.bound_generation != self.resources.generation:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GLM RecoverSSM cache must be bound before graph capture"
                )
            self.bind()

    def stage(self, source, cu, seq_lens, request_rows, batch):
        if batch == 0:
            return
        self.ensure_bound()
        if batch > self.resources.plan_capacity:
            raise ValueError("GLM RecoverSSM metadata exceeds reserved batch capacity")
        if request_rows is None:
            self.request_rows[:batch].copy_(self.resources.identity[:batch])
        else:
            self.request_rows[:batch].copy_(request_rows, non_blocking=True)
        _stage_record_metadata[(triton.cdiv(batch, 256),)](
            source,
            self.source,
            self.record_rows,
            self.request_rows,
            cu,
            seq_lens,
            self.verified_lens,
            self.computed,
            self.computed_by_request,
            B=batch,
            R=self.resources.record_capacity,
            T=self.resources.spec_query_len,
            SOURCE_STRIDE=source.stride(0),
            BLOCK=256,
        )

    def commit(self, num_sampled, batch, block_table, block_size, idx_mapping=None):
        if batch == 0:
            return
        self.ensure_bound()
        align = block_table is not None
        _prepare_commit_plan[(triton.cdiv(batch, 256),)](
            self.source,
            self.request_rows,
            self.verified_lens,
            self.computed,
            num_sampled,
            idx_mapping,
            block_table,
            self.final,
            self.commit_lens,
            self.boundary,
            self.boundary_lens,
            self.final_cols,
            self.request_slots,
            B=batch,
            ALIGN=align,
            HAS_IDX=idx_mapping is not None,
            NUM_SAMPLED=num_sampled.numel(),
            NUM_IDX=idx_mapping.numel() if idx_mapping is not None else 0,
            S=block_size or 1,
            BT_ROW_STRIDE=block_table.stride(0) if align else 0,
            BT_COL_STRIDE=block_table.stride(1) if align else 0,
            BT_WIDTH=block_table.shape[1] if align else 0,
            BLOCK=256,
        )
        kda_recoverssm_fold(
            self.context,
            self.source[:batch],
            self.final[:batch],
            self.commit_lens[:batch],
            record_rows=self.record_rows[:batch],
            boundary_state_indices=self.boundary[:batch] if align else None,
            boundary_lens=self.boundary_lens[:batch] if align else None,
        )

    def finish(self, batch, running_cols, accepted_out):
        if batch == 0:
            return
        _finish_commit[(triton.cdiv(batch, 256),)](
            self.request_slots,
            self.commit_lens,
            self.final_cols,
            running_cols,
            accepted_out,
            B=batch,
            ALIGN=running_cols is not None,
            BLOCK=256,
        )


class GLMRecoverSSMResources:
    """Persistent per-model storage, allocated inside model loading's budget."""

    def __init__(self, layers, config):
        self.layers = tuple(layers)
        self.layers_by_name = {layer.prefix: layer for layer in layers}
        self.record_capacity = config.scheduler_config.max_num_seqs
        self.spec_query_len = config.num_speculative_tokens + 1
        self.plan_capacity = self.record_capacity * self.spec_query_len
        first = layers[0]
        geometry = (
            first.local_num_heads,
            first.kda_lower_bound,
            first.get_state_dtype(),
        )
        if any(
            (layer.local_num_heads, layer.kda_lower_bound, layer.get_state_dtype())
            != geometry
            for layer in layers
        ):
            raise ValueError("GLM RecoverSSM requires homogeneous KDA layer geometry")
        self.context = KDARecoverSSMFoldContext.reserve(
            len(layers),
            first.local_num_heads,
            first.A_log.device,
            spec_query_len=self.spec_query_len,
            lower_bound=first.kda_lower_bound,
            with_conv=True,
        )
        self.plans = torch.empty(
            len(layers),
            len(_PLAN_FIELDS),
            self.plan_capacity,
            dtype=torch.int32,
            device=first.A_log.device,
        )
        self.identity = torch.arange(
            self.plan_capacity, dtype=torch.int32, device=first.A_log.device
        )
        self.neutral_accepted = torch.ones_like(self.identity)
        self.groups = {}
        self._layer_cursor = 0
        self._assigned_layers = set()
        self.generation = 0
        self.ready = False
        self.binding = False
        self.is_profiling = True
        self.captured_generation = None
        for layer in layers:
            layer._recoverssm_resources_ref = weakref.ref(self)
        record_bytes = sum(
            layer._recoverssm_correction.numel() * 4
            + layer._recoverssm_key_gate.numel() * 2
            for layer in layers
        )
        metadata_bytes = (
            self.context.metadata_nbytes
            + self.plans.numel() * 4
            + self.identity.numel() * 8
        )
        logger.info(
            "GLM RecoverSSM reserved %d layers, R=%d T=%d: records %.2f MiB, "
            "metadata %.2f MiB; speculative state blocks=0",
            len(layers),
            self.record_capacity,
            self.spec_query_len,
            record_bytes / 2**20,
            metadata_bytes / 2**20,
        )

    def begin_cache_binding(self, is_profiling):
        self.release_cache_binding()
        self.binding = True
        self.is_profiling = is_profiling

    def release_cache_binding(self):
        for group in self.groups.values():
            context = group.context
            context.state_pools = ()
            context.conv_pools = ()
            context.corrections = ()
            context.key_gates = ()
            context.bound = False
            group.bound_generation = -1
        self.groups.clear()
        self._layer_cursor = 0
        self._assigned_layers.clear()
        self.ready = False
        self.binding = False
        self.captured_generation = None
        self.generation += 1

    def before_layer_binding(self):
        if not self.binding and self.captured_generation is not None:
            raise RuntimeError(
                "Reinitialize KV cache and recapture graphs before rebinding GLM KDA storage"
            )

    def after_layer_binding(self):
        self.generation += 1

    def finish_cache_binding(self):
        for group in self.groups.values():
            group.bind()
        self.ready = True
        self.binding = False
        if not self.is_profiling:
            conv, state = self.layers[0].kv_cache
            logger.info(
                "GLM RecoverSSM cache bound: state=%s, conv=%s, layers=%d, "
                "generation=%d",
                state.dtype,
                conv.dtype,
                len(self.layers),
                self.generation,
            )

    def get_group(self, layer_names):
        key = tuple(layer_names)
        if key not in self.groups:
            if self._assigned_layers.intersection(key):
                raise RuntimeError("GLM RecoverSSM attention groups share a layer")
            stop = self._layer_cursor + len(key)
            if stop > len(self.layers):
                raise RuntimeError(
                    "GLM RecoverSSM attention groups overlap or exceed reservation"
                )
            group = KDARecoverSSMGroup(self, key, len(self.groups), self._layer_cursor)
            self.groups[key] = group
            self._layer_cursor = stop
            self._assigned_layers.update(key)
        return self.groups[key]

    def refresh_parameters(self):
        for layer in self.layers:
            layer._prepare_recoverssm_parameters()
        for group in self.groups.values():
            if group.context.bound:
                group.context.refresh_parameters(
                    [layer.A_log_view for layer in group.layers],
                    [layer.dt_bias_view for layer in group.layers],
                )


def initialize_recoverssm_resources(model, config):
    if not config.cache_config.use_kda_recoverssm:
        return
    layers = [
        layer
        for layer in model.modules()
        if getattr(layer, "use_recoverssm", False)
        and hasattr(layer, "_recoverssm_correction")
    ]
    if not layers:
        raise RuntimeError(
            "GLM RecoverSSM was enabled but no plugin KDA layer was found"
        )
    existing = getattr(layers[0], "_recoverssm_resources_ref", None)
    resources = existing() if existing is not None else None
    if resources is None:
        resources = GLMRecoverSSMResources(layers, config)
    model._recoverssm_resources = resources


def get_recoverssm_resources(model):
    return getattr(model, "_recoverssm_resources", None)


def begin_recoverssm_cache_binding(runner, is_profiling):
    resources = get_recoverssm_resources(runner.model)
    if resources is not None:
        resources.begin_cache_binding(is_profiling)
        state = runner.model_state
        state.recoverssm.clear()
        if state._align_mode:
            state._mamba_ctx = None
            state._mamba_spec = None
            state._mamba_group_ids = []
            state._mamba_state_copy_funcs = None


def finish_recoverssm_cache_binding(runner):
    resources = get_recoverssm_resources(runner.model)
    if resources is not None:
        resources.finish_cache_binding()
