"""Host identity and lifetime for one leased main-KV selection family.

All GPU preparation is performed by the attention caller. These keys inspect
tensor metadata only and never make a new step's cache contents ready.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import torch


def compact_owner_capacity(entries: int, physical_capacity: int) -> int | None:
    """Bound hash owners from selection geometry and the actual physical pool.

    Compact entry IDs are int32. Calls outside that representation keep the
    existing reader rather than narrowing a physical key or a selection ID.
    This resource bound never inspects the current device selections.
    """
    values = []
    for value in (entries, physical_capacity):
        try:
            size = operator.index(value)
        except TypeError as error:
            raise ValueError(
                "Compact geometry must use nonnegative integers"
            ) from error
        if isinstance(value, bool) or not 0 <= size <= (1 << 64) - 1:
            raise ValueError("Compact geometry must use nonnegative integers")
        values.append(size)
    entries, physical_capacity = values
    if not entries or not physical_capacity or entries > (1 << 31) - 1:
        return None
    bound = min(entries, physical_capacity)
    return 1 << (2 * bound - 1).bit_length()


@dataclass(frozen=True)
class TensorLayoutKey:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    alignment: int

    @classmethod
    def of(cls, tensor):
        return cls(
            tuple(tensor.shape),
            tensor.stride(),
            tensor.dtype,
            tensor.device,
            # Native layout dispatch distinguishes alignments up to64bytes.
            # Allocation identity belongs only to actual main/mapping views.
            tensor.data_ptr() % 64,
        )


@dataclass(frozen=True)
class TensorViewKey:
    layout: TensorLayoutKey
    storage_address: int
    storage_offset: int

    @classmethod
    def of(cls, tensor):
        return cls(
            TensorLayoutKey.of(tensor),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
        )


@dataclass(frozen=True)
class RestoreFamilyKey:
    owner: str
    index_source_layer_id: int
    payload: TensorViewKey
    scales: TensorViewKey
    workspace: TensorViewKey
    query: TensorLayoutKey
    output: TensorLayoutKey
    indices: TensorLayoutKey
    format_identity: tuple
    mapping_layout: tuple
    reader_layouts: tuple
    restore_start: int
    restore_tokens: int


def family_key(
    *,
    owner,
    index_source_layer_id,
    payload,
    scales,
    workspace,
    query,
    output,
    indices,
    format,
    mapping_layout=(),
    reader_layouts=(),
    restore_start,
    restore_tokens,
):
    """Key actual main views and reader sizing layouts, excluding Q contents.

    Per-layer SWA ownership/addresses are excluded: selected rows contain only
    main KV. Reader layouts include its geometry/alignment for exact scratch
    sizing. Equal layouts determine the same native scratch requirement; the
    first consumer queries it and every native reader checks its supplied span.
    """
    from ixformer.inference.functions.deepseek.quantized_kv_format import (
        resolve_quant_format,
    )

    fmt = resolve_quant_format(format)
    if (
        workspace.dtype is not torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
    ):
        raise ValueError("Expected the actual yielded contiguous uint8 workspace")
    if payload.dtype is not torch.uint8 or scales.dtype is not torch.float16:
        raise ValueError("Main KV needs uint8 payload and FP16 scales")
    if (
        query.ndim != 3
        or output.shape != query.shape
        or indices.ndim != 2
        or query.shape[0] != indices.shape[0]
        or query.dtype is not torch.bfloat16
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected full packed BF16 queries and integer main indices")
    if not owner or index_source_layer_id < 0:
        raise ValueError("Selected plans need actual owner and index-family identities")
    if (
        restore_start < 0
        or restore_tokens < 0
        or restore_start + restore_tokens > query.shape[0]
    ):
        raise ValueError("Restore range exceeds the full packed-token batch")
    if (
        len(
            {
                tensor.device
                for tensor in (payload, scales, workspace, query, output, indices)
            }
        )
        != 1
    ):
        raise ValueError("Selected plan tensors must share a device")
    return RestoreFamilyKey(
        owner,
        index_source_layer_id,
        TensorViewKey.of(payload),
        TensorViewKey.of(scales),
        TensorViewKey.of(workspace),
        TensorLayoutKey.of(query),
        TensorLayoutKey.of(output),
        TensorLayoutKey.of(indices),
        (fmt.name, fmt.bits, fmt.group_size, 1, "page_row_payload+page_row_group"),
        mapping_layout,
        reader_layouts,
        restore_start,
        restore_tokens,
    )


@dataclass(frozen=True)
class CompressedFamilySelection:
    indices: torch.Tensor
    lengths: torch.Tensor
    plan: Any
    scratch_bytes: int
    compact_plan: Any = None
    _query_views: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if self.plan is not None and self.compact_plan is not None:
            raise ValueError("A compressed family may own only one prepared plan")

    def for_queries(self, token_slice):
        """Reuse at most the old/new views while this prepared family is active.

        Views alias the same GPU map and retained rows. A new preparation
        creates a new selection, so host view reuse never skips GPU work or
        preserves a previous forward's data. Other slices remain supported
        without growing the bounded cache. Complete linear new-query views
        use the SDK's affine proof; partial or owner-table plans stay mapped.
        Compact views keep their separate SDK type and device query flags.
        """
        key = (token_slice.start, token_slice.stop, token_slice.step)
        if key in self._query_views:
            return self._query_views[key]
        view = None
        if self.compact_plan is not None:
            view = self.compact_plan.for_queries(token_slice)
        elif self.plan is not None:
            plan = self.plan
            view = plan.for_queries(token_slice)
            if (
                plan.mode == 0
                and plan.selected_flags is None
                and plan.row_map.shape[1] == 1
                and plan.rows.shape[0] >= plan.restore_tokens * plan.row_map.shape[-1]
            ):
                start, stop, step = token_slice.indices(plan.row_map.shape[0])
                if (
                    step == 1
                    and start >= plan.restore_start
                    and stop <= plan.restore_start + plan.restore_tokens
                    and stop > start
                ):
                    view = view.as_affine()
        result = (
            self.indices[token_slice],
            self.lengths[token_slice],
            view,
        )
        if len(self._query_views) < 2:
            self._query_views[key] = result
        return result


@dataclass(frozen=True)
class PreparationTicket:
    family: RestoreFamilyKey
    generation: int


class RestorePlanTracker:
    """One active family; all its plan views occupy the same worker arena."""

    def __init__(self):
        self.step_epoch = self.main_write_epoch = self.generation = 0
        self._family = self._selection = self._pending = None

    def invalidate(self):
        self.generation += 1
        self._family = self._selection = self._pending = None

    def begin_forward(self):
        # Identical layouts on a new eager/capture forward still run prepare.
        self.step_epoch += 1
        self.invalidate()

    def main_cache_written(self):
        self.main_write_epoch += 1
        self.invalidate()

    def lookup(self, family):
        return self._selection if family == self._family else None

    def begin_prepare(self, family):
        self.invalidate()
        self._pending = PreparationTicket(family, self.generation)
        return self._pending

    def publish(self, ticket, selection):
        if ticket != self._pending or ticket.generation != self.generation:
            raise RuntimeError("Selected family preparation became stale")
        if selection is None:
            raise ValueError("Cannot publish an absent compressed family")
        self._family, self._selection, self._pending = ticket.family, selection, None
