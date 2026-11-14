# SPDX-License-Identifier: Apache-2.0
"""Pack immutable CPU tensor views into Torch-owned pinned storage.

The installed host caching allocator rounds each request to a power of two.
Combining whole planes reduces that slack without changing their shapes or
strides. This module never registers memory, replaces an allocator, or owns
an external pointer: ordinary Torch tensors retain and release every storage.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class TensorPlacement:
    arena: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class PinnedTensorLayout:
    arena_bytes: tuple[int, ...]
    placements: tuple[TensorPlacement, ...]

    @property
    def storage_bytes(self) -> int:
        return sum(self.arena_bytes)


def _positive_integer(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Tensor storage dimensions must be positive integers")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("Tensor storage dimensions must be positive integers") from exc
    if value <= 0:
        raise ValueError("Tensor storage dimensions must be positive integers")
    return value


def plan_pinned_tensors(sizes, alignment: int = 64) -> PinnedTensorLayout:
    """Place whole planes deterministically, preserving original tensor order."""
    alignment = _positive_integer(alignment)
    if alignment & (alignment - 1):
        raise ValueError("Tensor storage alignment must be a power of two")
    sizes = tuple(_positive_integer(size) for size in sizes)
    if not sizes:
        return PinnedTensorLayout((), ())
    padded = tuple((size + alignment - 1) // alignment * alignment for size in sizes)
    capacities, used, placements = [], [], {}
    for index in sorted(range(len(sizes)), key=lambda i: (-padded[i], i)):
        fits = [
            arena for arena, capacity in enumerate(capacities)
            if capacity - used[arena] >= padded[index]
        ]
        if fits:
            arena = min(fits, key=lambda a: (capacities[a] - used[a], a))
        else:
            arena = len(capacities)
            capacities.append(1 << (padded[index] - 1).bit_length())
            used.append(0)
        placements[index] = TensorPlacement(arena, used[arena], sizes[index])
        used[arena] += padded[index]

    # Some mixtures pack better into a single larger bucket. Choose by bytes,
    # then allocation count; neither strategy depends on model/test shapes.
    combined = 1 << (sum(padded) - 1).bit_length()
    if (combined, 1) < (sum(capacities), len(capacities)):
        offset = 0
        single = []
        for size, width in zip(sizes, padded):
            single.append(TensorPlacement(0, offset, size))
            offset += width
        return PinnedTensorLayout((combined,), tuple(single))
    return PinnedTensorLayout(
        tuple(capacities), tuple(placements[index] for index in range(len(sizes)))
    )


def allocate_pinned_tensor_views(shapes, *, pin_memory: bool):
    """Create zeroed INT8 planes with standard Torch storage ownership.

    Each returned view retains its underlying Torch allocation even if every
    other view is destroyed. Callers must keep their views alive until their
    asynchronous transfers finish, just as for independently allocated planes.
    """
    shapes = tuple(tuple(_positive_integer(n) for n in shape) for shape in shapes)
    if any(not shape for shape in shapes):
        raise ValueError("Tensor storage shapes must have at least one dimension")
    layout = plan_pinned_tensors(prod(shape) for shape in shapes)
    arenas = [
        torch.zeros(size, dtype=torch.int8, device="cpu", pin_memory=pin_memory)
        for size in layout.arena_bytes
    ]
    tensors = [
        arenas[place.arena].narrow(0, place.offset, place.nbytes).view(shape)
        for shape, place in zip(shapes, layout.placements)
    ]
    return tensors, layout
