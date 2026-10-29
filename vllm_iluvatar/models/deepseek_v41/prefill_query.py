# SPDX-License-Identifier: Apache-2.0
"""Host-described retained queries with unchanged full-token KV ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def producer_query_compaction_enabled(additional_config) -> bool:
    policy = additional_config.get("ilu_prefill", {})
    if not isinstance(policy, dict):
        raise ValueError("additional_config.ilu_prefill must be a mapping")
    enabled = policy.get("producer_query_compaction", False)
    if type(enabled) is not bool:
        raise ValueError("producer_query_compaction must be a boolean")
    return enabled


@dataclass(frozen=True)
class ProducerQuerySelection:
    """Ordered host row spans; tensor slicing never reads device metadata.

    Each span addresses the original token carrier. Packed query order is the
    concatenation of those spans, while cache inserts retain the full carrier.
    """

    spans: tuple[tuple[int, int], ...]
    count: int

    @classmethod
    def from_rows(cls, rows: tuple[int, ...], total: int):
        spans: list[tuple[int, int]] = []
        previous = -1
        for row in rows:
            if type(row) is not int or not 0 <= row < total or row <= previous:
                raise ValueError(
                    "Retained query rows must be sorted, unique, and in range"
                )
            if spans and row == spans[-1][1]:
                spans[-1] = (spans[-1][0], row + 1)
            else:
                spans.append((row, row + 1))
            previous = row
        return cls(tuple(spans), len(rows))

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.spans:
            return tensor[:0]
        if len(self.spans) == 1:
            begin, end = self.spans[0]
            return tensor[begin:end]
        import torch

        return torch.cat([tensor[begin:end] for begin, end in self.spans], dim=0)

    def select_activation(self, activation):
        if isinstance(activation, tuple):
            values, scales, dtype = activation
            return self.select(values), self.select(scales), dtype
        return self.select(activation)

    def scatter(self, packed: torch.Tensor, output: torch.Tensor) -> None:
        offset = 0
        for begin, end in self.spans:
            count = end - begin
            output[begin:end].copy_(packed[offset : offset + count])
            offset += count

    def intersections(self, token_start: int, token_end: int):
        """Yield original row bounds and their offset in packed query order."""
        offset = 0
        for begin, end in self.spans:
            lower, upper = max(begin, token_start), min(end, token_end)
            if lower < upper:
                yield lower, upper, offset + lower - begin
            offset += end - begin
