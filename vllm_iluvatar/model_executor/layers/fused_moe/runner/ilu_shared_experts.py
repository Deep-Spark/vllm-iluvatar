# SPDX-License-Identifier: Apache-2.0
"""Iluvatar SharedExperts override.

Baseline : community ``runner/shared_experts.py`` @ v0.31.0 (db9527a468)
Delta    : override ``_disable_shared_experts_overlap`` to always allow
           multi-stream overlap (controlled by VLLM_DISABLE_SHARED_EXPERTS_STREAM
           env var instead of the community eplb / flashinfer checks).
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.utils.torch_utils import aux_stream, current_stream


class IluSharedExperts(SharedExperts):
    """Iluvatar override — always allow shared-experts overlap."""

    # [ILU override] _disable_shared_experts_overlap — delta: community checks
    # eplb backend + flashinfer conditions to disable overlap; Iluvatar
    # controls this solely via VLLM_DISABLE_SHARED_EXPERTS_STREAM env var.
    @property
    def _disable_shared_experts_overlap(self) -> bool:
        return False

    def begin_dispatch(self, inputs, *, overlap: bool):
        """Start shared work before dispatch; caller keeps inputs through join."""
        assert self._output[self._output_idx] is None
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if x.shape[0] == 0:
            dtype = inputs[2] if isinstance(inputs, tuple) else x.dtype
            self._output[self._output_idx] = torch.empty_like(x, dtype=dtype)
            return None
        stream = aux_stream() if overlap else None
        if stream is None:
            self._output[self._output_idx] = self._layer(inputs)
        else:
            stream.wait_stream(current_stream())
            with torch.cuda.stream(stream):
                self._output[self._output_idx] = self._layer(inputs)
        return stream

    def end_dispatch(self, stream):
        """Join immediately after dispatch and consume the normal output slot."""
        if stream is not None:
            current_stream().wait_stream(stream)
        return self.output
