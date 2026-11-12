# SPDX-License-Identifier: Apache-2.0
"""Capture DSpark's backbone while its distributed heads remain eager."""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)


class DSparkBackboneGraph:
    """Keep graph outputs alive and sample only the local request rows."""

    def __init__(self, speculator: Any) -> None:
        self.speculator = speculator
        self.outputs: dict[int, Any] = {}
        self.capturing = False
        self.num_reqs: int | None = None
        self._install_methods()

    def bind_manager(self, manager: Any) -> None:
        # KV-memory profiling recreates the manager before the final capture.
        self.outputs.clear()
        replay = manager.run_fullgraph
        reported = False

        @wraps(replay)
        def run_fullgraph(desc):
            nonlocal reported
            if self.num_reqs is None:
                raise RuntimeError("Backbone replay requires a proposal scope")
            hidden = self.outputs[desc.num_tokens]
            if self.num_reqs * self.speculator.num_query_per_req > hidden.shape[0]:
                raise RuntimeError("Local draft rows exceed the captured backbone")
            replay(desc)
            self.speculator._sample_sequential(self.num_reqs, hidden)
            if not reported:
                logger.info(
                    "DSpark backbone graph replay: tokens=%d local_requests=%d; "
                    "head sampled eagerly.",
                    desc.num_tokens,
                    self.num_reqs,
                )
                reported = True

        manager.run_fullgraph = run_fullgraph

    def _install_methods(self) -> None:
        spec = self.speculator
        original_capture = spec.capture
        original_generate = spec._generate_draft
        original_propose = spec.propose

        @wraps(original_capture)
        def capture(*args, **kwargs):
            if self.capturing:
                raise RuntimeError("DSpark backbone capture must not nest")
            self.outputs.clear()
            self.capturing = True
            try:
                return original_capture(*args, **kwargs)
            finally:
                self.capturing = False

        @wraps(original_generate)
        def generate(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            if not self.capturing:
                return original_generate(
                    num_reqs,
                    num_tokens_padded,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode,
                )
            self.outputs[num_tokens_padded] = spec._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )

        @wraps(original_propose)
        def propose(*args, **kwargs):
            if self.num_reqs is not None:
                raise RuntimeError("DSpark proposals must not overlap")
            batch = kwargs.get("input_batch") if not args else args[0]
            self.num_reqs = batch.num_reqs
            try:
                return original_propose(*args, **kwargs)
            finally:
                self.num_reqs = None

        spec.capture = capture
        spec._generate_draft = generate
        spec.propose = propose
