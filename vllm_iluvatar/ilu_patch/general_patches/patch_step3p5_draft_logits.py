# SPDX-License-Identifier: Apache-2.0
"""Pass the draft step into Step3.5 MTP logits.

Corresponds to: vllm/v1/worker/gpu/spec_decode/speculator.py
Target   : DraftModelSpeculator.sample_draft
Baseline : v0.31.0 (341bf7d4d1)
Why      : V2 calls ``compute_logits(hidden)``. The default ``spec_step_idx``
           is 0, so every draft step scores MTP layer 0. Step3.5 MTP-3 uses
           the draft step to select layers 0, 1, and 2.
How      : wrap ``DraftModelSpeculator.sample_draft``. When the draft model sets
           ``ilu_pass_spec_step_idx``, call ``compute_logits`` with that step.
           Models without the flag use the community call unchanged.
Scope    : ``IluStep3p5MTP`` only. DeepSeek and other MTP models also accept
           ``spec_step_idx`` and must not start rotating because of this wrap.
Remove   : when community ``sample_draft`` passes the draft step itself.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Step3.5 draft logits receive spec_step_idx",
    )


def _step_of(draft_step: Any) -> int:
    return int(draft_step.reshape(-1)[0].item())


def _step_for(self: Any, draft_step: Any) -> int:
    """Resolve the current draft step without breaking CUDA graph capture.

    ``.item()`` is a host sync and is illegal while a stream is being
    captured. The autoregressive draft capture runs the steps 0..N-1 in
    order within ONE capture session, so while capturing the step is
    derived from a session-local counter (reset on the eager->capturing
    transition); outside capture the host sync is legal and the true step
    is read from the device tensor.
    """
    capturing = torch.cuda.is_current_stream_capturing()
    prev = getattr(self, "_ilu_prev_capturing", False)
    if capturing:
        if not prev:
            self._ilu_capture_step = 0
        self._ilu_prev_capturing = True
        num_steps = getattr(self, "num_speculative_steps", 3)
        step = self._ilu_capture_step % num_steps
        self._ilu_capture_step = step + 1
        return step
    self._ilu_prev_capturing = False
    return _step_of(draft_step)


def _wrap_sample_draft(original: Callable[..., Any]) -> Callable[..., Any]:
    def sample_draft(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = getattr(self, "model", None)
        if model is None or not getattr(model, "ilu_pass_spec_step_idx", False):
            return original(self, *args, **kwargs)
        draft_step = kwargs.get("draft_step")
        if draft_step is None and len(args) >= 6:
            draft_step = args[5]
        if draft_step is None:
            return original(self, *args, **kwargs)
        orig_fn = model.compute_logits
        step = _step_for(self, draft_step)

        def compute_logits(
            hidden_states: Any,
            spec_step_idx: int = 0,
            *a: Any,
            **k: Any,
        ):
            del spec_step_idx
            if "spec_step_idx" in inspect.signature(orig_fn).parameters:
                return orig_fn(hidden_states, *a, spec_step_idx=step, **k)
            return orig_fn(hidden_states, *a, **k)

        model.compute_logits = compute_logits
        try:
            return original(self, *args, **kwargs)
        finally:
            model.compute_logits = orig_fn

    return sample_draft


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    # [ILU override] DraftModelSpeculator.sample_draft — delta: pass
    # spec_step_idx only when the draft sets ilu_pass_spec_step_idx.
    DraftModelSpeculator.sample_draft = _wrap_sample_draft(  # type: ignore[method-assign]
        DraftModelSpeculator.sample_draft
    )