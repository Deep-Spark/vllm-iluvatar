# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``AutoRegressiveSpeculator`` Option B override.

Corresponds to: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.v1.worker.gpu.spec_decode.autoregressive.speculator``; this
module only performs register-time wiring.

Target   : vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.
           AutoRegressiveSpeculator.__init__ / capture / _multi_step_decode
           (+ new _generate_draft_graph_part / _generate_draft_tail methods)
Baseline : v0.31.0 (db9527a468)
Why      : ixfd TP all_gather (inside sample_draft/compute_logits) is not
           CUDA-graph-safe; capturing it into the draft decode graph corrupts
           the shared ixfd communicator and produces garbage output. all_reduce
           is graph-safe (in-place), so only the draft — which captured the
           whole _generate_draft — broke; the target was never affected.
How      : wrap __init__ to add two persistent graph-output buffers; wrap
           capture to route the non-fused decode_fn to the graph-safe part via
           a temporary instance-attr swap; wrap _multi_step_decode to call the
           eager tail after each run_fullgraph replay; add the two new methods.
           No long upstream methods are copied (drift-free).
Scope    : non-fused draft decode on Iluvatar (MTP/Eagle). Fused path untouched.
Remove   : when ixfd exposes a graph-safe all_gather (stream-bound or
           buffer-registered) so the collective can stay in the captured graph.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar AutoRegressiveSpeculator Option B override."""
    p.register_once(
        __name__,
        _apply,
        detail="AutoRegressiveSpeculator Option B (graph-safe draft decode)",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )

    from vllm_iluvatar.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _generate_draft_graph_part,
        _generate_draft_tail,
        make_capture,
        make_init,
        make_multi_step_decode,
    )

    orig_init = AutoRegressiveSpeculator.__init__
    orig_capture = AutoRegressiveSpeculator.capture
    orig_msd = AutoRegressiveSpeculator._multi_step_decode

    wrapped_init = make_init(orig_init)
    wrapped_capture = make_capture(orig_capture)
    wrapped_msd = make_multi_step_decode(orig_msd)

    p.check_signatures(
        AutoRegressiveSpeculator,
        SimpleNamespace(
            __init__=wrapped_init,
            capture=wrapped_capture,
            _multi_step_decode=wrapped_msd,
        ),
        ["__init__", "capture", "_multi_step_decode"],
        strict=True,
    )

    # [ILU override] __init__ - delta: allocate two persistent graph-output
    # buffers (_cg_out_last_hidden / _cg_out_hidden) for the captured part.
    AutoRegressiveSpeculator.__init__ = wrapped_init
    # [ILU override] capture - delta: non-fused decode_fn routes to
    # _generate_draft_graph_part (graph-safe) instead of _generate_draft.
    AutoRegressiveSpeculator.capture = wrapped_capture
    # [ILU override] _multi_step_decode - delta: each FULL replay is followed
    # by the eager tail _generate_draft_tail (all_gather + argmax + update).
    AutoRegressiveSpeculator._multi_step_decode = wrapped_msd
    AutoRegressiveSpeculator._generate_draft_graph_part = _generate_draft_graph_part
    AutoRegressiveSpeculator._generate_draft_tail = _generate_draft_tail
