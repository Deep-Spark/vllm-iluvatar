# SPDX-License-Identifier: Apache-2.0
"""Forward persistent ``positions`` when building draft attention metadata.

Corresponds to: vllm/v1/worker/gpu/spec_decode/speculator.py
Target   : DraftModelSpeculator._build_attn_metadata
Baseline : v0.30.0
Why      : community omits ``positions``. Iluvatar V4.1 needs them.
How      : setattr the copied method onto ``DraftModelSpeculator``.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Draft attention metadata forwards persistent positions",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    from vllm_iluvatar.v1.worker.gpu.spec_decode import draft_attn_metadata as impl

    if getattr(DraftModelSpeculator, "_ilu_draft_attn_positions_patch", False):
        return

    p.check_signatures(
        DraftModelSpeculator,
        impl,
        ["_build_attn_metadata"],
        strict=True,
    )
    # Class methods only need setattr on the class.
    DraftModelSpeculator._build_attn_metadata = impl._build_attn_metadata
    DraftModelSpeculator._ilu_draft_attn_positions_patch = True
