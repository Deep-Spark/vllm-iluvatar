# SPDX-License-Identifier: Apache-2.0
"""Wire V2 DP decode-only publication onto the 0.31 sync.

Corresponds to: vllm/v1/worker/gpu/dp_utils.py
                vllm/v1/worker/gpu/cudagraph_utils.py
Baseline : v0.31.0 (db9527a468)
Why      : hybrid EP selects the LL decode backend from
           ``DPMetadata.all_ranks_decode_only``. Keep the community V2 sync and
           wrap it so the per-rank ``uniform_decode`` vote is all-reduced into
           ``set_last_all_ranks_decode_only``. Wrap ``CudaGraphManager.capture``
           so each captured descriptor publishes the same bit.
Scope    : V2 DP+EP MoE on Iluvatar.
Remove   : when community DP sync writes all-ranks decode-only itself.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="V2 DP decode-only sync and capture propagation",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu import dp_utils as dp_mod
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    import vllm_iluvatar.v1.worker.gpu.dp_utils as impl

    if impl._ORIGINAL_SYNC is None:
        p.check_signatures(
            dp_mod,
            impl,
            ["sync_cudagraph_and_dp_padding"],
            strict=True,
        )
        impl._ORIGINAL_SYNC = dp_mod.sync_cudagraph_and_dp_padding
        p.rebind_module_func(
            dp_mod,
            "sync_cudagraph_and_dp_padding",
            impl.sync_cudagraph_and_dp_padding,
        )
    if getattr(impl, "_ORIGINAL_DISPATCH", None) is None:
        p.check_signatures(
            dp_mod,
            impl,
            ["dispatch_cg_and_sync_dp"],
            strict=True,
        )
        impl._ORIGINAL_DISPATCH = dp_mod.dispatch_cg_and_sync_dp
        p.rebind_module_func(
            dp_mod,
            "dispatch_cg_and_sync_dp",
            impl.dispatch_cg_and_sync_dp,
        )

    if getattr(CudaGraphManager, "_ilu_decode_only_capture_patch", False):
        return
    CudaGraphManager.capture = impl.make_capture(CudaGraphManager.capture)
    CudaGraphManager._ilu_decode_only_capture_patch = True
