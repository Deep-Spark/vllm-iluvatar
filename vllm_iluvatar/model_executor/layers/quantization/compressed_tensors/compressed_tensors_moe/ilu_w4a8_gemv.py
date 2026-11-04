# SPDX-License-Identifier: Apache-2.0
"""Per-call W4A8 NN implementation policy, shared by both expert paths."""

from functools import cache

import ixformer.inference.functions as ixf
import torch
from ixformer.inference.backend_registry import registry

from vllm_iluvatar import envs


@cache
def _nn_perf_enabled() -> bool:
    """Snapshot the worker-wide opt-in at first NN GEMV; restart to change it."""
    return envs.VLLM_ILU_MOE_W4A8_GEMV_PERF


def w4a8_group_gemv(**kwargs):
    """Run using the actual packed layout; never retry a failed computation.

    NN is application-managed: explicit backend selection takes precedence
    over registry overrides. Other layouts retain normal registry selection.
    G1/G2 are planned independently and all optional features are preserved.
    NN defaults to vendor. Set VLLM_ILU_MOE_W4A8_GEMV_PERF=1 before
    worker startup to enable supported perf calls; this does not repack weights.
    The opt-in is frozen at first NN call, including for eager draft calls,
    so existing captured graphs and eager execution share the same policy.
    """
    backend = None
    if kwargs.get("format", 0) == 0:
        available = registry.available_backends("moe_w4a8_group_gemv")
        if (
            _nn_perf_enabled()
            and "ixformer_perf" in available
            and ixf.moe_w4a8_group_gemv_perf_supported(**kwargs)
        ):
            backend = "ixformer_perf"
        elif "ixinfer" in available:
            # A failed perf query may mean invalid cross-device operands,
            # which the vendor bridge does not comprehensively validate.
            owner = kwargs.get("input")
            if isinstance(owner, torch.Tensor):
                for name, tensor in kwargs.items():
                    if (
                        isinstance(tensor, torch.Tensor)
                        and tensor.device != owner.device
                    ):
                        raise ValueError(
                            f"NN W4A8 GEMV {name} must share the input device"
                        )
            # The vendor bridge supports NN bias/remapping and validates its
            # shape/version/group contract before launching. It receives the
            # original kwargs, including dst_to_src and the NN-packed weight.
            backend = "ixinfer"
        else:
            raise RuntimeError(
                "No available W4A8 GEMV backend supports this NN call "
                "under the configured perf opt-in"
            )
    return ixf.moe_w4a8_group_gemv(**kwargs, backend=backend)
