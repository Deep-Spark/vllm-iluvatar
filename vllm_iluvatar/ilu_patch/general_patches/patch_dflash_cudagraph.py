# SPDX-License-Identifier: Apache-2.0
"""Wire DFlash capture that forwards dummy positions for V4.1.

Corresponds to: vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py
Target   : _prepare_dflash_inputs_to_capture
Baseline : v0.30.0
Why      : community omits ``positions`` / ``seq_lens_cpu_upper_bound``.
How      : rebind the module-level helper. DFlashCudaGraphManager imports the
           name by value, so ``rebind_module_func`` is required.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DFlash CUDA-graph capture forwards positions",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as community

    from vllm_iluvatar.v1.worker.gpu.spec_decode import dflash_cudagraph as impl

    p.check_signatures(
        community,
        impl,
        ["_prepare_dflash_inputs_to_capture"],
        strict=True,
    )
    p.rebind_module_func(
        community,
        "_prepare_dflash_inputs_to_capture",
        impl._prepare_dflash_inputs_to_capture,
    )
