# SPDX-License-Identifier: Apache-2.0
"""Early import hooks for plugin-owned FLy configuration and MRV2 verification.

Configuration must be extended before EngineArgs and VllmConfig capture its
schema. The verifier class must be replaced before ModelRunnerV2 imports it.
General-plugin registration is too late for those two consumers; register
lightweight post-import callbacks from the existing startup hook instead.
"""

from types import ModuleType

from vllm_iluvatar.ilu_patch import import_hook

_registered = False


def _patch_speculative_config(module: ModuleType) -> None:
    from vllm_iluvatar.config.speculative import SpeculativeConfig

    module.SpeculativeConfig = SpeculativeConfig


def _patch_rejection_sampler(module: ModuleType) -> None:
    from vllm_iluvatar.v1.worker.gpu.spec_decode.rejection_sampler import (
        RejectionSampler,
    )

    module.RejectionSampler = RejectionSampler


def register() -> None:
    """Register callbacks without importing vLLM, Torch or Triton."""
    global _registered
    if _registered:
        return
    _registered = True
    import_hook.register("vllm.config.speculative", _patch_speculative_config)
    import_hook.register(
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler", _patch_rejection_sampler
    )
