# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar speculative rejection-sampler chunk limit.

Corresponds to: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.v1.worker.gpu.spec_decode.rejection_sampler.MAX_CHUNK_BYTES
Baseline : main_6e448d0
Why      : community sizes chunks only from the FP32 logits buffer. On Iluvatar
           the top-k/top-p path falls back to PyTorch sorting, which also
           materializes sorted values and int64 indices.
How      : override ``MAX_CHUNK_BYTES`` at register time using the value
           registered in ``vllm_iluvatar.envs``.
Scope    : V2 speculative rejection-sampler verification chunks on Iluvatar.
Remove   : when the Iluvatar sampler no longer needs the PyTorch sort fallback,
           or community chunk sizing accounts for the extra sort buffers.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)


def register() -> None:
    """Register the V2 speculative rejection-sampler chunk limit override."""
    p.register_once(
        __name__,
        _apply,
        detail="spec decode rejection sampler chunk limit",
    )


def _apply() -> None:
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rejection_sampler
    from vllm import envs

    max_chunk_bytes = envs.VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES
    rejection_sampler.MAX_CHUNK_BYTES = max_chunk_bytes
    logger.debug(
        "ILU: set spec decode rejection sampler MAX_CHUNK_BYTES=%d",
        max_chunk_bytes,
    )
