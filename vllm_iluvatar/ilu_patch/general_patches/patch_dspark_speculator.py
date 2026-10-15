# SPDX-License-Identifier: Apache-2.0
"""Wire Iluvatar vocab-sharded DSpark sampling.

Draft ``enforce_eager`` is applied by ``patch_draft_cudagraph``, which wraps
``DFlashSpeculator.init_cudagraph_manager``. DSpark inherits that method.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSpark vocab-sharded draft logits",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from vllm_iluvatar.v1.worker.gpu.spec_decode import dspark as impl

    if getattr(DSparkSpeculator, "_ilu_dspark_sharded_logits_patch", False):
        return

    # Keep the community body for top-k / models without local-head APIs.
    DSparkSpeculator._ilu_community_sample_sequential = (
        DSparkSpeculator._sample_sequential
    )
    DSparkSpeculator._sample_sequential = impl.sample_sequential
    DSparkSpeculator._ilu_dspark_sharded_logits_patch = True
