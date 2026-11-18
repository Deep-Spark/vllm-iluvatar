# SPDX-License-Identifier: Apache-2.0
"""Include Iluvatar sparse backends in upstream's local warmup selection."""

from vllm_iluvatar.ilu_patch import general_patch as p


def _apply() -> None:
    from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as warmup

    warmup._DEEPSEEK_V4_SPARSE_MLA_BACKENDS |= {
        "ILU_DEEPSEEK_SPARSE_SWA",
        "ILU_DEEPSEEK_SPARSE_SWA_FP8",
        "FLASHMLA_SPARSE_DSV4_FP8",
    }


def register() -> None:
    p.register_once(__name__, _apply, detail="register Iluvatar sparse MLA backends")
