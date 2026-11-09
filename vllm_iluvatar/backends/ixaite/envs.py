# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Environment compatibility owned by the ixaite process."""

import os

from vllm_iluvatar.envs import _TRUE_VALS, _get


# NCCL reads this when the communicator is created. Call it after --nnodes
# is known and before that init. The variable is process-wide.
def configure_peer_access(nnodes: int) -> None:
    # A missing count stays on the single-node path, which keeps peer access.
    if int(nnodes or 1) <= 1:
        return
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")


# Mainline's environment readers and defaults stay unchanged in baseline mode.
def apply() -> None:
    import vllm.envs as envs

    # Peer access waits for configure_peer_access. A single node keeps the
    # GPU peer link; forcing it off here would stage the payload through
    # host memory before the node count is known.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # Both names share one reader so explicit canonical values keep precedence.
    gdn_reader = lambda: (
        _get("VLLM_ILU_GDN_ATTN_BACKEND", "VLLM_GDN_ATTN_BACKEND", "1").lower()
        in _TRUE_VALS
    )
    envs.environment_variables.update(
        {
            "VLLM_ILU_GDN_ATTN_BACKEND": gdn_reader,
            "VLLM_GDN_ATTN_BACKEND": gdn_reader,
            "VLLM_ILU_ATTN_OPT_LEVEL": lambda: int(
                _get("VLLM_ILU_ATTN_OPT_LEVEL", "VLLM_ATTN_OPT_LEVEL", "0")
            ),
            "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH": lambda: _get(
                "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
                "VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
                "",
            ),
            "VLLM_ILU_USE_INT8_MLA": lambda: (
                _get("VLLM_ILU_USE_INT8_MLA", "VLLM_USE_INT8_MLA", "0").lower()
                in ("1", "true")
            ),
        }
    )
