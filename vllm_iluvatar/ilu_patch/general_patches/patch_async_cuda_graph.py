# SPDX-License-Identifier: Apache-2.0
"""Wire opt-in MR graph replay without the legacy device-wide completion wait."""

from __future__ import annotations

import os

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__, _apply, detail="MR asynchronous stateless CUDA graph replay"
    )


def _apply() -> None:
    setting = os.environ.get("VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY", "0")
    if setting not in ("0", "1"):
        raise ValueError("VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY must be 0 or 1")
    if setting == "0":
        return

    import torch
    import ixformer.cuda

    if ixformer.cuda.cuda_arch != "ivcore11":
        raise RuntimeError("Asynchronous CUDA graph replay is supported on MR")
    if os.environ.get("NCCL_GRAPH_REGISTER") != "0":
        raise ValueError("Asynchronous graph replay requires NCCL_GRAPH_REGISTER=0")
    if "NCCL_CUMEM_ENABLE" in os.environ:
        raise ValueError("Asynchronous graph replay requires NCCL_CUMEM_ENABLE unset")

    from ixformer.cuda import AsyncCUDAGraph

    torch.cuda.CUDAGraph = AsyncCUDAGraph
    torch.cuda.graphs.CUDAGraph = AsyncCUDAGraph
