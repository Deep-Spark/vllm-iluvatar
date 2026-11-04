# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/kv_offload/cpu/spec.py

Target   : CPUOffloadingSpec.__init__, CPUOffloadingSpec.create_worker
Baseline : v0.31.0 (341bf7d4d124)
Why      : Preserve paired KV pages and rank-local CPU offload storage on CoreX.
How      : Wrappers delegate unchanged paths to the community implementation.
Scope    : DeepSeek V4.1 paired / non-uniform PP adapters.
Remove   : When community provides equivalent geometry and storage ownership.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.models.deepseek_v41.offloading import (
    _GEOMETRY_ATTR,
    _RANK_SAFE_REQUIRED_ATTR,
)

logger = init_logger(__name__)

_ORIGINAL_SPEC_INIT = None
_ORIGINAL_CREATE_WORKER = None


# [ILU override] CPUOffloadingSpec.__init__ — delta: propagate rank-local PP geometry.
def _ilu_cpu_offloading_spec_init(self, config) -> None:
    assert _ORIGINAL_SPEC_INIT is not None
    geometry = getattr(config, _GEOMETRY_ATTR, None)
    needs_geometry = getattr(config, _RANK_SAFE_REQUIRED_ATTR, False)
    if ilu_envs.VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED and (
        geometry is None or needs_geometry is not True
    ):
        raise RuntimeError(
            "Required DSV4.1 PP CPU offload spec geometry/marker is missing"
        )
    if needs_geometry and geometry is None:
        raise RuntimeError("DSV4.1 PP CPU offload geometry was not propagated")
    _ORIGINAL_SPEC_INIT(self, config)
    if geometry is None:
        return
    if config.canonical_layout:
        raise ValueError(
            "DSV4.1 non-uniform PP CPU offload uses rank-local direct pages; "
            "canonical_layout is not supported."
        )
    if len(geometry) != config.parallel.world_size:
        raise ValueError(
            "PP CPU offload geometry does not match world size: "
            f"{len(geometry)} vs {config.parallel.world_size}."
        )
    rank = config.parallel.rank
    if rank < 0 or rank >= config.parallel.world_size:
        raise ValueError(
            "PP CPU offload rank is outside the propagated geometry: "
            f"rank={rank}, world_size={config.parallel.world_size}."
        )

    from vllm.utils.math_utils import round_up

    cpu_bytes_to_use = int(self.extra_config["cpu_bytes_to_use"])
    bytes_per_chunk = sum(geometry) * self.blocks_per_chunk
    if any(value < 0 for value in geometry) or bytes_per_chunk <= 0:
        raise ValueError("PP CPU offload requires a positive aggregate chunk size")
    aligned_bytes_per_chunk = round_up(
        bytes_per_chunk,
        self.BLOCK_SIZE_ALIGNMENT,
    )
    self.num_chunks = cpu_bytes_to_use // aligned_bytes_per_chunk
    if self.num_chunks <= 0:
        raise ValueError(
            "cpu_bytes_to_use is smaller than one aggregate PP KV chunk: "
            f"{cpu_bytes_to_use} < {aligned_bytes_per_chunk}."
        )
    self.kv_bytes_per_chunk = aligned_bytes_per_chunk
    self.cpu_page_size_per_worker = geometry[rank] * self.blocks_per_chunk
    self.replicated_layout = False
    self._ilu_nonuniform_pp_private = True


# [ILU override] CPUOffloadingSpec.create_worker — delta: propagate rank-local PP geometry.
def _ilu_create_cpu_offloading_worker(self, kv_caches):
    if not getattr(self, "_ilu_nonuniform_pp_private", False):
        if ilu_envs.VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED:
            raise RuntimeError(
                "Required DSV4.1 PP CPU offload cannot use generic worker path"
            )
        assert _ORIGINAL_CREATE_WORKER is not None
        return _ORIGINAL_CREATE_WORKER(self, kv_caches)

    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    rank = self.config.parallel.rank
    geometry = getattr(self.config, _GEOMETRY_ATTR, None)
    if geometry is None:
        raise RuntimeError("DSV4.1 PP CPU offload geometry was not propagated")
    actual_worker_bytes_per_block = sum(
        tensor.page_size_bytes for tensor in kv_caches.tensors
    )
    if actual_worker_bytes_per_block != geometry[rank]:
        raise RuntimeError(
            "DSV4.1 PP CPU offload canonical worker bytes do not match "
            "propagated geometry: "
            f"rank={rank}, propagated={geometry[rank]}, "
            f"actual={actual_worker_bytes_per_block}."
        )
    logger.info(
        "DSV4.1 PP CPU offload private worker: rank=%d, "
        "propagated_bytes_per_block=%d, actual_canonical_bytes_per_block=%d, "
        "cpu_page_size_per_worker=%d, "
        "aggregate_chunk_stride=%d, num_chunks=%d, mmap_region=None, "
        "storage=%s",
        rank,
        geometry[rank],
        actual_worker_bytes_per_block,
        self.cpu_page_size_per_worker,
        self.kv_bytes_per_chunk,
        self.num_chunks,
        "empty" if actual_worker_bytes_per_block == 0 else "private_pinned",
    )
    if actual_worker_bytes_per_block == 0:
        if kv_caches.tensors or any(kv_caches.group_data_refs):
            raise RuntimeError(
                "Zero-byte PP offload rank has nonempty cache references"
            )
        from vllm_iluvatar.v1.worker.empty_offloading import EmptyOffloadingWorker

        # Every PP worker must acknowledge each global offload job, including
        # ranks with no local payload. Skipping the worker would leave loads
        # and stores pending forever in the scheduler.
        return EmptyOffloadingWorker()
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=self.blocks_per_chunk,
        num_cpu_chunks=self.num_chunks,
        mmap_region=None,
    )
    if getattr(worker, "_mmap_region", None) is not None:
        raise RuntimeError(
            "DSV4.1 PP CPU offload private worker unexpectedly owns shared mmap"
        )
    return worker
