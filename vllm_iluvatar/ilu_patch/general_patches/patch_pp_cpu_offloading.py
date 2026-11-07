# SPDX-License-Identifier: Apache-2.0
"""Make native CPU KV offload rank-safe for non-uniform DSV4.1 PP."""

from __future__ import annotations

import contextlib
import errno
import mmap
import os
import sys
import time
import weakref
from multiprocessing import current_process

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.dsv41_utils import is_ilu_dsv41_config
from vllm_iluvatar.ilu_util.mmap_utils import mmap_byte_array

logger = init_logger("vllm.ilu_pp_cpu_offloading")

_GEOMETRY_ATTR = "_ilu_offload_worker_bytes_by_rank"
_RANK_SAFE_REQUIRED_ATTR = "_ilu_rank_safe_offload_required"
_ORIGINAL_GET_CONFIGS = None
_ORIGINAL_BUILD_CONFIG = None
_ORIGINAL_SPEC_INIT = None
_ORIGINAL_CREATE_WORKER = None
_ORIGINAL_REGISTER_KV = None
_ORIGINAL_WORKER_INIT = None
_ORIGINAL_WORKER_SHUTDOWN = None
_REQUIRED_ENV = "VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED"
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_REVIEWED_PARTITIONS = {
    (4, 2, 8): ((5, 5, 5, 25),),
    # Both PP8 cuts are reviewed. 3,3,3,3,2,3,2,21 is the 256K profile and is
    # 0.19 GiB cheaper per 1M request (PP5 starts on KV source 14, so it needs no
    # 7th forwarder replica). 3,3,3,3,3,2,2,21 is what
    # prefill_state/configuration.py:layer_partitions derives by default for
    # depth=21 over PP8, and is the 1M profile; accepting only the former made
    # a legitimate cut fail with "expected the reviewed ... partition".
    (8, 1, 8): (
        (3, 3, 3, 3, 2, 3, 2, 21),
        (3, 3, 3, 3, 3, 2, 2, 21),
    ),
    (8, 2, 16): (
        (3, 3, 3, 3, 2, 3, 2, 21),
        (3, 3, 3, 3, 3, 2, 2, 21),
    ),
}


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSV4.1 non-uniform PP native CPU offload geometry",
    )


def _has_native_offloading(vllm_config) -> bool:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None:
        return False
    if transfer.kv_connector == "OffloadingConnector":
        return True
    if transfer.kv_connector != "MultiConnector":
        return False
    extra_config = getattr(transfer, "kv_connector_extra_config", None) or {}
    return any(
        child.get("kv_connector") == "OffloadingConnector"
        for child in extra_config.get("connectors", ())
    )


def _needs_rank_safe_geometry(vllm_config) -> bool:
    return (
        is_ilu_dsv41_config(vllm_config)
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and _has_native_offloading(vllm_config)
    )


def _required() -> bool:
    return os.getenv(_REQUIRED_ENV) == "1"


def attest_required_bindings() -> None:
    if not _required():
        return

    from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import config
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    bindings = {
        "planner": kv_cache_utils.get_kv_cache_configs is _ilu_get_kv_cache_configs,
        "config_builder": config.build_offloading_config
        is _ilu_build_offloading_config,
        "connector_builder": offloading_connector.build_offloading_config
        is _ilu_build_offloading_config,
        "spec_init": CPUOffloadingSpec.__init__ is _ilu_cpu_offloading_spec_init,
        "spec_worker": CPUOffloadingSpec.create_worker
        is _ilu_create_cpu_offloading_worker,
        "worker_init": CPUOffloadingWorker.__init__ is _ilu_cpu_offloading_worker_init,
        "worker_shutdown": CPUOffloadingWorker.shutdown
        is _ilu_cpu_offloading_worker_shutdown,
        "paired_registration": OffloadingConnectorWorker.register_kv_caches
        is _ilu_register_kv_caches,
        "handshake": (
            OffloadingConnector.set_xfer_handshake_metadata_pp_aware
            is _ilu_offloading_set_xfer_handshake_metadata_pp_aware
        ),
    }
    engine_core = sys.modules.get("vllm.v1.engine.core")
    engine_planner: bool | str = "not_loaded"
    if engine_core is not None:
        engine_planner = engine_core.get_kv_cache_configs is _ilu_get_kv_cache_configs
        bindings["engine_planner"] = engine_planner

    failed = [name for name, bound in bindings.items() if bound is not True]
    if failed:
        raise RuntimeError(
            "Required DSV4.1 PP CPU offload bindings are missing: "
            f"{failed}; bindings={bindings}"
        )
    logger.info(
        "DSV4.1 PP CPU offload bindings attested: pid=%d, process=%s, "
        "patch_file=%s, planner=%s, engine_planner=%s, "
        "config_builder=%s, connector_builder=%s, spec_init=%s, "
        "spec_worker=%s, handshake=%s",
        os.getpid(),
        current_process().name,
        __file__,
        bindings["planner"],
        engine_planner,
        bindings["config_builder"],
        bindings["connector_builder"],
        bindings["spec_init"],
        bindings["spec_worker"],
        bindings["handshake"],
    )


def _worker_bytes_per_block(config) -> int:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        get_offloading_group_ids,
    )

    from vllm_iluvatar.models.deepseek_v41.offloading import (
        has_paired_pages,
        plan_offload_layout,
    )

    # A PP stage containing only bounded-replay SWA still allocates GPU pages,
    # but none of those pages belongs to the CPU prefix cache. Do not charge
    # that backing allocation to the offload layout.
    groups = getattr(config, "kv_cache_groups", None)
    if groups is not None:
        group_ids = (
            get_offloading_group_ids(config)
            if hasattr(config, "hisparse_host_num_blocks")
            else range(len(groups))
        )
        if not any(groups[group_id].layer_names for group_id in group_ids):
            return 0
    if has_paired_pages(config):
        return plan_offload_layout(config).bytes_per_block
    if config.num_blocks <= 0 or not config.kv_cache_tensors:
        raise ValueError("PP CPU offload requires an allocated GPU KV cache.")
    total_sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(total_sizes) != 1:
        raise ValueError(
            "PP CPU offload requires every KV tensor descriptor to reference "
            f"one backing allocation size, got {sorted(total_sizes)}."
        )
    (total_size,) = total_sizes
    if total_size % config.num_blocks:
        raise ValueError(
            f"KV backing size {total_size} is not divisible by "
            f"num_blocks={config.num_blocks}."
        )
    return total_size // config.num_blocks


def _ilu_register_kv_caches(self, kv_caches):
    from vllm_iluvatar.models.deepseek_v41.offloading import (
        canonical_paired_pages,
        has_paired_pages,
    )

    if not has_paired_pages(self.kv_cache_config):
        assert _ORIGINAL_REGISTER_KV is not None
        return _ORIGINAL_REGISTER_KV(self, kv_caches)
    context = self.vllm_config.compilation_config.static_forward_context
    self._init_worker(canonical_paired_pages(self.kv_cache_config, kv_caches, context))


def _validate_target_geometry(vllm_config, geometry: tuple[int, ...]) -> None:
    parallel = vllm_config.parallel_config
    world_size = parallel.world_size
    pp_size = parallel.pipeline_parallel_size
    tp_size = parallel.tensor_parallel_size
    if len(geometry) != world_size:
        raise ValueError(
            "PP CPU offload expected one KV geometry per worker: "
            f"got {len(geometry)} for world_size={world_size}."
        )
    if world_size != pp_size * tp_size:
        raise ValueError(
            "DSV4.1 PP CPU offload requires world_size == pp_size * tp_size: "
            f"{world_size} != {pp_size} * {tp_size}."
        )
    if any(value < 0 for value in geometry) or not any(geometry):
        raise ValueError(
            "DSV4.1 PP CPU offload geometry must contain nonnegative byte "
            "sizes and at least one positive size: "
            f"{geometry}."
        )

    for pp_rank in range(pp_size):
        start = pp_rank * tp_size
        stage_values = geometry[start : start + tp_size]
        if len(set(stage_values)) != 1:
            raise ValueError(
                "DSV4.1 PP CPU offload TP ranks within one PP stage must have "
                f"identical geometry: pp_rank={pp_rank}, values={stage_values}."
            )

    expected_partitions = _REVIEWED_PARTITIONS.get((pp_size, tp_size, world_size))
    if expected_partitions is None:
        return
    partition_text = os.environ.get("VLLM_PP_LAYER_PARTITION", "")
    try:
        partition = tuple(int(value) for value in partition_text.split(","))
    except ValueError as exc:
        raise ValueError(
            "DSV4.1 PP CPU offload requires an integer VLLM_PP_LAYER_PARTITION."
        ) from exc
    if partition not in expected_partitions:
        expected_text = " or ".join(
            ",".join(str(value) for value in candidate)
            for candidate in expected_partitions
        )
        raise ValueError(
            f"DSV4.1 PP CPU offload expected the reviewed {expected_text} "
            f"partition, got {partition_text!r}."
        )


def _ilu_get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    assert _ORIGINAL_GET_CONFIGS is not None
    needs_geometry = _needs_rank_safe_geometry(vllm_config)
    if _required():
        model = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model, "hf_config", None)
        parallel = vllm_config.parallel_config
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        logger.info(
            "DSV4.1 PP CPU offload target gate: architecture=%s, "
            "model_type=%s, pp_size=%d, tp_size=%d, world_size=%d, "
            "connector=%s, offloading_child=%s, "
            "needs_rank_safe_geometry=%s",
            getattr(model, "architectures", None),
            getattr(hf_config, "model_type", None),
            parallel.pipeline_parallel_size,
            parallel.tensor_parallel_size,
            parallel.world_size,
            getattr(transfer, "kv_connector", None),
            _has_native_offloading(vllm_config),
            needs_geometry,
        )
        if not needs_geometry:
            raise RuntimeError(
                "Required DSV4.1 PP CPU offload target predicate is false"
            )
    configs = _ORIGINAL_GET_CONFIGS(
        vllm_config,
        kv_cache_specs,
        available_memory,
    )
    if not needs_geometry:
        return configs

    geometry = tuple(_worker_bytes_per_block(config) for config in configs)
    _validate_target_geometry(vllm_config, geometry)
    for config in configs:
        setattr(config, _GEOMETRY_ATTR, geometry)
    logger.info(
        "DSV4.1 PP CPU offload rank geometry: bytes_per_block=%s, "
        "aggregate_bytes_per_block=%d, world_size=%d, pp_size=%d, tp_size=%d",
        geometry,
        sum(geometry),
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.pipeline_parallel_size,
        vllm_config.parallel_config.tensor_parallel_size,
    )
    return configs


def _ilu_build_offloading_config(vllm_config, kv_cache_config):
    assert _ORIGINAL_BUILD_CONFIG is not None
    needs_geometry = _needs_rank_safe_geometry(vllm_config)
    geometry = getattr(kv_cache_config, _GEOMETRY_ATTR, None)
    if needs_geometry and geometry is None:
        raise RuntimeError("DSV4.1 PP CPU offload geometry was not propagated")
    config = _ORIGINAL_BUILD_CONFIG(vllm_config, kv_cache_config)
    if geometry is not None:
        object.__setattr__(config, _GEOMETRY_ATTR, tuple(geometry))
    if needs_geometry:
        object.__setattr__(config, _RANK_SAFE_REQUIRED_ATTR, True)
    return config


def _ilu_cpu_offloading_spec_init(self, config) -> None:
    assert _ORIGINAL_SPEC_INIT is not None
    geometry = getattr(config, _GEOMETRY_ATTR, None)
    needs_geometry = getattr(config, _RANK_SAFE_REQUIRED_ATTR, False)
    if _required() and (geometry is None or needs_geometry is not True):
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


def _populate_anonymous_mmap(mmap_obj: mmap.mmap, length: int) -> None:
    if length <= 0:
        return
    try:
        mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, length)
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
        arr = np.frombuffer(mmap_obj, dtype=np.uint8)
        arr[: length : mmap.PAGESIZE] = 0


def _cudart_host_register(address: int, size: int) -> int:
    cudart = torch.cuda.cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is unavailable for host registration")
    result = cudart.cudaHostRegister(address, size, 0)
    return int(getattr(result, "value", result))


def _cudart_host_unregister(address: int) -> int:
    cudart = torch.cuda.cudart()
    if cudart is None:
        return 0
    result = cudart.cudaHostUnregister(address)
    return int(getattr(result, "value", result))


def _clear_cuda_last_error() -> None:
    with contextlib.suppress(Exception):
        cudart = torch.cuda.cudart()
        if cudart is not None and hasattr(cudart, "cudaGetLastError"):
            cudart.cudaGetLastError()


def _unregister_addresses(addresses: list[int]) -> None:
    while addresses:
        address = addresses.pop()
        try:
            code = _cudart_host_unregister(address)
        except Exception:
            logger.warning(
                "cudaHostUnregister failed for private offload address=%#x",
                address,
                exc_info=True,
            )
            continue
        if code != 0:
            _clear_cuda_last_error()
            logger.warning(
                "cudaHostUnregister failed for private offload address=%#x (code=%d)",
                address,
                code,
            )


def _close_mmap_quietly(mmap_obj: mmap.mmap | None) -> None:
    if mmap_obj is not None and not mmap_obj.closed:
        with contextlib.suppress(Exception):
            mmap_obj.close()


def _cleanup_private_mmap(
    addresses: list[int],
    mmap_obj: mmap.mmap | None,
) -> None:
    _unregister_addresses(addresses)
    _close_mmap_quietly(mmap_obj)


class _PrivateOffloadBuffer:
    """Page-aligned anonymous private mmap pinned in-place via cudaHostRegister.

    Bypasses PyTorch's CUDACachingHostAllocator (which rounds every
    `pin_memory=True` allocation up to the next power of 2 and allocates from
    POSIX shared memory on CoreX).
    """

    def __init__(
        self,
        mmap_obj: mmap.mmap | None,
        owner: np.ndarray | None,
        addresses: list[int],
        *,
        is_pinned: bool,
        tensor: torch.Tensor | None = None,
    ) -> None:
        self.mmap_obj = mmap_obj
        self.owner = owner
        self.pinned_addresses = addresses
        self.is_pinned = is_pinned
        self._tensor_ref = None
        if tensor is not None:
            with contextlib.suppress(TypeError):
                self._tensor_ref = weakref.ref(tensor)
        self._finalizer = None
        if owner is not None and mmap_obj is not None:
            self._finalizer = weakref.finalize(
                owner, _cleanup_private_mmap, addresses, mmap_obj
            )
            self._finalizer.atexit = False

    def cleanup(self) -> None:
        self.is_pinned = False
        if self._tensor_ref is not None:
            tensor = self._tensor_ref()
            self._tensor_ref = None
            if tensor is not None and hasattr(tensor, "set_"):
                with contextlib.suppress(Exception):
                    tensor.set_(torch.empty(0, dtype=tensor.dtype, device="cpu"))
            tensor = None
        self.owner = None
        _unregister_addresses(self.pinned_addresses)
        _close_mmap_quietly(self.mmap_obj)
        if self._finalizer is not None and (
            self.mmap_obj is None or self.mmap_obj.closed
        ):
            self._finalizer.detach()


def _allocate_private_cpu_tensor(
    num_cpu_chunks: int,
    cpu_page_size_bytes: int,
    *,
    pin_memory: bool,
    max_register_chunk_bytes: int,
) -> tuple[torch.Tensor, _PrivateOffloadBuffer]:
    from vllm.platforms import current_platform

    nbytes = int(num_cpu_chunks) * int(cpu_page_size_bytes)
    if nbytes <= 0:
        empty = torch.zeros(
            (num_cpu_chunks, cpu_page_size_bytes),
            dtype=torch.int8,
            device="cpu",
        )
        return empty, _PrivateOffloadBuffer(
            None, None, [], is_pinned=bool(pin_memory)
        )

    page_size = mmap.PAGESIZE
    aligned_bytes = ((nbytes + page_size - 1) // page_size) * page_size
    mmap_obj = mmap.mmap(
        -1,
        aligned_bytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    owner = None
    tensor = None
    try:
        _populate_anonymous_mmap(mmap_obj, aligned_bytes)
        owner = mmap_byte_array(mmap_obj, nbytes).view(np.int8)
        tensor = torch.from_numpy(owner).view(
            num_cpu_chunks, cpu_page_size_bytes
        )
    except BaseException:
        tensor = None
        owner = None
        mmap_obj.close()
        raise

    addresses: list[int] = []
    is_pinned = False
    if pin_memory and current_platform.is_cuda_alike():
        base_ptr = tensor.data_ptr()
        chunk_bytes = max((max_register_chunk_bytes // page_size) * page_size, page_size)
        try:
            for offset in range(0, aligned_bytes, chunk_bytes):
                address = base_ptr + offset
                size = min(chunk_bytes, aligned_bytes - offset)
                code = _cudart_host_register(address, size)
                if code != 0:
                    _clear_cuda_last_error()
                    logger.warning(
                        "cudaHostRegister failed at %.2f of %.2f GiB (code=%d); "
                        "private offload tensor stays pageable",
                        offset / (1 << 30),
                        aligned_bytes / (1 << 30),
                        code,
                    )
                    _unregister_addresses(addresses)
                    break
                addresses.append(address)
            else:
                is_pinned = True
        except Exception:
            logger.warning(
                "Could not register private offload tensor with CUDA runtime; "
                "the tensor stays pageable",
                exc_info=True,
            )
            _unregister_addresses(addresses)

    return tensor, _PrivateOffloadBuffer(
        mmap_obj,
        owner,
        addresses,
        is_pinned=is_pinned,
        tensor=tensor,
    )


def _ilu_cpu_offloading_worker_init(
    self,
    kv_caches,
    blocks_per_chunk: int,
    num_cpu_chunks: int,
    mmap_region=None,
    canonical_layout: bool = False,
) -> None:
    if mmap_region is not None:
        assert _ORIGINAL_WORKER_INIT is not None
        _ORIGINAL_WORKER_INIT(
            self,
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_chunks=num_cpu_chunks,
            mmap_region=mmap_region,
            canonical_layout=canonical_layout,
        )
        return

    from vllm.v1.kv_offload.cpu import gpu_worker

    assert not canonical_layout
    self._mmap_region = None
    self._ilu_private_regions = []
    pin_memory = bool(gpu_worker.PIN_MEMORY)
    gpu_worker.logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))

    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    gpu_tensor = None
    cpu_tensor = None
    region = None
    try:
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk
            t0 = time.monotonic()
            cpu_tensor, region = _allocate_private_cpu_tensor(
                num_cpu_chunks,
                cpu_page_size_bytes,
                pin_memory=pin_memory,
                max_register_chunk_bytes=gpu_worker.MAX_HOST_REGISTER_CHUNK_BYTES,
            )
            gpu_worker.logger.debug(
                "anonymous mmap pinned tensor %d×%d (%.2f GiB, pinned=%s): %.3f s",
                num_cpu_chunks,
                cpu_page_size_bytes,
                num_cpu_chunks * cpu_page_size_bytes / (1 << 30),
                region.is_pinned,
                time.monotonic() - t0,
            )
            self._ilu_private_regions.append(region)
            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)
            gpu_tensor = None
            cpu_tensor = None
            region = None

        host_memory_is_pinned = pin_memory and all(
            r.is_pinned for r in self._ilu_private_regions
        )
        self._store_handler = gpu_worker.SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=False,
            host_memory_is_pinned=host_memory_is_pinned,
        )
        self._load_handler = gpu_worker.SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=False,
            host_memory_is_pinned=host_memory_is_pinned,
        )
    except BaseException:
        gpu_tensor = None
        cpu_tensor = None
        region = None
        gpu_tensors.clear()
        cpu_tensors.clear()
        for attr in ("_store_handler", "_load_handler"):
            handler = getattr(self, attr, None)
            if handler is not None:
                with contextlib.suppress(Exception):
                    handler.shutdown()
                for list_attr in (
                    "src_tensors",
                    "dst_tensors",
                    "gpu_tensors",
                    "cpu_tensors",
                ):
                    tensor_list = getattr(handler, list_attr, None)
                    if isinstance(tensor_list, list):
                        tensor_list.clear()
                setattr(self, attr, None)
        for priv_region in reversed(self._ilu_private_regions):
            priv_region.cleanup()
        self._ilu_private_regions.clear()
        raise


def _ilu_cpu_offloading_worker_shutdown(self) -> None:
    regions = getattr(self, "_ilu_private_regions", None)
    if not regions:
        assert _ORIGINAL_WORKER_SHUTDOWN is not None
        _ORIGINAL_WORKER_SHUTDOWN(self)
        return

    handler_failed = False
    try:
        store_handler = getattr(self, "_store_handler", None)
        if store_handler is not None:
            try:
                store_handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down store offloading handler")
                handler_failed = True

        load_handler = getattr(self, "_load_handler", None)
        if load_handler is not None:
            try:
                load_handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down load offloading handler")
                handler_failed = True
    except BaseException:
        handler_failed = True
        raise
    finally:
        for handler in (
            getattr(self, "_store_handler", None),
            getattr(self, "_load_handler", None),
        ):
            if handler is not None:
                for list_attr in (
                    "src_tensors",
                    "dst_tensors",
                    "gpu_tensors",
                    "cpu_tensors",
                ):
                    tensor_list = getattr(handler, list_attr, None)
                    if isinstance(tensor_list, list):
                        tensor_list.clear()
        if handler_failed:
            try:
                torch.accelerator.synchronize()
            except Exception:
                logger.warning(
                    "Device sync before private offload cleanup failed; "
                    "proceeding with cleanup anyway",
                    exc_info=True,
                )
        for region in reversed(regions):
            region.cleanup()
        regions.clear()


def _ilu_create_cpu_offloading_worker(self, kv_caches):
    if not getattr(self, "_ilu_nonuniform_pp_private", False):
        if _required():
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


def _ilu_offloading_set_xfer_handshake_metadata_pp_aware(
    self,
    metadata,
) -> None:
    """Ignore a sibling OptNixl connector's remote PP handshake metadata."""


def _bind_offloading_handshake_noop() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    OffloadingConnector.set_xfer_handshake_metadata_pp_aware = (
        _ilu_offloading_set_xfer_handshake_metadata_pp_aware
    )


def _apply() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import config
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    global _ORIGINAL_GET_CONFIGS, _ORIGINAL_BUILD_CONFIG
    global _ORIGINAL_SPEC_INIT, _ORIGINAL_CREATE_WORKER
    global _ORIGINAL_REGISTER_KV
    global _ORIGINAL_WORKER_INIT, _ORIGINAL_WORKER_SHUTDOWN

    current_get_configs = kv_cache_utils.get_kv_cache_configs
    if current_get_configs is not _ilu_get_kv_cache_configs:
        _ORIGINAL_GET_CONFIGS = current_get_configs
    p.rebind_module_func(
        kv_cache_utils,
        "get_kv_cache_configs",
        _ilu_get_kv_cache_configs,
    )

    current_build = config.build_offloading_config
    if current_build is not _ilu_build_offloading_config:
        _ORIGINAL_BUILD_CONFIG = current_build
    p.rebind_module_func(
        config,
        "build_offloading_config",
        _ilu_build_offloading_config,
    )
    p.rebind_module_func(
        offloading_connector,
        "build_offloading_config",
        _ilu_build_offloading_config,
    )

    current_init = CPUOffloadingSpec.__init__
    if current_init is not _ilu_cpu_offloading_spec_init:
        _ORIGINAL_SPEC_INIT = current_init
    CPUOffloadingSpec.__init__ = _ilu_cpu_offloading_spec_init

    current_create = CPUOffloadingSpec.create_worker
    if current_create is not _ilu_create_cpu_offloading_worker:
        _ORIGINAL_CREATE_WORKER = current_create
    CPUOffloadingSpec.create_worker = _ilu_create_cpu_offloading_worker

    current_worker_init = CPUOffloadingWorker.__init__
    if current_worker_init is not _ilu_cpu_offloading_worker_init:
        _ORIGINAL_WORKER_INIT = current_worker_init
    CPUOffloadingWorker.__init__ = _ilu_cpu_offloading_worker_init

    current_worker_shutdown = CPUOffloadingWorker.shutdown
    if current_worker_shutdown is not _ilu_cpu_offloading_worker_shutdown:
        _ORIGINAL_WORKER_SHUTDOWN = current_worker_shutdown
    CPUOffloadingWorker.shutdown = _ilu_cpu_offloading_worker_shutdown

    current_register = OffloadingConnectorWorker.register_kv_caches
    if current_register is not _ilu_register_kv_caches:
        _ORIGINAL_REGISTER_KV = current_register
    OffloadingConnectorWorker.register_kv_caches = _ilu_register_kv_caches

    _bind_offloading_handshake_noop()
