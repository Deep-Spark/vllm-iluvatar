# SPDX-License-Identifier: Apache-2.0
"""Wiring for rank-safe native CPU offload and paired-page registration.

Corresponds to: vllm/v1/{core/kv_cache_utils,kv_offload/cpu/{spec,gpu_worker}}.py
               vllm/distributed/kv_transfer/kv_connector/v1/offloading{,_connector}
Target   : Planner, config builder, spec, CPU worker, registration and handshake.
Baseline : v0.31.0 (341bf7d4d1244424c94296ce4fc5626ad14b827a)
Why      : Preserve DSV4.1 non-uniform PP geometry and CoreX private pinned storage.
How      : Lazy mirror-module binding; unchanged paths delegate to community.
Scope    : Geometry adapters are model-specific; private worker storage is generic.
Remove   : When community owns equivalent geometry and private-storage lifetime.
Fingerprint: CPUOffloadingWorker.__init__ and shutdown copies are checked against
             the verbatim community snapshots below before any binding changes.
"""

from __future__ import annotations

import os
import sys
from multiprocessing import current_process

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim pinned community source; update snapshots and worker copies together.
# fmt: off
WORKER_INIT = '''    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)
        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned
        )

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_chunks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_chunks,
                    cpu_page_size_bytes,
                    num_cpu_chunks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )
'''

WORKER_SHUTDOWN = '''    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        try:
            self._load_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down load offloading handler")
            handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None
'''
# fmt: on


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSV4.1 non-uniform PP native CPU offload geometry",
    )


def attest_required_bindings() -> None:
    from vllm_iluvatar import envs as ilu_envs

    if not ilu_envs.VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED:
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

    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.config as config_impl
    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.worker as reg_impl
    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading_connector as connector_impl
    import vllm_iluvatar.v1.core.kv_cache_utils as kv_impl
    import vllm_iluvatar.v1.kv_offload.cpu.gpu_worker as gpu_impl
    import vllm_iluvatar.v1.kv_offload.cpu.spec as spec_impl

    bindings = {
        "planner": kv_cache_utils.get_kv_cache_configs
        is kv_impl._ilu_get_kv_cache_configs,
        "config_builder": config.build_offloading_config
        is config_impl._ilu_build_offloading_config,
        "connector_builder": offloading_connector.build_offloading_config
        is config_impl._ilu_build_offloading_config,
        "spec_init": CPUOffloadingSpec.__init__
        is spec_impl._ilu_cpu_offloading_spec_init,
        "spec_worker": CPUOffloadingSpec.create_worker
        is spec_impl._ilu_create_cpu_offloading_worker,
        "worker_init": CPUOffloadingWorker.__init__
        is gpu_impl._ilu_cpu_offloading_worker_init,
        "worker_shutdown": CPUOffloadingWorker.shutdown
        is gpu_impl._ilu_cpu_offloading_worker_shutdown,
        "paired_registration": OffloadingConnectorWorker.register_kv_caches
        is reg_impl._ilu_register_kv_caches,
        "handshake": (
            OffloadingConnector.set_xfer_handshake_metadata_pp_aware
            is connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware
        ),
    }
    engine_core = sys.modules.get("vllm.v1.engine.core")
    engine_planner: bool | str = "not_loaded"
    if engine_core is not None:
        engine_planner = (
            engine_core.get_kv_cache_configs is kv_impl._ilu_get_kv_cache_configs
        )
        bindings["engine_planner"] = engine_planner

    failed = [name for name, bound in bindings.items() if bound is not True]
    if failed:
        raise RuntimeError(
            "Required DSV4.1 PP CPU offload bindings are missing: "
            f"{failed}; bindings={bindings}"
        )
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    logger.info(
        "DSV4.1 PP CPU offload bindings attested: pid=%d, process=%s, "
        "patch_file=%s, planner=%s, engine_planner=%s, "
        "config_builder=%s, connector_builder=%s, spec_init=%s, "
        "spec_worker=%s, handshake=%s, planner_impl=%s, worker_impl=%s",
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
        kv_impl.__file__,
        gpu_impl.__file__,
    )


def _bind_offloading_handshake_noop() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading_connector as connector_impl

    OffloadingConnector.set_xfer_handshake_metadata_pp_aware = (
        connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware
    )


def _apply() -> None:
    import inspect
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import config
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.config as config_impl
    import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.worker as reg_impl
    import vllm_iluvatar.v1.core.kv_cache_utils as kv_impl
    import vllm_iluvatar.v1.kv_offload.cpu.gpu_worker as gpu_impl
    import vllm_iluvatar.v1.kv_offload.cpu.spec as spec_impl

    if CPUOffloadingWorker.__init__ is not gpu_impl._ilu_cpu_offloading_worker_init:
        p.check_fingerprint(
            "CPUOffloadingWorker.__init__",
            inspect.getsource(CPUOffloadingWorker.__init__),
            WORKER_INIT,
            resync_hint="Reconcile the private worker with the new community baseline.",
        )
        p.check_signatures(
            SimpleNamespace(__init__=CPUOffloadingWorker.__init__),
            SimpleNamespace(__init__=gpu_impl._ilu_cpu_offloading_worker_init),
            ["__init__"],
            strict=True,
        )
    if CPUOffloadingWorker.shutdown is not gpu_impl._ilu_cpu_offloading_worker_shutdown:
        p.check_fingerprint(
            "CPUOffloadingWorker.shutdown",
            inspect.getsource(CPUOffloadingWorker.shutdown),
            WORKER_SHUTDOWN,
            resync_hint="Reconcile the private worker with the new community baseline.",
        )
        p.check_signatures(
            SimpleNamespace(shutdown=CPUOffloadingWorker.shutdown),
            SimpleNamespace(shutdown=gpu_impl._ilu_cpu_offloading_worker_shutdown),
            ["shutdown"],
            strict=True,
        )

    current_get_configs = kv_cache_utils.get_kv_cache_configs
    if current_get_configs is not kv_impl._ilu_get_kv_cache_configs:
        kv_impl._ORIGINAL_GET_CONFIGS = current_get_configs
    p.rebind_module_func(
        kv_cache_utils,
        "get_kv_cache_configs",
        kv_impl._ilu_get_kv_cache_configs,
    )

    current_build = config.build_offloading_config
    if current_build is not config_impl._ilu_build_offloading_config:
        config_impl._ORIGINAL_BUILD_CONFIG = current_build
    p.rebind_module_func(
        config,
        "build_offloading_config",
        config_impl._ilu_build_offloading_config,
    )
    p.rebind_module_func(
        offloading_connector,
        "build_offloading_config",
        config_impl._ilu_build_offloading_config,
    )

    current_init = CPUOffloadingSpec.__init__
    if current_init is not spec_impl._ilu_cpu_offloading_spec_init:
        spec_impl._ORIGINAL_SPEC_INIT = current_init
    CPUOffloadingSpec.__init__ = spec_impl._ilu_cpu_offloading_spec_init

    current_create = CPUOffloadingSpec.create_worker
    if current_create is not spec_impl._ilu_create_cpu_offloading_worker:
        spec_impl._ORIGINAL_CREATE_WORKER = current_create
    CPUOffloadingSpec.create_worker = spec_impl._ilu_create_cpu_offloading_worker

    current_worker_init = CPUOffloadingWorker.__init__
    if current_worker_init is not gpu_impl._ilu_cpu_offloading_worker_init:
        gpu_impl._ORIGINAL_WORKER_INIT = current_worker_init
    CPUOffloadingWorker.__init__ = gpu_impl._ilu_cpu_offloading_worker_init

    current_worker_shutdown = CPUOffloadingWorker.shutdown
    if current_worker_shutdown is not gpu_impl._ilu_cpu_offloading_worker_shutdown:
        gpu_impl._ORIGINAL_WORKER_SHUTDOWN = current_worker_shutdown
    CPUOffloadingWorker.shutdown = gpu_impl._ilu_cpu_offloading_worker_shutdown

    current_register = OffloadingConnectorWorker.register_kv_caches
    if current_register is not reg_impl._ilu_register_kv_caches:
        reg_impl._ORIGINAL_REGISTER_KV = current_register
    OffloadingConnectorWorker.register_kv_caches = reg_impl._ilu_register_kv_caches

    _bind_offloading_handshake_noop()
