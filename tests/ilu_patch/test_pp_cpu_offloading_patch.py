from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from vllm_iluvatar.ilu_patch.general_patches import patch_pp_cpu_offloading


def _config(
    *,
    pp_size: int = 2,
    tp_size: int = 1,
    world_size: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["IluDeepseekV41ForCausalLM"],
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            world_size=world_size,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "OptNixlConnector",
                        "kv_role": "kv_producer",
                    },
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                    },
                ]
            },
        ),
    )


def _kv_config(num_blocks: int, total_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[SimpleNamespace(size=total_size)],
    )


def test_direct_and_multi_connector_detection() -> None:
    multi = _config()
    direct = _config()
    direct.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={},
    )
    nixl_only = _config()
    nixl_only.kv_transfer_config = SimpleNamespace(
        kv_connector="OptNixlConnector",
        kv_connector_extra_config={},
    )

    assert patch_pp_cpu_offloading._has_native_offloading(multi)
    assert patch_pp_cpu_offloading._has_native_offloading(direct)
    assert not patch_pp_cpu_offloading._has_native_offloading(nixl_only)


def test_nonuniform_pp_geometry_is_attached_to_every_worker(
    monkeypatch,
) -> None:
    configs = [_kv_config(10, 1000), _kv_config(10, 3000)]
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )

    result = patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
        _config(),
        [{}, {}],
        [1, 1],
    )

    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == (100, 300) for config in configs
    )


def test_dsv41_pp8_partition_is_accepted(monkeypatch) -> None:
    per_stage = (100, 200, 300, 400, 500, 600, 700, 800)
    geometry = tuple(value for value in per_stage for _ in range(2))
    configs = [_kv_config(10, value * 10) for value in geometry]
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    result = patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
        _config(pp_size=8, tp_size=2, world_size=16),
        [{}] * 16,
        [1] * 16,
    )

    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == geometry for config in configs
    )


def test_dsv41_pp8_accepts_the_1m_partition(monkeypatch) -> None:
    """1M Prefill uses the PP8 cut that ``layer_partitions`` derives itself.

    depth=21 over pp=8 gives 3,3,3,3,3,2,2 plus the 19 unexecuted suffix
    layers on the last stage. Regression guard: the allow-list once held only
    the 256K cut, so a legitimate 1M launch died in _validate_target_geometry
    before KV admission.
    """
    per_stage = (100, 200, 300, 400, 500, 600, 700, 800)
    geometry = tuple(value for value in per_stage for _ in range(2))
    configs = [_kv_config(10, value * 10) for value in geometry]
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,3,2,2,21",
    )

    result = patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
        _config(pp_size=8, tp_size=2, world_size=16),
        [{}] * 16,
        [1] * 16,
    )

    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == geometry for config in configs
    )


def test_dsv41_pp8_rejects_wrong_partition(monkeypatch) -> None:
    configs = [_kv_config(10, 1000) for _ in range(16)]
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "5,5,5,5,5,5,5,5",
    )

    with pytest.raises(
        ValueError,
        match="expected the reviewed 3,3,3,3,2,3,2,21 or 3,3,3,3,3,2,2,21",
    ):
        patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
            _config(pp_size=8, tp_size=2, world_size=16),
            [{}] * 16,
            [1] * 16,
        )


def test_dsv41_rejects_tp_peer_geometry_mismatch(monkeypatch) -> None:
    configs = [_kv_config(10, 1000) for _ in range(16)]
    configs[5] = _kv_config(10, 2000)
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    with pytest.raises(ValueError, match="TP ranks within one PP stage"):
        patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
            _config(pp_size=8, tp_size=2, world_size=16),
            [{}] * 16,
            [1] * 16,
        )


def test_cpu_capacity_uses_sum_of_rank_geometry(monkeypatch) -> None:
    cpu_bytes = 10 * 4096

    def original_init(spec, _config):
        spec.extra_config = {"cpu_bytes_to_use": cpu_bytes}
        spec.blocks_per_chunk = 2
        spec.BLOCK_SIZE_ALIGNMENT = 4096
        spec.replicated_layout = False

    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_SPEC_INIT",
        original_init,
    )
    spec = SimpleNamespace()
    config = SimpleNamespace(
        canonical_layout=False,
        parallel=SimpleNamespace(world_size=2, rank=1),
        _ilu_offload_worker_bytes_by_rank=(512, 1536),
        _ilu_rank_safe_offload_required=True,
    )

    patch_pp_cpu_offloading._ilu_cpu_offloading_spec_init(spec, config)

    assert spec.num_chunks == 10
    assert spec.kv_bytes_per_chunk == 4096
    assert spec.cpu_page_size_per_worker == 3072
    assert spec._ilu_nonuniform_pp_private


def test_geometry_propagates_into_offloading_config(monkeypatch) -> None:
    result_config = SimpleNamespace()
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_BUILD_CONFIG",
        lambda *_args: result_config,
    )
    kv_config = SimpleNamespace(
        _ilu_offload_worker_bytes_by_rank=(100, 300),
    )

    result = patch_pp_cpu_offloading._ilu_build_offloading_config(
        _config(),
        kv_config,
    )

    assert result is result_config
    assert result._ilu_offload_worker_bytes_by_rank == (100, 300)
    assert result._ilu_rank_safe_offload_required is True


def test_target_build_fails_closed_when_geometry_is_missing(
    monkeypatch,
) -> None:
    original = MagicMock()
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_BUILD_CONFIG",
        original,
    )

    with pytest.raises(RuntimeError, match="geometry was not propagated"):
        patch_pp_cpu_offloading._ilu_build_offloading_config(
            _config(),
            SimpleNamespace(),
        )

    original.assert_not_called()


def test_nonuniform_pp_worker_uses_private_rank_local_cpu_storage(
    monkeypatch,
) -> None:
    worker = SimpleNamespace(_mmap_region=None)
    create_worker = MagicMock(return_value=worker)
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.gpu_worker.CPUOffloadingWorker",
        create_worker,
    )
    spec = SimpleNamespace(
        _ilu_nonuniform_pp_private=True,
        blocks_per_chunk=1,
        num_chunks=17,
        cpu_page_size_per_worker=3072,
        kv_bytes_per_chunk=4096,
        config=SimpleNamespace(
            parallel=SimpleNamespace(rank=1),
            _ilu_offload_worker_bytes_by_rank=(512, 1536),
        ),
    )
    caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(page_size_bytes=512),
            SimpleNamespace(page_size_bytes=1024),
        ]
    )

    result = patch_pp_cpu_offloading._ilu_create_cpu_offloading_worker(
        spec,
        caches,
    )

    assert result is worker
    create_worker.assert_called_once_with(
        kv_caches=caches,
        blocks_per_chunk=1,
        num_cpu_chunks=17,
        mmap_region=None,
    )


def test_local_offloading_ignores_pp_handshake_without_state() -> None:
    connector = object.__new__(OffloadingConnector)
    connector.sentinel = object()
    state_before = vars(connector).copy()

    patch_pp_cpu_offloading._ilu_offloading_set_xfer_handshake_metadata_pp_aware(
        connector,
        {(0, 0): object(), (3, 1): object()},
    )

    assert vars(connector) == state_before


def test_multi_connector_fanout_preserves_opt_nixl_pp_metadata() -> None:
    class PPAwareFake:
        received = None

        def set_xfer_handshake_metadata_pp_aware(self, metadata) -> None:
            self.received = metadata

    class LocalOffloadingFake:
        set_xfer_handshake_metadata_pp_aware = (
            patch_pp_cpu_offloading._ilu_offloading_set_xfer_handshake_metadata_pp_aware
        )

    remote = PPAwareFake()
    local = LocalOffloadingFake()
    multi = object.__new__(MultiConnector)
    multi._connectors = [remote, local]
    metadata = {(0, 0): object(), (3, 1): object()}

    multi.set_xfer_handshake_metadata_pp_aware(metadata)

    assert remote.received is metadata
    assert vars(local) == {}


def test_unadapted_base_connector_still_rejects_pp_metadata() -> None:
    with pytest.raises(
        ValueError,
        match="does not support PP-disaggregated KV transfer",
    ):
        KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware(
            SimpleNamespace(),
            {(0, 0): object(), (1, 0): object()},
        )


def test_required_worker_cannot_enter_generic_path(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", "1")
    original = MagicMock()
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_CREATE_WORKER",
        original,
    )

    with pytest.raises(RuntimeError, match="cannot use generic worker path"):
        patch_pp_cpu_offloading._ilu_create_cpu_offloading_worker(
            SimpleNamespace(_ilu_nonuniform_pp_private=False),
            object(),
        )

    original.assert_not_called()


def test_runtime_offloading_class_can_bind_local_noop(monkeypatch) -> None:
    monkeypatch.setattr(
        OffloadingConnector,
        "set_xfer_handshake_metadata_pp_aware",
        KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware,
    )

    patch_pp_cpu_offloading._bind_offloading_handshake_noop()

    expected = (
        patch_pp_cpu_offloading._ilu_offloading_set_xfer_handshake_metadata_pp_aware
    )
    assert OffloadingConnector.set_xfer_handshake_metadata_pp_aware is expected


def test_required_spec_rejects_missing_geometry_or_marker(
    monkeypatch,
) -> None:
    original = MagicMock()
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_SPEC_INIT",
        original,
    )
    monkeypatch.setenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", "1")

    with pytest.raises(RuntimeError, match="geometry/marker is missing"):
        patch_pp_cpu_offloading._ilu_cpu_offloading_spec_init(
            SimpleNamespace(),
            SimpleNamespace(
                _ilu_offload_worker_bytes_by_rank=(512, 1536),
            ),
        )

    original.assert_not_called()


def test_direct_offloading_without_extra_config_is_detected() -> None:
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config=None,
    )

    assert patch_pp_cpu_offloading._has_native_offloading(config)


def test_non_target_model_does_not_need_rank_safe_geometry() -> None:
    config = _config()
    config.model_config.architectures = ["OtherArchitecture"]
    config.model_config.hf_config.model_type = "other"

    assert not patch_pp_cpu_offloading._needs_rank_safe_geometry(config)


def test_pp1_does_not_need_rank_safe_geometry() -> None:
    config = _config(pp_size=1, tp_size=2, world_size=2)

    assert not patch_pp_cpu_offloading._needs_rank_safe_geometry(config)


def test_invalid_world_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="world_size == pp_size"):
        patch_pp_cpu_offloading._validate_target_geometry(
            _config(pp_size=8, tp_size=2, world_size=15),
            (100,) * 15,
        )


def test_cpu_capacity_rejects_less_than_one_aggregate_chunk(
    monkeypatch,
) -> None:
    def original_init(spec, _config):
        spec.extra_config = {"cpu_bytes_to_use": 4095}
        spec.blocks_per_chunk = 2
        spec.BLOCK_SIZE_ALIGNMENT = 4096
        spec.replicated_layout = False

    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_SPEC_INIT",
        original_init,
    )

    with pytest.raises(ValueError, match="smaller than one aggregate"):
        patch_pp_cpu_offloading._ilu_cpu_offloading_spec_init(
            SimpleNamespace(),
            SimpleNamespace(
                canonical_layout=False,
                parallel=SimpleNamespace(world_size=2, rank=0),
                _ilu_offload_worker_bytes_by_rank=(512, 1536),
                _ilu_rank_safe_offload_required=True,
            ),
        )


def test_canonical_layout_is_rejected(monkeypatch) -> None:
    def original_init(spec, _config):
        spec.extra_config = {"cpu_bytes_to_use": 1 << 20}
        spec.blocks_per_chunk = 1
        spec.BLOCK_SIZE_ALIGNMENT = 4096
        spec.replicated_layout = False

    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_SPEC_INIT",
        original_init,
    )

    with pytest.raises(ValueError, match="canonical_layout is not supported"):
        patch_pp_cpu_offloading._ilu_cpu_offloading_spec_init(
            SimpleNamespace(),
            SimpleNamespace(
                canonical_layout=True,
                parallel=SimpleNamespace(world_size=2, rank=0),
                _ilu_offload_worker_bytes_by_rank=(512, 1536),
                _ilu_rank_safe_offload_required=True,
            ),
        )


def test_private_worker_rejects_geometry_mismatch(monkeypatch) -> None:
    create_worker = MagicMock()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.gpu_worker.CPUOffloadingWorker",
        create_worker,
    )
    spec = SimpleNamespace(
        _ilu_nonuniform_pp_private=True,
        blocks_per_chunk=1,
        num_chunks=17,
        cpu_page_size_per_worker=1536,
        kv_bytes_per_chunk=4096,
        config=SimpleNamespace(
            parallel=SimpleNamespace(rank=1),
            _ilu_offload_worker_bytes_by_rank=(512, 1536),
        ),
    )
    caches = SimpleNamespace(tensors=[SimpleNamespace(page_size_bytes=1024)])

    with pytest.raises(RuntimeError, match="do not match propagated geometry"):
        patch_pp_cpu_offloading._ilu_create_cpu_offloading_worker(
            spec,
            caches,
        )

    create_worker.assert_not_called()


def test_generic_worker_path_is_preserved_when_not_required(
    monkeypatch,
) -> None:
    original = MagicMock(return_value="generic-worker")
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_CREATE_WORKER",
        original,
    )
    monkeypatch.delenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", raising=False)
    spec = SimpleNamespace(_ilu_nonuniform_pp_private=False)
    caches = object()

    result = patch_pp_cpu_offloading._ilu_create_cpu_offloading_worker(
        spec,
        caches,
    )

    assert result == "generic-worker"
    original.assert_called_once_with(spec, caches)


def test_runtime_noop_does_not_call_base_rejection() -> None:
    connector = object.__new__(OffloadingConnector)

    patch_pp_cpu_offloading._ilu_offloading_set_xfer_handshake_metadata_pp_aware(
        connector,
        {(7, 1): object()},
    )


def test_cpu_offloading_spec_type_is_available() -> None:
    assert CPUOffloadingSpec is not None


def test_dsv41_pp8_tp1_partition_is_accepted(monkeypatch) -> None:
    per_stage = (100, 200, 300, 400, 500, 600, 700, 800)
    configs = [_kv_config(10, value * 10) for value in per_stage]
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    result = patch_pp_cpu_offloading._ilu_get_kv_cache_configs(
        _config(pp_size=8, tp_size=1, world_size=8),
        [{}] * 8,
        [1] * 8,
    )

    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == per_stage for config in configs
    )


def test_private_worker_init_pins_exact_anonymous_mmap_and_unregisters_on_shutdown(
    monkeypatch,
) -> None:
    import mmap

    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    registered: list[tuple[int, int, int]] = []
    unregistered: list[int] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append((ptr, size, flags))
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    class FakeHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, host_memory_is_pinned, **kw):
            self.gpu_tensors = list(gpu_tensors)
            self.cpu_tensors = list(cpu_tensors)
            self.host_memory_is_pinned = host_memory_is_pinned

        def shutdown(self) -> None:
            self.gpu_tensors.clear()
            self.cpu_tensors.clear()

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(gpu_worker, "MAX_HOST_REGISTER_CHUNK_BYTES", 2 * mmap.PAGESIZE)
    monkeypatch.setattr(
        gpu_worker, "SingleDirectionOffloadingHandler", FakeHandler
    )
    monkeypatch.setattr(
        patch_pp_cpu_offloading,
        "_ORIGINAL_WORKER_SHUTDOWN",
        lambda self: (self._store_handler.shutdown(), self._load_handler.shutdown()),
    )

    gpu_t0 = torch.zeros((4, 3000), dtype=torch.int8)
    gpu_t1 = torch.zeros((4, 1500), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=gpu_t0, page_size_bytes=3000),
            SimpleNamespace(tensor=gpu_t1, page_size_bytes=1500),
        ],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=3000)]],
    )

    worker = SimpleNamespace()
    patch_pp_cpu_offloading._ilu_cpu_offloading_worker_init(
        worker,
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_chunks=3,
        mmap_region=None,
    )

    assert worker._mmap_region is None
    assert len(worker._ilu_private_regions) == 2
    assert worker._store_handler.host_memory_is_pinned is True
    assert worker._load_handler.host_memory_is_pinned is True

    # Tensor 0: 3 * 3000 = 9000 bytes -> 3 pages (12288 bytes), chunked at 2 pages + 1 page.
    # Tensor 1: 3 * 1500 = 4500 bytes -> 2 pages (8192 bytes), 1 chunk of 2 pages.
    assert [size for _, size, _ in registered] == [
        2 * mmap.PAGESIZE,
        1 * mmap.PAGESIZE,
        2 * mmap.PAGESIZE,
    ]
    assert all(ptr % mmap.PAGESIZE == 0 for ptr, _, _ in registered)

    # Verify private anonymous mapping (rw-p, not /dev/shm) in /proc/self/maps.
    ptr0 = worker._store_handler.cpu_tensors[0].data_ptr()
    with open("/proc/self/maps") as maps:
        for line in maps:
            lo, hi = (int(addr, 16) for addr in line.split()[0].split("-"))
            if lo <= ptr0 < hi:
                assert line.split()[1] == "rw-p"
                assert "/dev/shm" not in line
                break
        else:
            raise AssertionError("private anonymous mmap not found in /proc/self/maps")

    mmaps = [region.mmap_obj for region in worker._ilu_private_regions]
    patch_pp_cpu_offloading._ilu_cpu_offloading_worker_shutdown(worker)

    assert worker._ilu_private_regions == []
    assert unregistered == [ptr for ptr, _, _ in reversed(registered)]
    assert all(m.closed for m in mmaps)


def test_private_worker_shutdown_synchronizes_device_when_handler_fails(
    monkeypatch,
) -> None:
    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    events: list[str] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            events.append("unregister")
            return SimpleNamespace(value=0)

    class FailingStoreHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, gpu_to_cpu, **kw):
            self.src_tensors = list(gpu_tensors if gpu_to_cpu else cpu_tensors)
            self.dst_tensors = list(cpu_tensors if gpu_to_cpu else gpu_tensors)
            self.gpu_to_cpu = gpu_to_cpu

        def shutdown(self) -> None:
            if self.gpu_to_cpu:
                events.append("store_shutdown_fail")
                raise RuntimeError("transfer event sync failed")
            events.append("load_shutdown_ok")
            self.src_tensors.clear()
            self.dst_tensors.clear()

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: events.append("device_sync"),
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(
        gpu_worker, "SingleDirectionOffloadingHandler", FailingStoreHandler
    )

    gpu_t0 = torch.zeros((2, 1024), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[SimpleNamespace(tensor=gpu_t0, page_size_bytes=1024)],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=1024)]],
    )

    worker = SimpleNamespace()
    patch_pp_cpu_offloading._ilu_cpu_offloading_worker_init(
        worker,
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_chunks=2,
        mmap_region=None,
    )
    mmap_obj = worker._ilu_private_regions[0].mmap_obj

    patch_pp_cpu_offloading._ilu_cpu_offloading_worker_shutdown(worker)

    assert events == [
        "store_shutdown_fail",
        "load_shutdown_ok",
        "device_sync",
        "unregister",
    ]
    assert mmap_obj.closed
    assert worker._ilu_private_regions == []


def test_private_worker_init_rollback_closes_mmaps_and_unregisters_on_error(
    monkeypatch,
) -> None:
    import mmap

    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    registered: list[int] = []
    unregistered: list[int] = []
    created_mmaps: list[mmap.mmap] = []
    real_mmap = mmap.mmap

    def tracking_mmap(*args, **kwargs):
        m = real_mmap(*args, **kwargs)
        created_mmaps.append(m)
        return m

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append(ptr)
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    class FailOnLoadHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, gpu_to_cpu, **kw):
            self.src_tensors = list(gpu_tensors if gpu_to_cpu else cpu_tensors)
            self.dst_tensors = list(cpu_tensors if gpu_to_cpu else gpu_tensors)
            if not gpu_to_cpu:
                raise RuntimeError("load handler init boom")

        def shutdown(self) -> None:
            self.src_tensors.clear()
            self.dst_tensors.clear()

    monkeypatch.setattr(mmap, "mmap", tracking_mmap)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(
        gpu_worker, "SingleDirectionOffloadingHandler", FailOnLoadHandler
    )

    gpu_t0 = torch.zeros((2, 1024), dtype=torch.int8)
    gpu_t1 = torch.zeros((2, 2048), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=gpu_t0, page_size_bytes=1024),
            SimpleNamespace(tensor=gpu_t1, page_size_bytes=2048),
        ],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=1024)]],
    )

    worker = SimpleNamespace()
    with pytest.raises(RuntimeError, match="load handler init boom"):
        patch_pp_cpu_offloading._ilu_cpu_offloading_worker_init(
            worker,
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_chunks=2,
            mmap_region=None,
        )

    assert len(created_mmaps) == 2
    assert all(m.closed for m in created_mmaps)
    assert unregistered == list(reversed(registered))
    assert worker._ilu_private_regions == []


def test_private_offload_buffer_gc_finalizer_closes_mmap_without_buffer_error(
    monkeypatch,
) -> None:
    import gc

    import torch

    events: list[str] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            events.append("register")
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            events.append("unregister")
            return SimpleNamespace(value=0)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )

    tensor, region = patch_pp_cpu_offloading._allocate_private_cpu_tensor(
        2,
        1024,
        pin_memory=True,
        max_register_chunk_bytes=1 << 20,
    )
    mmap_obj = region.mmap_obj
    assert mmap_obj is not None and not mmap_obj.closed
    assert events == ["register"]

    del tensor, region
    gc.collect()

    assert events == ["register", "unregister"]
    assert mmap_obj.closed


