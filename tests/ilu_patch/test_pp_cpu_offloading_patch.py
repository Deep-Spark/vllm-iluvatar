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

import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.config as config_impl
import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading.worker as reg_impl
import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.offloading_connector as connector_impl
import vllm_iluvatar.models.deepseek_v41.offloading as model_offloading
import vllm_iluvatar.v1.core.kv_cache_utils as kv_impl
import vllm_iluvatar.v1.kv_offload.cpu.gpu_worker as gpu_impl
import vllm_iluvatar.v1.kv_offload.cpu.spec as spec_impl
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

    assert model_offloading._has_native_offloading(multi)
    assert model_offloading._has_native_offloading(direct)
    assert not model_offloading._has_native_offloading(nixl_only)


@pytest.mark.parametrize("empty_stage", [False, True])
def test_nonuniform_pp_geometry_is_attached_to_every_worker(
    monkeypatch,
    empty_stage,
) -> None:
    configs = [_kv_config(10, 1000), _kv_config(10, 3000)]
    if empty_stage:
        configs[0].kv_cache_groups = [SimpleNamespace(layer_names=[])]
        configs[0].hisparse_host_num_blocks = None
        configs[0].prefix_cacheable_group_ids = ()
    monkeypatch.setattr(
        kv_impl,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )

    result = kv_impl._ilu_get_kv_cache_configs(
        _config(),
        [{}, {}],
        [1, 1],
    )

    expected = (0, 300) if empty_stage else (100, 300)
    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == expected for config in configs
    )


def test_dsv41_pp8_partition_is_accepted(monkeypatch) -> None:
    per_stage = (100, 200, 300, 400, 500, 600, 700, 800)
    geometry = tuple(value for value in per_stage for _ in range(2))
    configs = [_kv_config(10, value * 10) for value in geometry]
    monkeypatch.setattr(
        kv_impl,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    result = kv_impl._ilu_get_kv_cache_configs(
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
        kv_impl,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,3,2,2,21",
    )

    result = kv_impl._ilu_get_kv_cache_configs(
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
        kv_impl,
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
        kv_impl._ilu_get_kv_cache_configs(
            _config(pp_size=8, tp_size=2, world_size=16),
            [{}] * 16,
            [1] * 16,
        )


def test_dsv41_rejects_tp_peer_geometry_mismatch(monkeypatch) -> None:
    configs = [_kv_config(10, 1000) for _ in range(16)]
    configs[5] = _kv_config(10, 2000)
    monkeypatch.setattr(
        kv_impl,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    with pytest.raises(ValueError, match="TP ranks within one PP stage"):
        kv_impl._ilu_get_kv_cache_configs(
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
        spec_impl,
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

    spec_impl._ilu_cpu_offloading_spec_init(spec, config)

    assert spec.num_chunks == 10
    assert spec.kv_bytes_per_chunk == 4096
    assert spec.cpu_page_size_per_worker == 3072
    assert spec._ilu_nonuniform_pp_private


def test_geometry_propagates_into_offloading_config(monkeypatch) -> None:
    result_config = SimpleNamespace()
    monkeypatch.setattr(
        config_impl,
        "_ORIGINAL_BUILD_CONFIG",
        lambda *_args: result_config,
    )
    kv_config = SimpleNamespace(
        _ilu_offload_worker_bytes_by_rank=(100, 300),
    )

    result = config_impl._ilu_build_offloading_config(
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
        config_impl,
        "_ORIGINAL_BUILD_CONFIG",
        original,
    )

    with pytest.raises(RuntimeError, match="geometry was not propagated"):
        config_impl._ilu_build_offloading_config(
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

    result = spec_impl._ilu_create_cpu_offloading_worker(
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


def test_zero_byte_rank_acknowledges_offload_jobs() -> None:
    from vllm_iluvatar.v1.worker.empty_offloading import EmptyOffloadingWorker

    spec = SimpleNamespace(
        _ilu_nonuniform_pp_private=True,
        num_chunks=17,
        cpu_page_size_per_worker=0,
        kv_bytes_per_chunk=4096,
        config=SimpleNamespace(
            parallel=SimpleNamespace(rank=0),
            _ilu_offload_worker_bytes_by_rank=(0, 1536),
        ),
    )
    caches = SimpleNamespace(tensors=[], group_data_refs=[[]])

    worker = spec_impl._ilu_create_cpu_offloading_worker(spec, caches)

    assert isinstance(worker, EmptyOffloadingWorker)
    assert worker.submit_store(7, MagicMock(), MagicMock())
    worker.wait({7})
    (result,) = worker.get_finished()
    assert result.job_id == 7 and result.success


def test_local_offloading_ignores_pp_handshake_without_state() -> None:
    connector = object.__new__(OffloadingConnector)
    connector.sentinel = object()
    state_before = vars(connector).copy()

    connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware(
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
            connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware
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
        spec_impl,
        "_ORIGINAL_CREATE_WORKER",
        original,
    )

    with pytest.raises(RuntimeError, match="cannot use generic worker path"):
        spec_impl._ilu_create_cpu_offloading_worker(
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

    expected = connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware
    assert OffloadingConnector.set_xfer_handshake_metadata_pp_aware is expected


def test_required_spec_rejects_missing_geometry_or_marker(
    monkeypatch,
) -> None:
    original = MagicMock()
    monkeypatch.setattr(
        spec_impl,
        "_ORIGINAL_SPEC_INIT",
        original,
    )
    monkeypatch.setenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", "1")

    with pytest.raises(RuntimeError, match="geometry/marker is missing"):
        spec_impl._ilu_cpu_offloading_spec_init(
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

    assert model_offloading._has_native_offloading(config)


def test_non_target_model_does_not_need_rank_safe_geometry() -> None:
    config = _config()
    config.model_config.architectures = ["OtherArchitecture"]
    config.model_config.hf_config.model_type = "other"

    assert not model_offloading._needs_rank_safe_geometry(config)


def test_pp1_does_not_need_rank_safe_geometry() -> None:
    config = _config(pp_size=1, tp_size=2, world_size=2)

    assert not model_offloading._needs_rank_safe_geometry(config)


def test_invalid_world_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="world_size == pp_size"):
        model_offloading._validate_target_geometry(
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
        spec_impl,
        "_ORIGINAL_SPEC_INIT",
        original_init,
    )

    with pytest.raises(ValueError, match="smaller than one aggregate"):
        spec_impl._ilu_cpu_offloading_spec_init(
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
        spec_impl,
        "_ORIGINAL_SPEC_INIT",
        original_init,
    )

    with pytest.raises(ValueError, match="canonical_layout is not supported"):
        spec_impl._ilu_cpu_offloading_spec_init(
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
        spec_impl._ilu_create_cpu_offloading_worker(
            spec,
            caches,
        )

    create_worker.assert_not_called()


def test_generic_worker_path_is_preserved_when_not_required(
    monkeypatch,
) -> None:
    original = MagicMock(return_value="generic-worker")
    monkeypatch.setattr(
        spec_impl,
        "_ORIGINAL_CREATE_WORKER",
        original,
    )
    monkeypatch.delenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", raising=False)
    spec = SimpleNamespace(_ilu_nonuniform_pp_private=False)
    caches = object()

    result = spec_impl._ilu_create_cpu_offloading_worker(
        spec,
        caches,
    )

    assert result == "generic-worker"
    original.assert_called_once_with(spec, caches)


def test_runtime_noop_does_not_call_base_rejection() -> None:
    connector = object.__new__(OffloadingConnector)

    connector_impl._ilu_offloading_set_xfer_handshake_metadata_pp_aware(
        connector,
        {(7, 1): object()},
    )


def test_cpu_offloading_spec_type_is_available() -> None:
    assert CPUOffloadingSpec is not None


def test_dsv41_pp8_tp1_partition_is_accepted(monkeypatch) -> None:
    per_stage = (100, 200, 300, 400, 500, 600, 700, 800)
    configs = [_kv_config(10, value * 10) for value in per_stage]
    monkeypatch.setattr(
        kv_impl,
        "_ORIGINAL_GET_CONFIGS",
        lambda *_args: configs,
    )
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        "3,3,3,3,2,3,2,21",
    )

    result = kv_impl._ilu_get_kv_cache_configs(
        _config(pp_size=8, tp_size=1, world_size=8),
        [{}] * 8,
        [1] * 8,
    )

    assert result is configs
    assert all(
        config._ilu_offload_worker_bytes_by_rank == per_stage for config in configs
    )


@pytest.mark.parametrize(
    "raw, required", [("1", True), ("true", True), ("0", False), ("false", False)]
)
def test_required_flag_uses_registered_boolean_reader(monkeypatch, raw, required):
    monkeypatch.setenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", raw)
    sentinel = object()
    monkeypatch.setattr(spec_impl, "_ORIGINAL_CREATE_WORKER", lambda *_: sentinel)
    if required:
        with pytest.raises(RuntimeError, match="generic worker path"):
            spec_impl._ilu_create_cpu_offloading_worker(SimpleNamespace(), None)
    else:
        assert (
            spec_impl._ilu_create_cpu_offloading_worker(SimpleNamespace(), None)
            is sentinel
        )


def test_required_flag_respects_explicit_registered_disable(monkeypatch):
    from vllm_iluvatar import envs

    monkeypatch.setenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", "true")
    monkeypatch.setitem(envs.__dict__, "VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", False)
    sentinel = object()
    monkeypatch.setattr(spec_impl, "_ORIGINAL_CREATE_WORKER", lambda *_: sentinel)
    assert (
        spec_impl._ilu_create_cpu_offloading_worker(SimpleNamespace(), None) is sentinel
    )


def test_wiring_preserves_aliases_idempotence_and_provenance():
    # Isolate process-wide rebind propagation from other tests' imports.
    import os
    import subprocess
    import sys

    code = r"""
import sys
from types import ModuleType
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm_iluvatar.ilu_patch.general_patches import patch_pp_cpu_offloading as patch
from vllm_iluvatar.v1.kv_offload.cpu import gpu_worker as impl

alias = ModuleType('vllm._ilu_offload_alias_test')
alias.get_kv_cache_configs = kv_cache_utils.get_kv_cache_configs
sys.modules[alias.__name__] = alias
patch._apply()
original = impl._ORIGINAL_WORKER_INIT
patch._apply()
assert impl._ORIGINAL_WORKER_INIT is original
assert CPUOffloadingWorker.__init__ is impl._ilu_cpu_offloading_worker_init
assert CPUOffloadingWorker.__init__.__module__ == impl.__name__
assert alias.get_kv_cache_configs is kv_cache_utils.get_kv_cache_configs
patch.attest_required_bindings()
"""
    env = os.environ.copy()
    env.update(
        VLLM_ILUVATAR_ENABLE_STARTUP_PATCH="0",
        VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED="true",
        VLLM_ILU_PATCH_CHECK_FINGERPRINT="1",
        VLLM_ILU_PATCH_CHECK_SIGNATURE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("method", ["__init__", "shutdown"])
def test_worker_source_drift_only_blocks_private_allocation(monkeypatch, method):
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    monkeypatch.setenv("VLLM_ILU_PATCH_CHECK_FINGERPRINT", "1")
    snapshot = "WORKER_INIT" if method == "__init__" else "WORKER_SHUTDOWN"
    monkeypatch.setattr(patch_pp_cpu_offloading, snapshot, "def changed(): pass")
    patch_pp_cpu_offloading.validate_private_worker_baseline.cache_clear()
    patch_pp_cpu_offloading._apply()
    assert CPUOffloadingWorker.__init__ is gpu_impl._ilu_cpu_offloading_worker_init

    original_init = MagicMock()
    with monkeypatch.context() as shared:
        shared.setattr(gpu_impl, "_ORIGINAL_WORKER_INIT", original_init)
        CPUOffloadingWorker(object(), 1, 1, mmap_region=object())
    original_init.assert_called_once()

    with pytest.raises(RuntimeError, match=f"{method} source changed"):
        CPUOffloadingWorker(object(), 1, 1)


@pytest.mark.parametrize("paired", [False, True])
def test_registration_selects_paired_pages_or_community(monkeypatch, paired):
    from vllm_iluvatar.models.deepseek_v41 import offloading

    pages, caches, context, config = object(), object(), object(), object()
    calls = []
    worker = SimpleNamespace(
        kv_cache_config=config,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        ),
        _init_worker=lambda value: calls.append(value),
    )
    monkeypatch.setattr(
        offloading, "has_paired_pages", lambda value: value is config and paired
    )

    def canonical(actual_config, actual_caches, actual_context):
        assert (actual_config, actual_caches, actual_context) == (
            config,
            caches,
            context,
        )
        return pages

    monkeypatch.setattr(offloading, "canonical_paired_pages", canonical)
    monkeypatch.setattr(
        reg_impl,
        "_ORIGINAL_REGISTER_KV",
        lambda self, value: calls.append((self, value)),
    )
    reg_impl._ilu_register_kv_caches(worker, caches)
    assert calls == ([pages] if paired else [(worker, caches)])
