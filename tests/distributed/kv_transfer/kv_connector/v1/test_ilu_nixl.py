# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.request import RequestStatus

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl import (
    base_scheduler as base_scheduler_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl import (
    base_worker as base_worker_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_scheduler import (
    IluNixlBaseConnectorScheduler,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_worker import (
    IluNixlWorkerMixin,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
    IluNixlConnectorMetadata,
    IluNixlTransferRegion,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pp_utils import (
    infer_pp_split,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pull_scheduler import (
    IluNixlPullConnectorScheduler,
)


class _FakeUpstreamNixlWorker:
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        self.vllm_config = vllm_config
        self.pp_seen_during_init = vllm_config.parallel_config.pipeline_parallel_size
        is_hma_required = any(
            not isinstance(group.kv_cache_spec, FullAttentionSpec)
            for group in kv_cache_config.kv_cache_groups
        )
        if self.pp_seen_during_init > 1 and is_hma_required:
            raise NotImplementedError("upstream PP + HMA guard")
        if (
            self.pp_seen_during_init > 1
            and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        ):
            raise NotImplementedError("upstream consumer PP guard")
        self.pp_size = self.pp_seen_during_init


class _TestIluNixlWorker(IluNixlWorkerMixin, _FakeUpstreamNixlWorker):
    pass


def _region(
    region_id: int,
    block_len: int,
    identities: list[str],
) -> IluNixlTransferRegion:
    return IluNixlTransferRegion(
        region_id=region_id,
        block_len=block_len,
        shared_by=[],
        layer_indices=[],
        group_ids=[],
        cache_kinds=[],
        identities=identities,
    )


@pytest.mark.parametrize(
    ("pp_size", "role", "requires_hma", "expected_init_pp", "mask_register"),
    [
        (2, "kv_consumer", False, 1, False),
        (2, "kv_producer", False, 2, False),
        (2, "kv_producer", True, 1, True),
        (1, "kv_consumer", False, 1, False),
    ],
)
def test_upstream_pp_guards_are_masked_by_phase(
    monkeypatch,
    pp_size,
    role,
    requires_hma,
    expected_init_pp,
    mask_register,
):
    monkeypatch.setenv(
        "VLLM_PP_LAYER_PARTITION",
        ",".join("1" for _ in range(pp_size)),
    )
    monkeypatch.setattr(
        base_worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=pp_size),
    )

    full_attention_spec = object.__new__(FullAttentionSpec)
    kv_cache_spec = SimpleNamespace() if requires_hma else full_attention_spec
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_cache_spec)]
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        kv_transfer_config=SimpleNamespace(kv_role=role, kv_connector="NixlConnector"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )

    worker = _TestIluNixlWorker(vllm_config, "engine", kv_cache_config)

    assert worker.pp_seen_during_init == expected_init_pp
    assert vllm_config.parallel_config.pipeline_parallel_size == pp_size
    assert worker.pp_size == pp_size
    assert worker._mask_pp_during_register is mask_register


def test_dsv2_mla_consumer_pp_bypasses_upstream_init_guard(monkeypatch):
    """DSV2 MLA is a FullAttentionSpec but decode PP is rejected upstream."""
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "14,13")
    monkeypatch.setattr(
        base_worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2),
    )

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object.__new__(MLAAttentionSpec))
        ]
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )

    worker = _TestIluNixlWorker(vllm_config, "dsv2-decode", kv_cache_config)

    assert worker.pp_seen_during_init == 1
    assert vllm_config.parallel_config.pipeline_parallel_size == 2
    assert worker.pp_size == 2
    assert worker._mask_pp_during_register is False


def test_pp_layer_range_and_targets():
    assert IluNixlWorkerMixin._pp_layer_range("2,3,4", 1) == (2, 5)
    assert IluNixlWorkerMixin._target_pp_ranks("3,3", 0, "2,2,2") == [0, 1]
    assert IluNixlWorkerMixin._overlap_pp_count(1, "2,2,2", "3,3") == 2


def test_compatibility_hash_is_independent_of_stage_local_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_worker_module,
        "compute_nixl_compatibility_hash",
        lambda config, backend, transfer_mode: calls.append(
            (config, backend, transfer_mode)
        )
        or f"{backend}:{transfer_mode}",
    )
    worker = object.__new__(IluNixlWorkerMixin)
    worker.vllm_config = object()

    first = worker._compute_compatibility_hash(False)
    worker.backend_name = "A_DIFFERENT_LOCAL_PP_BACKEND"
    second = worker._compute_compatibility_hash(False)

    assert first == second == "ILU_NIXL_SEMANTIC_REGIONS_V1:pull"
    assert calls[0][1] == calls[1][1] == "ILU_NIXL_SEMANTIC_REGIONS_V1"


@pytest.mark.parametrize("method", ["dspark", "mtp"])
def test_compatibility_hash_matches_when_only_decode_enables_dspark_or_mtp(
    method: str,
):
    def _make_worker(spec_cfg):
        worker = object.__new__(IluNixlWorkerMixin)
        worker.fp8_wire_staging_enabled = False
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="deepseek-v41",
                dtype=torch.bfloat16,
                get_total_num_kv_heads=lambda: 1,
                get_head_size=lambda: 512,
                get_total_num_hidden_layers=lambda: 40,
            ),
            cache_config=SimpleNamespace(cache_dtype="bfloat16", block_size=64),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            speculative_config=spec_cfg,
        )
        return worker

    prefill_hash = _make_worker(None)._compute_compatibility_hash(False)
    decode_spec = SimpleNamespace(
        method=method,
        use_eagle=lambda: True,
        kv_cache_dtype=None,
        parallel_drafting=False,
        draft_model_config=SimpleNamespace(
            model="draft",
            revision=None,
            code_revision=None,
            hf_config=SimpleNamespace(),
        ),
    )
    decode_hash = _make_worker(decode_spec)._compute_compatibility_hash(False)
    assert prefill_hash == decode_hash


def test_detects_qwen_kv_first_fa_layout():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.use_mla = False
    worker._has_mamba = False
    worker.num_blocks = 2
    worker._layer_specs = {"model.layers.0.self_attn.attn": object()}

    assert worker._uses_kv_first_fa_layout(
        {
            "model.layers.0.self_attn.attn": torch.zeros(
                (2, 2, 1, 16, 8), dtype=torch.bfloat16
            )
        }
    )
    assert not worker._uses_kv_first_fa_layout(
        {
            "model.layers.0.self_attn.attn": torch.zeros(
                (2, 1, 16, 16), dtype=torch.bfloat16
            )
        }
    )


def test_registers_qwen_kv_first_components_as_separate_regions():
    class FakeNixlWrapper:
        def __init__(self):
            self.registered = None

        def get_reg_descs(self, data, _memory_type):
            return data

        def register_memory(self, descs, backends):
            self.registered = (descs, backends)

        def get_agent_metadata(self):
            return b"agent"

    layer_name = "model.layers.0.self_attn.attn"
    cache = torch.zeros((2, 2, 1, 16, 8), dtype=torch.bfloat16)
    component_block_len = cache[0].stride(0) * cache.element_size()
    layer_spec = SimpleNamespace(page_size_bytes=2 * component_block_len)
    group = SimpleNamespace(
        layer_names=[layer_name],
        kv_cache_spec=layer_spec,
    )

    worker = object.__new__(IluNixlWorkerMixin)
    worker.use_host_buffer = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker.num_blocks = 2
    worker.world_size = 1
    worker.tp_rank = 0
    worker.engine_id = "engine"
    worker.block_size = 16
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 1)
    worker.vllm_config = object()
    worker._has_mamba = False
    worker._layer_specs = {layer_name: layer_spec}
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    worker._compute_compatibility_hash = lambda _: "stable-hash"
    worker.nixl_memory_type = "DRAM"
    worker.nixl_backends = ["UCX"]
    worker.nixl_wrapper = FakeNixlWrapper()
    worker._registered_descs = []
    worker.kv_caches_base_addr = {"engine": {0: []}}
    worker.dst_num_blocks = {}
    worker.region_mem_types = []
    worker.dst_region_mem_types = {}
    worker.src_xfer_handles_by_block_size = {}
    worker.register_local_xfer_handler = lambda _: (
        7,
        np.zeros((4, 3), dtype=np.uint64),
    )
    worker.backend_name = "FLASH_ATTN"
    worker.kv_cache_layout = "NHD"
    worker._mamba_ssm_size = (0, 0)
    worker.pp_split = "1"
    worker.pp_rank = 0

    worker._register_kv_first_fa_caches({layer_name: cache})

    assert worker.block_len_per_layer == [
        component_block_len,
        component_block_len,
    ]
    assert worker.layer_region_ids == {layer_name: [0, 1]}
    assert [ids.tolist() for ids in worker.region_ids_by_group] == [[0, 1]]
    assert worker.num_regions == 2
    assert worker.num_descs == 4
    assert worker.compat_hash == "stable-hash"
    assert worker.xfer_handshake_metadata.compatibility_hash == "stable-hash"
    registered, backends = worker.nixl_wrapper.registered
    assert [entry[1] for entry in registered] == [
        2 * component_block_len,
        2 * component_block_len,
    ]
    assert backends == ["UCX"]


def test_infer_pp_split_single_stage_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=None,
            hf_config=SimpleNamespace(num_hidden_layers=43),
        ),
    )

    assert infer_pp_split(vllm_config) == "43"


def test_infer_pp_split_respects_env(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "21,22")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(
            hf_text_config=None,
            hf_config=SimpleNamespace(num_hidden_layers=43),
        ),
    )

    assert infer_pp_split(vllm_config) == "21,22"


def test_remote_transfer_regions_uses_registered_block_lengths():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._is_hma_required = False
    worker.transfer_topo = SimpleNamespace(block_size_ratio=lambda _: 2)
    worker._group_spec_types = [FullAttentionSpec]
    worker.tp_mappings = {
        "remote": SimpleNamespace(
            source_ranks_per_group=[[0, 1]],
            rank_offset_factor=1,
        )
    }
    worker.block_len_per_layer = [2048]
    worker._is_region_replicated = lambda _: False
    remote_meta = SimpleNamespace(block_size=128)

    assert worker._remote_transfer_regions("remote", remote_meta, [0], [3]) == [
        (3, 512, 1024)
    ]


def test_cache_identity_and_region_range_selection():
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name("model.layers.16.attn.swa_cache")
        == "swa_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.16.attn.indexer.k_cache"
        )
        == "indexer_k_cache"
    )
    assert (
        IluNixlWorkerMixin._transfer_identity("model.layers.16.attn.swa_cache")
        == "16:swa_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.2.attn.kv_cache_scales"
        )
        == "main_mla_scales"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.2.attn.indexer.k_cache.kv_cache_scales"
        )
        == "indexer_k_cache_scales"
    )

    layer_region_ids = {
        "model.layers.2.attn": [20],
        "model.layers.0.attn": [0, 1],
        "model.layers.1.attn.swa_cache": [10, 11],
        "model.embed_tokens": [99],
    }
    assert IluNixlWorkerMixin._region_ids_for_layer_range(layer_region_ids, 1, 3) == [
        10,
        11,
        20,
    ]


def test_mamba_regions_use_upstream_sub_descriptors_without_virtual_split():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.block_len_per_layer = [262144, 262144]
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=True)
    worker._has_mamba = True
    worker._is_region_replicated = lambda _: False

    assert worker._actual_to_virtual_region_ids() == {0: [0], 1: [1]}
    assert worker._block_len_for_virtual_region(1) == 262144


def test_filter_group_region_ids_preserves_order_and_compacts():
    groups = [[4, 1, 7, 4], np.asarray([2, 7, 9])]
    selected = [7, 1]

    filtered = IluNixlWorkerMixin._filter_group_region_ids(
        groups, selected, compact=False
    )
    assert [group.tolist() for group in filtered] == [[1, 7], [7]]

    compacted = IluNixlWorkerMixin._filter_group_region_ids(
        groups, selected, compact=True
    )
    assert [group.tolist() for group in compacted] == [[1, 0], [0]]


def test_heterogeneous_pp_rejects_non_mla_layout_mismatch():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.transfer_topo = object()
    worker.use_host_buffer = False
    worker.kv_cache_layout = "NHD"
    worker.use_mla = False
    remote = SimpleNamespace(kv_cache_layout="HND")

    with pytest.raises(RuntimeError, match="same non-MLA KV cache layout"):
        worker._validate_pp_selected_region_lengths(remote, [0], [0], 1)


def test_remote_transfer_regions_uses_registered_region_lengths():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._is_hma_required = False
    worker.transfer_topo = SimpleNamespace(block_size_ratio=lambda _: 1)
    worker._group_spec_types = (FullAttentionSpec,)
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            source_ranks_per_group=((0, 1),),
            rank_offset_factor=1,
        )
    }
    worker.block_len_per_layer = [256]
    worker._region_is_mla = [False]
    worker._is_region_replicated = lambda _: False
    remote = SimpleNamespace(block_size=16)

    assert worker._remote_transfer_regions(
        "prefill",
        remote,
        selected_local_region_ids=[0],
        selected_remote_region_ids=[7],
    ) == [(7, 128, 256)]


def test_hma_remote_transfer_regions_slices_larger_prefill_tp_page():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.transfer_topo = SimpleNamespace(block_size_ratio=lambda _: 1)
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            source_ranks_per_group=((0,), (0,)),
            rank_offset_factor=1,
        )
    }
    worker.block_len_per_layer = [256]
    worker._is_region_replicated = lambda _: False
    remote = SimpleNamespace(block_size=16)

    assert worker._remote_transfer_regions(
        "prefill",
        remote,
        selected_local_region_ids=[0],
        selected_remote_region_ids=[7],
    ) == [(7, 256, 256)]


def test_mamba_region_desc_ids_follow_fa_descriptors():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._conv_decomp = SimpleNamespace(local_conv_offsets=[(0, 1)] * 3)

    desc_ids = worker._compute_mamba_desc_ids_for_regions(
        block_ids=[1],
        region_ids=np.asarray([1], dtype=np.int64),
        num_physical_blocks=10,
        physical_blocks_per_logical=5,
        num_fa_regions=2,
    )

    assert desc_ids.tolist() == [29, 31, 33, 35]


def test_hybrid_fa_desc_ids_select_k_and_v_components():
    desc_ids = IluNixlWorkerMixin._compute_fa_component_desc_ids(
        block_ids=[1],
        region_ids=np.asarray([1], dtype=np.int64),
        num_blocks=10,
    )

    assert desc_ids.tolist() == [21, 31]


def test_mamba_desc_ids_follow_hybrid_kv_component_descriptors():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._conv_decomp = SimpleNamespace(local_conv_offsets=[(0, 1)] * 3)

    desc_ids = worker._compute_mamba_desc_ids_for_regions(
        block_ids=[1],
        region_ids=np.asarray([1], dtype=np.int64),
        num_physical_blocks=10,
        physical_blocks_per_logical=5,
        num_fa_regions=2,
        num_fa_components=2,
    )

    assert desc_ids.tolist() == [49, 51, 53, 55]


def test_align_hma_transfer_regions_by_identity():
    local_regions = [
        _region(0, 2048, ["15:main_mla:group0"]),
        _region(6, 65536, ["15:swa_cache:group1", "16:main_mla:group0"]),
    ]
    remote_regions = [
        _region(26, 2048, ["15:main_mla:group0"]),
        _region(16, 65536, ["15:swa_cache:group1", "16:main_mla:group0"]),
    ]

    local_groups, remote_groups, selected_local, selected_remote = (
        IluNixlWorkerMixin._align_hma_transfer_regions(
            [0, 6], [26, 16], local_regions, remote_regions, num_groups=2
        )
    )

    assert [group.tolist() for group in local_groups] == [[0, 6], [6]]
    assert [group.tolist() for group in remote_groups] == [[0, 1], [1]]
    assert selected_local == [0, 6]
    assert selected_remote == [26, 16]


def test_align_hma_transfer_regions_allows_heterogeneous_tp_lengths():
    local_regions = [_region(1, 131072, ["0:unknown:group0"])]
    remote_regions = [_region(1, 262144, ["0:unknown:group0"])]

    local_groups, remote_groups, selected_local, selected_remote = (
        IluNixlWorkerMixin._align_hma_transfer_regions(
            [1], [1], local_regions, remote_regions, num_groups=1
        )
    )

    assert [group.tolist() for group in local_groups] == [[1]]
    assert [group.tolist() for group in remote_groups] == [[0]]
    assert selected_local == [1]
    assert selected_remote == [1]


def test_align_hma_transfer_regions_rejects_conflicts():
    local_regions = [_region(0, 2048, ["1:main_mla:group0", "1:swa_cache:group0"])]
    remote_regions = [
        _region(10, 2048, ["1:main_mla:group0"]),
        _region(11, 2048, ["1:swa_cache:group0"]),
    ]

    with pytest.raises(RuntimeError, match="slice/scatter transfer"):
        IluNixlWorkerMixin._align_hma_transfer_regions(
            [0], [10, 11], local_regions, remote_regions, num_groups=1
        )


def test_metadata_add_new_req_carries_remote_pp_topology():
    meta = IluNixlConnectorMetadata()
    req_meta = meta._add_new_req(
        local_block_ids=[[1, 2], [3]],
        kv_transfer_params={
            "tp_size": 4,
            "remote_block_size": 16,
            "pp_size": 2,
            "pp_split": "11,10",
        },
    )

    assert req_meta.local_block_ids == [[1, 2], [3]]
    assert req_meta.local_physical_block_ids == [[1, 2], [3]]
    assert req_meta.tp_size == 4
    assert req_meta.remote_block_size == 16
    assert req_meta.remote_pp_size == 2
    assert req_meta.remote_pp_split == "11,10"


def test_metadata_save_carries_num_tokens():
    meta = IluNixlConnectorMetadata()
    meta.add_new_req_to_save(
        request_id="req",
        local_block_ids=([1, 2],),
        kv_transfer_params={"tp_size": 8},
        num_tokens=257,
    )

    assert meta.reqs_to_save["req"].num_tokens == 257


def test_read_blocks_skips_decode_only_eagle_group():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda _: SimpleNamespace(
            remote_tp_size=1,
            remote_physical_blocks_per_logical=1,
            remote_block_size=16,
        ),
        tp_ratio=lambda _: 1,
    )
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            source_ranks_per_group=((0,), (0,), (0,)),
        )
    }
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(is_eagle_group=False),
            SimpleNamespace(is_eagle_group=False),
            SimpleNamespace(is_eagle_group=True),
        ]
    )
    worker._engine_last_active = {}
    worker._remote_agents = {"prefill": {0: "agent"}}
    worker.src_xfer_handles_by_block_size = {16: 100}
    worker.dst_xfer_side_handles = {"prefill": {0: 200}}
    worker._is_heterogeneous_pp_engine = lambda _: True
    worker._logical_to_kernel_block_ids = lambda block_ids, _: block_ids

    captured = []
    worker._read_blocks = lambda **kwargs: captured.append(kwargs["read_spec"])
    meta = SimpleNamespace(
        remote=SimpleNamespace(
            engine_id="prefill",
            request_id="remote-req",
            block_ids=([10], [20]),
        ),
        local_physical_block_ids=([1], [2], [3]),
    )

    worker._read_blocks_for_req("req", meta)

    assert len(captured) == 1
    assert captured[0].local_block_ids == [[1], [2]]
    assert captured[0].remote_block_ids == [[10], [20]]


def test_scheduler_staging_accumulates_chunks_reuses_swa_and_exports(monkeypatch):
    scheduler = object.__new__(IluNixlBaseConnectorScheduler)
    scheduler.fp8_wire_staging_enabled = True
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    request = SimpleNamespace(
        request_id="req",
        kv_transfer_params={"tp_size": 8},
        num_computed_tokens=0,
        num_prompt_tokens=8,
    )
    scheduler._reqs_need_save = {"req": request}
    scheduler._fp8_staging_block_ids = {"req": [[1], [10]]}
    scheduler._fp8_exported_block_ids = {}
    scheduler.get_exchange_clipped_blocks = lambda groups, clip_ssm=True: (
        groups[0],
        groups[1][-1:],
    )
    output = SimpleNamespace(num_scheduled_tokens={"req": 4})
    monkeypatch.setattr(
        base_scheduler_module,
        "yield_req_data",
        lambda _: iter(()),
        raising=False,
    )
    community_scheduler = __import__(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler",
        fromlist=["yield_req_data"],
    )
    monkeypatch.setattr(
        community_scheduler,
        "yield_req_data",
        lambda _: iter([("req", ([2], [11]), None)]),
    )

    first_meta = IluNixlConnectorMetadata()
    scheduler._build_save_meta(first_meta, output)
    assert not first_meta.reqs_to_save

    request.num_computed_tokens = 4
    monkeypatch.setattr(
        community_scheduler,
        "yield_req_data",
        lambda _: iter([("req", ([3], [10]), None)]),
    )
    final_meta = IluNixlConnectorMetadata()
    scheduler._build_save_meta(final_meta, output)

    saved = final_meta.reqs_to_save["req"]
    assert saved.local_block_ids == ([1, 2, 3], [10])
    assert saved.num_tokens == 8
    assert scheduler._fp8_exported_block_ids["req"] == (
        [1, 2, 3],
        [10],
    )
    assert "req" not in scheduler._fp8_staging_block_ids


def test_scheduler_finished_exports_staged_block_ids(monkeypatch):
    captured = {}

    def finish(_self, _request, block_ids):
        captured["block_ids"] = block_ids
        return True, {"remote_block_ids": block_ids}

    monkeypatch.setattr(NixlPullConnectorScheduler, "request_finished", finish)
    scheduler = object.__new__(IluNixlPullConnectorScheduler)
    scheduler.fp8_wire_staging_enabled = True
    scheduler._fp8_staging_block_ids = {}
    scheduler._fp8_exported_block_ids = {"req": ([1, 2], [7])}
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(
            hf_text_config=None,
            hf_config=SimpleNamespace(num_hidden_layers=4),
        ),
    )
    request = SimpleNamespace(
        request_id="req",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={"do_remote_decode": True},
    )

    delay, params = scheduler.request_finished(request, ([99], [98]))

    assert delay
    assert captured["block_ids"] == ([1, 2], [7])
    assert params["remote_block_ids"] == ([1, 2], [7])
    assert params["pp_size"] == 2


def test_scheduler_abort_cleans_staging_state(monkeypatch):
    def finish(_self, _request, _block_ids):
        return False, None

    monkeypatch.setattr(NixlPullConnectorScheduler, "request_finished", finish)
    scheduler = object.__new__(IluNixlPullConnectorScheduler)
    scheduler.fp8_wire_staging_enabled = True
    scheduler._fp8_staging_block_ids = {"req": [[1]]}
    scheduler._fp8_exported_block_ids = {"req": ([1],)}
    request = SimpleNamespace(
        request_id="req",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={"do_remote_decode": True},
    )

    assert scheduler.request_finished(request, ([1],)) == (False, None)
    assert "req" not in scheduler._fp8_staging_block_ids
    assert "req" not in scheduler._fp8_exported_block_ids


def test_virtual_slot_descriptor_address_math():
    assert IluNixlWorkerMixin._virtual_slot_desc(
        backing_base=1000,
        block_id=3,
        block_stride=256,
        slot_offset=48,
        page_size=64,
        device_id=2,
    ) == (1816, 64, 2)

    # Layer-major packing: layer 2 starts two block strides past the base.
    assert IluNixlWorkerMixin._virtual_slot_desc(
        backing_base=1000,
        block_id=3,
        block_stride=256,
        slot_offset=512,
        page_size=256,
        device_id=2,
    ) == (2280, 256, 2)

    # Layer-outer mixed-page pool: layer base need not be stride-aligned.
    assert IluNixlWorkerMixin._virtual_slot_desc(
        backing_base=0,
        block_id=1,
        block_stride=131072,
        slot_offset=2682322944,
        page_size=131072,
        device_id=0,
    ) == (2682322944 + 131072, 131072, 0)

    with pytest.raises(ValueError, match="outside its block stride"):
        IluNixlWorkerMixin._virtual_slot_desc(1000, 0, 64, 32, 64, 0)


def test_packed_virtual_layout_splits_overlaid_mixed_page_sizes():
    small_layer = "model.layers.2.attn.indexer.k_cache"
    large_layer = "model.layers.0.attn.swa_cache"
    small_spec = SimpleNamespace(page_size_bytes=16)
    large_spec = SimpleNamespace(page_size_bytes=32)
    groups = [
        SimpleNamespace(layer_names=[small_layer], kv_cache_spec=small_spec),
        SimpleNamespace(layer_names=[large_layer], kv_cache_spec=large_spec),
    ]
    layout_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=128,
                layers=[small_layer, large_layer],
                offset=0,
                block_stride=64,
            )
        ],
    )

    backing = torch.zeros(128, dtype=torch.uint8)
    kv_caches = {
        small_layer: torch.as_strided(
            backing, size=(2, 16), stride=(64, 1), storage_offset=0
        ),
        large_layer: torch.as_strided(
            backing, size=(2, 32), stride=(64, 1), storage_offset=0
        ),
    }
    worker = object.__new__(IluNixlWorkerMixin)
    worker.kv_cache_config = layout_config

    layer_region_ids, region_ids_by_group, regions, region_is_mla = (
        worker._build_packed_virtual_layout(kv_caches, layout_config, layout_config)
    )

    assert layer_region_ids == {small_layer: [0], large_layer: [1]}
    assert [ids.tolist() for ids in region_ids_by_group] == [[0], [1]]
    assert [region.block_len for region in regions] == [16, 32]
    assert [region.byte_offset for region in regions] == [0, 0]
    assert [region.block_stride for region in regions] == [64, 64]
    assert [region.physical_backing_id for region in regions] == [0, 0]
    assert [region.shared_by for region in regions] == [
        [small_layer],
        [large_layer],
    ]
    assert region_is_mla == [False, False]


def test_packed_virtual_layout_places_each_layer_at_its_page():
    page = 65536
    block_stride = page * 3
    num_blocks = 2
    layers = [
        "model.layers.0.attn",
        "model.layers.2.attn.swa_cache",
        "model.layers.2.attn.indexer.k_cache",
    ]
    specs = {name: SimpleNamespace(page_size_bytes=page) for name in layers}
    groups = [
        SimpleNamespace(layer_names=[name], kv_cache_spec=specs[name])
        for name in layers
    ]
    layout_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=block_stride * num_blocks,
                layers=layers,
                offset=0,
                layer_stride=page,
                block_stride=block_stride,
            )
        ],
    )

    backing = torch.zeros(block_stride * num_blocks, dtype=torch.uint8)
    kv_caches = {
        name: torch.as_strided(
            backing,
            size=(num_blocks, page),
            stride=(block_stride, 1),
            storage_offset=index * page,
        )
        for index, name in enumerate(layers)
    }
    worker = object.__new__(IluNixlWorkerMixin)
    worker.kv_cache_config = layout_config

    _layer_region_ids, _region_ids_by_group, regions, _region_is_mla = (
        worker._build_packed_virtual_layout(kv_caches, layout_config, layout_config)
    )

    assert {tuple(region.shared_by): region.byte_offset for region in regions} == {
        (layers[0],): 0,
        (layers[1],): page,
        (layers[2],): page * 2,
    }
    assert [region.block_len for region in regions] == [page, page, page]
    assert [region.block_stride for region in regions] == [block_stride] * 3


def test_packed_virtual_layout_layer_outer_mixed_pages_share_pool():
    """LBHNC-style shared pool: backing need not be a multiple of main stride*N.

    MiniMax packs main (large page) and indexer (small page) into one buffer.
    Under layer-outer, each tensor's block_stride is its own page, so
    bytes_per_block / main_page is not an integer when L is not a multiple of
    main_page/gcd. The old check ``backing % (stride * N) == 0`` rejected this.
    """
    num_blocks = 2
    main_page = 8
    idx_page = 1
    main_layers = [
        "model.layers.0.attn",
        "model.layers.1.attn",
    ]
    idx_layers = [
        "model.layers.0.attn.indexer.k_cache",
        "model.layers.1.attn.indexer.k_cache",
    ]
    bytes_per_block = len(main_layers) * main_page + len(idx_layers) * idx_page
    size = bytes_per_block * num_blocks
    assert size % (main_page * num_blocks) != 0

    main_spec = SimpleNamespace(page_size_bytes=main_page)
    idx_spec = SimpleNamespace(page_size_bytes=idx_page)
    # One group per layer so page sizes stay correct (not a single UniformType).
    groups = [
        *[
            SimpleNamespace(layer_names=[name], kv_cache_spec=main_spec)
            for name in main_layers
        ],
        *[
            SimpleNamespace(layer_names=[name], kv_cache_spec=idx_spec)
            for name in idx_layers
        ],
    ]
    main_layer_stride = main_page * num_blocks
    idx_layer_stride = idx_page * num_blocks
    idx_offset = len(main_layers) * main_page * num_blocks
    layout_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=size,
                layers=main_layers,
                offset=0,
                layer_stride=main_layer_stride,
                block_stride=main_page,
            ),
            SimpleNamespace(
                size=size,
                layers=idx_layers,
                offset=idx_offset,
                layer_stride=idx_layer_stride,
                block_stride=idx_page,
            ),
        ],
    )

    backing = torch.zeros(size, dtype=torch.uint8)
    kv_caches = {}
    for index, name in enumerate(main_layers):
        kv_caches[name] = torch.as_strided(
            backing,
            size=(num_blocks, main_page),
            stride=(main_page, 1),
            storage_offset=index * main_layer_stride,
        )
    for index, name in enumerate(idx_layers):
        kv_caches[name] = torch.as_strided(
            backing,
            size=(num_blocks, idx_page),
            stride=(idx_page, 1),
            storage_offset=idx_offset + index * idx_layer_stride,
        )

    worker = object.__new__(IluNixlWorkerMixin)
    worker.kv_cache_config = layout_config

    layer_region_ids, _region_ids_by_group, regions, _region_is_mla = (
        worker._build_packed_virtual_layout(kv_caches, layout_config, layout_config)
    )

    assert set(layer_region_ids) == set(main_layers + idx_layers)
    assert {region.block_stride for region in regions} == {main_page, idx_page}
    assert {region.block_len for region in regions} == {main_page, idx_page}


def test_finish_memory_registration_records_region_memory_type():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.nixl_backends = ["UCX"]
    worker.num_regions = 3
    worker.num_blocks = 2
    worker.block_size = 256
    worker.region_mem_types = []
    worker.dst_region_mem_types = {}
    worker.engine_id = "engine"
    worker.tp_rank = 0
    worker.kv_caches_base_addr = {"engine": {}}
    worker.dst_num_blocks = {}
    worker.src_xfer_handles_by_block_size = {}
    worker._registered_descs = []
    seen: dict = {}

    def register_local(block_size):
        seen["mem_types"] = list(worker.region_mem_types)
        return 1, None

    worker.nixl_wrapper = SimpleNamespace(
        get_reg_descs=lambda caches_data, memory_type: ("descs", memory_type),
        register_memory=lambda descs, backends: None,
    )
    worker.register_local_xfer_handler = register_local
    worker._set_handshake_metadata = lambda **kwargs: None

    worker._finish_memory_registration(
        {},
        [(0, 128, 0, "")],
        [0],
        memory_type="VRAM",
    )

    assert seen["mem_types"] == ["VRAM", "VRAM", "VRAM"]
    assert worker.dst_region_mem_types["engine"] == ["VRAM", "VRAM", "VRAM"]


def test_glm5_flash_cache_kind_recognition():
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.mla_attn"
        )
        == "main_mla"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.indexer.k_cache"
        )
        == "indexer_k_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.indexer.tail_cache"
        )
        == "kpool_tail_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name("model.layers.0.self_attn")
        == "mamba_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name("model.layers.1.linear_attn")
        == "mamba_cache"
    )
    assert (
        IluNixlWorkerMixin._cache_kind_from_layer_name(
            "model.layers.0.self_attn.ilu_deepseek_v4_multi_head_latent_attention"
        )
        == "main_mla"
    )


def test_glm5_flash_mamba_hma_routes_7_to_6_groups_and_cross_cycle_offset():
    """Verify P (7 groups: 6,6,6,6,6,6,5,4) -> D (6 groups: 23,22) routing."""
    worker = object.__new__(IluNixlWorkerMixin)
    worker._group_spec_types = (
        MLAAttentionSpec,
        MLAAttentionSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
    )
    worker.num_regions = 4
    # Even regions are 64KB MLA+Mamba shared; odd regions are 4KB indexer+tail.
    worker._ssm_region_indices = [0, 2]

    local_regions = [
        _region(
            0,
            65536,
            [
                "0:mamba_cache:group2",
                "1:mamba_cache:group3",
                "2:mamba_cache:group4",
                "3:main_mla:group0",
                "4:mamba_cache:group5",
            ],
        ),
        _region(
            1,
            4096,
            [
                "3:indexer_k_cache:group0",
                "3:kpool_tail_cache:group1",
            ],
        ),
        _region(
            2,
            65536,
            [
                "5:mamba_cache:group2",
                "6:mamba_cache:group3",
                "7:main_mla:group0",
                "8:mamba_cache:group4",
                "9:mamba_cache:group5",
            ],
        ),
        _region(
            3,
            4096,
            [
                "7:indexer_k_cache:group0",
                "7:kpool_tail_cache:group1",
            ],
        ),
    ]
    # P_PP0 owns layers 0..5 with 7 groups (group 6 holds layer 5 on P_R0).
    remote_regions = [
        _region(
            0,
            65536,
            [
                "0:mamba_cache:group2",
                "1:mamba_cache:group3",
                "2:mamba_cache:group4",
                "3:main_mla:group0",
                "4:mamba_cache:group5",
                "5:mamba_cache:group6",
            ],
        ),
        _region(
            1,
            4096,
            [
                "3:indexer_k_cache:group0",
                "3:kpool_tail_cache:group1",
            ],
        ),
    ]

    routes, fa_local_ids, fa_remote_ids, ssm_remote_ids = (
        worker._build_mamba_hma_transfer_routes(
            selected_local_region_ids=[0, 1, 2],
            selected_remote_region_ids=[0, 1],
            local_transfer_regions=local_regions,
            remote_transfer_regions=remote_regions,
            overlap_start=0,
            overlap_end=6,
        )
    )

    # FA channel only includes D_R0/D_R1 <-> P_R0/P_R1, not polluted by D_R2.
    assert fa_local_ids == [0, 1]
    assert fa_remote_ids == [0, 1]
    # SSM channel only includes 64KB P_R0, excluding 4KB P_R1.
    assert ssm_remote_ids == [0]

    route_map = {
        (r.local_group_idx, r.remote_group_idx): (
            r.is_ssm,
            r.local_desc_region_ids.tolist(),
            r.remote_desc_region_ids.tolist(),
        )
        for r in routes
    }
    assert route_map == {
        (0, 0): (False, [0, 1], [0, 1]),
        (1, 1): (False, [1], [1]),
        (2, 2): (True, [0], [0]),
        (3, 3): (True, [0], [0]),
        (4, 4): (True, [0], [0]),
        (5, 5): (True, [0], [0]),
        # Layer 5: P_g6 on P_R0 (slot 0) -> D_g2 on D_R2 (compact SSM slot 1).
        (2, 6): (True, [1], [0]),
    }


def test_validate_pp_hma_region_lengths_allows_shared_mla_mamba_under_hetero_tp():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._has_mamba = True
    worker.use_mla = True
    worker._group_spec_types = (
        MLAAttentionSpec,
        MLAAttentionSpec,
        MambaSpec,
    )
    worker.block_len_per_layer = [65536, 4096, 65536]
    worker._is_region_replicated = lambda _: True
    worker.dst_num_blocks = {"prefill": 140}
    worker.transfer_topo = SimpleNamespace(
        block_size_ratio=lambda _: 1,
        tp_ratio=lambda _: 4,
    )
    remote_meta = SimpleNamespace(
        engine_id="prefill",
        block_size=128,
        block_lens=[65536, 4096],
        num_blocks=140,
    )
    routes = [
        base_worker_module._HmaTransferRoute(
            local_group_idx=0,
            remote_group_idx=0,
            is_ssm=False,
            local_desc_region_ids=np.asarray([0, 1], dtype=np.int64),
            remote_desc_region_ids=np.asarray([0, 1], dtype=np.int64),
            raw_local_region_ids=(0, 1),
            raw_remote_region_ids=(0, 1),
        ),
        base_worker_module._HmaTransferRoute(
            local_group_idx=2,
            remote_group_idx=6,
            is_ssm=True,
            local_desc_region_ids=np.asarray([1], dtype=np.int64),
            remote_desc_region_ids=np.asarray([0], dtype=np.int64),
            raw_local_region_ids=(2,),
            raw_remote_region_ids=(0,),
        ),
    ]

    # Should not raise AssertionError (65536 != 262144).
    worker._validate_pp_hma_region_lengths(
        remote_meta,
        local_region_ids_by_group=[],
        remote_region_ids_by_group=[],
        selected_remote_region_ids=[0, 1],
        remote_tp_size=1,
        routes=routes,
    )


def test_glm5_flash_remote_handle_and_read_blocks_filter_non_ssm_regions():
    worker = object.__new__(IluNixlWorkerMixin)
    worker.engine_id = "decode"
    worker.world_size = 4
    worker.tp_rank = 0
    worker.pp_world_size = 2
    worker.pp_rank = 0
    worker.pp_split = "23,22"
    worker.use_mla = True
    worker._has_mamba = True
    worker._is_hma_required = True
    worker.num_regions = 4
    worker._ssm_region_indices = [0, 2]
    worker.block_len_per_layer = [65536, 4096, 65536, 4096]
    worker._is_region_replicated = lambda _: True
    worker._group_spec_types = (
        MLAAttentionSpec,
        MLAAttentionSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
    )
    worker._physical_blocks_per_logical_kv_block = 2
    worker.dst_num_blocks = {"decode": 4, "prefill": 4}
    worker._mamba_ssm_size = (300, 1000)
    worker._conv_decomp = SimpleNamespace(
        local_conv_offsets=[(0, 100), (100, 100), (200, 100)],
        remote_conv_offsets=lambda offset, tp_ratio: [
            (0, 100),
            (100, 100),
            (200, 100),
        ],
    )
    worker._stack_descs = lambda addrs, length, dev: np.column_stack(
        [
            addrs,
            np.full(len(addrs), length, dtype=np.uint64),
            np.full(len(addrs), dev, dtype=np.uint64),
        ]
    )
    worker.transfer_topo = SimpleNamespace(
        block_size_ratio=lambda _: 1,
        tp_ratio=lambda _: 4,
        get_engine_info=lambda _: SimpleNamespace(
            remote_tp_size=1,
            remote_physical_blocks_per_logical=2,
            remote_block_size=128,
        ),
    )
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            source_ranks_per_group=((0,), (0,), (0,), (0,), (0,), (0,)),
            all_source_ranks=(0,),
            rank_offset_factor=0,
        )
    }
    worker._remote_pp_size_by_engine = {"prefill": 8}
    worker._remote_pp_split_by_engine = {"prefill": "6,6,6,6,6,6,5,4"}
    worker._semantic_region_engines = {"prefill"}
    worker._engine_last_active = {}
    worker._remote_agents = {"prefill": {0: "prefill_agent_0"}}
    worker.src_xfer_handles_by_block_size = {128: 101}
    worker.dst_xfer_side_handles = {"prefill": {}}
    worker._recving_transfers = {"req-1": []}
    worker.kv_cache_config = SimpleNamespace(
        transfer_groups=[
            SimpleNamespace(kv_cache_spec=object.__new__(MLAAttentionSpec)),
            SimpleNamespace(kv_cache_spec=object.__new__(MLAAttentionSpec)),
            SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
            SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
            SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
            SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
        ]
    )

    prepped_descs = {}
    xfer_calls = []

    class FakeNixlWrapper:
        def get_xfer_descs(self, blocks_data, memory_type):
            return (np.asarray(blocks_data, dtype=np.uint64), memory_type)

        def prep_xfer_dlist(self, agent_name, descs):
            prepped_descs[agent_name] = descs[0]
            return 202

        def make_prepped_xfer(
            self,
            op,
            local_handle,
            local_ids,
            remote_handle,
            remote_ids,
            notif_msg=None,
        ):
            xfer_calls.append(
                (
                    op,
                    local_handle,
                    local_ids.tolist(),
                    remote_handle,
                    remote_ids.tolist(),
                    notif_msg,
                )
            )
            return 303

        def transfer(self, handle):
            pass

    worker.nixl_wrapper = FakeNixlWrapper()
    remote_meta = SimpleNamespace(
        engine_id="prefill",
        device_id=0,
        memory_type="VRAM",
        block_size=128,
        num_blocks=4,
        physical_blocks_per_logical_kv_block=2,
        kv_caches_base_addr=[1000000, 2000000],
        block_lens=[65536, 4096],
        block_strides=[65536, 4096],
        ssm_sizes=(1200, 4000),
        transfer_regions=None,
    )

    # Register remote handle: FA has [0, 1], SSM only has [0] (skips 4KB region 1).
    worker._register_pp_remote_xfer_handle(
        engine_id="prefill",
        remote_key=0,
        remote_agent_name="prefill_agent_0",
        nixl_agent_meta=remote_meta,
        selected_local_region_ids=[0, 1],
        selected_remote_region_ids=[0, 1],
        selected_ssm_remote_region_ids=[0],
    )

    remote_descs = prepped_descs["prefill_agent_0"]
    # 2 FA regions * 4 physical blocks + 1 SSM region * 4 sub-regions * 2 logical blocks = 16
    assert remote_descs.shape == (16, 3)
    # First 4 descs are 64KB single-page FA descs on region 0; next 4 are 4KB on region 1.
    assert remote_descs[:4, 1].tolist() == [65536] * 4
    assert remote_descs[4:8, 1].tolist() == [4096] * 4
    # Remaining 8 descs are Mamba sub-descs all within region 0 (base 1000000), none in region 1.
    assert all(1000000 <= addr < 2000000 for addr in remote_descs[8:, 0].tolist())

    worker._mamba_hma_routes = {
        "prefill": {
            0: [
                base_worker_module._HmaTransferRoute(
                    local_group_idx=0,
                    remote_group_idx=0,
                    is_ssm=False,
                    local_desc_region_ids=np.asarray([0, 1], dtype=np.int64),
                    remote_desc_region_ids=np.asarray([0, 1], dtype=np.int64),
                    raw_local_region_ids=(0, 1),
                    raw_remote_region_ids=(0, 1),
                ),
                base_worker_module._HmaTransferRoute(
                    local_group_idx=2,
                    remote_group_idx=6,
                    is_ssm=True,
                    local_desc_region_ids=np.asarray([1], dtype=np.int64),
                    remote_desc_region_ids=np.asarray([0], dtype=np.int64),
                    raw_local_region_ids=(2,),
                    raw_remote_region_ids=(0,),
                ),
            ]
        }
    }
    worker._apply_prefix_caching = lambda decode_block_ids, prefill_block_ids, **_: (
        decode_block_ids,
        prefill_block_ids,
    )
    worker._logical_to_kernel_block_ids = lambda block_ids, ratio: [
        [b * ratio + o for b in grp for o in range(ratio)] if idx < 2 else list(grp)
        for idx, grp in enumerate(block_ids)
    ]

    meta = SimpleNamespace(
        remote=SimpleNamespace(
            engine_id="prefill",
            request_id="r-prefill",
            # 7 remote groups on P; group 6 has logical block [1]
            block_ids=([0], [0], [0], [0], [0], [0], [1]),
        ),
        # 6 local groups on D (already expanded for FA: [0, 1]); group 2 has logical block [1]
        local_physical_block_ids=([0, 1], [0, 1], [1], [0], [0], [0]),
        awaiting_kvs=False,
    )

    worker._read_blocks_for_req("req-1", meta)

    assert len(xfer_calls) == 1
    _, local_handle, local_ids, remote_handle, remote_ids, _ = xfer_calls[0]
    assert local_handle == 101
    assert remote_handle == 202
    # FA desc IDs: local regions [0, 1] * 4 blocks + [0, 1] -> [0, 1, 4, 5]
    # SSM desc IDs (local): num_fa_regions(4)*4=16 + (slot(1)*4 + [0,1,2,3])*2 + block(1) -> [25, 27, 29, 31]
    assert local_ids == [0, 1, 4, 5, 25, 27, 29, 31]
    # FA desc IDs (remote): compact regions [0, 1] * 4 blocks + [0, 1] -> [0, 1, 4, 5]
    # SSM desc IDs (remote): num_fa_regions(2)*4=8 + (slot(0)*4 + [0,1,2,3])*2 + block(1) -> [9, 11, 13, 15]
    assert remote_ids == [0, 1, 4, 5, 9, 11, 13, 15]


def test_glm5_flash_6_vs_6_hetero_pp_cross_cycle_mounting():
    """Verify 6 vs 6 hetero PP where Mamba layers mount on offset MLA regions."""
    worker = object.__new__(IluNixlWorkerMixin)
    worker._group_spec_types = (
        MLAAttentionSpec,
        MLAAttentionSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
    )
    worker.num_regions = 8
    worker._ssm_region_indices = [0, 2, 4, 6]

    # D_PP0 holds layers 0..22; in cycle 3 (layers 12..15), layer 13 (group 2)
    # mounts on D_R4 (slot 2) while layer 15 MLA is on D_R6/D_R7.
    local_regions = [
        _region(4, 65536, ["11:main_mla:group0", "13:mamba_cache:group2"]),
        _region(6, 65536, ["15:main_mla:group0", "14:mamba_cache:group3"]),
        _region(
            7,
            4096,
            ["15:indexer_k_cache:group0", "15:kpool_tail_cache:group1"],
        ),
    ]
    # P stage holds layers 12..16 (1 MLA at layer 15 -> P_R0/P_R1).
    remote_regions = [
        _region(
            0,
            65536,
            [
                "13:mamba_cache:group2",
                "14:mamba_cache:group3",
                "15:main_mla:group0",
            ],
        ),
        _region(
            1,
            4096,
            ["15:indexer_k_cache:group0", "15:kpool_tail_cache:group1"],
        ),
    ]

    routes, fa_local_ids, fa_remote_ids, ssm_remote_ids = (
        worker._build_mamba_hma_transfer_routes(
            selected_local_region_ids=[4, 6, 7],
            selected_remote_region_ids=[0, 1],
            local_transfer_regions=local_regions,
            remote_transfer_regions=remote_regions,
            overlap_start=12,
            overlap_end=17,
        )
    )

    assert fa_local_ids == [6, 7]
    assert fa_remote_ids == [0, 1]
    assert ssm_remote_ids == [0]
    route_map = {
        (r.local_group_idx, r.remote_group_idx): (
            r.local_desc_region_ids.tolist(),
            r.remote_desc_region_ids.tolist(),
        )
        for r in routes
    }
    assert route_map == {
        (0, 0): ([6, 7], [0, 1]),
        (1, 1): ([7], [1]),
        # D_R4 is compact SSM slot 2; P_R0 is compact SSM slot 0.
        (2, 2): ([2], [0]),
        # D_R6 is compact SSM slot 3; P_R0 is compact SSM slot 0.
        (3, 3): ([3], [0]),
    }


def test_hma_failed_transfer_invalidates_all_non_empty_local_groups():
    import queue

    worker = object.__new__(IluNixlWorkerMixin)
    worker._is_hma_required = True
    worker._invalid_block_ids = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker._recving_metadata = {
        "req-hma": SimpleNamespace(local_block_ids=([3, 4], [], [11, 12]))
    }
    worker.xfer_stats = SimpleNamespace(record_failed_transfer=lambda: None)
    worker.nixl_wrapper = None

    worker._handle_failed_transfer("req-hma", None)

    assert worker._invalid_block_ids.get_nowait() == {3, 4, 11, 12}
    assert worker._failed_recv_reqs.get_nowait() == "req-hma"


def test_draining_session_retains_dlists_until_inflight_read_finishes():
    from collections import defaultdict

    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.session import (
        RemoteSessionState,
    )

    released_dlists = []
    removed_agents = []
    worker = object.__new__(IluNixlWorkerMixin)
    worker._remote_agents = defaultdict(dict, {"eng-old": {0: "agent-old"}})
    worker.dst_xfer_side_handles = defaultdict(dict, {"eng-old": {0: 501}})
    worker.kv_caches_base_addr = defaultdict(dict)
    worker.dst_num_blocks = {"eng-old": 8}
    worker.tp_mappings = {}
    worker.transfer_topo = None
    worker._engine_last_active = {"eng-old": 1.0}
    worker._consumer_disconnect_grace = 30.0
    worker.nixl_wrapper = SimpleNamespace(
        release_dlist_handle=released_dlists.append,
        remove_remote_agent=removed_agents.append,
    )

    session = worker._register_active_session(
        "eng-old", "10.0.0.1", 5600, {0: "agent-old"}
    )
    session.add_inflight(9001)

    # Peer restarts at same (host, port) with new engine_id -> old enters DRAINING.
    worker._remote_agents["eng-new"][0] = "agent-new"
    worker.dst_xfer_side_handles["eng-new"][0] = 502
    worker._register_active_session("eng-new", "10.0.0.1", 5600, {0: "agent-new"})

    assert session.state == RemoteSessionState.DRAINING
    assert released_dlists == []
    assert removed_agents == []

    # Once handle 9001 finishes, progress releases old NIXL handles cleanly.
    worker._on_handle_settled_for_session(9001, "eng-old")
    worker._progress_draining_sessions(100.0)

    assert session.state == RemoteSessionState.CLOSED
    assert released_dlists == [501]
    assert removed_agents == ["agent-old"]


def test_rle_block_ids_and_structured_notifications():
    from collections import defaultdict

    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
        encode_structured_notif,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pull_worker import (
        IluNixlPullConnectorWorker,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common import (
        PoolDirectory,
        encode_block_ids_rle,
        resolve_remote_block_ids,
    )

    rle = encode_block_ids_rle(([10, 11, 12, 20], []))
    assert rle == [[[10, 3], [20, 1]], []]
    params = {"remote_block_ids_rle": rle}
    assert resolve_remote_block_ids(params) == ([10, 11, 12, 20], [])

    pdir = PoolDirectory()
    pdir.record_exported(([1, 2],), [{1: "h1", 2: "h2"}])
    assert pdir.filter_blocks_needing_export(0, [1, 2, 3], {1: "h1", 2: "h2_new"}) == [
        2,
        3,
    ]

    pw = object.__new__(IluNixlPullConnectorWorker)
    pw.world_size = 1
    pw.transfer_topo = SimpleNamespace(tp_ratio=lambda _: -2)
    pw._reqs_to_send = {"r1": 100.0, "r2": 100.0}
    pw._reqs_to_process = {"r1", "r2"}
    pw.consumer_notification_counts_by_req = defaultdict(int)
    pw.xfer_stats = SimpleNamespace(record_failed_transfer=lambda: None)
    pw.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {
            "d": [
                encode_structured_notif(
                    "r1",
                    status="ok",
                    remote_key=0,
                    consumer_pp_rank=0,
                    consumer_tp_rank=0,
                    expected_consumers=2,
                ),
                # Duplicate from same sender -> ignored
                encode_structured_notif(
                    "r1",
                    status="ok",
                    remote_key=0,
                    consumer_pp_rank=0,
                    consumer_tp_rank=0,
                    expected_consumers=2,
                ),
                # Fail on r2 -> released immediately without waiting for 2nd consumer
                encode_structured_notif(
                    "r2",
                    status="fail",
                    remote_key=0,
                    consumer_pp_rank=0,
                    consumer_tp_rank=0,
                    expected_consumers=2,
                ),
            ]
        }
    )
    assert pw._get_new_notifs() == {"r2"}
    assert "r1" in pw._reqs_to_send and "r2" not in pw._reqs_to_send


def test_glm5_hetero_mamba_align_uses_kernel_pages_and_preserves_kpool_tail():
    """Regression test for GLM-5.3 hetero PD (D=2304/36, P=4480/70 -> 64-token pages).

    Covers:
    1) Shared-storage layout with ``_physical_blocks_per_logical_kv_block > 1``
       bypasses ``_register_virtual_packed_kv_cache`` so both sides advertise
       64-token kernel pages instead of 2304 vs 4480.
    2) Upstream ``register_kv_caches`` registers the full shared
       ``untyped_storage`` span, splits ``tokens_per_state=4`` indexer rows into
       dense 64-token pages, and skips co-located ``KpoolTailSpec`` views.
    3) ``_build_hma_transfer_regions`` emits ``kernel_blocks=1``,
       ``block_tokens=64``, and ``block_stride=block_len`` so peer handshake
       geometry checks succeed and ``virtual_packed`` stays False.
    4) Non-V4.1 scheduler clip rules do not zero out ``KpoolTailSpec`` groups,
       ``_align_group_blocks_with_start`` handles ``start_offset`` in scheduler
       blocks when ``d_ppl != p_ppl``, and ``post_process_device_kv_on_receive``
       zeroes stale physical pages without indexing out of bounds on
       ``KpoolTailSpec`` / compressed indexer views.
    """
    from collections import defaultdict
    from unittest.mock import MagicMock

    import msgspec
    from vllm.v1.kv_cache_interface import (
        KpoolTailSpec,
        KVCacheGroupSpec,
        UniformTypeKVCacheSpecs,
    )

    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_scheduler import (
        IluNixlBaseConnectorScheduler,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
        IluNixlAgentMetadata,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pull_worker import (
        IluNixlPullConnectorWorker,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.registration_mixin import (
        _BlocksFirstTopologyBackend,
    )

    def _build_side(*, sched_block: int, ppl: int, n_logical: int, tp_size: int):
        mla_name = "language_model.model.layers.3.self_attn.attn"
        idx_name = "language_model.model.layers.3.self_attn.indexer.k_cache"
        tail_name = "language_model.model.layers.3.self_attn.indexer.tail_cache"
        mamba_name = "language_model.model.layers.0.self_attn"

        mla_spec = MLAAttentionSpec(
            block_size=sched_block,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        idx_spec = MLAAttentionSpec(
            block_size=sched_block,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8,
            tokens_per_state=4,
        )
        tail_spec = KpoolTailSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=16,
            page_size_padded=idx_spec.page_size_bytes,
        )
        mamba_spec = MambaSpec(
            shapes=((512 // tp_size, 3), (32 // tp_size, 64, 64)),
            dtypes=(torch.bfloat16, torch.bfloat16),
            block_size=sched_block,
            page_size_padded=mla_spec.page_size_bytes,
        )
        attn_uniform = UniformTypeKVCacheSpecs.from_specs(
            {mla_name: mla_spec, idx_name: idx_spec}
        )
        tail_uniform = UniformTypeKVCacheSpecs.from_specs({tail_name: tail_spec})
        assert attn_uniform is not None and tail_uniform is not None

        groups = [
            KVCacheGroupSpec([mla_name, idx_name], attn_uniform),
            KVCacheGroupSpec([tail_name], tail_uniform),
            KVCacheGroupSpec([mamba_name], mamba_spec),
        ]
        mla_span = n_logical * mla_spec.page_size_bytes
        idx_span = n_logical * idx_spec.page_size_bytes
        total_bytes = mla_span + idx_span
        backing = torch.ones(total_bytes, dtype=torch.uint8)

        n_phys = n_logical * ppl
        mla_view = torch.as_strided(
            backing.view(torch.bfloat16),
            size=(n_phys, 64, 1, 512),
            stride=(64 * 512, 512, 512, 1),
            storage_offset=0,
        )
        idx_view = torch.as_strided(
            backing,
            size=(n_phys // 2, 1, 32, 256),
            stride=(8192, 8192, 256, 1),
            storage_offset=mla_span,
        )
        tail_view = torch.as_strided(
            backing.view(torch.bfloat16),
            size=(n_logical, 2, 16, 128),
            stride=(idx_spec.page_size_bytes // 2, 16 * 128, 128, 1),
            storage_offset=mla_span // 2,
        )
        mamba_view = torch.as_strided(
            backing,
            size=(n_logical, mla_spec.page_size_bytes),
            stride=(mla_spec.page_size_bytes, 1),
            storage_offset=0,
        )
        kv_caches = {
            mla_name: mla_view,
            idx_name: idx_view,
            tail_name: tail_view,
            mamba_name: mamba_view,
        }
        kv_cfg = SimpleNamespace(
            num_blocks=n_logical,
            kv_cache_groups=groups,
            transfer_groups=tuple(groups),
            transfer_group_ids=(0, 1, 2),
            transfer_group_index_by_layer={
                mla_name: 0,
                idx_name: 0,
                tail_name: 1,
                mamba_name: 2,
            },
            select_transfer_block_ids=lambda ids: tuple(ids),
            kv_cache_tensors=[
                SimpleNamespace(
                    size=total_bytes,
                    layers=[mla_name],
                    layer_stride=mla_span,
                    block_stride=mla_spec.page_size_bytes,
                    offset=0,
                ),
                SimpleNamespace(
                    size=total_bytes,
                    layers=[mamba_name],
                    layer_stride=mla_span,
                    block_stride=mla_spec.page_size_bytes,
                    offset=0,
                ),
                SimpleNamespace(
                    size=total_bytes,
                    layers=[idx_name],
                    layer_stride=idx_span,
                    block_stride=idx_spec.page_size_bytes,
                    offset=mla_span,
                ),
                SimpleNamespace(
                    size=total_bytes,
                    layers=[tail_name],
                    layer_stride=idx_span,
                    block_stride=idx_spec.page_size_bytes,
                    offset=mla_span,
                ),
            ],
        )

        worker = object.__new__(IluNixlPullConnectorWorker)
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="glm-5.3-flash",
                dtype=torch.bfloat16,
                hf_config=SimpleNamespace(model_type="glm5_next", num_hidden_layers=45),
                use_mla=True,
                get_total_num_kv_heads=lambda: 1,
                get_head_size=lambda: 512,
                get_total_num_hidden_layers=lambda: 45,
            ),
            cache_config=SimpleNamespace(
                block_size=sched_block, cache_dtype="bfloat16", mamba_cache_mode="align"
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            speculative_config=None,
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_consumer" if tp_size == 2 else "kv_producer",
                get_from_extra_config=lambda _k, d=None: d,
            ),
        )
        worker.dsv41_layout = None
        worker.fp8_wire_staging_enabled = False
        worker.use_host_buffer = False
        worker.host_xfer_buffers = {}
        worker.kv_buffer_device = "cpu"
        worker.device_type = "cpu"
        worker.device_id = 0
        worker._mixed_mem_types = False
        worker.nixl_memory_type = "DRAM"
        worker.nixl_backends = ["UCX"]
        worker.kv_cache_config = kv_cfg
        worker.engine_id = "decode" if tp_size == 2 else "prefill"
        worker.tp_rank = 0
        worker.transfer_tp_rank = 0
        worker.transfer_tp_size = tp_size
        worker.world_size = tp_size
        worker.pp_rank = 0
        worker.pp_world_size = 1
        worker.pp_size = 1
        worker.pp_split = "45"
        worker.dcp_size = 1
        worker.pcp_size = 1
        worker.use_mla = True
        worker._has_mamba = True
        worker._is_csa_linear = False
        worker._ple_region_index = None
        worker._supports_pp_hma = False
        worker._has_packed_cache = True
        worker._is_hma_required = True
        worker.enable_permute_local_kv = False
        worker.model_config = worker.vllm_config.model_config
        worker.attn_backends = [_BlocksFirstTopologyBackend]
        worker.backend_name = "FLASHMLA_SPARSE"
        worker.kv_cache_layout = "LBHNC"
        worker.host_buffer_kv_cache_layout = "LBHNC"
        worker._layer_specs = {
            mla_name: mla_spec,
            idx_name: idx_spec,
            tail_name: tail_spec,
            mamba_name: mamba_spec,
        }
        worker._group_spec_types = (MLAAttentionSpec, KpoolTailSpec, MambaSpec)
        worker._conv_decomp = SimpleNamespace(
            local_conv_offsets=[(0, 512 // tp_size)],
            remote_conv_offsets=lambda off, _r: [
                (off * (512 // tp_size), 512 // tp_size)
            ],
            ssm_sizes=(1536 // tp_size, 4096 // tp_size),
        )
        worker._mamba_ssm_size = (1536 // tp_size, 4096 // tp_size)
        worker._logical_num_blocks = n_logical
        worker._scheduler_block_size = sched_block
        worker._physical_blocks_per_logical_kv_block = ppl
        worker.block_size = 64
        worker.num_blocks = n_phys
        worker._virtual_packed_regions = False
        worker._registered_descs = []
        worker.block_len_per_layer = []
        worker.block_stride_per_layer = []
        worker.region_group_ids = []
        worker.region_names = []
        worker.region_num_blocks = []
        worker.region_members = []
        worker._region_is_mla = []
        worker._transfer_layer_region_indices = []
        worker.region_mem_types = []
        worker.dst_region_mem_types = {}
        worker.dst_region_num_blocks = {}
        worker.dst_region_group_ids = {}
        worker.dst_uses_region_group_mapping = {}
        worker.kv_caches_base_addr = defaultdict(dict)
        worker.dst_num_blocks = {}
        worker.src_xfer_handles_by_block_size = {}
        worker.src_xfer_handles_by_tp_ratio = {}
        worker.dst_xfer_side_handles = {worker.engine_id: {}, "prefill": {}}
        worker._remote_agents = {}
        worker.tp_mappings = {}
        worker.local_region_ids_by_group = {}
        worker.remote_region_ids_by_group = {}
        worker._remote_pp_size_by_engine = {}
        worker._remote_pp_split_by_engine = {}
        worker._semantic_region_engines = set()
        worker._compute_compatibility_hash = lambda _=None: "compat-glm53"

        reg_calls = []
        dlist_calls = []
        nixl = MagicMock()
        nixl.get_agent_metadata.return_value = b"agent"
        nixl.get_reg_descs.side_effect = lambda ranges, mt: (ranges, mt)
        nixl.register_memory.side_effect = (
            lambda descs, backends=None: reg_calls.append(descs)
        )
        nixl.get_xfer_descs.side_effect = lambda arr, mt: (arr, mt)
        nixl.prep_xfer_dlist.side_effect = lambda agent, descs: (
            dlist_calls.append((agent, descs)) or len(dlist_calls)
        )
        nixl.add_remote_agent.return_value = "remote-p"
        worker.nixl_wrapper = nixl
        return worker, kv_caches, total_bytes, reg_calls, dlist_calls

    d_worker, d_caches, d_total_bytes, d_regs, d_dlists = _build_side(
        sched_block=2304, ppl=36, n_logical=4, tp_size=2
    )
    p_worker, p_caches, p_total_bytes, p_regs, _ = _build_side(
        sched_block=4480, ppl=70, n_logical=4, tp_size=1
    )

    d_worker.register_kv_caches(d_caches)
    p_worker.register_kv_caches(p_caches)

    assert d_worker.block_size == 64 and p_worker.block_size == 64
    assert not d_worker._virtual_packed_regions and not p_worker._virtual_packed_regions
    assert d_regs[0][0] == [
        (
            d_caches["language_model.model.layers.3.self_attn.attn"]
            .untyped_storage()
            .data_ptr(),
            d_total_bytes,
            0,
            "",
        )
    ]
    assert p_regs[0][0] == [
        (
            p_caches["language_model.model.layers.3.self_attn.attn"]
            .untyped_storage()
            .data_ptr(),
            p_total_bytes,
            0,
            "",
        )
    ]
    reg_base, reg_len, _, _ = d_regs[0][0][0]
    local_blocks_arr = d_dlists[0][1][0]
    assert int(local_blocks_arr[:, 0].min()) >= reg_base
    assert (
        int((local_blocks_arr[:, 0] + local_blocks_arr[:, 1]).max())
        <= reg_base + reg_len
    )
    assert d_worker.block_len_per_layer == [65536, 4096]
    assert d_worker.block_stride_per_layer == [65536, 4096]

    p_meta = msgspec.msgpack.decode(
        p_worker.xfer_handshake_metadata.agent_metadata_bytes,
        type=IluNixlAgentMetadata,
    )
    assert p_meta.block_size == 64
    assert not d_worker._metadata_uses_virtual_packed_regions(p_meta)

    agent_name = d_worker.add_remote_agent(
        p_meta, remote_tp_rank=0, remote_tp_size=1, remote_pp_rank=0, remote_pp_size=1
    )
    assert agent_name == "remote-p"

    # Non-V4.1 scheduler must NOT clip KpoolTailSpec group to [].
    sched = object.__new__(IluNixlBaseConnectorScheduler)
    sched.dsv41_layout = None
    sched.kv_cache_config = d_worker.kv_cache_config
    sched._is_hma_required = True
    sched.blocks_per_sw = [0, 2, 0]
    sched._ssm_spec_blocks = [None, None, 0]
    sched._ssm_state_slots_are_positional = False
    sched._init_ilu_transfer_clip_rules()
    clipped = sched.get_exchange_clipped_blocks(([1, 2], [9], [3, 4]))
    assert clipped == ([1, 2], [9], [4])

    # _align_group_blocks_with_start handles start_offset with hetero ppl (36 vs 70).
    remote_info = d_worker.transfer_topo.get_engine_info("prefill")
    loc0, rem0 = d_worker._align_group_blocks_with_start(
        list(range(72)),
        list(range(70)),
        local_group_idx=0,
        num_local_groups=3,
        remote_info=remote_info,
        start_offset=0,
    )
    assert len(loc0) == len(rem0) == 70
    loc1, rem1 = d_worker._align_group_blocks_with_start(
        list(range(100, 136)),
        list(range(70)),
        local_group_idx=0,
        num_local_groups=3,
        remote_info=remote_info,
        start_offset=1,
    )
    assert loc1 == list(range(100, 134))
    assert rem1 == list(range(36, 70))

    # Stale physical page zeroing (e.g. pages 70..71 >= n_logical=4) does not OOB
    # on tail_cache (dim0=4) or compressed indexer_k_cache (dim0=72).
    d_worker.post_process_device_kv_on_receive(
        block_size_ratio=1,
        block_ids_list=[(list(range(72)), 70)],
        convert=False,
    )
    assert (
        int(
            d_caches["language_model.model.layers.3.self_attn.attn"][70]
            .view(torch.uint8)
            .sum()
            .item()
        )
        == 0
    )
    assert (
        int(
            d_caches["language_model.model.layers.3.self_attn.attn"][69]
            .view(torch.uint8)
            .sum()
            .item()
        )
        > 0
    )
