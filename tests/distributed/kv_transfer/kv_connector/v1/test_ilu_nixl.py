# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl import (
    base_worker as base_worker_module,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_worker import (
    IluNixlWorkerMixin,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.connector import (
    IluNixlBaseConnector,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
    IluNixlConnectorMetadata,
    IluNixlTransferRegion,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pp_utils import (
    infer_pp_split,
)


class _TestIluNixlBaseConnector(IluNixlBaseConnector):
    def start_load_kv(self, forward_context, **kwargs) -> None:
        pass


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
        kv_transfer_config=SimpleNamespace(kv_role=role),
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
        IluNixlWorkerMixin._transfer_identity("model.layers.16.attn.swa_cache", 1)
        == "16:swa_cache:group1"
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


def test_set_xfer_handshake_metadata_pp_aware_maps_to_global_rank():
    connector = object.__new__(_TestIluNixlBaseConnector)
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2)
    )
    captured = {}
    connector.set_xfer_handshake_metadata = captured.update

    connector.set_xfer_handshake_metadata_pp_aware(
        {
            (0, 0): "pp0-tp0",
            (1, 0): "pp1-tp0",
            (1, 1): "pp1-tp1",
        }
    )

    assert captured == {
        0: "pp0-tp0",
        2: "pp1-tp0",
        3: "pp1-tp1",
    }


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
