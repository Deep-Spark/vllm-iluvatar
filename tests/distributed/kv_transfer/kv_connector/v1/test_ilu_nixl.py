# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

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


class _TestIluNixlBaseConnector(IluNixlBaseConnector):
    def start_load_kv(self, forward_context, **kwargs) -> None:
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


def test_pp_layer_range_and_targets():
    assert IluNixlWorkerMixin._pp_layer_range("2,3,4", 1) == (2, 5)
    assert IluNixlWorkerMixin._target_pp_ranks("3,3", 0, "2,2,2") == [0, 1]
    assert IluNixlWorkerMixin._overlap_pp_count(1, "2,2,2", "3,3") == 2


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
