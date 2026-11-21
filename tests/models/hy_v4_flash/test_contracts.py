# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.hy_v4_flash.attention import FlashPageStorage
from vllm_iluvatar.models.hy_v4_flash.config import HYV4FlashConfig


def test_dfly_capture_warms_blas_on_capture_stream():
    from vllm_iluvatar.models.hy_v4_flash.speculator import FlashDFlySpeculator

    device = torch.device("cuda")
    linear = torch.nn.Linear(
        6144, 3072, bias=False, device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        linear.weight.fill_(1 / 1024)
    inputs = torch.full((8, 6144), 0.125, device=device, dtype=torch.bfloat16)
    output = torch.zeros((8, 3072), device=device, dtype=torch.bfloat16)

    def step(rows):
        output[:rows].copy_(linear(inputs[:rows]))

    owner = SimpleNamespace(
        use_graph=True,
        device=device,
        max_num_reqs=8,
        active=torch.zeros(8, device=device, dtype=torch.bool),
        graphs={},
        _capture_stream=None,
        _step=step,
    )
    FlashDFlySpeculator.capture(owner)
    capture_stream = owner._capture_stream
    FlashDFlySpeculator.capture(owner)
    assert owner._capture_stream is capture_stream
    assert set(owner.graphs) == {1, 2, 4, 8}
    for rows, graph in owner.graphs.items():
        for value in (0.125, 0.25):
            inputs.fill_(value)
            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            # Exact binary inputs give the analytic dot product 6144*x/1024.
            assert torch.all(output[:rows] == value * 6)
            assert torch.all(output[rows:] == 0)


def test_reject_unimplemented_index_layout():
    with pytest.raises(ValueError, match="compress_ratio=4"):
        HYV4FlashConfig(
            compressed_indexer_compress_ratio=2,
            compressed_indexer_candidate_topk=1024,
            index_topk=1024,
        )


@pytest.mark.parametrize(
    "width,dtype", [(128, torch.bfloat16), (32, torch.bfloat16), (1, torch.int32)]
)
def test_auxiliary_page_budget_matches_storage(width, dtype):
    owner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64), head_dim=width, dtype=dtype
    )
    spec = FlashPageStorage.get_kv_cache_spec(owner, None)
    cache = torch.empty(3, 1, 64, width, dtype=dtype)
    FlashPageStorage.bind_kv_cache(owner, cache)
    assert spec.page_size_bytes * 3 == owner.kv_cache.nbytes
    assert owner.kv_cache.data_ptr() == cache.data_ptr()


def test_parallel_config_is_checked_before_allocating_model():
    from vllm_iluvatar.models.hy_v4_flash.model import HYV4FlashForCausalLM

    config = HYV4FlashConfig()
    parallel = SimpleNamespace(
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_eplb=False,
        enable_dbo=False,
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        enable_elastic_ep=False,
        enable_fault_tolerance=False,
        use_ubatching=False,
        nnodes=1,
        expert_placement_strategy="linear",
        all2all_backend="deepep_low_latency",
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config), parallel_config=parallel
    )
    with pytest.raises(ValueError, match="allgather_reducescatter"):
        HYV4FlashForCausalLM(vllm_config=vc)


def test_ihc_3072_old_width_guard(monkeypatch):
    import ixformer.inference.functions as ops

    from vllm_iluvatar.models.hy_v4.hc import _native_supports_3072

    def rejected(*args, **kwargs):
        raise RuntimeError("hidden dim must be 6144 (HYV4 production) or 4096")

    monkeypatch.setattr(ops, "fuse_ihc_post", rejected)
    assert not _native_supports_3072.__wrapped__(torch.device("cpu"))

    def failed(*args, **kwargs):
        raise RuntimeError("operator launch failed")

    monkeypatch.setattr(ops, "fuse_ihc_post", failed)
    with pytest.raises(RuntimeError, match="operator launch failed"):
        _native_supports_3072.__wrapped__(torch.device("cpu"))


@pytest.mark.parametrize(
    "starts,capacity", [([0, 1, 4, 7], 9), ([0, 0, 2, 3], 8), ([0], 0)]
)
def test_sparse_request_mapping_matches_default(starts, capacity):
    import numpy as np

    from vllm_iluvatar.models.hy_v4_flash.attention import FlashMetadataBuilder
    from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
        IluFlashMLASparseMetadataBuilder,
    )

    offsets = torch.tensor(starts, dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.cuda(),
        num_actual_tokens=capacity,
    )
    host = torch.empty(max(capacity, 1), dtype=torch.int32, pin_memory=True)
    default = SimpleNamespace(
        req_id_per_token_buffer=torch.empty(
            max(capacity, 1), dtype=torch.int32, device="cuda"
        ),
        _req_id_per_token_slot=-1,
        _req_id_per_token_depth=1,
        _req_id_per_token_hosts=[host],
        _req_id_per_token_nps=[host.numpy()],
    )
    flash = SimpleNamespace(
        req_id_per_token_buffer=torch.empty_like(default.req_id_per_token_buffer)
    )
    expected = IluFlashMLASparseMetadataBuilder._stage_req_id_per_token(
        default, metadata
    )
    actual = FlashMetadataBuilder._stage_req_id_per_token(flash, metadata)
    assert torch.equal(actual, expected)
    oracle = np.repeat(np.arange(len(starts) - 1), np.diff(starts)).tolist() + [0] * (
        capacity - starts[-1]
    )
    assert actual.cpu().tolist() == oracle


def test_sparse_request_mapping_replay_uses_changed_offsets():
    from vllm_iluvatar.models.hy_v4_flash.attention import FlashMetadataBuilder

    offsets = torch.tensor([0, 1, 4, 7], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(query_start_loc=offsets, num_actual_tokens=8)
    builder = SimpleNamespace(
        req_id_per_token_buffer=torch.empty(8, dtype=torch.int32, device="cuda")
    )
    FlashMetadataBuilder._stage_req_id_per_token(builder, metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = FlashMetadataBuilder._stage_req_id_per_token(builder, metadata)
    for starts, expected in [
        ([0, 3, 3, 5], [0, 0, 0, 2, 2, 0, 0, 0]),
        ([0, 0, 2, 8], [1, 1, 2, 2, 2, 2, 2, 2]),
    ]:
        offsets.copy_(torch.tensor(starts, dtype=torch.int32, device="cuda"))
        graph.replay()
        assert result.cpu().tolist() == expected
