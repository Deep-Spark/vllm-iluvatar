# SPDX-License-Identifier: Apache-2.0
"""Graph layout, address stability and dispatch contracts (CPU only)."""

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_iluvatar.v1.spec_decode import draft_cudagraph as graphs
from vllm_iluvatar.v1.worker import dp_utils


def cad(lengths):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    return NS(
        num_reqs=len(lengths),
        num_actual_tokens=int(lengths.sum()),
        max_query_len=int(lengths.max()) if len(lengths) else 0,
        query_start_loc_cpu=torch.cat(
            (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0))
        ),
    )


def test_equal_token_counts_do_not_alias_index_or_query_variants():
    keys = [
        graphs.uniform_key(cad([4] * 4), False, (1, 4)),
        graphs.uniform_key(cad([1] * 16), False, (1, 4)),
        graphs.uniform_key(cad([1] * 16), True, (1,)),
    ]
    assert len(set(keys)) == 3
    assert {key.num_tokens for key in keys} == {16}


@pytest.mark.parametrize("lengths", [[], [0, 0], [4, 0], [1, 3], [5, 5]])
def test_ragged_empty_and_prefill_layouts_fall_back(lengths):
    assert graphs.uniform_key(cad(lengths), False, (1, 4)) is None


@dataclass
class Metadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    decode: object
    num_reqs: int


def test_metadata_replay_updates_values_without_rebinding_storage():
    src = {
        "mla": Metadata(
            torch.tensor([20, 31]),
            torch.tensor([[2, 3], [4, 5]]),
            Metadata(torch.tensor([19, 30]), torch.tensor([6, 7]), None, 2),
            2,
        )
    }
    dst = graphs._clone_tree(src)
    ptrs = dst["mla"].seq_lens.data_ptr(), dst["mla"].decode.block_table.data_ptr()
    src["mla"].seq_lens = torch.tensor([18, 35])  # includes a rejection rollback
    src["mla"].block_table = torch.tensor([[9, 8], [6, 4]])
    src["mla"].decode.block_table = torch.tensor([8, 9])
    graphs._stage_tree(dst, src)
    assert ptrs == (
        dst["mla"].seq_lens.data_ptr(),
        dst["mla"].decode.block_table.data_ptr(),
    )
    torch.testing.assert_close(dst["mla"].seq_lens, src["mla"].seq_lens)
    torch.testing.assert_close(dst["mla"].block_table, src["mla"].block_table)
    torch.testing.assert_close(
        dst["mla"].decode.block_table, src["mla"].decode.block_table
    )


@pytest.mark.parametrize(
    "replacement", [torch.ones(3, dtype=torch.int64), torch.ones(2)]
)
def test_staging_rejects_shape_or_dtype_change(replacement):
    with pytest.raises(ValueError, match="layout"):
        graphs._stage_tree(torch.ones(2, dtype=torch.int64), replacement)


def test_staging_rejects_unrepresented_scalar_change():
    with pytest.raises(ValueError, match="static metadata"):
        graphs._stage_tree({"num_reqs": 4}, {"num_reqs": 3})


@pytest.mark.parametrize(
    "counts,modes,expected",
    [
        ([4, 4], [2, 2], 2),
        ([4, 8], [2, 2], 0),
        ([4, 4], [2, 0], 0),
        ([0, 4], [0, 2], 0),
    ],
)
def test_dp_graph_fallback_keeps_eager_sizes_and_phase(
    monkeypatch, counts, modes, expected
):
    payload = torch.tensor([counts, counts, counts, [0, 0], modes, [0, 0]])
    monkeypatch.setattr(dp_utils, "_run_ar", lambda **kwargs: payload)
    pc = NS(
        all2all_backend="allgather_reducescatter", data_parallel_size=2, num_ubatches=1
    )
    ubatch, actual, mode = dp_utils.coordinate_batch_across_dp(
        counts[0], False, pc, cudagraph_mode=modes[0], require_uniform_graph_tokens=True
    )
    assert not ubatch and mode == expected
    assert actual.tolist() == counts
    assert not dp_utils.get_last_all_ranks_decode_only()


def test_existing_dp_callers_still_pad_unequal_graph_batches(monkeypatch):
    payload = torch.tensor([[4, 8], [4, 8], [4, 8], [0, 0], [2, 2], [0, 0]])
    monkeypatch.setattr(dp_utils, "_run_ar", lambda **kwargs: payload)
    pc = NS(
        all2all_backend="allgather_reducescatter", data_parallel_size=2, num_ubatches=1
    )
    _, counts, mode = dp_utils.coordinate_batch_across_dp(
        4, False, pc, cudagraph_mode=2
    )
    assert mode == 2 and counts.tolist() == [8, 8]


def test_manager_preserves_nested_model_api():
    manager = object.__new__(graphs.DraftGraphManager)
    nested = NS(set_skip_topk=lambda skip: None)
    manager.runnable = NS(model=nested, compute_logits=lambda x: x + 1)
    assert manager.model is nested
    assert manager.compute_logits(2) == 3


def test_clear_graphs_causes_eager_fallback(monkeypatch):
    manager = object.__new__(graphs.DraftGraphManager)
    key = graphs.DraftGraphKey(4, 4, False)
    manager.entries = {key: NS(wrapper=NS(concrete_cudagraph_entries={}))}
    manager.first_key, manager.ready, manager.coordination_count = key, True, 0
    manager.proposer = NS(
        vllm_config=NS(
            parallel_config=NS(
                all2all_backend="allgather_reducescatter", data_parallel_size=1
            )
        )
    )
    manager.calibrated = lambda: True
    mode, n, _ = manager.coordinate(16)
    assert mode == CUDAGraphMode.NONE and n == 16
    assert not manager.first_replay


def test_dummy_run_does_not_reuse_last_request_graph_state():
    calls = []

    def orig(self, n, use_cudagraphs=True):
        calls.append(n)
        return CUDAGraphMode.NONE, n, None

    manager = NS(
        ready=True,
        in_propose=False,
        coordinate=lambda *a: pytest.fail("dummy must use original dispatch"),
    )
    proposer = NS(_ilu_draft_graph=manager)
    assert graphs.make_determine_batch(orig)(proposer, 1)[0] == CUDAGraphMode.NONE
    assert calls == [1]


def test_propose_scope_is_cleared_on_exception():
    manager = NS(in_propose=False, reset=lambda: None)

    def orig(self):
        assert manager.in_propose
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        graphs.make_propose(orig)(NS(_ilu_draft_graph=manager))
    assert not manager.in_propose


def supported_proposer():
    return NS(
        method="mtp",
        parallel_drafting=False,
        supports_mm_inputs=False,
        speculative_config=NS(enforce_eager=False),
        draft_model_config=NS(
            hf_config=NS(
                architectures=["IluGlmMoeDsaMTPModel"],
                num_nextn_predict_layers=1,
                index_share_for_mtp_iteration=True,
            )
        ),
        vllm_config=NS(
            model_config=NS(enforce_eager=False),
            lora_config=None,
            compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
            parallel_config=NS(
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                all2all_backend="allgather_reducescatter",
                enable_dbo=False,
                enable_eplb=False,
            ),
        ),
    )


def test_graph_is_opt_in_and_target_mode_is_preserved(monkeypatch):
    p = supported_proposer()
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    monkeypatch.setenv("VLLM_ILU_GLM_DRAFT_CUDAGRAPH", "0")
    assert not graphs.supported(p)
    monkeypatch.setenv("VLLM_ILU_GLM_DRAFT_CUDAGRAPH", "1")
    assert graphs.supported(p)
    assert (
        p.vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )
    p.speculative_config.enforce_eager = True
    assert not graphs.supported(p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("all2all_backend", "deepep_low_latency"),
        ("enable_dbo", True),
        ("enable_eplb", True),
    ],
)
def test_unsupported_parallel_modes_keep_existing_path(monkeypatch, field, value):
    monkeypatch.setenv("VLLM_ILU_GLM_DRAFT_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    p = supported_proposer()
    setattr(p.vllm_config.parallel_config, field, value)
    assert not graphs.supported(p)


@pytest.mark.parametrize("depth", [2, 3, 5])
def test_target_capture_rounding_keeps_followup_request_graph(depth):
    from vllm.config import CompilationConfig

    q = depth + 1
    cfg = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8, 4 * q],
        max_cudagraph_capture_size=4 * q,
    )
    cfg.adjust_cudagraph_sizes_for_spec_decode(q, tensor_parallel_size=1)
    if depth == 2:
        assert 4 not in cfg.cudagraph_capture_sizes
    keys = graphs._capture_keys(cfg.cudagraph_capture_sizes, 4, q)
    assert graphs.DraftGraphKey(4, q, False) in keys
    assert graphs.DraftGraphKey(4, 1, True) in keys
    assert all(0 < key.num_reqs <= 4 for key in keys)


@pytest.mark.parametrize("src", [{"other": 1}, None])
def test_metadata_layer_structure_is_checked(src):
    with pytest.raises(ValueError, match="layer set"):
        graphs._stage_tree({"layer": 1}, src)


def test_skipped_dp_collective_cannot_authorize_graph(monkeypatch):
    monkeypatch.setattr(dp_utils, "should_skip_dp_coordination", lambda: True)
    pc = NS(
        all2all_backend="allgather_reducescatter", data_parallel_size=2, num_ubatches=1
    )
    for require, expected in [(False, 2), (True, 0)]:
        _, counts, mode = dp_utils.coordinate_batch_across_dp(
            4, False, pc, cudagraph_mode=2, require_uniform_graph_tokens=require
        )
        assert mode == expected
        assert counts.tolist() == [4, 4]
