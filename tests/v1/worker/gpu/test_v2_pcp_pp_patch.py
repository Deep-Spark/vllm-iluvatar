# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for PCP segment rows crossing the PP sampling boundary."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

from vllm_iluvatar.v1.worker.gpu.model_runner import make_pcp_pp_sample_tokens
from vllm_iluvatar.v1.worker.gpu.pcp_manager import make_validate_config


def config(pcp=8, pp=2):
    return NS(
        parallel_config=NS(
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=pp,
            decode_context_parallel_size=1,
            dcp_comm_backend="ag_rs",
        ),
        model_config=NS(use_mla=True, is_encoder_decoder=False, hf_text_config=NS()),
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.NONE),
        lora_config=None,
        speculative_config=None,
    )


def test_validate_preserves_live_topology_and_other_rejections():
    validate = make_validate_config(PCPManager.validate_config)
    cfg = config()
    validate(cfg, False)
    assert cfg.parallel_config.pipeline_parallel_size == 2
    for field, value, error in [
        ("lora_config", object(), "LoRA"),
        (
            "speculative_config",
            NS(
                use_dspark=lambda: False,
                method="attention",
                use_multi_module_mtp=lambda: False,
            ),
            "speculative",
        ),
    ]:
        cfg = config()
        setattr(cfg, field, value)
        with pytest.raises(NotImplementedError, match=error):
            validate(cfg, False)
    with pytest.raises(NotImplementedError, match="MM"):
        validate(config(), True)
    cfg = config()
    cfg.model_config.use_mla = False
    with pytest.raises(NotImplementedError, match="MLA"):
        validate(cfg, False)
    cfg = config()
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(NotImplementedError, match="eager"):
        validate(cfg, False)


@pytest.mark.parametrize("pcp,pp", [(1, 2), (8, 1), (1, 1)])
def test_uncombined_validation_delegates_unchanged(pcp, pp):
    cfg = config(pcp, pp)
    calls = []
    make_validate_config(lambda c, mm: calls.append((c, mm)))(cfg, False)
    assert calls == [(cfg, False)]
    assert calls[0][0] is cfg


def batch(indices, computed, scheduled, prefill):
    return NS(
        num_reqs=len(indices),
        idx_mapping=np.array(indices),
        num_computed_tokens_np=np.array(computed),
        num_scheduled_tokens=np.array(scheduled),
        prefill_len_np=np.array(prefill),
        max_seq_len_np=np.array(prefill) + 40,
        is_prefilling_np=np.array([c < p for c, p in zip(computed, prefill)]),
    )


@pytest.mark.parametrize("local_rows", [[0, 0], [], [1, 0, 0]])
@pytest.mark.parametrize("scheduled", [[16, 1], [8, 1]])
def test_non_last_stage_receives_and_updates_global_requests(local_rows, scheduled):
    # A chunked prefill alongside a decode. Rank-local segments may duplicate
    # the prefill, omit it entirely, or reorder it after the decode.
    global_batch = batch([0, 1], [0, 20], scheduled, [16, 10])
    local_batch = batch(
        local_rows, [0] * len(local_rows), [2] * len(local_rows), [16] * len(local_rows)
    )
    state = ExecuteModelState(local_batch, None, None, None, None, None, set(), None, None)
    computed = np.array([0, 20])
    received = []
    postprocessed = []

    def receive(b):
        mask = compute_need_sampled_mask(b)
        received.append((b.num_reqs, mask.tolist()))
        return bool(mask.all())

    def advance(b):
        # CPU reference for request-level progress; never count PCP segments.
        np.add.at(computed, b.idx_mapping, b.num_scheduled_tokens)

    runner = NS(
        execute_model_state=state,
        pcp_manager=NS(_global_batch=global_batch),
        is_last_pp_rank=False,
        pp_handler=NS(receive=receive),
        postprocess_num_computed_tokens=advance,
        model_state=NS(postprocess_state=lambda idx, _: postprocessed.append(idx)),
        kv_connector=NS(post_forward=lambda _: None),
        eplb=NS(step=lambda **_: None),
    )
    make_pcp_pp_sample_tokens(GPUModelRunner.sample_tokens)(runner, None)
    assert runner.execute_model_state is None
    assert received == [(2, [scheduled[0] == 16, True])]
    np.testing.assert_array_equal(computed, np.array([0, 20]) + scheduled)
    assert len(postprocessed) == int(scheduled[0] < 16)
    assert state.input_batch is local_batch


@pytest.mark.parametrize(
    "last,pcp,has_state",
    [(True, True, True), (False, False, True), (False, True, False)],
)
def test_other_sampling_paths_unchanged(last, pcp, has_state):
    state = (
        ExecuteModelState(object(), None, None, None, None, None, set(), None, None)
        if has_state
        else None
    )
    runner = NS(
        execute_model_state=state,
        is_last_pp_rank=last,
        pcp_manager=NS(_global_batch=object()) if pcp else None,
    )
    sentinel = object()

    def original(runner, grammar):
        assert runner.execute_model_state is state
        assert grammar is sentinel
        return sentinel

    assert make_pcp_pp_sample_tokens(original)(runner, sentinel) is sentinel


@pytest.mark.parametrize("stage", [0, 1])
def test_nn_expert_range_is_stage_local(monkeypatch, stage):
    from vllm_iluvatar.model_executor.layers.quantization import (
        ilu_w4a8_nn_experts as nn,
    )

    monkeypatch.setattr(
        nn, "get_ep_group", lambda: NS(rank=stage * 8 + 5, rank_in_group=5)
    )
    experts = nn.IluMoEStandardExpertsW4A8NN.__new__(nn.IluMoEStandardExpertsW4A8NN)
    experts.group_size = -1
    experts.w13_input_scale = None
    experts.local_num_experts = 32

    class RouteChecked(Exception):
        pass

    def route(**kwargs):
        assert kwargs["start_expert_id"] == 160
        assert kwargs["end_expert_id"] == 192
        assert kwargs["group_layout"].shape == (32, 2)
        raise RouteChecked

    monkeypatch.setattr(nn.ixff, "moe_compute_token_index_ep_packed", route)
    hidden = torch.zeros(2, 4, dtype=torch.bfloat16)
    with pytest.raises(RouteChecked):
        experts._apply_nn(
            hidden,
            hidden,
            None,
            None,
            torch.ones(2, 2),
            torch.tensor([[0, 160], [191, 255]]),
            None,
            256,
        )
