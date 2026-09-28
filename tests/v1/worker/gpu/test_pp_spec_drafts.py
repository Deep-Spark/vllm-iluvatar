# SPDX-License-Identifier: Apache-2.0
"""CPU tests for PP+spec real-draft staging / attach helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm_iluvatar.v1.worker.gpu import model_runner as impl


def test_stage_scheduler_drafts_copies_non_negative_ids():
    draft_buf = torch.full((4, 5), -1, dtype=torch.int64)
    last_buf = torch.zeros(4, dtype=torch.int64)
    last_buf[:] = 7
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        req_states=SimpleNamespace(
            draft_tokens=draft_buf,
            last_sampled_tokens=last_buf,
            req_id_to_index={"r0": 1, "r1": 2},
        ),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "r0": [10, 11, 12, 13, 14],
            "r1": [-1, -1, -1, -1, -1],  # must not wipe local drafts
        },
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            # PP catch-up token for r0 (leading verify token).
            new_token_ids=[[128822], []],
        ),
    )
    draft_buf[2] = torch.tensor([99, 98, 97, 96, 95])
    impl.stage_scheduler_drafts(runner, scheduler_output)
    assert draft_buf[1].tolist() == [10, 11, 12, 13, 14]
    assert draft_buf[2].tolist() == [99, 98, 97, 96, 95]
    assert int(last_buf[1]) == 128822
    assert int(last_buf[2]) == 7  # r1 had no real drafts → untouched


def test_stage_scheduler_drafts_skips_async():
    draft_buf = torch.full((2, 5), -1, dtype=torch.int64)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        req_states=SimpleNamespace(
            draft_tokens=draft_buf,
            req_id_to_index={"r0": 0},
        ),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [1, 2, 3, 4, 5]}
    )
    impl.stage_scheduler_drafts(runner, scheduler_output)
    assert draft_buf[0].tolist() == [-1, -1, -1, -1, -1]


def test_attach_draft_token_ids_on_async_output():
    draft_buf = torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=2)
        ),
        num_speculative_steps=5,
        draft_tokens_handler=SimpleNamespace(req_ids=["r0"]),
        req_states=SimpleNamespace(
            draft_tokens=draft_buf,
            req_id_to_index={"r0": 0},
        ),
    )
    inner = SimpleNamespace()
    output = SimpleNamespace(model_runner_output=inner)
    impl.attach_draft_token_ids(runner, output)
    assert inner.draft_token_ids.req_ids == ["r0"]
    assert inner.draft_token_ids.draft_token_ids == [[7, 8, 9, 10, 11]]


def test_attach_skipped_for_pp1():
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1)
        ),
        num_speculative_steps=5,
        draft_tokens_handler=SimpleNamespace(req_ids=["r0"]),
        req_states=SimpleNamespace(
            draft_tokens=torch.tensor([[1, 2, 3, 4, 5]]),
            req_id_to_index={"r0": 0},
        ),
    )
    inner = SimpleNamespace()
    impl.attach_draft_token_ids(runner, SimpleNamespace(model_runner_output=inner))
    assert not hasattr(inner, "draft_token_ids")


def test_patch_registers():
    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_drafts as patch

    patch.register()
    patch._apply()
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    assert getattr(Scheduler.update_from_output, "_ilu_pp_spec_drafts_patched", False)
    assert getattr(EngineCore.post_step, "_ilu_pp_spec_drafts_patched", False)
    assert getattr(GPUModelRunner.prepare_inputs, "_ilu_pp_spec_drafts_patched", False)
    assert getattr(GPUModelRunner.sample_tokens, "_ilu_pp_spec_drafts_patched", False)
