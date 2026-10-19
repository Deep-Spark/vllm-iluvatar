# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the DeepSeek-V4.1 raw-token-ids model state."""

from __future__ import annotations

import torch
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

from vllm_iluvatar.models.deepseek_v41.model_state import (
    RAW_INPUT_IDS_KWARG,
    IluDeepseekV41ModelState,
)


def _state(max_num_tokens: int = 8) -> IluDeepseekV41ModelState:
    # The parent __init__ wants a VllmConfig and a built model; only the two
    # fields the overrides touch matter here.
    state = IluDeepseekV41ModelState.__new__(IluDeepseekV41ModelState)
    state.max_num_tokens = max_num_tokens
    state.device = torch.device("cpu")
    return state


def test_prepare_inputs_adds_the_batch_ids_under_the_raw_name(monkeypatch):
    monkeypatch.setattr(
        DeepseekV41ModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {"input_ids": input_batch.input_ids},
    )
    input_ids = torch.tensor([3, 1, 4], dtype=torch.int32)
    input_batch = type("InputBatch", (), {"input_ids": input_ids})()

    model_inputs = _state().prepare_inputs(input_batch, req_states=None)

    # Same tensor, not a copy: the runner blanks "input_ids" right after this.
    assert model_inputs[RAW_INPUT_IDS_KWARG] is input_ids
    assert model_inputs["input_ids"] is input_ids


def test_prepare_dummy_inputs_hands_out_real_ids_sliced_to_the_batch(monkeypatch):
    monkeypatch.setattr(
        DeepseekV41ModelState,
        "prepare_dummy_inputs",
        lambda self, num_reqs, num_tokens: {},
    )
    state = _state(max_num_tokens=8)

    ids = state.prepare_dummy_inputs(num_reqs=2, num_tokens=3)[RAW_INPUT_IDS_KWARG]

    assert ids.shape == (3,)
    assert ids.dtype == torch.int32
    # Zero is a real token: the engram token map is indexed by these.
    assert torch.equal(ids, torch.zeros(3, dtype=torch.int32))


def test_prepare_dummy_inputs_reuses_one_buffer(monkeypatch):
    monkeypatch.setattr(
        DeepseekV41ModelState,
        "prepare_dummy_inputs",
        lambda self, num_reqs, num_tokens: {},
    )
    state = _state(max_num_tokens=8)

    first = state.prepare_dummy_inputs(num_reqs=1, num_tokens=2)[RAW_INPUT_IDS_KWARG]
    second = state.prepare_dummy_inputs(num_reqs=1, num_tokens=4)[RAW_INPUT_IDS_KWARG]

    assert first.data_ptr() == second.data_ptr()


def test_native_request_and_attention_callbacks_keep_parent_behavior(monkeypatch):
    from types import SimpleNamespace

    events = []
    state = _state()
    state.prefix_state = SimpleNamespace(
        add_request=lambda idx, req: events.append(("prefix_add", idx, req)),
        remove_request=lambda rid: events.append(("prefix_remove", rid)),
        prepare_attn=lambda batch, tables, slots: events.append(
            ("prefix_attn", batch, tables, slots)
        ),
    )
    monkeypatch.setattr(
        DeepseekV41ModelState,
        "add_request",
        lambda self, idx, req: events.append(("base_add", idx, req)),
    )
    monkeypatch.setattr(
        DeepseekV41ModelState,
        "remove_request",
        lambda self, rid: events.append(("base_remove", rid)),
    )
    monkeypatch.setattr(
        DeepseekV41ModelState, "prepare_attn", lambda self, *a, **kw: "base_metadata"
    )
    req = object()
    state.add_request(3, req)
    state.remove_request("r")
    assert events == [
        ("base_add", 3, req),
        ("prefix_add", 3, req),
        ("prefix_remove", "r"),
        ("base_remove", "r"),
    ]
    assert state.prepare_attn("batch", "mode", "tables", "slots") == "base_metadata"
    assert events[-1] == ("prefix_attn", "batch", "tables", "slots")
