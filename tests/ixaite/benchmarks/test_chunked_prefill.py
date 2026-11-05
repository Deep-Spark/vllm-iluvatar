# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the chunked-prefill profile and real-step acceptance gate."""

import pytest

from tests.ixaite.backend_isolation.chunked_prefill import (
    MODEL,
    check_steps,
    read_steps,
)
from tests.ixaite.benchmarks.core.model_catalog import e2e_spec
from tests.ixaite.benchmarks.suites.attention import AttentionSuite


# One request must continue prefill and then decode alongside another request.
def test_accepts_observed_continuation_and_mixed_step():
    def req(computed):
        return dict(
            request_id="long", query_tokens=512, computed=computed, prompt_tokens=1024
        )

    rows = [
        dict(step=0, prefills=1, decodes=1, requests=[req(0)]),
        dict(step=1, prefills=1, decodes=1, requests=[req(512)]),
        dict(step=2, prefills=0, decodes=2, requests=[req(1024)]),
    ]
    result = check_steps(rows)
    assert result["mixed_steps"] == [0, 1]
    assert result["long_requests"][0]["chunks"] == 2
    # Two concurrent prefills alone cannot serve as mixed-path evidence.
    with pytest.raises(AssertionError, match="mixed"):
        check_steps([{**r, "decodes": 0} for r in rows])
    # Completing prefill without subsequently exercising decode is incomplete.
    with pytest.raises(AssertionError, match="never entered decode"):
        check_steps(rows[:2])


# Reading a live JSONL file must not manufacture a completed scheduler step.
def test_read_steps_ignores_partial_last_line(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"step": 0}\n{"step":')
    assert read_steps(path) == [{"step": 0}]


# Lock the actual serving flags, independently of the original spec profile.
def test_chunked_profile_is_non_spec_and_exceeds_token_budget():
    spec = e2e_spec(MODEL)
    args = spec["serve_args"]
    assert args[args.index("--max-num-batched-tokens") + 1] == "512"
    assert args[args.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert "--enable-chunked-prefill" in args
    assert "--no-enable-prefix-caching" in args
    assert "--speculative-config" not in args
    assert "gdn_spec_delta_update" not in spec["ops"]
    assert {"flash_attn", "gdn_chunk", "gdn_packed_decode"} <= set(spec["ops"])
    assert {w["input_len"] for w in spec["workloads"]} == {2048, 2305}
    assert {w["max_concurrency"] for w in spec["workloads"]} == {1, 8}
    assert all(w["input_len"] + w["output_len"] <= 4096 for w in spec["workloads"])


# The FA comparison must have different query and key lengths to test cached prefixes.
def test_attention_continuation_uses_suffix_queries():
    import torch

    cases = AttentionSuite().iter_cases(MODEL, "paged_continuation")
    assert len(cases) == 4
    for case in cases:
        wl = case.make_workload(torch.device("cpu"), torch.bfloat16)
        assert wl.kwargs["max_seqlen_q"] < wl.kwargs["max_seqlen_k"]
        assert wl.kwargs["cu_seqlens_q"][-1] == wl.kwargs["q"].shape[0]
        assert wl.kwargs["cu_seqlens_k"][-1] == case.batch * case.context_len
