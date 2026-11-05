# SPDX-License-Identifier: Apache-2.0
"""GPU verification of the diagnostic capture, replay and trace pipeline."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
def test_chunked_capture_replays_both_ops_and_attributes_kernels(tmp_path):
    import ixaite

    ixaite.initialize("ivcore11")
    from tests.ixaite.backend_isolation.chunked_prefill import (
        MODEL,
        check_trace,
        replay,
    )
    from tests.ixaite.backend_isolation.chunked_worker import ChunkedPrefillWorker
    from tests.ixaite.benchmarks.suites.attention import AttentionSuite
    from tests.ixaite.benchmarks.suites.chunk_gated_delta_rule import (
        ChunkGatedDeltaRuleSuite,
    )
    from vllm_iluvatar.backends.ixaite.layers.gdn import (
        _adapt_ixaite_gdn_chunk,
    )

    # Only exercise the observation methods; no model/worker lifecycle is mocked.
    worker = object.__new__(ChunkedPrefillWorker)
    worker._audit_active = True
    worker._audit_seen = set()
    worker._audit_step = 0
    worker._audit_dir = tmp_path / "ixaite/rank0"
    worker._audit_dir.mkdir(parents=True)
    worker._audit_gdn = _adapt_ixaite_gdn_chunk(ixaite.chunk_gated_delta_rule)
    worker._audit_fa = ixaite.flash_attn_varlen_func
    gdn_case = ChunkGatedDeltaRuleSuite().iter_cases(MODEL, None)[0]
    gdn = gdn_case.make_workload(torch.device("cuda"), torch.bfloat16)
    fa_case = AttentionSuite().iter_cases(MODEL, "paged_continuation")[0]
    fa = fa_case.make_workload(torch.device("cuda"), torch.bfloat16)
    # Warm kernels before tracing; profiling records actual provider calls only.
    worker._audit_active = False
    worker._capture_gdn(*gdn.args, None)
    worker._capture_fa(**fa.kwargs)
    worker._audit_active = True
    with torch.profiler.profile(
        record_shapes=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        worker._capture_gdn(*gdn.args, None)
        worker._capture_fa(**fa.kwargs)
    path = worker._audit_dir / "mixed-trace.json"
    prof.export_chrome_trace(str(path))
    # Both saved op inputs must survive compacting and compare to independent baseline math.
    assert len(replay(tmp_path)) == 2
    kernels = check_trace(path)
    assert any("gdn_" in name for name in kernels)
    assert any("flash_attn" in name for name in kernels)

    # A numerical failure must remain visible in a complete replay report.
    packet = worker._audit_dir / "0000-gdn.pt"
    saved = torch.load(packet, weights_only=True)
    saved["final_state"].add_(10)
    torch.save(saved, packet)
    results = replay(tmp_path, fail_fast=False)
    assert len(results) == 2
    assert sum(not row["passed"] for row in results) == 1
    with pytest.raises(AssertionError):
        replay(tmp_path)


# Check full WY tiles followed by a partial serving chunk without resetting state.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
def test_gdn_weak_decay_rounding_matches_baseline_across_chunks():
    """Correlated keys and large value channels expose bf16 WY rounding drift."""
    import ixaite
    import ixformer._C as ixf

    ixaite.initialize("ivcore11")
    torch.manual_seed(3266)
    state = torch.randn(1, 24, 128, 128, device="cuda") * 0.03
    expected_state = state.clone()
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    mask = torch.ones(1, device="cuda", dtype=torch.int32)
    for tokens in (511, 17):
        q = torch.randn(tokens, 8, 128, device="cuda")
        k = torch.randn_like(q) * 0.07
        k[:, :, 112] += 0.7
        q = torch.nn.functional.normalize(q, dim=-1).bfloat16()
        k = torch.nn.functional.normalize(k, dim=-1).bfloat16()
        v = torch.randn(tokens, 24, 128, device="cuda") * 0.1
        v[:, :, 65] += torch.rand(tokens, 24, device="cuda") * 30
        v = v.bfloat16()
        g = -torch.rand(tokens, 24, device="cuda") * 0.01
        beta = (0.5 + torch.rand(tokens, 24, device="cuda") * 0.5).bfloat16()
        cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        expected = torch.empty_like(v)
        ixf.infer.chunk_gated_delta_rule(
            q,
            k,
            v,
            expected,
            g,
            beta,
            expected_state,
            mask,
            cu,
            cu.cpu(),
            indices,
            128**-0.5,
            None,
        )
        actual, _ = ixaite.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            initial_state_indices=indices,
            has_initial_state=mask,
            cu_seqlens=cu,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state, expected_state, rtol=3e-2, atol=3e-2)
