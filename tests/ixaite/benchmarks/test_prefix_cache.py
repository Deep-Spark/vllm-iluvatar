# SPDX-License-Identifier: Apache-2.0
"""The all-mode acceptance matrix needs explicit cascade, non-spec and bounded graph sizes."""

import pytest

from tests.ixaite.backend_isolation.prefix_cache import MODEL
from tests.ixaite.benchmarks.core.model_catalog import e2e_spec
from tests.ixaite.benchmarks.e2e.runner import check_workload_result


def test_prefix_profile_covers_parent_matrix_with_bounded_request_capacity():
    """Do not allocate default 512-request Mamba profiling pools for a BS32 matrix."""
    spec = e2e_spec(MODEL)
    args = spec["serve_args"]
    assert args[args.index("--mamba-cache-mode") + 1] == "all"
    assert "--enable-prefix-caching" in args and "--no-disable-cascade-attn" in args
    assert "--speculative-config" not in args
    cases = {
        (w["max_concurrency"], w["input_len"], w["output_len"])
        for w in spec["workloads"]
    }
    assert cases == {
        (b, i, o)
        for b in (1, 8, 16, 32)
        for i, o in ((128, 128), (128, 2048), (2048, 128), (2048, 2048), (1024, 1024))
    }
    assert int(args[args.index("--max-num-seqs") + 1]) == max(b for b, _, _ in cases)
    # Top-k/top-p sorting needs runtime memory beyond the model/KV/graph profile.
    assert float(args[args.index("--gpu-memory-utilization") + 1]) <= 0.85


@pytest.mark.parametrize(
    "change",
    ({"completed": 31}, {"failed": 1}, {"total_output_tokens": 61751}),
)
def test_matrix_rejects_partial_results_even_with_successful_http(change):
    """A server OOM can leave HTTP-successful streams with truncated output."""
    workload = {"id": "bs32_128_2048", "num_prompts": 32, "output_len": 2048}
    result = {"completed": 32, "failed": 0, "total_output_tokens": 65536, **change}
    with pytest.raises(RuntimeError, match="expected"):
        check_workload_result(workload, result)


def test_matrix_accepts_complete_results():
    """Require the full generated token budget for all requests."""
    check_workload_result(
        {"id": "bs32_128_2048", "num_prompts": 32, "output_len": 2048},
        {"completed": 32, "failed": 0, "total_output_tokens": 65536},
    )
