# SPDX-License-Identifier: Apache-2.0
"""V1 adapter contracts; CPU native spies isolate RNG and dispatch from kernels."""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference import functions as native

from vllm_iluvatar.v1.sample import rejection_sampler as impl


def inputs():
    return dict(
        draft_token_ids=torch.tensor([1, 2, -1], dtype=torch.int32),
        num_draft_tokens=[2, 0, 1],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2, 2, 3], dtype=torch.int32),
        draft_probs=None,
        target_logits=torch.randn(3, 8),
        bonus_token_ids=torch.tensor([[3], [4], [5]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(all_random=True, generators={}),
    )


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "probabilistic",
        "mixed",
        "synthetic",
        "fp64",
        "width",
        "unsupported",
        "capture",
    ],
)
def test_unsupported_contract_delegates_unchanged(monkeypatch, case):
    args = inputs()
    monkeypatch.setenv("VLLM_ILU_MTP_REJECTION_FUSION", "1")
    monkeypatch.setattr(
        impl, "_native_rejection_available", lambda _: case != "unsupported"
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: case == "capture"
    )
    if case == "disabled":
        monkeypatch.setenv("VLLM_ILU_MTP_REJECTION_FUSION", "0")
    elif case == "probabilistic":
        args["draft_probs"] = torch.ones(3, 8)
    elif case == "mixed":
        args["sampling_metadata"].all_random = False
    elif case == "synthetic":
        args["synthetic_mode"] = True
    elif case == "fp64":
        args["use_fp64_gumbel"] = True
    elif case == "width":
        args["max_spec_len"] = 65
    calls = []
    monkeypatch.setattr(
        impl,
        "_original_rejection_sample",
        lambda *a, **kw: calls.append((a, kw)) or "upstream",
    )
    assert impl.rejection_sample(**args) == "upstream"
    assert len(calls) == 1
    assert calls[0][0][0] is args["draft_token_ids"]
    assert calls[0][0][5] is args["target_logits"]
    assert calls[0][1]["use_fp64_gumbel"] == args.get("use_fp64_gumbel", False)


@pytest.mark.parametrize("reject", [False, True])
def test_rng_matches_upstream_policy_including_empty_and_accepted(monkeypatch, reject):
    args = inputs()
    generators = {i: torch.Generator().manual_seed(10 + i) for i in range(3)}
    args["sampling_metadata"].generators = generators
    monkeypatch.setenv("VLLM_ILU_MTP_REJECTION_FUSION", "1")
    monkeypatch.setattr(impl, "_native_rejection_available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    seen = {}

    def accept(probs, cu, draft, bonus, uniform, output):
        seen["uniform"] = uniform.clone()
        seen["probs"] = probs.clone()
        return torch.tensor(
            [[0, 1] if reject else [-1, -1], [-1, -1], [-1, -1]], dtype=torch.int32
        )

    def recover(probs, draft, rejected, requests, noise, output):
        seen["noise"] = noise.clone()
        assert requests.tolist() == [0]

    monkeypatch.setattr(native, "rejection_accept", accept, raising=False)
    monkeypatch.setattr(native, "rejection_recover", recover, raising=False)
    torch.manual_seed(123)
    impl.rejection_sample(**args)
    actual_default = torch.get_rng_state()
    actual_generators = {i: g.get_state() for i, g in generators.items()}
    torch.manual_seed(123)
    for i, g in generators.items():
        g.manual_seed(10 + i)
    uniform = impl.generate_uniform_probs(3, [2, 0, 1], generators, torch.device("cpu"))
    noise = torch.empty(3, 8).exponential_()
    for i, g in generators.items():
        if args["num_draft_tokens"][i] > 0:
            noise[i].exponential_(generator=g)
    assert torch.equal(actual_default, torch.get_rng_state())
    for i, g in generators.items():
        assert torch.equal(actual_generators[i], g.get_state())
    torch.testing.assert_close(seen["uniform"], uniform)
    torch.testing.assert_close(seen["probs"], args["target_logits"].softmax(-1))
    if reject:
        torch.testing.assert_close(seen["noise"], noise[:1])
    else:
        assert "noise" not in seen


def test_invalid_native_sentinel_raises_before_recovery(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_MTP_REJECTION_FUSION", "1")
    monkeypatch.setattr(impl, "_native_rejection_available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        native, "rejection_accept", lambda *a: torch.tensor([[-2, -1]]), raising=False
    )
    monkeypatch.setattr(
        native,
        "rejection_recover",
        lambda *a: pytest.fail("invalid output consumed"),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="requests \\[0\\]"):
        impl.rejection_sample(**inputs())


@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("draft_kind", ["likely", "padding", "empty_request"])
def test_cq_output_and_rng_match_upstream(monkeypatch, seeded, draft_kind, mask):
    from conftest import _corex30_available

    if not _corex30_available():
        pytest.skip("CQ device required")
    # Missing pinned APIs must fail rather than quietly testing fallback.
    assert callable(native.rejection_accept)
    assert callable(native.rejection_recover)
    torch.manual_seed(741)
    counts = [2, 0, 2] if draft_kind == "empty_request" else [2, 2, 2]
    x = torch.randn(sum(counts), 8192, device="cuda") * 4
    if mask:
        from vllm_iluvatar.v1.sample.ops import topk_topp_sampler

        monkeypatch.setenv("VLLM_ILU_TOP_P_RADIX", "1")
        x = topk_topp_sampler.apply_top_k_top_p(
            x, None, torch.full((sum(counts),), 0.95, device="cuda")
        )
    draft = x.argmax(-1).int()
    if draft_kind == "padding":
        draft[0] = -1
    generators = {0: torch.Generator(device="cuda").manual_seed(81)} if seeded else {}
    args = dict(
        draft_token_ids=draft,
        num_draft_tokens=counts,
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor(counts, dtype=torch.int32, device="cuda")
        .cumsum(0)
        .int(),
        draft_probs=None,
        target_logits=x,
        bonus_token_ids=torch.tensor([[3], [4], [5]], dtype=torch.int32, device="cuda"),
        sampling_metadata=SimpleNamespace(
            all_random=True,
            all_greedy=False,
            temperature=torch.ones(3, device="cuda"),
            generators=generators,
        ),
    )
    default = torch.cuda.get_rng_state()
    states = {i: g.get_state() for i, g in generators.items()}
    results = []
    for flag in ["0", "1"]:
        monkeypatch.setenv("VLLM_ILU_MTP_REJECTION_FUSION", flag)
        torch.cuda.set_rng_state(default)
        for i, g in generators.items():
            g.set_state(states[i])
        results.append(
            (
                impl.rejection_sample(**args),
                torch.cuda.get_rng_state(),
                {i: g.get_state() for i, g in generators.items()},
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    for i in generators:
        assert torch.equal(results[0][2][i], results[1][2][i])
