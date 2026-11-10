# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow checks; native arithmetic/TP are tested separately."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import speculator as impl


@pytest.mark.parametrize("arch", ["ivcore30", "ivcore11"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize(
    "layout", ["dense", "fp32_bias", "broadcast_bias", "strided_bias", "broadcast_base"]
)
@pytest.mark.parametrize("confidence", [False, True])
@pytest.mark.parametrize("full_logits", [False, True])
def test_sequential_selection_and_fallbacks(
    monkeypatch, arch, dtype, layout, confidence, full_logits
):
    monkeypatch.setattr(impl, "cuda_arch", arch)
    generator = torch.Generator().manual_seed(73)

    def random(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    embedding, projection, bias_weight = random(31, 8), random(8, 31), random(8, 31)
    hidden = random(40, 8)
    fused_calls = []
    gather_calls = []

    def fused(base, bias, start, padding):
        # Enforce the native input contract so unsupported layouts must take
        # the original path, rather than silently passing via this CPU oracle.
        assert base.shape == bias.shape and base.dtype == bias.dtype
        assert base.dtype in (torch.bfloat16, torch.float32)
        assert base.stride(1) == bias.stride(1) == 1
        assert base.stride(0) >= base.shape[1]
        assert bias.stride(0) >= bias.shape[1]
        fused_calls.append(base.shape[0])
        return (base + bias)[:, : base.shape[1] - padding].argmax(-1) + start

    monkeypatch.setattr(impl, "add_bias_vocab_parallel_argmax", fused)

    def gather(logits, vocab, padding):
        gather_calls.append(logits.shape[0])
        return logits[:, :vocab]

    monkeypatch.setattr(impl, "gather_vocab_parallel_logits", gather)
    monkeypatch.setattr(
        impl,
        "vocab_parallel_argmax",
        lambda x, start, p: x[:, : x.shape[1] - p].argmax(-1) + start,
    )

    def project(hidden):
        result = hidden @ projection
        return result[:1].expand_as(result) if layout == "broadcast_base" else result

    def bias(hidden):
        result = hidden @ bias_weight
        if layout == "fp32_bias":
            return result.float()
        if layout == "broadcast_bias":
            return result[:1]
        if layout == "strided_bias":
            return torch.stack((result, result), -1).flatten(-2)[:, ::2]
        return result

    model = SimpleNamespace(
        compute_draft_logits_local=project,
        get_draft_vocab_shard=lambda: (0, 0),
        markov_embed=lambda ids: embedding[ids],
        markov_bias_local=bias,
        map_draft_to_target=lambda ids: (ids * 7 + 3) % 31,
        compute_confidence=lambda h, e: (h.float() + e.float()).sum(-1),
    )

    def state(local):
        owner = SimpleNamespace(
            _draft_topk=None,
            model=model,
            num_speculative_steps=5,
            sample_indices=torch.arange(40),
            _d2t_scatter_index=None,
            vocab_size=31,
            sample_idx_mapping=torch.arange(40),
            sample_pos=torch.arange(40),
            use_confidence_head=confidence,
            input_buffers=SimpleNamespace(input_ids=torch.arange(8)),
            _anchor_idx=torch.arange(8),
            draft_logits=torch.empty(0) if full_logits else None,
            use_local_argmax_reduction=local,
            draft_tokens=torch.full((8, 5), -7),
            draft_token_confidence_probs=torch.full((8, 5), -7.0),
        )
        owner._sample_logits = lambda logits, *args: model.map_draft_to_target(
            logits.argmax(-1)
        )
        return owner

    control, candidate = state(False), state(True)
    for requests in [4, 1, 7, 8, 4]:
        indices = torch.randperm(40, generator=generator)
        anchors = torch.randperm(8, generator=generator)
        input_ids = torch.randint(31, (8,), generator=generator)
        for owner in (control, candidate):
            owner.sample_indices.copy_(indices)
            owner._anchor_idx.copy_(anchors)
            owner.input_buffers.input_ids.copy_(input_ids)
            prior_gathers = len(gather_calls)
            impl.sample_sequential(owner, requests, hidden)
            if arch == "ivcore30" and requests < 8:
                # Low CQ batches must restore full gather, not merely fall
                # back from the fused kernel to the old local reduction.
                assert len(gather_calls) - prior_gathers == 5
        assert torch.equal(control.draft_tokens, candidate.draft_tokens)
        assert torch.equal(
            control.draft_token_confidence_probs, candidate.draft_token_confidence_probs
        )
    if layout == "dense" and dtype != torch.float16 and not full_logits:
        expected = (
            [8] * 5
            if arch == "ivcore30"
            else [4] * 5 + [1] * 5 + [7] * 5 + [8] * 5 + [4] * 5
        )
        assert fused_calls == expected
    if (
        full_logits
        or layout in ("strided_bias", "broadcast_base")
        or dtype == torch.float16
    ):
        assert not fused_calls
