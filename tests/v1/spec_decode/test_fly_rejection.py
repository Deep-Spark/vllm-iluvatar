# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred-window and probability-gate regressions for MRV2 FLy."""

import pytest
import torch

from vllm_iluvatar.v1.worker.gpu.spec_decode.fly_rejection import (
    rejection_sample,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for FLy rejection tests", allow_module_level=True)


def _build_rejection_sample_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    num_speculative_steps: int,
    temperature: float,
    num_trials: int,
) -> dict:
    """Build rejection_sample kwargs from a fixed target and draft distribution.

    target_logits_1d must already have temperature applied (the sampler applies
    sampling params before verification), whereas draft_logits_1d must not:
    rejection_sample divides the draft logits by the temperature on load.
    """
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = (
        draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()
    )

    scaled_draft_logits_1d = draft_logits_1d.float()
    if temperature > 0:
        scaled_draft_logits_1d = scaled_draft_logits_1d / temperature
    draft_probs = torch.softmax(scaled_draft_logits_1d, dim=0)
    draft_tokens = torch.multinomial(
        draft_probs.expand(num_trials, -1), K, replacement=True
    )
    draft_sampled_2d = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    draft_sampled_2d[:, 1:] = draft_tokens
    draft_sampled = draft_sampled_2d.reshape(-1)

    cu_num_logits = torch.arange(num_trials + 1, dtype=torch.int32, device=device) * (
        K + 1
    )
    pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.arange(
        num_trials, dtype=torch.int32, device=device
    ).repeat_interleave(K + 1)
    expanded_local_pos = torch.arange(K + 1, dtype=torch.int32, device=device).repeat(
        num_trials
    )
    temp_tensor = torch.full(
        (num_trials,), temperature, dtype=torch.float32, device=device
    )
    seed = torch.arange(num_trials, dtype=torch.int64, device=device)

    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temp_tensor,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("temperature", "has_draft_logits", "probability_floor"),
    [
        (0.0, False, 0.0),
        (0.0, True, 0.0),
        (1.0, False, 0.0),
        (1.0, True, 0.0),
        (1.0, True, 0.01),
    ],
)
@pytest.mark.parametrize("top_k", [1, 3])
def test_fly_requires_entropy_and_native_lookahead(
    temperature: float,
    has_draft_logits: bool,
    probability_floor: float,
    top_k: int,
):
    """Rescue only drafts passing the entropy, lookahead and probability gates."""
    device = "cuda"
    inputs = _build_rejection_sample_inputs(
        torch.zeros(10, device=device),
        torch.zeros(10, device=device),
        num_speculative_steps=4,
        temperature=temperature,
        num_trials=7,
    )
    # Full window, low entropy, masked token, consecutive rejections,
    # short request, placeholder draft, and bonus-only request.
    counts = [5, 5, 5, 5, 3, 5, 1]
    target_probs = torch.zeros(7, 5, 10, device=device)
    target_probs[:, 0, 7:10] = torch.tensor([1e-12, 0.5, 0.5], device=device)
    for row, token in enumerate([2, 3, 4, 8], start=1):
        target_probs[:, row, token] = 1.0
    target_probs[1, 0, 8:10] = torch.tensor([1.0, 0.0], device=device)
    target_probs[2, 0, 7] = 0.0
    target_probs[3, 1, 2] = 1e-12
    target_probs[3, 1, 8:10] = 0.5
    draft_tokens = inputs["draft_sampled"].view(7, 5)
    draft_tokens[:, 1:] = torch.tensor([7, 2, 3, 4], device=device)
    draft_tokens[5, 2] = -1
    if has_draft_logits:
        inputs["draft_logits"].fill_(float("-inf"))
        for step, token in enumerate([7, 2, 3, 4]):
            inputs["draft_logits"][:, step, token] = 0.0
    else:
        inputs["draft_logits"] = None
    rows = torch.tensor(
        [req * 5 + row for req, count in enumerate(counts) for row in range(count)],
        device=device,
    )
    inputs["target_logits"] = target_probs.log().view(-1, 10)[rows]
    for key in ("draft_sampled", "pos", "expanded_idx_mapping", "expanded_local_pos"):
        inputs[key] = inputs[key][rows]
    inputs["cu_num_logits"] = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device
    )

    sampled, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=4,
        fly_window_size=2,
        fly_entropy_threshold=0.5,
        fly_entropy_top_k=top_k,
        fly_min_target_probability=probability_floor,
    )
    rescued = top_k == 3 and probability_floor == 0.0
    assert num_sampled.tolist() == [5 if rescued else 1, 1, 1, 1, 1, 1, 1]
    if rescued:
        assert sampled[0].tolist() == [7, 2, 3, 4, 8]
    else:
        assert sampled[0, 0].item() >= 8
    assert sampled[1:6, 0].min().item() >= 8


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("failing_step", [None, 2, 5])
def test_fly_pending_window_preserves_recovery_position(
    temperature: float, failing_step: int | None
):
    """A failed window recovers at its first rejection using that position's p/q."""
    device = "cuda"
    num_steps, vocab_size, num_trials = 7, 16, 32
    draft_ids = torch.arange(1, num_steps + 1, device=device)
    steps = torch.arange(num_steps, device=device)
    draft_probs = torch.zeros(num_steps, vocab_size, device=device)
    draft_probs[steps, draft_ids] = 0.8
    draft_probs[:, -1] = 0.2
    target_probs = torch.zeros(num_steps + 1, vocab_size, device=device)
    target_probs[steps, draft_ids] = 0.9
    target_probs[:-1, -1] = 0.1
    target_probs[-1, -2:] = torch.tensor([0.7, 0.3], device=device)
    # Distinct normalizers expose accidentally overwritten recovery statistics.
    shifts = torch.arange(num_steps + 1, device=device)[:, None] * 11
    native_logits = target_probs.log() + shifts
    rejections = [0, 3] + ([] if failing_step is None else [failing_step])
    target_probs[rejections] = 0
    target_probs[rejections, draft_ids[rejections]] = 1e-12
    target_probs[rejections, -2:] = torch.tensor([0.7, 0.3], device=device)
    inputs = _build_rejection_sample_inputs(
        native_logits[0], draft_probs[0].log(), num_steps, temperature, num_trials
    )
    inputs["target_logits"] = (target_probs.log() + shifts).repeat(num_trials, 1)
    inputs["draft_logits"] = (
        draft_probs.log() * (temperature if temperature > 0 else 1) + steps[:, None] * 7
    ).repeat(num_trials, 1, 1)
    inputs["draft_sampled"].view(num_trials, num_steps + 1)[:, 1:] = draft_ids
    rejected_step = (
        num_steps if failing_step is None else (0 if failing_step == 2 else 3)
    )
    reference_inputs = dict(inputs)
    reference_logits = (
        inputs["target_logits"].clone().view(num_trials, num_steps + 1, vocab_size)
    )
    for step in (0, 3):
        if step < rejected_step:
            reference_logits[:, step] = native_logits[step]
    reference_inputs["target_logits"] = reference_logits.flatten(0, 1)
    expected, expected_counts = rejection_sample(
        **reference_inputs, num_speculative_steps=num_steps
    )
    sampled, counts = rejection_sample(
        **inputs, num_speculative_steps=num_steps, fly_window_size=2
    )
    assert counts.tolist() == [rejected_step + 1] * num_trials
    assert torch.equal(counts, expected_counts)
    assert torch.equal(
        sampled[:, : rejected_step + 1], expected[:, : rejected_step + 1]
    )
