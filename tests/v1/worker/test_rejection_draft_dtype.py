# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for aligning draft logits to the target dtype."""

from __future__ import annotations

import torch

from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    rejection_sample_with_aligned_draft,
)


def test_rejection_sample_casts_bf16_draft_to_target_fp32():
    seen = {}

    def original(target_logits, draft_logits, *args, **kwargs):
        seen["draft"] = draft_logits
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    wrapped = rejection_sample_with_aligned_draft(original)
    target = torch.zeros(2, 4, dtype=torch.float32)
    draft = torch.ones(1, 2, 4, dtype=torch.bfloat16)

    assert wrapped(target, draft, "rest", use_fp64=False) == "ok"
    assert seen["draft"].dtype == torch.float32
    assert torch.equal(seen["draft"], torch.ones(1, 2, 4, dtype=torch.float32))
    assert seen["args"] == ("rest",)
    assert seen["kwargs"] == {"use_fp64": False}


def test_rejection_sample_keeps_matching_draft_dtype():
    draft = torch.ones(1, 2, 4, dtype=torch.float32)
    seen = {}

    def original(target_logits, draft_logits, *args, **kwargs):
        seen["draft"] = draft_logits

    wrapped = rejection_sample_with_aligned_draft(original)
    wrapped(torch.zeros(2, 4), draft)
    assert seen["draft"] is draft


def test_rejection_sample_keyword_draft_and_none():
    seen = {}

    def original(*args, **kwargs):
        seen["kwargs"] = kwargs

    wrapped = rejection_sample_with_aligned_draft(original)
    target = torch.zeros(2, 4, dtype=torch.float32)
    draft = torch.ones(1, 2, 4, dtype=torch.bfloat16)

    wrapped(target_logits=target, draft_logits=draft)
    assert seen["kwargs"]["draft_logits"].dtype == torch.float32

    wrapped(target_logits=target, draft_logits=None)
    assert seen["kwargs"]["draft_logits"] is None
