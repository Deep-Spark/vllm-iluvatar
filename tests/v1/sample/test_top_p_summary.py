# SPDX-License-Identifier: Apache-2.0
"""Sampling gates must preserve distributions and TP ownership boundaries."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

from vllm_iluvatar.v1.sample.ops.top_p_summary import eligible, verify_eligible


def sampler():
    bias = object.__new__(LogitBiasState)
    bias.use_logit_bias = np.zeros(3, dtype=bool)
    penalties = object.__new__(PenaltiesState)
    penalties.use_penalty = np.zeros(3, dtype=bool)
    bad = object.__new__(BadWordsState)
    bad.num_bad_words = NS(np=np.zeros(3, dtype=np.int32))
    states = NS(
        temperature=NS(np=np.array([0.0, 0.6, 1.0], dtype=np.float32)),
        top_p=NS(np=np.array([1.0, 0.95, 0.8], dtype=np.float32)),
        top_k=NS(np=np.full(3, 154880)),
        min_p=NS(np=np.zeros(3)),
        vocab_size=154880,
        max_num_logprobs=lambda _: -1,
    )
    return NS(
        _ilu_summary_replicated_batch=True,
        use_fp64_gumbel=False,
        return_sampling_mask=False,
        logits_processors=[bias, penalties, bad],
        thinking_budget_state=NS(enabled=False),
        sampling_states=states,
        logprobs_mode="raw_logprobs",
    )


def logits():
    return NS(
        is_cuda=True,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(18, 154880),
        is_contiguous=lambda: True,
    )


def test_supported_and_processed_logprob_gate():
    s = sampler()
    ids = np.arange(3)
    assert eligible(s, logits(), ids)
    s.logprobs_mode = "processed_logprobs"
    assert eligible(s, logits(), ids)
    assert not eligible(s, logits(), ids, return_logprobs=True)


def test_unsupported_architecture_keeps_original_sampler(monkeypatch):
    from vllm_iluvatar.v1.sample.ops import top_p_summary

    monkeypatch.setattr(top_p_summary, "current_cuda_arch", lambda: "ivcore40")
    assert not eligible(sampler(), logits(), np.arange(3))


@pytest.mark.parametrize(
    "mode",
    [
        "bias",
        "penalty",
        "bad_words",
        "custom",
        "temperature",
        "top_k",
        "min_p",
        "thinking",
        "sharded",
        "fp64",
        "mask",
    ],
)
def test_fallback_modes(mode):
    s = sampler()
    ids = np.arange(3)
    if mode == "bias":
        s.logits_processors[0].use_logit_bias[1] = True
    elif mode == "penalty":
        s.logits_processors[1].use_penalty[1] = True
    elif mode == "bad_words":
        s.logits_processors[2].num_bad_words.np[1] = 1
    elif mode == "custom":
        s.logits_processors.append(object())
    elif mode == "temperature":
        s.sampling_states.temperature.np[1] = np.nan
    elif mode == "top_k":
        s.sampling_states.top_k.np[1] = 40
    elif mode == "min_p":
        s.sampling_states.min_p.np[1] = 0.1
    elif mode == "thinking":
        s.thinking_budget_state = NS(
            enabled=True, use_thinking_budget=np.ones(3, dtype=bool)
        )
    elif mode == "sharded":
        s._ilu_summary_replicated_batch = False
    elif mode == "fp64":
        s.use_fp64_gumbel = True
    elif mode == "mask":
        s.return_sampling_mask = True
    assert not eligible(s, logits(), ids)


@pytest.mark.parametrize("mode", ["probabilistic", "watermark", "synthetic", "block"])
def test_verification_gate(mode):
    v = NS(
        sampler=sampler(),
        watermark_key=None,
        use_block_verification=False,
        synthetic_conditional_rates=None,
        num_speculative_steps=5,
    )
    draft = None
    assert verify_eligible(v, logits(), draft, np.arange(3))
    if mode == "probabilistic":
        draft = object()
    elif mode == "watermark":
        v.watermark_key = 1
    elif mode == "synthetic":
        v.synthetic_conditional_rates = object()
    elif mode == "block":
        v.use_block_verification = True
    assert not verify_eligible(v, logits(), draft, np.arange(3))
