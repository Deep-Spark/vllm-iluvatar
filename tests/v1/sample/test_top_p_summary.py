# SPDX-License-Identifier: Apache-2.0
"""Sampling gates must preserve distributions and TP ownership boundaries."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates

from vllm_iluvatar.v1.sample.ops import top_p_summary
from vllm_iluvatar.v1.sample.ops.top_p_summary import eligible, verify_eligible


@pytest.fixture(autouse=True)
def supported_architecture(monkeypatch):
    monkeypatch.setattr(top_p_summary, "current_cuda_arch", lambda: "ivcore30")


def sampler():
    bias = object.__new__(LogitBiasState)
    bias.use_logit_bias = np.zeros(3, dtype=bool)
    penalties = object.__new__(PenaltiesState)
    penalties.use_penalty = np.zeros(3, dtype=bool)
    bad = object.__new__(BadWordsState)
    bad.num_bad_words = NS(np=np.zeros(3, dtype=np.int32))
    states = object.__new__(SamplingStates)
    states.temperature = NS(np=np.array([0.0, 0.6, 1.0], dtype=np.float32))
    states.top_p = NS(np=np.array([1.0, 0.95, 0.8], dtype=np.float32))
    states.top_k = NS(np=np.full(3, 154880))
    states.min_p = NS(np=np.zeros(3))
    states.vocab_size = 154880
    states.seeds_set = np.zeros(3, dtype=bool)
    states.num_logprobs = np.full(3, NO_LOGPROBS)
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


def test_explicit_seed_falls_back_independently_of_batch_companions():
    s = sampler()
    s.sampling_states.seeds_set[1] = True
    assert not eligible(s, logits(), np.array([1]))
    assert not eligible(s, logits(), np.arange(3))
    assert eligible(s, logits(), np.array([0, 2]))


@pytest.mark.parametrize(
    "mode", ["probabilistic", "watermark", "synthetic", "block", "seed"]
)
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
    elif mode == "seed":
        v.sampler.sampling_states.seeds_set[1] = True
    assert not verify_eligible(v, logits(), draft, np.arange(3))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or getattr(torch, "corex_arch", None) not in ("ivcore11", "ivcore30"),
    reason="native top-p summary requires ivcore11 or ivcore30",
)
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_ordinary_summary_all_masked_row_updates_only_valid_ids(
    monkeypatch, temperature
):
    from vllm.v1.worker.gpu.input_batch import post_update
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    from vllm_iluvatar.ilu_patch.general_patches import patch_gpu_top_p_summary

    # Exercise the registered sampler entry point and restore the patches on exit.
    monkeypatch.setattr(Sampler, "__init__", Sampler.__init__)
    monkeypatch.setattr(Sampler, "sample", Sampler.sample)
    monkeypatch.setattr(RejectionSampler, "_verify", RejectionSampler._verify)
    patch_gpu_top_p_summary._apply()
    monkeypatch.setattr(
        top_p_summary,
        "get_tp_group",
        lambda: NS(rank_in_group=0, broadcast=lambda tensor, src: None),
    )

    device = torch.device("cuda")
    vocab_size = 4096
    s = object.__new__(Sampler)
    s.__dict__.update(vars(sampler()))
    s.compute_nans = False
    s.trace_replay_state = None
    s.logprob_token_ids_state = NS(max_num_token_ids=lambda mapping: 0)
    states = s.sampling_states
    states.temperature.np[:] = temperature
    states.temperature.gpu = torch.full(
        (3,), temperature, device=device, dtype=torch.float32
    )
    states.top_p.np[:] = 1.0
    states.top_p.gpu = torch.ones(3, device=device, dtype=torch.float32)
    states.top_k.np[:] = vocab_size
    states.vocab_size = vocab_size
    states.seeds = NS(gpu=torch.zeros(3, device=device, dtype=torch.int64))

    def i32(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    idx_mapping = i32([0, 1])
    s.req_states = NS(prefill_len=NS(gpu=i32([2, 2])))
    batch = NS(
        num_reqs=2,
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=np.array([0, 1]),
        expanded_local_pos=i32([0, 0]),
        cu_num_logits=i32([0, 1, 2]),
        cu_num_logits_np=np.arange(3),
        positions=torch.tensor([1, 1], dtype=torch.int64, device=device),
        logits_indices=torch.arange(2, device=device),
        input_ids=torch.tensor([3, 3], dtype=torch.int64, device=device),
        seq_lens=i32([2, 2]),
        seq_lens_cpu_upper_bound=torch.tensor([2, 2], dtype=torch.int32),
    )
    scores = torch.full(
        (2, vocab_size), -torch.inf, dtype=torch.bfloat16, device=device
    )
    scores[0, 11] = 0

    result = s(scores, batch)
    assert s._ilu_summary_logged
    assert result.sampled_token_ids.flatten().tolist() == [11, 0]
    assert result.num_sampled.tolist() == [1, 1]
    assert result.num_rejected.tolist() == [0, 0]

    computed = i32([1, 1])
    generated = i32([2, 2])
    history = torch.full((2, 4), -99, dtype=torch.int32, device=device)
    last_sampled = i32([3, 3])
    bins = torch.zeros((2, vocab_size), dtype=torch.int32, device=device)
    post_update(
        idx_mapping,
        computed,
        last_sampled,
        bins,
        result.sampled_token_ids,
        result.num_sampled,
        result.num_rejected,
        i32([0, 1, 2]),
        history,
        generated,
    )
    assert history[:, 2].tolist() == [11, 0]
    assert last_sampled.tolist() == [11, 0]
    assert generated.tolist() == [3, 3]
    assert computed.tolist() == [2, 2]
    expected = torch.zeros_like(bins)
    expected[0, 11] = expected[1, 0] = 1
    torch.testing.assert_close(bins, expected)
