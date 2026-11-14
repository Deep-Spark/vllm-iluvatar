# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm_iluvatar.v1.spec_decode.fly import compute_fly_entropy


def test_entropy_keeps_full_vocabulary_normalization():
    probabilities = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    expected = -0.4 * math.log(0.4) - 0.3 * math.log(0.3)

    actual = compute_fly_entropy(probabilities, 2)

    torch.testing.assert_close(actual, torch.tensor([expected]))


def test_entropy_keeps_equal_probability_multiplicity():
    probabilities = torch.full((2, 5), 0.2)

    actual = compute_fly_entropy(probabilities, 3)

    expected = torch.full((2,), -3 * 0.2 * math.log(0.2))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("top_k", [1, 3, 8, 100])
def test_entropy_logits_match_probabilities_with_masks_and_strides(dtype, top_k):
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 34, generator=generator).to(dtype)[:, ::2]
    logits[0, 3] = -math.inf
    logits[1] = -math.inf
    logits[1, 1] = 0
    probabilities = logits.float().softmax(-1)

    actual = compute_fly_entropy(logits, top_k, from_logits=True)
    expected = compute_fly_entropy(probabilities, top_k)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("from_logits", [False, True])
def test_entropy_empty_rows_preserve_shape_and_output_dtype(from_logits):
    actual = compute_fly_entropy(
        torch.empty(0, 17, dtype=torch.bfloat16), 3, from_logits=from_logits
    )

    assert actual.shape == (0,)
    assert actual.dtype == torch.float32


def test_entropy_all_masked_logits_and_zero_probabilities_are_zero():
    logits = torch.full((2, 17), -math.inf)

    torch.testing.assert_close(
        compute_fly_entropy(logits, 3, from_logits=True), torch.zeros(2)
    )
    torch.testing.assert_close(
        compute_fly_entropy(torch.zeros_like(logits), 3), torch.zeros(2)
    )


@pytest.mark.parametrize(
    ("shape", "top_k", "message"),
    [
        ((3,), 3, "2-D"),
        ((1, 0), 3, "non-empty"),
        ((1, 3), 0, "greater than zero"),
        ((1, 3), -1, "greater than zero"),
    ],
)
def test_entropy_rejects_invalid_metadata(shape, top_k, message):
    with pytest.raises(ValueError, match=message):
        compute_fly_entropy(torch.empty(shape), top_k)
