# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from conftest import requires_cuda
from vllm_iluvatar.models.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

pytestmark = requires_cuda

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6
ATOL = 5e-3
RTOL = 1.5e-2
RMSE = 1e-3


def grouped_gemma_rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    grouped = x.float().unflatten(-1, (num_groups, x.shape[-1] // num_groups))
    variance = grouped.square().mean(-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


def hc_silu_torch(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.nn.functional.silu(x.float() / hc_count).to(x.dtype)


def hc_gate_mix_torch(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    hidden_size = x.shape[-1] // hc_count
    mixed = (
        torch.sigmoid(gate.float().unflatten(-1, (hc_count, hidden_size)))
        * x.float().unflatten(-1, (hc_count, hidden_size))
    ).mean(-2)
    return mixed.to(x.dtype)


def hc_combine_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    hidden_size = residual.shape[-1] // hc_count
    injection_weight = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    combined = residual.float().unflatten(-1, (hc_count, hidden_size))
    combined += block_output.float().unsqueeze(-2) * injection_weight.unsqueeze(-1)
    return combined.flatten(-2).to(residual.dtype)


def hc_combine_norm_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = hc_combine_torch(
        residual,
        block_output,
        injection_logits,
        hc_count,
    )
    normalized = grouped_gemma_rmsnorm_torch(
        combined,
        norm_weight,
        eps,
        hc_count,
    )
    return combined, normalized


def _check(got: torch.Tensor, ref: torch.Tensor, tag: str) -> None:
    err = (got.float() - ref.float()).abs().max().item()
    d = got.float() - ref.float()
    rmse = torch.sqrt((d * d).mean()).item()
    if (
        not torch.allclose(got.float(), ref.float(), atol=ATOL, rtol=RTOL)
    ) or rmse > RMSE:
        raise AssertionError(f"{tag} max={err:.4e} rmse={rmse:.4e}")


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)
    expected = grouped_gemma_rmsnorm_torch(x, weight, EPS, HC)
    _check(actual, expected, "grouped_gemma")


def test_hc_silu() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 320, dtype=torch.bfloat16, device="cuda")
    actual = hc_silu(x, HC)
    expected = hc_silu_torch(x, HC)
    _check(actual, expected, "silu")


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = hc_gate_mix_torch(x, gate, HC)
    _check(actual, expected, "gate_mix")


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    expected = hc_combine_torch(residual, block_output, injection, HC)
    _check(actual, expected, "combine")


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )
    expected, expected_norm = hc_combine_norm_torch(
        residual, block_output, injection, weight, EPS, HC
    )
    _check(actual, expected, "combine_norm hidden")
    _check(actual_norm, expected_norm, "combine_norm xn")
