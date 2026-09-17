from __future__ import annotations

import math

import ixformer.inference.functions as ixfops
import torch

_FREQS_CIS_CACHE: dict[tuple, torch.Tensor] = {}
_COS_SIN_CACHE: dict[tuple, torch.Tensor] = {}


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    ``original_seq_len > 0`` enables YaRN. Sliding-window layers pass 0.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        freqs = freqs / factor * ramp + freqs * (1 - ramp)

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def get_shared_freqs_cis(key: tuple, device: torch.device) -> torch.Tensor:
    cached = _FREQS_CIS_CACHE.get((key, device))
    if cached is None:
        cached = precompute_freqs_cis(*key).to(device)
        _FREQS_CIS_CACHE[(key, device)] = cached
    return cached


def get_shared_cos_sin(key: tuple, device: torch.device) -> torch.Tensor:
    cached = _COS_SIN_CACHE.get((key, device))
    if cached is None:
        f = get_shared_freqs_cis(key, device)
        cached = torch.cat([f.real, f.imag], dim=-1).float().contiguous()
        _COS_SIN_CACHE[(key, device)] = cached
    return cached


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Rotate ``x`` in place. ``x`` is ``[1, s, d]`` or ``[1, s, h, d]``."""
    v = x.squeeze(0)
    if v.ndim == 2:
        v = v.unsqueeze(1)
    op = ixfops.ds4_inv_rope if inverse else ixfops.ds4_rope
    op(v, positions, cos_sin, True, None)
