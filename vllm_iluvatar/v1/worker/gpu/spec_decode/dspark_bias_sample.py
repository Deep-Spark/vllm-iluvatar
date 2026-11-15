# SPDX-License-Identifier: Apache-2.0
"""Fuse Markov bias into the native DSpark sampler and draft-cache write."""

import torch


def fused_bias_gumbel_sample(
    logits: torch.Tensor,
    bias: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    logits_cache: torch.Tensor,
    logits_cache_col: torch.Tensor,
    use_fp64: bool = False,
    num_vocab_padding: int = 0,
) -> torch.Tensor:
    """Sample without storing the sum, preserving the native sampler's RNG."""
    if hasattr(torch.ops._C, "ilu_dspark_bias_sample_v2"):
        rows = logits.shape[0]
        threads = 128 if 1 < rows <= 3 else 256
        return torch.ops._C.ilu_dspark_bias_sample_v2(
            logits,
            bias,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            logits_cache,
            logits_cache_col,
            use_fp64,
            num_vocab_padding,
            1024,
            threads,
        )
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return logits.new_empty((0,), dtype=torch.int64)
    if logits_cache.size(-1) < vocab_size:
        raise ValueError("draft logits cache is narrower than the sampled vocabulary")
    block_size = 1024
    num_blocks = (vocab_size + block_size - 1) // block_size
    local_argmax = logits.new_empty((num_tokens, num_blocks), dtype=torch.int64)
    local_max = logits.new_empty(
        (num_tokens, num_blocks),
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    torch.ops._C.ilu_fused_bias_gumbel_sample(
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0),
        logits_cache.stride(1),
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        block_size,
        True,
        True,
        use_fp64,
        logits_cache_col.dim() > 0,
        bias,
        bias.stride(0),
        num_vocab_padding,
    )
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)


class FusedBiasWorkspace:
    """Reusable partials and strided output for the native Markov sampler."""

    def __init__(
        self, max_rows: int, vocab_size: int, device: torch.device, use_fp64: bool
    ) -> None:
        self.vocab_size = vocab_size
        self.use_fp64 = use_fp64
        blocks = (vocab_size + 1023) // 1024
        ids = torch.empty((max_rows, blocks), dtype=torch.int64, device=device)
        values = torch.empty(
            (max_rows, blocks),
            dtype=torch.float64 if use_fp64 else torch.float32,
            device=device,
        )
        self.partials = [(ids[:rows], values[:rows]) for rows in range(max_rows + 1)]

    def sample_into(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        predicted_pos: torch.Tensor,
        cache: torch.Tensor,
        column: torch.Tensor,
        output: torch.Tensor,
        num_vocab_padding: int = 0,
    ) -> torch.Tensor:
        rows = logits.shape[0]
        if logits.shape[1] != self.vocab_size or rows >= len(self.partials):
            raise ValueError("DSpark sampler workspace geometry changed")
        ids, values = self.partials[rows]
        threads = 128 if 1 < rows <= 3 else 256
        torch.ops._C.ilu_dspark_bias_sample_v4(
            logits,
            bias,
            mapping,
            temperature,
            seed,
            predicted_pos,
            cache,
            column,
            self.use_fp64,
            num_vocab_padding,
            1024,
            threads,
            -1,
            ids,
            values,
            output,
        )
        return output
