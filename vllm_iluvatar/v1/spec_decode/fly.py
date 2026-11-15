# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary normalized top-k entropy for FLy verification."""

import torch
from vllm.triton_utils import tl, triton


def _compute_fly_entropy_torch(values, top_k, from_logits):
    values = values.to(torch.float32)
    top_values = torch.topk(values, k=top_k, dim=-1).values
    if from_logits:
        top_log_probs = top_values - values.logsumexp(dim=-1, keepdim=True)
        top_probs = top_log_probs.exp()
    else:
        top_probs = top_values
        top_log_probs = top_probs.log()

    entropy_terms = torch.where(
        top_probs > 0,
        top_probs * top_log_probs,
        torch.zeros_like(top_probs),
    )
    return -entropy_terms.sum(dim=-1)


@triton.jit
def _fly_order_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    # Match torch.topk's NaN-before-number ordering without inspecting the input
    # on the host. Equal-value tie order has no effect on entropy.
    return tl.where(value != value, 0xFFFFFFFF, key).to(tl.uint32)


@triton.jit
def _fly_entropy_tiles(
    values_ptr,
    partial_ptr,
    output_ptr,
    logsumexp_ptr,
    row_stride,
    col_stride,
    vocab_size: tl.constexpr,
    num_tiles: tl.constexpr,
    TOP_K: tl.constexpr,
    FROM_LOGITS: tl.constexpr,
    RETURN_LOGSUMEXP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    value = tl.load(
        values_ptr + row * row_stride + offsets * col_stride,
        offsets < vocab_size,
        other=-float("inf"),
    ).to(tl.float32)
    partial_base = (row * num_tiles + tile) * (TOP_K + 2)
    log_normalizer = tl.full((), 0.0, tl.float32)
    if FROM_LOGITS:
        maximum = tl.max(value, 0)
        safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        denominator = tl.sum(tl.exp(value - safe_maximum), 0)
        if num_tiles == 1:
            log_normalizer = safe_maximum + tl.log(denominator)
            if RETURN_LOGSUMEXP:
                tl.store(logsumexp_ptr + row, log_normalizer)
        else:
            tl.store(partial_ptr + partial_base + TOP_K, maximum)
            tl.store(partial_ptr + partial_base + TOP_K + 1, denominator)

    keys = tl.where(offsets < vocab_size, _fly_order_key(value), 0)
    entropy = tl.full((), 0.0, tl.float32)
    for k in range(TOP_K):
        max_key, index = tl.max(keys, 0, return_indices=True)
        selected_offset = tile * BLOCK + index
        selected = tl.load(
            values_ptr + row * row_stride + selected_offset * col_stride,
            (selected_offset < vocab_size) & (max_key != 0),
            other=-float("inf"),
        ).to(tl.float32)
        if num_tiles == 1:
            if FROM_LOGITS:
                log_probability = selected - log_normalizer
                probability = tl.exp(log_probability)
            else:
                probability = selected
                log_probability = tl.log(probability)
            entropy -= tl.where(probability > 0, probability * log_probability, 0.0)
        else:
            tl.store(partial_ptr + partial_base + k, selected)
        keys = tl.where(tl.arange(0, BLOCK) == index, 0, keys)
    if num_tiles == 1:
        tl.store(output_ptr + row, entropy)


@triton.jit
def _fly_entropy_merge(
    partial_ptr,
    output_ptr,
    logsumexp_ptr,
    num_tiles: tl.constexpr,
    TOP_K: tl.constexpr,
    FROM_LOGITS: tl.constexpr,
    RETURN_LOGSUMEXP: tl.constexpr,
    BLOCK: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    partial_base = row * num_tiles * (TOP_K + 2)
    top_offsets = (offsets // TOP_K) * (TOP_K + 2) + offsets % TOP_K
    selected = tl.load(
        partial_ptr + partial_base + top_offsets,
        offsets < num_tiles * TOP_K,
        other=-float("inf"),
    )
    log_normalizer = tl.full((), 0.0, tl.float32)
    if FROM_LOGITS:
        tiles = tl.arange(0, LSE_BLOCK).to(tl.int64)
        maximum = tl.load(
            partial_ptr + partial_base + tiles * (TOP_K + 2) + TOP_K,
            tiles < num_tiles,
            other=-float("inf"),
        )
        denominator = tl.load(
            partial_ptr + partial_base + tiles * (TOP_K + 2) + TOP_K + 1,
            tiles < num_tiles,
            other=0.0,
        )
        global_maximum = tl.max(maximum, 0)
        safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
        total = tl.sum(denominator * tl.exp(maximum - safe_maximum), 0)
        log_normalizer = safe_maximum + tl.log(total)
        if RETURN_LOGSUMEXP:
            tl.store(logsumexp_ptr + row, log_normalizer)

    keys = tl.where(offsets < num_tiles * TOP_K, _fly_order_key(selected), 0)
    entropy = tl.full((), 0.0, tl.float32)
    for k in range(TOP_K):
        max_key, index = tl.max(keys, 0, return_indices=True)
        address = partial_base + (index // TOP_K) * (TOP_K + 2) + index % TOP_K
        top_value = tl.load(partial_ptr + address, max_key != 0, other=-float("inf"))
        if FROM_LOGITS:
            log_probability = top_value - log_normalizer
            probability = tl.exp(log_probability)
        else:
            probability = top_value
            log_probability = tl.log(probability)
        entropy -= tl.where(probability > 0, probability * log_probability, 0.0)
        keys = tl.where(offsets == index, 0, keys)
    tl.store(output_ptr + row, entropy)


def compute_fly_entropy(
    values: torch.Tensor,
    entropy_top_k: int,
    *,
    from_logits: bool = False,
    return_logsumexp: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k terms of FLy's full-distribution entropy in FP32.

    The selected probabilities retain their full-vocabulary normalization.
    FP16/BF16/FP32 CUDA inputs with k <= 8 use tiled selection/normalization
    and one small merge. Other inputs retain the upstream Torch path.
    Neither path normalizes only the selected k entries.

    Logit inputs may also return their full-vocabulary log-normalizer. The
    tiled kernels already compute it for entropy, so the probability gate can
    reuse it without another vocabulary scan or kernel launch.
    """
    if values.ndim != 2:
        raise ValueError("FLy expects 2-D target probabilities or logits")
    if values.shape[-1] == 0:
        raise ValueError("FLy requires a non-empty target vocabulary")
    if entropy_top_k <= 0:
        raise ValueError("fly_entropy_top_k must be greater than zero")
    if return_logsumexp and not from_logits:
        raise ValueError("FLy logsumexp output requires logits")
    top_k = min(entropy_top_k, values.shape[-1])
    if (
        not values.is_cuda
        or values.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or top_k > 8
        or values.requires_grad
    ):
        output = _compute_fly_entropy_torch(values, top_k, from_logits)
        if return_logsumexp:
            return output, values.float().logsumexp(dim=-1)
        return output
    rows, vocab_size = values.shape
    output = torch.empty(rows, device=values.device, dtype=torch.float32)
    logsumexp = torch.empty_like(output) if return_logsumexp else None
    if rows == 0:
        if return_logsumexp:
            return output, logsumexp
        return output
    block = min(4096, triton.next_power_of_2(vocab_size))
    num_tiles = triton.cdiv(vocab_size, block)
    partial = torch.empty(
        (rows, num_tiles, top_k + 2), device=values.device, dtype=torch.float32
    )
    with torch.cuda.device(values.device):
        _fly_entropy_tiles[(rows, num_tiles)](
            values,
            partial,
            output,
            logsumexp,
            values.stride(0),
            values.stride(1),
            vocab_size,
            num_tiles,
            top_k,
            from_logits,
            return_logsumexp,
            block,
            num_warps=4,
        )
        if num_tiles > 1:
            _fly_entropy_merge[(rows,)](
                partial,
                output,
                logsumexp,
                num_tiles,
                top_k,
                from_logits,
                return_logsumexp,
                triton.next_power_of_2(num_tiles * top_k),
                triton.next_power_of_2(num_tiles),
                num_warps=1,
            )
    if return_logsumexp:
        return output, logsumexp
    return output
