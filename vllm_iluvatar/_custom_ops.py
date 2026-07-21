# SPDX-License-Identifier: Apache-2.0
"""
_custom_ops — Iluvatar custom operator implementations (slim, ILU-only).

Implementation layer — no side effects on import.
Corresponds to: vllm/_custom_ops.py
Registration:   vllm_iluvatar/ilu_patch/patch_custom_ops.py

Only contains functions that differ from the community implementation.
The patch_custom_ops.py patch layer splats these into vllm._custom_ops.
"""

from __future__ import annotations

import math

import ixformer.inference.functions as ixf_ops  # type: ignore[import]
import torch
import torch.nn.functional as F
from ixformer.core import config
from vllm.logger import init_logger

logger = init_logger(__name__)
_USE_TORCH_OPS = config.IXFORMER_USE_TORCH_OPS

# ---------------------------------------------------------------------------
# Manifest split into two groups (see ilu_patch/patch_custom_ops.py):
#   _ILU_OVERRIDES — functions that EXIST in community _custom_ops.py and whose
#                    ILU implementation replaces the community one. These are
#                    signature-validated against community (strict).
#   _ILU_ADDED     — ILU-only functions that community _custom_ops.py does NOT
#                    have. They are added (not overrides), so no community
#                    signature to validate against.
# ---------------------------------------------------------------------------
_ILU_OVERRIDES: list[str] = [
    "apply_repetition_penalties",
    "awq_gemm",
    "concat_and_cache_mla",
    "cp_gather_cache",
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_supports_block_fp8",
    "cutlass_scaled_mm_supports_fp8",
    "fused_add_rms_norm",
    "reshape_and_cache_flash",
    "rms_norm",
    "scaled_int8_quant",
    "topk_hash_softplus_sqrt",
    "topk_sigmoid",
    "topk_softmax",
]

_ILU_ADDED: list[str] = [
    "concat_and_cache_mla_int8",
    "custom_gptq_marlin_gemm",
    "dsa_convert_req_index_to_global_index",
    "fatrelu_and_mul",
    "gather_cache_int8",
    "gelu_and_mul",
    "gelu_fast",
    "gelu_new",
    "gelu_quick",
    "gelu_tanh_and_mul",
    "indexer_k_cache",
    "mul_and_silu",
    "quant_kv",
    "ref_mqa_logits",
    "ref_paged_mqa_logits",
    "rejection_greedy_sample_torch",
    "rejection_random_sample_torch",
    "reshape_and_cache_flash_mix",
    "rms_norm_qk",
    "router_gemm_bf16_fp32",
    "silu_and_mul",
    "silu_and_mul_with_clamp",
    "sparse_prefill_fwd",
]


# [ILU]
# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_ops.silu_and_mul(x, out)


def silu_and_mul_with_clamp(
    out: torch.Tensor,
    x: torch.Tensor,
    limit: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    if alpha == 1.0 and beta == 0.0:
        ixf_ops.silu_and_mul_with_clamp(input=x, output=out, limit=limit)
        return
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=limit)
    up = torch.clamp(x[..., d:], min=-limit, max=limit)
    out.copy_(gate * torch.sigmoid(alpha * gate) * (up + beta))


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_ops.gelu_and_mul(x, out)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_ops.gelu_tanh_and_mul(x, out)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    x = 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    out.copy_(x)


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    x = (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )
    out.copy_(x)


def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(x)
    out.mul_(torch.sigmoid(x * 1.702))


def fatrelu_and_mul(out: torch.Tensor, x: torch.Tensor, threshold: float = 0.0) -> None:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x1 = torch.where(x1 > threshold, x1, torch.zeros_like(x1))
    out.copy_(x1 * x2)


def mul_and_silu(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(x[..., :d] * torch.nn.functional.silu(x[..., d:]))


# pos encoding ops
# rotary_embedding is now provided by the plugin's native torch.ops._C kernel
# (csrc/pos_encoding_kernels.cu), which matches the community signature exactly,
# so community vllm._custom_ops.rotary_embedding is used unchanged (no override).


# layer norm ops
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    ixf_ops.rms_norm(input, weight, epsilon, out)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return ixf_ops.residual_rms_norm(input, weight, epsilon, 1.0, residual)


def rms_norm_qk(
    output_q: torch.Tensor,
    output_k: torch.Tensor,
    input_q: torch.Tensor,
    input_k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> None:
    ixf_ops.rms_norm_qk(
        input_q, input_k, weight_q, weight_k, epsilon, output_q, output_k
    )


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    # if logits.is_cuda and logits.is_contiguous():
    #     apply_repetition_penalties_cuda(
    #         logits, prompt_mask, output_mask, repetition_penalties
    #     )
    # else:
    apply_repetition_penalties_torch(
        logits, prompt_mask, output_mask, repetition_penalties
    )


# [ILU]
def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    pack_factor,
    group_size: int = 128,
) -> torch.Tensor:
    if _USE_TORCH_OPS:
        return torch.ops.ixf_ops.wui4a16(
            input, qweight, scales, qzeros, None, group_size, "NN"
        )
    return ixf_ops.wui4a16(input, qweight, scales, qzeros, None, group_size, "NN")


def custom_gptq_marlin_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    pack_factor,
    group_size: int = 128,
    bias=None,
) -> torch.Tensor:
    return ixf_ops.wui4a16(input, qweight, scales, qzeros, bias, group_size, "NN")


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return False


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return False


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    format: str = "TN",
    is_w4a8_linear: bool = False,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert a.ndim <= 3 and b.ndim == 2
    assert bias is None or (
        bias.dtype == out_dtype and bias.numel() == b.shape[1]
        if not is_w4a8_linear
        else bias.numel() == b.shape[1] * 2
    )

    if format == "NN" and a.ndim == 2 and b.ndim == 2 and a.shape[-1] < b.shape[0]:
        pad_size = b.shape[0] - a.shape[-1]
        a = torch.nn.functional.pad(a, (0, pad_size), mode="constant", value=0)
    m = a.shape[0]
    if format == "TN":
        assert not is_w4a8_linear, "W4A8 linear only supports NN format"
        n = b.shape[0]
    elif is_w4a8_linear:
        n = b.shape[1] * 2
    else:
        n = b.shape[1]
    if format == "TN":
        if a.shape[-1] != b.shape[1]:
            padding = b.shape[1] - a.shape[-1]
            a = torch.nn.functional.pad(a, (0, padding), mode="constant", value=0)
    else:
        if a.shape[-1] != b.shape[0]:
            padding = b.shape[0] - a.shape[-1]
            a = torch.nn.functional.pad(a, (0, padding), mode="constant", value=0)

    if a.ndim == 2:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    else:
        bs = a.shape[1]
        out = torch.empty((m, bs, n), dtype=out_dtype, device=a.device)

    if is_w4a8_linear:
        assert format == "NN", "W4A8 linear only supports NN format"
        ixf_ops.w4a8(
            a,
            b,
            scale_a,
            scale_b,
            bias=bias,
            format=0,
            output=out.view(-1, n),
            output_dtype=out_dtype,
        )
    else:
        ixf_ops.w8a8(
            a, b, scale_a, scale_b, bias, format=format, output=out, out_dtype=out_dtype
        )

    return out


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and
    scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        if scale.dtype != torch.float32:
            scale = scale.to(dtype=torch.float32)
        ixf_ops.static_scaled_int8_quant(output, input, scale)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    ixf_ops.dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales, input_azp


# [ILU]
def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM. weight shape: (N, K).

    On Iluvatar there is no native ``_moe_C::router_gemm_bf16_fp32`` kernel.
    We dispatch to the ixformer fused op ``gemm_bf16_bf16_fp32`` which calls
    ``cuinfer_gemm`` with ``CUDA_R_16BF x CUDA_R_16BF -> CUDA_R_32F`` -- this
    is numerically equivalent to the upstream cuBLAS kernel (fp32 accumulate)
    but runs natively on Iluvatar instead of falling back to a pure-torch
    fp32 cast + matmul.

    ``weight`` is (N, K) and we want ``input @ weight.T`` so we pass the
    ``"TN"`` format to the ixformer op (weight rows are ``N``, weight cols
    are ``K``, matching the ``TN`` interpretation in
    ``ixformer_gemm_bf16_bf16_fp32``).
    """
    return ixf_ops.gemm_bf16_bf16_fp32(input, weight, format="TN")


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    ixf_ops.moe_fused_topk_bias(
        gating_output=gating_output,
        topk=topk_weights.shape[-1],
        scoring_func="softmax",
        routed_scaling_factor=1,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> None:
    ixf_ops.moe_fused_topk_bias(
        gating_output=gating_output,
        topk=topk_weights.shape[-1],
        scoring_func="sigmoid",
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
    )


def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> None:
    """PyTorch equivalent of ``torch.ops._moe_C.topk_softplus_sqrt``.

    The native CUDA kernel is unavailable on Iluvatar, so we reproduce the
    same semantics in pure PyTorch. The function writes the three
    pre-allocated output buffers in-place:

    * ``topk_weights``         (num_tokens, top_k) fp32
    * ``topk_indices``         (num_tokens, top_k) int32/int64
    * ``token_expert_indices`` (num_tokens, top_k) int32/int64

    Pipeline:
      1. score = sqrt(softplus(gating_output))
      2. expert selection - either ``hash_indices_table[input_tokens]`` for
         hash routing, or ``topk(score + e_score_correction_bias)``.
      3. gather scores at chosen indices, optional renormalize, then
         multiply by ``routed_scaling_factor`` (matches the kernel which
         applies the scale internally).
      4. ``token_expert_indices[t, j] = j`` (positional helper).
    """
    # ixformer expects int32 index tensors. topk_indices/token_expert_indices
    # are output buffers, so use temporaries only when needed and copy back into
    # the original buffers after the kernel writes them.
    orig_topk_indices = topk_indices
    orig_token_expert_indices = token_expert_indices
    if orig_topk_indices.dtype != torch.int32:
        topk_indices = torch.empty_like(orig_topk_indices, dtype=torch.int32)
    if orig_token_expert_indices.dtype != torch.int32:
        token_expert_indices = torch.empty_like(
            orig_token_expert_indices, dtype=torch.int32
        )
    if input_tokens is not None and input_tokens.dtype != torch.int32:
        input_tokens = input_tokens.to(dtype=torch.int32)
    if hash_indices_table is not None and hash_indices_table.dtype != torch.int32:
        hash_indices_table = hash_indices_table.to(dtype=torch.int32)

    ixf_ops.topk_hash_softplus_sqrt(
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        token_expert_indices=token_expert_indices,
        gating_output=gating_output,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        input_tokens=input_tokens,
        hash_indices_table=hash_indices_table,
    )
    if topk_indices is not orig_topk_indices:
        orig_topk_indices.copy_(topk_indices)
    if token_expert_indices is not orig_token_expert_indices:
        orig_token_expert_indices.copy_(token_expert_indices)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    ixf_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        1.0,
        1.0,
    )


def reshape_and_cache_flash_mix(
    key: torch.Tensor,
    value: torch.Tensor,
    key_scale: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
):
    ixf_ops.reshape_and_cache_flash_mix(
        key,
        value,
        key_scale,
        key_cache,
        value_cache,
        key_scale_cache,
        slot_mapping,
        kv_cache_dtype,
    )


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    ixf_ops.vllm_concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


def quant_kv(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax_, _ = torch.max(torch.abs(kv), dim=-1, keepdim=True)
    f_scale = amax_.float() / 127.0
    scales = f_scale.view(kv.shape[:-1])
    kv = kv / f_scale
    kv = torch.clamp(torch.round(kv), -127, 127).to(torch.int8)
    return kv, scales


def concat_and_cache_mla_int8(
    kv_c_int8: torch.Tensor,
    kv_c_scale: torch.Tensor,
    k_pe_int8: torch.Tensor,
    k_pe_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    ixf_ops.vllm_concat_and_cache_mla_int8(
        kv_c_int8,
        kv_c_scale,
        k_pe_int8,
        k_pe_scale,
        kv_cache,
        kv_cache_scale,
        slot_mapping,
        kv_cache_dtype,
        scale,
    )


def gather_cache_int8(
    src_cache: torch.Tensor,
    src_cache_scale: torch.Tensor,
    kv_lora_rank: int,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    ixf_ops.vllm_gather_cache_int8(
        src_cache,
        src_cache_scale,
        kv_lora_rank,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        seq_starts,
    )


def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    ixf_ops.vllm_cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


def indexer_k_cache(
    k: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    ixf_ops.indexer_k_cache(k, kv_cache, slot_mapping)


def dsa_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, num_topk_tokens]
    block_size: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """
    Convert request-local token indices to global KV cache indices.

    For each (token_id, indice_id):
        tok = token_indices[token_id, indice_id]
        if tok < 0 or tok // block_size >= max_num_blocks_per_req:
            output[token_id, indice_id] = -1
        else:
            req = req_id[token_id]
            block_idx = block_table[req, tok // block_size]
            output[token_id, indice_id] = block_idx * block_size + tok % block_size
    """
    return ixf_ops.dsa_convert_req_index_to_global_index(
        req_id, block_table, token_indices, block_size, output
    )


def ref_mqa_logits(
    q: torch.Tensor,  # [num_tokens, n_head, head_dim] - 可能已量化
    k: torch.Tensor,  # [num_blocks, block_size, head_dim] 或展开形式 - 可能已量化
    weights: torch.Tensor,  # [num_tokens, n_head, 1] - 权重
    cu_seqlen_ks: torch.Tensor,  # 序列起始位置
    cu_seqlen_ke: torch.Tensor,  # 序列结束位置
) -> torch.Tensor:
    """
    多查询注意力logits计算的PyTorch等价实现
    """

    M, H, D = q.shape
    N = k.shape[0]
    device = q.device
    # 初始化输出logits [M, N]
    logits = torch.full((M, N), -float("inf"), device=device, dtype=torch.float32)
    for i in range(M):
        seq_start = cu_seqlen_ks[i]
        seq_end = cu_seqlen_ke[i]

        if seq_start >= seq_end:
            continue

        # 当前查询的Q [H, D]
        q_i = q[i]  # [H, D]

        seq_k = k[seq_start:seq_end]  # [seq_len, head_dim]

        # 计算注意力分数 [H, seq_len]
        attention_scores = torch.matmul(q_i, seq_k.T)  # BF16计算
        attention_scores = F.relu(attention_scores)

        # 应用权重 [H, seq_len]
        attention_scores_f32 = attention_scores.float()
        weights_i = weights[i].unsqueeze(1)  # [H, 1]
        weighted_scores = attention_scores_f32 * weights_i  # [H, seq_len]

        # 汇总所有头的logits [seq_len]
        logits_i = torch.sum(weighted_scores, dim=0)  # [seq_len]

        # 将结果填充到输出logits的对应位置
        logits[i, seq_start:seq_end] = logits_i

    return logits


def ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """使用分页KV缓存计算FP8多查询注意力logits的PyTorch实现

    Args:
        q: 查询张量 [B, next_n, H, D]
        kv_cache: 分页KV缓存 [num_blocks, block_size, 1, D]
        weights: 权重张量 [B * next_n, H], dtype=torch.float32
        context_lens: 上下文长度 [B], dtype=int32
        block_tables: 块映射表 [B, max_blocks], dtype=int32
        schedule_metadata: 调度元数据
        max_model_len: 最大序列长度，用于确定输出logits大小

    Returns:
        Logits张量 [B * next_n, max_model_len], dtype=torch.float32
    """

    def reassemble_k_from_paged_cache(
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
        head_dim: int,
        block_size: int,
    ) -> torch.Tensor:
        """从分页缓存中重组K值"""
        num_blocks_needed = (context_len + block_size - 1) // block_size
        valid_blocks = block_table[:num_blocks_needed]
        device = kv_cache.device
        # 初始化输出K序列 [context_len, head_dim]
        k_sequence = torch.zeros(
            context_len, head_dim, device=device, dtype=kv_cache.dtype
        )
        token_offset = 0
        for block_idx in valid_blocks:
            if block_idx < 0:
                break
            # 当前块中的token数量
            tokens_in_block = min(block_size, context_len - token_offset)
            if tokens_in_block <= 0:
                break
            # 从缓存块中提取K值
            block_data = kv_cache[block_idx]  # [block_size, 1, D]

            # 提取K值
            k_sequence[token_offset : token_offset + tokens_in_block] = block_data[
                :tokens_in_block, 0, :head_dim
            ]  # [tokens_in_block, D]
            token_offset += tokens_in_block

        return k_sequence

    def compute_mqa_logits(
        q: torch.Tensor,  # [next_n, H, D]
        k: torch.Tensor,  # [context_len, D]
        weights: torch.Tensor,  # [next_n, H]
        context_len: int,
        max_model_len: int,
    ) -> torch.Tensor:
        """计算多查询注意力logits"""
        next_n, H, D = q.shape
        device = q.device

        # 初始化批次logits [next_n, max_model_len]
        batch_logits = torch.full(
            (next_n, max_model_len), -float("inf"), device=device, dtype=torch.float32
        )

        # 扩展K以匹配多头 [context_len, H, D]
        k_expanded = k.unsqueeze(1).expand(-1, H, -1)  # [context_len, H, D]

        # 转置以便矩阵乘法
        q_transposed = q.transpose(0, 1)  # [H, next_n, D]
        k_transposed = k_expanded.transpose(0, 1)  # [H, context_len, D]
        # 批量计算注意力分数 [H, next_n, context_len]
        attention_scores = torch.bmm(
            q_transposed, k_transposed.transpose(1, 2)
        )  # [H, next_n, context_len]
        attention_scores = F.relu(attention_scores)
        # 应用权重并汇总所有头 [next_n, context_len]
        weights_expanded = weights.transpose(0, 1).unsqueeze(2)  # [H, next_n, 1]
        weighted_scores = (
            attention_scores * weights_expanded
        )  # [H, next_n, context_len]
        logits_per_token = weighted_scores.sum(dim=0)  # [next_n, context_len]

        # 填充到输出logits中
        batch_logits[:, :context_len] = logits_per_token

        return batch_logits

    def clean_logits_tensor(
        logits: torch.Tensor,
        context_lens: torch.Tensor,
        next_n: int,
        max_model_len: int,
    ) -> torch.Tensor:
        """清理logits张量,将超出上下文长度的位置设为负无穷"""
        B = len(context_lens)

        for batch_idx in range(B):
            context_len = context_lens[batch_idx].item()
            if context_len >= max_model_len:
                continue

            # 当前批次在logits中的位置
            batch_start = batch_idx * next_n
            batch_end = (batch_idx + 1) * next_n

            # 将超出上下文长度的位置设为负无穷
            logits[batch_start:batch_end, context_len:] = -float("inf")

        return logits

    B, next_n, H, D = q.shape
    num_blocks, block_size, _, cache_stride = kv_cache.shape
    device = q.device

    # 初始化输出logits [B * next_n, max_model_len]
    logits = torch.full(
        (B * next_n, max_model_len), -float("inf"), device=device, dtype=torch.float32
    )

    # 处理每个批次
    for batch_idx in range(B):
        context_len = context_lens[batch_idx].item()
        if context_len == 0:
            continue

        # 当前批次的查询 [next_n, H, D]
        batch_q = q[batch_idx]  # [next_n, H, D]

        # 当前批次的权重 [next_n, H]
        batch_weights_start = batch_idx * next_n
        batch_weights_end = (batch_idx + 1) * next_n
        batch_weights = weights[batch_weights_start:batch_weights_end]  # [next_n, H]

        # 从分页缓存中重组K值
        batch_k = reassemble_k_from_paged_cache(
            kv_cache, block_tables[batch_idx], context_len, D, block_size
        )  # [context_len, D]
        # 计算多查询注意力logits
        batch_logits = compute_mqa_logits(
            batch_q, batch_k, batch_weights, context_len, max_model_len
        )  # [next_n, max_model_len]

        # 填充到输出logits中
        logits[batch_weights_start:batch_weights_end] = batch_logits

    if clean_logits:
        # 清理logits：将超出上下文长度的位置设为负无穷
        logits = clean_logits_tensor(logits, context_lens, next_n, max_model_len)

    return logits


def sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
):
    return ixf_ops.sparse_flash_attn(q, kv, indices, sm_scale, d_v)
