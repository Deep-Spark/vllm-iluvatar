# SPDX-License-Identifier: Apache-2.0
"""Numeric equivalence test for the two Iluvatar FP4/FP8 MoE paths."""

from types import SimpleNamespace

import ixformer.inference.functions as ixff
import pytest
import torch

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_batch_experts import (
    IluMoEBatchedExpertsFp8,
)


def _has_ixmm_ivcore30() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        return is_ivcore30() and _ixmm_backend_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_ixmm_ivcore30(), reason="requires an ivcore30 GPU with ixmm backend"
)
def test_batched_fp8_experts_match_legacy_fp4_fp8_path():
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_tokens, num_experts, topk = 8, 2, 2
    hidden_size = intermediate_size = 256
    max_tokens_per_expert = num_tokens

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = torch.tensor(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(
        num_tokens,
        topk,
        dtype=torch.float32,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    topk_weights *= 3.0

    w1 = torch.randint(
        0,
        256,
        (num_experts, hidden_size // 2, 2 * intermediate_size),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, intermediate_size // 2, hidden_size),
        dtype=torch.uint8,
        device=device,
    )
    w1_scale = torch.full(
        (num_experts, hidden_size // 256, 2 * intermediate_size),
        0.01,
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.full(
        (num_experts, intermediate_size // 256, hidden_size),
        0.01,
        dtype=torch.float32,
        device=device,
    )

    legacy_output = ilu_mxfp4_fused_experts(
        hidden_states=hidden_states,
        w13_weight=w1,
        w13_weight_scale=w1_scale,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        swiglu_limit=7.0,
        backend="ixmm",
        swiglu_variant="step3_swiglustep",
    )

    batched_input = torch.zeros(
        num_experts,
        max_tokens_per_expert,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    expert_num_tokens = torch.zeros(
        num_experts,
        dtype=torch.int32,
        device=device,
    )
    route_slots = []
    for token_idx in range(num_tokens):
        token_slots = []
        for topk_idx in range(topk):
            expert_idx = int(topk_ids[token_idx, topk_idx])
            slot = int(expert_num_tokens[expert_idx])
            batched_input[expert_idx, slot].copy_(hidden_states[token_idx])
            expert_num_tokens[expert_idx] += 1
            token_slots.append((expert_idx, slot))
        route_slots.append(token_slots)

    experts = object.__new__(IluMoEBatchedExpertsFp8)
    experts.out_dtype = torch.bfloat16
    experts.swiglu_limit = 7.0
    experts.quant_config = SimpleNamespace(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    metadata = SimpleNamespace(expert_num_tokens=expert_num_tokens)

    def run_batched(
        inputs: torch.Tensor,
        input_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        output = torch.empty(
            num_experts,
            max_tokens_per_expert,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        experts.apply(
            output=output,
            hidden_states=inputs,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=SimpleNamespace(value="swiglustep"),
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=input_scales,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=metadata,
        )
        combined = torch.zeros_like(hidden_states)
        for token_idx, token_slots in enumerate(route_slots):
            for topk_idx, (expert_idx, slot) in enumerate(token_slots):
                combined[token_idx] += (
                    output[expert_idx, slot] * topk_weights[token_idx, topk_idx]
                )
        return combined

    batched_bf16_output = run_batched(batched_input, None)
    dispatched_fp8, dispatched_scales = ixff.scaled_fp8_quant(
        batched_input.reshape(-1, hidden_size)
    )
    batched_fp8_output = run_batched(
        dispatched_fp8.reshape_as(batched_input),
        dispatched_scales,
    )

    torch.testing.assert_close(batched_fp8_output, batched_bf16_output)
    torch.testing.assert_close(
        batched_fp8_output,
        legacy_output,
        atol=0.25,
        rtol=0.02,
    )
