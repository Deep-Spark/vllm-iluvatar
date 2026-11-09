# SPDX-License-Identifier: Apache-2.0
"""Verify the independent vision and Step attention paths on real hardware."""

from types import SimpleNamespace

import pytest
import torch

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Iluvatar GPU required"
)


# Linear outputs are identity inputs so the test isolates attention wiring.
def _identity_projection(x):
    return x, None


# Gemma's existing ixaite entry owns the per-head normalization in this fixture.
class _GemmaNorm:
    # Weight and epsilon stay fixed across capture and replay.
    def __init__(self, weight):
        self.weight = weight
        self.variance_epsilon = 1e-6

    # Call the library directly instead of relying on the test process's OOT mode.
    def __call__(self, x):
        from vllm_iluvatar.backends.ixaite import ops

        return ops.gemma_rms_norm(x, self.weight, self.variance_epsilon)


# Compute Gemma norm in fp32 independently of the model's implementation.
def _gemma_reference(x, weight):
    y = x.float()
    return (
        y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())
    ).to(x.dtype)


# A changed input after capture checks live normalization and per-head broadcast gates.
def test_step_qk_norm_and_head_gate_graph():
    from vllm_iluvatar.backends.ixaite.models.step3p5 import IluStep3p5Attention

    torch.manual_seed(21)
    with forbid_ixformer_calls():
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        layer = SimpleNamespace(
            qkv_proj=_identity_projection,
            q_size=256,
            kv_size=128,
            head_dim=128,
            num_heads=2,
            q_norm=_GemmaNorm(qw),
            k_norm=_GemmaNorm(kw),
            use_rope=False,
            use_head_wise_attn_gate=True,
            attn=lambda q, k, v: q,
            g_proj=lambda value: (value[:, :2], None),
            o_proj=_identity_projection,
        )
        positions = torch.arange(3, device="cuda")

        # This oracle keeps the gate multiply in fp32 until the final BF16 store.
        def reference():
            q = _gemma_reference(x[:, :256].reshape(3, 2, 128), qw)
            return (
                (q.float() * torch.sigmoid(x[:, :2].float()).unsqueeze(-1))
                .to(x.dtype)
                .flatten(1)
            )

        out = IluStep3p5Attention.forward(layer, positions, x)
        torch.testing.assert_close(out, reference(), rtol=1e-2, atol=1e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = IluStep3p5Attention.forward(layer, positions, x)
        for _ in range(2):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(captured, reference(), rtol=1e-2, atol=1e-2)


# Non-causal vision attention must preserve image boundaries and work in graph replay.
def test_pixtral_vision_attention_graph():
    from vllm_iluvatar.backends.ixaite.attention.flash_attn import (
        configure_flash_attn_provider,
    )
    from vllm_iluvatar.backends.ixaite.models.pixtral import Attention

    torch.manual_seed(23)
    with forbid_ixformer_calls():
        configure_flash_attn_provider()
        x = torch.randn(2, 4, 768, device="cuda", dtype=torch.bfloat16)
        starts = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
        seqinfo = SimpleNamespace(seqstart=starts, max_seqlen=4)
        mask = SimpleNamespace(q_seqinfo=seqinfo, k_seqinfo=seqinfo)
        freqs = torch.ones(4, 64, device="cuda", dtype=torch.complex64)
        layer = SimpleNamespace(
            qkv_proj=_identity_projection,
            o_proj=_identity_projection,
            n_heads=2,
            head_dim=128,
        )

        # Identity RoPE makes the fp32 dense attention equation an independent oracle.
        def reference():
            q, k, v = [
                part.reshape(2, 4, 2, 128).transpose(1, 2).float()
                for part in x.chunk(3, -1)
            ]
            scores = q @ k.transpose(-1, -2) * 128**-0.5
            return (
                (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 4, 256).to(x.dtype)
            )

        out = Attention.forward(layer, x, mask, freqs)
        torch.testing.assert_close(out, reference(), rtol=3e-2, atol=3e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = Attention.forward(layer, x, mask, freqs)
        for _ in range(2):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(captured, reference(), rtol=3e-2, atol=3e-2)
