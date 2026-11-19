# SPDX-License-Identifier: Apache-2.0
"""Flash FP8 producer and attention with TP-local heads and DFly-sized indices."""

from types import SimpleNamespace

import ixformer.inference.functions as ops
import pytest
import torch

from tests.conftest import requires_corex30
from vllm_iluvatar.models.hy_v4_flash.attention import FP8_KV_WIDTH, FlashAttention


@requires_corex30
@pytest.mark.parametrize("backend", ["ixinfer", "ixattn"])
@pytest.mark.parametrize(
    "heads,width", [(16, 2048), (32, 2048), (32, 512), (64, 512), (32, 2112)]
)
@torch.inference_mode()
def test_fp8_cache_scale_one(heads, width, backend):
    torch.manual_seed(42)
    page = 64
    rows = 6
    tokens = width + page
    device = "cuda"
    bf16 = torch.bfloat16
    k = torch.randn(tokens, 512, device=device, dtype=bf16)
    kpe = torch.randn(tokens, 64, device=device, dtype=bf16)
    q = torch.randn(rows, heads, 512, device=device, dtype=bf16)
    qpe = torch.randn(rows, heads, 64, device=device, dtype=bf16)
    sink = torch.linspace(-1, 3, heads, device=device)
    scale = torch.ones(1, device=device)
    slots = torch.randperm(tokens, device=device)
    cache = torch.empty(
        tokens // page, page, FP8_KV_WIDTH, device=device, dtype=torch.uint8
    )
    FlashAttention.store_fp8_kv(SimpleNamespace(kv_scale=scale), k, kpe, cache, slots)

    codes = k.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    expected_cache = torch.cat((codes.view(torch.uint8), kpe.view(torch.uint8)), dim=-1)
    flat = cache.flatten(0, 1)
    torch.testing.assert_close(flat[slots], expected_cache, atol=0, rtol=0)

    packed = torch.empty(rows, heads, FP8_KV_WIDTH, device=device, dtype=torch.uint8)
    q_scale = torch.empty(rows, heads, device=device)
    indices = slots[:width].to(torch.int32).repeat(rows, 1).unsqueeze(1)
    lengths = torch.tensor(
        [1, 63, 64, 65, width - 1, 0], device=device, dtype=torch.int32
    )

    def launch():
        ops.glm_dsa_fp8_q_pack(q, qpe, packed, q_scale, backend="ixformer_native")
        return ops.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
            packed.unsqueeze(1),
            cache.unsqueeze(2),
            indices,
            lengths,
            q_scale,
            scale,
            1 / 16,
            backend=backend,
            attn_sink=sink,
        ).squeeze(1)

    def check(output, lens):
        query_codes = packed[..., :512].contiguous().view(torch.float8_e4m3fn)
        assert torch.isfinite(q_scale).all()
        reconstructed = query_codes.float() * q_scale.unsqueeze(-1)
        # The reference sees the same quantized values, testing only the MLA.
        expected = []
        for i, length in enumerate(lens):
            keys = codes[:length].float()
            logits = (
                reconstructed[i] @ keys.T + qpe[i].float() @ kpe[:length].float().T
            ) / 16
            probabilities = torch.cat((logits, sink[:, None]), dim=-1).softmax(-1)
            expected.append(probabilities[:, :-1] @ keys)
        reference = torch.stack(expected)
        # Match the existing IxFormer packed FP8 kernel contract. This budget
        # includes the kernel's reduced-precision accumulation in QK and PV.
        torch.testing.assert_close(output.float(), reference, atol=0.05, rtol=0.05)
        assert (
            torch.nn.functional.cosine_similarity(
                output.float().flatten(), reference.flatten(), dim=0
            )
            > 0.999
        )
        assert output[lens.index(0)].count_nonzero() == 0

    out = launch()
    check(out, lengths.tolist())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = launch()
    q.normal_()
    lengths.copy_(lengths.roll(1))
    graph.replay()
    check(out, lengths.tolist())
