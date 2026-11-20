import math

import ixformer.inference.functions as ops
import pytest
import torch
from ixformer.cuda import cuda_arch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_iluvatar.models.glm5next.attention import (
    IluIndexer,
    prepare_indexer_projection_weights,
)

pytestmark = pytest.mark.skipif(
    cuda_arch != "ivcore30", reason="CQ projection dispatch"
)


def make_indexer(dtype):
    indexer = IluIndexer.__new__(IluIndexer)
    nn.Module.__init__(indexer)
    indexer.head_dim, indexer.n_head = 128, 32
    indexer.wk_weights_proj = nn.Linear(
        4096, 160, bias=False, device="cuda", dtype=dtype
    )
    indexer.wk_weights_proj.quant_method = UnquantizedLinearMethod()
    indexer.index_kpool_compress_gate = nn.Parameter(
        torch.randn(128, 4096, device="cuda", dtype=dtype) / math.sqrt(4096),
        requires_grad=False,
    )
    indexer.register_buffer("_kw_gate_weight", None, persistent=False)
    root = nn.Module()
    root.indexer = indexer
    return root, indexer


def project(indexer, x):
    k, weight, gate = indexer._project_k_weights_gate(x)
    if gate is None:
        gate = ops.linear(x, indexer.index_kpool_compress_gate)
    return torch.cat((k, weight, gate), dim=-1)


def reference(indexer, x):
    w = torch.cat(
        (
            indexer.wk_weights_proj.weight.detach().cpu(),
            indexer.index_kpool_compress_gate.detach().cpu(),
        )
    )
    return (x.float().cpu() @ w.float().t()).to(x.dtype)


@pytest.mark.parametrize("tokens", [16, 48, 72, 96, 128, 256, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_projection_numerics(tokens, dtype):
    torch.manual_seed(42)
    root, indexer = make_indexer(dtype)
    prepare_indexer_projection_weights(root)
    x = torch.randn(tokens, 4096, device="cuda", dtype=dtype)
    out = project(indexer, x)
    tol = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(out.cpu(), reference(indexer, x), atol=tol, rtol=tol)


@torch.inference_mode()
def test_graph_replay_after_weight_reload():
    root, indexer = make_indexer(torch.bfloat16)
    prepare_indexer_projection_weights(root)
    x = torch.randn(96, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        project(indexer, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = project(indexer, x)
    for replace in (False, True):
        wk = torch.randn_like(indexer.wk_weights_proj.weight) / math.sqrt(4096)
        gate = torch.randn_like(indexer.index_kpool_compress_gate) / math.sqrt(4096)
        if replace:
            indexer.wk_weights_proj.weight = nn.Parameter(wk, requires_grad=False)
            indexer.index_kpool_compress_gate = nn.Parameter(gate, requires_grad=False)
        else:
            indexer.wk_weights_proj.weight.copy_(wk)
            indexer.index_kpool_compress_gate.copy_(gate)
        prepare_indexer_projection_weights(root)
        graph.replay()
        torch.testing.assert_close(
            output.cpu(), reference(indexer, x), atol=0.02, rtol=0.02
        )


@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
@torch.inference_mode()
def test_merged_gate_stride_cache_numerics(decode):
    from ixformer.inference.functions.models.glm5next import (
        kpool_compress_insert,
        kpool_decode_update_and_maybe_write_cache_batched,
    )

    root, indexer = make_indexer(torch.bfloat16)
    prepare_indexer_projection_weights(root)
    x = torch.randn(96, 4096, device="cuda", dtype=torch.bfloat16)
    k, _, gate = indexer._project_k_weights_gate(x)
    k = k.contiguous()
    ape = torch.randn(4, 128, device="cuda")
    if not decode:
        slots = torch.full((96,), -1, device="cuda", dtype=torch.int64)
        slots[3::4] = torch.arange(24, device="cuda")
        actual = torch.zeros(1, 32, 128, device="cuda", dtype=k.dtype)
        expected = torch.zeros_like(actual)
        kpool_compress_insert(k, gate, ape, actual, slots, 4, 128, False)
        kpool_compress_insert(k, gate.contiguous(), ape, expected, slots, 4, 128, False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        positions = (
            torch.arange(6, device="cuda", dtype=torch.int32).expand(16, 6).contiguous()
        )
        requests = torch.arange(16, device="cuda", dtype=torch.int64)[:, None]
        tail_slots = requests * 8 + positions
        slots = requests * 32 + positions // 4
        actual = torch.zeros(16, 32, 128, device="cuda", dtype=k.dtype)
        expected = torch.zeros_like(actual)
        actual_tail = torch.zeros(16, 2, 8, 128, device="cuda", dtype=k.dtype)
        expected_tail = torch.zeros_like(actual_tail)
        kpool_decode_update_and_maybe_write_cache_batched(
            actual,
            actual_tail,
            tail_slots,
            k.view(16, 6, 128),
            gate.view(16, 6, 128),
            ape,
            slots,
            positions,
            4,
            128,
            False,
        )
        kpool_decode_update_and_maybe_write_cache_batched(
            expected,
            expected_tail,
            tail_slots,
            k.view(16, 6, 128),
            gate.contiguous().view(16, 6, 128),
            ape,
            slots,
            positions,
            4,
            128,
            False,
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_tail, expected_tail, atol=0, rtol=0)
