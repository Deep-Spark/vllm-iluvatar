"""Native plugin dataflow smoke on caller-owned buffers and synthetic input."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_cache import quantize_paged_kv
from ixformer.inference.functions.deepseek.quantized_kv_format import (
    I4G32,
    I8G64,
    dequantize_reference,
    quantize_reference,
)
from test_compressed_kv_integration import attention
from test_compressed_kv_state import cache_config

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    CompressedKVState,
    move_quantized_records,
    plan_scale_layout,
)
from vllm_iluvatar.models.deepseek_v41.compressor import IluDeepseekCompressor

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicitly leased GPU and KV_COMPRESSED_KV_GPU_TESTS=1",
)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("ratio", [1, 2])
def test_native_pp_bytes_scales_boundary_guards_and_neighbor_canaries(fmt, ratio):
    pages, rows, width, groups = (
        3,
        fmt.rows_per_page(512),
        fmt.payload_bytes(512),
        fmt.scale_groups(512),
    )
    raw = torch.full(
        (pages * rows * width * 2,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    scale_raw = torch.full(
        (pages * rows * groups * 2,), 3, dtype=torch.float16, device="cuda"
    )
    cache = raw.as_strided((pages, rows, width), (rows * width * 2, width, 1))
    scales = scale_raw.as_strided((pages, rows, groups), (rows * groups * 2, groups, 1))
    cpu_records = (
        torch.arange(6 * width, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .view(6, width)
    )
    cpu_scale_bytes = (
        torch.arange(6 * groups * 2, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .view(6, groups * 2)
    )
    cpu_scale_records = cpu_scale_bytes.view(torch.float16)
    slots_cpu = torch.tensor(
        [rows - 1, rows, rows * 2 + 2, -1, pages * rows + 1, rows * 2],
        dtype=torch.int64,
    )
    positions_cpu = torch.arange(6, dtype=torch.int64)
    expected_data, expected_scale = raw.cpu(), scale_raw.cpu()
    expected_cache = expected_data.as_strided(cache.shape, cache.stride())
    expected_scales = expected_scale.as_strided(scales.shape, scales.stride())
    for index, slot in enumerate(slots_cpu.tolist()):
        if 0 <= slot < pages * rows and (index + 1) % ratio == 0:
            page, row = divmod(slot, rows)
            expected_cache[page, row].copy_(cpu_records[index])
            expected_scales[page, row].copy_(cpu_scale_records[index])
    move_quantized_records(
        cpu_records.to("cuda"),
        cpu_scale_records.to("cuda"),
        cache,
        scales,
        slots_cpu.to("cuda"),
        positions_cpu.to("cuda"),
        ratio,
        format=fmt,
    )
    ready = torch.cuda.Event()
    ready.record()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        copied_data, copied_scales = raw.clone(), scale_raw.clone()
    stream.synchronize()
    assert torch.equal(copied_data.cpu(), expected_data)
    assert torch.equal(
        copied_scales.cpu().view(torch.uint8), expected_scale.view(torch.uint8)
    )


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("prefill", [False, True])
def test_actual_compressor_to_quantized_attention_dataflow(fmt, prefill, monkeypatch):
    torch.manual_seed(1014)
    layer = attention(fmt, ratio=1)
    layer.rotary_emb.cos_sin_cache = layer.cos_sin.cuda()
    config, names = cache_config(fmt)
    # The first view in this actual allocation is the source being exercised.
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    context[names[0]] = layer
    state = CompressedKVState(plan_scale_layout(config, fmt), fmt, "cuda", context)
    layer.kv_cache = layer.kv_cache.to("cuda")
    layer.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.randn(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    )
    layer.scale, layer.attn_sink = 512**-0.5, torch.zeros(32, device="cuda")
    layer.compressed_kv_workspace = state.workspace
    compressor = IluDeepseekCompressor.__new__(IluDeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.k_cache_prefix = names[0]
    compressor._static_forward_context = context
    compressor.state_cache = None
    compressor.head_dim, compressor.compress_ratio = 512, 1
    compressor.norm = SimpleNamespace(
        weight=torch.ones(512, dtype=torch.bfloat16, device="cuda")
    )
    compressor.rms_norm_eps = 1e-20
    slots = torch.tensor(
        [0, fmt.rows_per_page(512) - 1, fmt.rows_per_page(512)],
        dtype=torch.int64,
        device="cuda",
    )
    metadata = SimpleNamespace(slot_mapping=slots, compressed_kv_format=fmt)
    monkeypatch.setattr(
        sys.modules[IluDeepseekCompressor.__module__],
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={names[0]: metadata}),
    )
    positions = torch.arange(3, dtype=torch.int64, device="cuda")
    cos_sin = torch.zeros(8, 64, device="cuda")
    cos_sin[:, :32].fill_(1)
    records = torch.empty(3, fmt.payload_bytes(512), dtype=torch.uint8, device="cuda")
    scale_records = torch.empty(
        3, fmt.scale_groups(512), dtype=torch.float16, device="cuda"
    )
    latent = compressor(
        torch.randn(3, 512, device="cuda"),
        positions,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        row_out=records,
        scale_out=scale_records,
    )
    expected_data, expected_scales = quantize_reference(
        latent.cpu().float().numpy(), format=fmt
    )
    assert np.array_equal(records.cpu().numpy(), expected_data)
    assert np.array_equal(scale_records.cpu().numpy(), expected_scales)
    indices = slots.to(torch.int32).view(1, 1, 3).expand(3, 1, 3).contiguous()
    swa_indices = (
        torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        .view(1, 1, 2)
        .expand(3, 1, 2)
        .contiguous()
    )
    lengths = torch.full((3,), 3, dtype=torch.int32, device="cuda")
    swa_lengths = torch.full((3,), 2, dtype=torch.int32, device="cuda")
    q = torch.randn(3, 32, 512, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(q)
    layer._sparse_mla(
        q,
        swa_indices,
        swa_lengths,
        layer.kv_cache,
        indices,
        lengths,
        output,
        positions=positions,
        prefill=prefill,
    )
    main_rows = torch.from_numpy(
        dequantize_reference(expected_data, expected_scales, head_dim=512, format=fmt)
    ).to(torch.bfloat16)
    kv = torch.cat([layer.swa_cache_layer.kv_cache[0, :2].cpu(), main_rows]).float()
    logits = torch.einsum("bhd,kd->bhk", q.cpu().float(), kv) * layer.scale
    maximum = torch.maximum(logits.amax(-1), torch.zeros_like(logits[..., 0]))
    probabilities = torch.exp(logits - maximum.unsqueeze(-1))
    denominator = probabilities.sum(-1) + torch.exp(-maximum)
    expected = (
        torch.einsum("bhk,kd->bhd", probabilities, kv) / denominator.unsqueeze(-1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        output.cpu().float(),
        expected.float(),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
    state.release()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("ratio", [1, 2])
def test_actual_mixed_forward_route_translates_main_pages_and_replays(
    fmt, ratio, monkeypatch
):
    """Run forward_mqa's actual prefill-as-decode and old-request branches."""
    torch.manual_seed(1014 + ratio)
    monkeypatch.setenv("VLLM_ILU_DS4_PREFILL_OPT", "0")
    monkeypatch.setenv("VLLM_ILU_DS4_PREFILL_AS_DECODE", "1")
    layer = attention(fmt, ratio=ratio)
    layer.rotary_emb.cos_sin_cache = layer.cos_sin.cuda()
    config, names = cache_config(fmt)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    context[names[0]] = layer
    state = CompressedKVState(plan_scale_layout(config, fmt), fmt, "cuda", context)
    layer.kv_cache = layer.kv_cache.to("cuda")
    layer.compressed_cache_prefix = "main"
    layer.swa_cache_layer = SimpleNamespace(
        prefix="swa",
        kv_cache=torch.randn(3, 64, 512, device="cuda", dtype=torch.bfloat16),
    )
    layer.scale, layer.attn_sink = 512**-0.5, torch.zeros(32, device="cuda")
    page_rows = fmt.rows_per_page(512)
    # Logical page0 is physical page2; logical page1 is physical page1.
    slots = torch.tensor(
        [3 * page_rows - 1, page_rows, page_rows + 1], dtype=torch.int64, device="cuda"
    )
    main_rows = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
    logical_topk = torch.tensor(
        [page_rows - 1, -1, page_rows, page_rows + 1, 3 * page_rows, -1],
        dtype=torch.int32,
    )
    layer.topk_indices_buffer = logical_topk.repeat(5, 1).to("cuda")
    positions = torch.arange(
        ratio * (page_rows + 2) - 1,
        ratio * (page_rows + 2) + 4,
        dtype=torch.int64,
        device="cuda",
    )
    valid = torch.tensor([True, False, True, True, True], device="cuda")
    # The invalid token must not dereference this intentionally stale request.
    req_ids = torch.tensor([0, -999, 0, 0, 0], dtype=torch.int32, device="cuda")
    window_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda").repeat(5, 1)
    window_lengths = torch.tensor([2, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    main_metadata = SimpleNamespace(
        block_table=torch.tensor([[2, 1]], dtype=torch.int32, device="cuda"),
        block_size=page_rows * ratio,
        compressed_kv_format=fmt,
    )
    window_metadata = SimpleNamespace(
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=2,
        num_prefill_tokens=3,
        token_to_req_indices=req_ids,
        is_valid_token=valid,
        decode_swa_indices=window_indices[:2],
        decode_swa_lens=window_lengths[:2],
        prefill_swa_indices=window_indices[2:],
        prefill_swa_lens=window_lengths[2:],
    )
    monkeypatch.setattr(
        sys.modules[IluDeepseekV41Attention.__module__],
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"main": main_metadata, "swa": window_metadata}
        ),
    )
    q = torch.randn(5, 32, 512, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(q)
    kv = torch.empty(5, 512, dtype=torch.bfloat16, device="cuda")

    def publish():
        quantize_paged_kv(
            main_rows, slots, layer.kv_cache, layer.kv_cache_scales, format=fmt
        )

    def check():
        payload, scales = quantize_reference(
            main_rows.cpu().float().numpy(), format=fmt
        )
        restored = (
            torch.from_numpy(
                dequantize_reference(payload, scales, head_dim=512, format=fmt)
            )
            .to(torch.bfloat16)
            .float()
        )
        q_cpu, indices, window = (
            q.cpu().float(),
            layer.topk_indices_buffer.cpu(),
            layer.swa_cache_layer.kv_cache[0, :2].cpu().float(),
        )
        valid_cpu, lengths = valid.cpu(), window_lengths.cpu()
        physical_to_row = dict(zip(slots.cpu().tolist(), range(3)))
        expected = torch.empty_like(q_cpu)
        for token in range(5):
            selected = []
            if valid_cpu[token]:
                for logical in indices[token].tolist():
                    if 0 <= logical < 2 * page_rows:
                        physical_page = 2 if logical < page_rows else 1
                        physical = physical_page * page_rows + logical % page_rows
                        selected.append(restored[physical_to_row[physical]])
            records = torch.cat(
                [
                    window[: int(lengths[token])],
                    torch.stack(selected) if selected else restored[:0],
                ]
            )
            if not records.numel():
                expected[token].zero_()
                continue
            logits = q_cpu[token] @ records.T * layer.scale
            maximum = torch.maximum(logits.amax(-1), torch.zeros(32))
            probabilities = torch.exp(logits - maximum[:, None])
            denominator = probabilities.sum(-1) + torch.exp(-maximum)
            expected[token] = probabilities @ records / denominator[:, None]
        torch.testing.assert_close(
            output.cpu().float(),
            expected.to(torch.bfloat16).float(),
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=2 * torch.finfo(torch.bfloat16).eps,
        )

    publish()
    with state.workspace.acquire():
        layer.forward_mqa(q, kv, positions, output)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), state.workspace.acquire(nested=True):
        layer.forward_mqa(q, kv, positions, output)
    q.copy_(torch.randn_like(q))
    main_rows.neg_()
    layer.topk_indices_buffer[:, 0].fill_(page_rows + 1)
    layer.topk_indices_buffer[:, 2].fill_(-1)
    publish()
    with state.workspace.acquire():
        graph.replay()
    check()
    state.release()
