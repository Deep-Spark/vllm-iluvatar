"""Native plugin dataflow smoke on caller-owned buffers and synthetic input."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ixformer.inference.functions.models.deepseek_v41.kv.common.quantized_kv_cache import (
    quantize_paged_kv,
)
from ixformer.inference.functions.models.deepseek_v41.kv.common.quantized_kv_format import (
    I4G32,
    I8G64,
    dequantize_reference,
    quantize_reference,
)

from tests.models.deepseek_v41.test_compressed_kv_integration import attention
from tests.models.deepseek_v41.test_compressed_kv_state import cache_config
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
@pytest.mark.parametrize("active_records", [0, 3, 6])
def test_native_pp_bytes_scales_boundary_guards_and_neighbor_canaries(
    fmt, ratio, active_records
):
    from ixformer.inference.backend_registry import registry

    assert registry.resolve("move_paged_records") == "ixformer_native"
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
        (
            torch.arange(6 * width, dtype=torch.int32).view(6, width)
            + 17 * torch.arange(6, dtype=torch.int32)[:, None]
        )
        .remainder(256)
        .to(torch.uint8)
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
    slots_cpu = slots_cpu[:active_records]
    # Positions may have still more capacity than the payload staging buffer.
    positions_cpu = torch.arange(8, dtype=torch.int64)
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


@pytest.mark.parametrize("invalid", ["positions", "rows", "device", "rank"])
def test_pp_record_metadata_rejects_without_writes(invalid):
    fmt = I4G32
    rows = torch.zeros((3, fmt.payload_bytes(512)), dtype=torch.uint8, device="cuda")
    scale_rows = torch.ones(
        (3, fmt.scale_groups(512)), dtype=torch.float16, device="cuda"
    )
    cache = torch.full(
        (1, fmt.rows_per_page(512), fmt.payload_bytes(512)),
        165,
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.full(
        (1, fmt.rows_per_page(512), fmt.scale_groups(512)),
        3,
        dtype=torch.float16,
        device="cuda",
    )
    slots = torch.arange(3, dtype=torch.int64, device="cuda")
    positions = slots.clone()
    if invalid == "positions":
        positions = positions[:2]
    elif invalid == "rows":
        rows, scale_rows = rows[:2], scale_rows[:2]
    elif invalid == "device":
        positions = positions.cpu()
    else:
        positions = positions.view(1, 3)
    before_data, before_scales = cache.clone(), scales.clone()
    with pytest.raises(ValueError, match="PP records|share a device"):
        move_quantized_records(
            rows, scale_rows, cache, scales, slots, positions, 1, format=fmt
        )
    torch.cuda.synchronize()
    assert torch.equal(cache, before_data)
    assert torch.equal(scales, before_scales)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("prefill", [False, True])
def test_actual_compressor_to_quantized_attention_dataflow(fmt, prefill, monkeypatch):
    torch.manual_seed(1014)
    layer = attention(fmt, ratio=1)
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


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize(
    "case",
    ["cr1_real_arena", "cr2_independent", "cr2_disjoint_views", "cr2_real_arena"],
)
def test_real_v2_producer_alias_route_and_changed_input_graph(fmt, case, monkeypatch):
    """Real 40-layer groups, small physical pool, independent numeric oracle.

    State and main groups own different block IDs inside a shared pool. The
    fused entry cannot prove this from tensor byte ranges; two ordered kernels
    must preserve both selected roles and every unselected byte. No model or
    context-capacity admission claim is made by the four-block allocation.
    """
    from ixformer.inference.backend_registry import registry
    from ixformer.inference.functions.models.deepseek_v41.kv.common import (
        quantized_kv_cache as native,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.worker.utils import allocate_kv_cache

    from tests.models.deepseek_v41.test_compressed_kv_v2_planning import (
        config,
        specs_for_rank,
    )
    from vllm_iluvatar.models.deepseek_v41.common.ops import (
        fused_compress_quant_cache as pooling,
    )
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import CacheBudget
    from vllm_iluvatar.models.deepseek_v41.compressor import _overlapping_byte_ranges

    torch.manual_seed(42044)
    assert registry.resolve("compress_norm_rope_quant_store") == "ixformer_native"
    assert registry.resolve("rope_quant_store") == "ixformer_native"
    ratio = 1 if case == "cr1_real_arena" else 2
    cfg = config()
    specs, context = specs_for_rank(cfg, fmt, 0, 40)
    budget = CacheBudget.create(8 * 1024**3, fmt)
    full = kv_cache_utils.get_kv_cache_configs(cfg, [specs], [budget.payload_bytes])[0]
    # Keep the real planner's full-layer grouping. Construct only four physical
    # blocks via its normal per-group helper, rather than bypassing service
    # admission with an undersized num_gpu_blocks_override.
    cfg.cache_config.num_gpu_blocks_override = 4
    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        cfg, full.kv_cache_groups, budget.payload_bytes
    )
    views = allocate_kv_cache(
        cache_config,
        torch.device("cuda"),
        cfg.cache_config.get_resolved_kv_cache_layout(),
    )
    key = "model.layers.20.attn" if ratio == 1 else "model.layers.2.attn"
    ring_key = key + ".compressor.state_cache"
    layer = context[key]
    cache = views[key].squeeze(1)
    ring = None if ratio == 1 else views[ring_key].squeeze(1)
    if case == "cr2_independent":
        cache, ring = cache.clone(), ring.clone()
    elif case == "cr2_disjoint_views":
        cache_bytes, ring_bytes = cache.numel(), ring.numel() * ring.element_size()
        arena = torch.empty(
            cache_bytes + 32 + ring_bytes, dtype=torch.uint8, device="cuda"
        )
        cache = arena[:cache_bytes].view_as(cache)
        ring = arena[cache_bytes + 32 :].view(torch.float32).view_as(ring)
        assert cache.untyped_storage()._cdata == ring.untyped_storage()._cdata
    layer.kv_cache = cache
    scale_state = CompressedKVState(
        plan_scale_layout(cache_config, fmt), fmt, "cuda", context
    )
    scales = layer.kv_cache_scales
    scales.fill_(3)
    if ring is not None:
        ring.normal_(mean=0.0, std=0.4)
    else:
        cache.fill_(0xA5)
    shared = case == "cr2_real_arena"
    if ring is not None:
        assert _overlapping_byte_ranges(ring, cache) is shared
    if shared:
        assert ring.data_ptr() == cache.data_ptr()
        assert ring.numel() * 4 == cache.numel() == 4 * 65536
    compressor = IluDeepseekCompressor.__new__(IluDeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.k_cache_prefix, compressor._static_forward_context = key, context
    compressor.head_dim, compressor.compress_ratio = 512, ratio
    compressor.rms_norm_eps = 1e-6
    compressor.norm = SimpleNamespace(
        weight=(1 + 0.1 * torch.randn(512, device="cuda")).to(torch.bfloat16)
    )
    compressor._latent = torch.full((8, 512), -99, dtype=torch.bfloat16, device="cuda")
    compressor.state_cache = (
        None if ring is None else SimpleNamespace(prefix=ring_key, kv_cache=ring)
    )
    count, state_page, main_page = 4, 1, 2
    positions = torch.arange(1, 5, dtype=torch.int64, device="cuda")
    starts = torch.tensor([0, count], dtype=torch.int32, device="cuda")
    requests = torch.zeros(count, dtype=torch.int32, device="cuda")
    slots = main_page * fmt.rows_per_page(512) + positions // ratio
    state_slots = (
        slots
        if ring is None
        else state_page * ring.shape[1] + positions % ring.shape[1]
    )
    metadata = {key: SimpleNamespace(slot_mapping=slots, compressed_kv_format=fmt)}
    if ring is not None:
        metadata[ring_key] = SimpleNamespace(
            capacity=ring.shape[1],
            slot_mapping=state_slots,
            query_start_loc=starts,
            token_to_req_indices=requests,
        )
    monkeypatch.setattr(
        sys.modules[IluDeepseekCompressor.__module__],
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    angles = (
        torch.arange(64, device="cuda").float()[:, None]
        * torch.linspace(0.005, 0.12, 32, device="cuda")[None]
    )
    table = torch.cat((angles.cos(), angles.sin()), 1)
    rotary = SimpleNamespace(cos_sin_cache=table)
    raw = torch.randn(count, 512 * ratio, device="cuda")
    pp_rows = torch.full(
        (8, fmt.payload_bytes(512)), 0xA5, dtype=torch.uint8, device="cuda"
    )
    pp_scales = torch.full(
        (8, fmt.scale_groups(512)), -123, dtype=torch.float16, device="cuda"
    )
    initial_cache = cache.clone()
    initial_scales = scales.clone()
    latent_pointer, workspace_pointer = (
        compressor._latent.data_ptr(),
        scale_state.workspace.storage.data_ptr(),
    )
    calls = dict(fused=0, pool=0, rope=0)

    def counted(name, function):
        def run(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return run

    monkeypatch.setattr(
        native,
        "compress_norm_rope_quant_store",
        counted("fused", native.compress_norm_rope_quant_store),
    )
    monkeypatch.setattr(
        native, "rope_quant_store", counted("rope", native.rope_quant_store)
    )
    monkeypatch.setattr(
        pooling.ixfops,
        "compress_pool_norm",
        counted("pool", pooling.ixfops.compress_pool_norm),
    )
    producer_ref = registry.get_reference_impl("compress_norm_rope_quant_store")
    rope_ref = registry.get_reference_impl("rope_quant_store")

    def reference_pool():
        gold_ring = None if ring is None else ring.clone()
        gold_latent = torch.zeros((count, 512), dtype=torch.bfloat16, device="cuda")
        producer_ref(
            raw,
            positions,
            gold_ring,
            state_slots,
            None if ring is None else starts,
            None if ring is None else requests,
            compressor.norm.weight,
            compressor.rms_norm_eps,
            gold_latent,
            table,
            cache.clone(),
            scales.clone(),
            slots,
            ratio,
            format=fmt,
        )
        return gold_ring, gold_latent

    def check(gold_ring, gold_latent, latent):
        valid = (positions >= 0) & (positions % ratio == ratio - 1)
        torch.testing.assert_close(
            latent[valid], gold_latent[valid], atol=1e-2, rtol=1e-2
        )
        assert torch.count_nonzero(latent[~valid]) == 0
        assert (compressor._latent[count:] == -99).all()
        assert latent.data_ptr() == latent_pointer
        assert scale_state.workspace.storage.data_ptr() == workspace_pointer
        encoded = initial_cache.clone()
        encoded_scales = initial_scales.clone()
        expected_pp = torch.full_like(pp_rows, 0xA5)
        expected_pp_scales = torch.full_like(pp_scales, -123)
        # As in the existing producer tests: norm error is isolated above;
        # codec/RoPE compare the exact same BF16 latent against the oracle.
        rope_ref(
            latent,
            positions,
            table,
            encoded,
            encoded_scales,
            slots,
            ratio,
            format=fmt,
            row_out=expected_pp[:count],
            scale_out=expected_pp_scales[:count],
        )
        if shared:
            expected_bytes = initial_cache.view(torch.uint8).flatten().clone()
            expected_bytes.view(torch.float32).view_as(ring)[state_page].copy_(
                gold_ring[state_page]
            )
            expected_bytes.view_as(cache)[main_page].copy_(encoded[main_page])
            assert torch.equal(cache.view(torch.uint8).flatten(), expected_bytes)
        else:
            assert torch.equal(cache, encoded)
            if ring is not None:
                assert torch.equal(ring.view(torch.uint8), gold_ring.view(torch.uint8))
        assert torch.equal(scales.view(torch.uint8), encoded_scales.view(torch.uint8))
        assert torch.equal(pp_rows, expected_pp)
        assert torch.equal(
            pp_scales.view(torch.uint8), expected_pp_scales.view(torch.uint8)
        )

    try:
        gold_ring, gold_latent = reference_pool()
        before = torch.cuda.memory_allocated()
        latent = compressor(
            raw, positions, rotary_emb=rotary, row_out=pp_rows, scale_out=pp_scales
        )
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        check(gold_ring, gold_latent, latent)
        assert calls == (
            dict(fused=0, pool=1, rope=1) if shared else dict(fused=1, pool=0, rope=0)
        )
        if shared:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), scale_state.workspace.acquire(nested=True):
                latent = compressor(
                    raw,
                    positions,
                    rotary_emb=rotary,
                    row_out=pp_rows,
                    scale_out=pp_scales,
                )
            # New odd start, ring wrap, changed payload, new cache rows, and PAD.
            cache.copy_(initial_cache)
            scales.copy_(initial_scales)
            raw.neg_()
            positions.copy_(torch.tensor([17, 18, 19, -1], device="cuda"))
            state_slots.copy_(
                torch.tensor(
                    [
                        state_page * ring.shape[1] + 1,
                        state_page * ring.shape[1] + 2,
                        state_page * ring.shape[1] + 3,
                        -1,
                    ],
                    device="cuda",
                )
            )
            slots.copy_(
                torch.tensor(
                    [
                        main_page * fmt.rows_per_page(512) + 8,
                        main_page * fmt.rows_per_page(512) + 9,
                        main_page * fmt.rows_per_page(512) + 9,
                        -1,
                    ],
                    device="cuda",
                )
            )
            starts.copy_(torch.tensor([0, 3], dtype=torch.int32, device="cuda"))
            requests[-1] = 2**31 - 1
            pp_rows.fill_(0xA5)
            pp_scales.fill_(-123)
            gold_ring, gold_latent = reference_pool()
            with scale_state.workspace.acquire():
                graph.replay()
            torch.cuda.synchronize()
            check(gold_ring, gold_latent, latent)
    finally:
        scale_state.release()
