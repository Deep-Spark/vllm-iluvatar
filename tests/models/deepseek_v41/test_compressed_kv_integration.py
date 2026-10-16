"""Plugin call-path checks using explicit native facade substitutes on CPU.

These verify allocation/metadata/routing contracts, not kernel accuracy.
"""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv import cache_group_geometry
from vllm_iluvatar.models.deepseek_v41.compressor import IluDeepseekCompressor
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model
from vllm_iluvatar.models.deepseek_v41.sparse_mla import IluDeepseekV4FlashMLABackend


def attention(fmt, ratio=2, owner=True):
    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 512
    layer.compress_ratio = ratio
    layer.index_source_layer_id = 2 if ratio == 2 else 20
    layer.is_kv_source = owner
    layer.is_kv_forwarder = False
    layer.compressed_kv_format = fmt
    layer.kv_cache_dtype = "bfloat16"
    layer.kv_cache_torch_dtype = torch.bfloat16
    layer.kv_cache = torch.zeros(
        3, fmt.rows_per_page(512), fmt.payload_bytes(512), dtype=torch.uint8
    )
    layer.kv_cache_scales = torch.zeros(
        3, fmt.rows_per_page(512), fmt.scale_groups(512), dtype=torch.float16
    )
    return layer


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("ratio", [1, 2])
def test_main_spec_has_fixed_payload_page_without_changing_swa_dtype(
    fmt, ratio, monkeypatch
):
    layer = attention(fmt, ratio)
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    spec = layer.get_kv_cache_spec(config)
    assert spec.page_size_bytes == 65536
    assert spec.state_content_size_bytes == fmt.payload_bytes(512)
    assert spec.block_size == fmt.rows_per_page(512) * ratio
    assert spec.dtype is torch.uint8
    assert layer.kv_cache_torch_dtype is torch.bfloat16
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", fmt.name)
    assert (
        spec.block_size
        in IluDeepseekV4FlashMLABackend.get_supported_kernel_block_sizes()
    )
    receipt = cache_group_geometry(
        SimpleNamespace(kv_cache_spec=spec, layer_names=["main"])
    )
    assert receipt["compressed_main_kv"]["formats"] == {"main": fmt.name}
    assert receipt["compressed_main_kv"]["scale_dtype"] == "float16"


def test_none_retains_bf16_main_spec_and_receipt(monkeypatch):
    layer = attention(I8G64)
    layer.compressed_kv_format = None
    spec = layer.get_kv_cache_spec(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    )
    assert spec.dtype is torch.bfloat16 and spec.state_content_size_bytes == 1024
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    assert IluDeepseekV4FlashMLABackend.get_supported_kernel_block_sizes() == [64]
    receipt = cache_group_geometry(
        SimpleNamespace(kv_cache_spec=spec, layer_names=["main"])
    )
    assert set(receipt) == {"block_size", "layer_names", "dtype"}


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_shared_consumer_uses_exact_owner_payload_and_scale_views(fmt):
    owner, consumer = attention(fmt), attention(fmt, owner=False)
    consumer.compressed_cache_prefix = "owner"
    consumer._static_forward_context = {"owner": owner}
    assert consumer.get_kv_cache_spec(SimpleNamespace()) is None
    assert consumer._compressed_kv_cache() is owner.kv_cache
    assert consumer._compressed_kv_scales() is owner.kv_cache_scales
    owner.kv_cache_scales = None
    with pytest.raises(ValueError, match="scale plane"):
        consumer._compressed_kv_scales()


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_sparse_reader_reaches_the_selected_native_facade(fmt, prefill, monkeypatch):
    layer = attention(fmt)
    layer.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(3, 64, 512, dtype=torch.bfloat16)
    )
    layer.scale = 512**-0.5
    layer.attn_sink = torch.zeros(2)
    workspace = torch.empty(256, dtype=torch.uint8)

    @contextmanager
    def acquire(*, nested):
        assert nested
        yield workspace

    layer.compressed_kv_workspace = SimpleNamespace(acquire=acquire)
    calls = []

    def reader(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["output"].fill_(9)

    name = "ixformer.inference.functions.deepseek." + (
        "quantized_sparse_mla_prefill" if prefill else "quantized_sparse_mla"
    )
    facade = ModuleType(name)
    setattr(
        facade,
        "quantized_sparse_mla_prefill" if prefill else "quantized_sparse_mla",
        reader,
    )
    monkeypatch.setitem(sys.modules, name, facade)
    q = torch.zeros(2, 2, 512, dtype=torch.bfloat16)
    indices = torch.zeros(2, 1, 4, dtype=torch.int32)
    lengths = torch.full((2,), 4, dtype=torch.int32)
    output = torch.empty_like(q)
    layer._sparse_mla(
        q, indices, lengths, layer.kv_cache, indices, lengths, output, prefill=prefill
    )
    args, kwargs = calls[0]
    assert args[0].shape == (2, 1, 2, 512)
    assert args[1].dtype is torch.bfloat16
    assert args[4] is layer.kv_cache and args[5] is layer.kv_cache_scales
    assert kwargs["workspace"] is workspace and kwargs["format"] is fmt
    assert (output == 9).all()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("capacity", [3, 8])
def test_fused_producer_retains_latent_and_writes_paired_records_once(fmt, capacity, monkeypatch):
    layer = attention(fmt, ratio=1)
    compressor = IluDeepseekCompressor.__new__(IluDeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.k_cache_prefix = "owner"
    compressor._static_forward_context = {"owner": layer}
    compressor.state_cache = None
    compressor.head_dim = 512
    compressor.compress_ratio = 1
    compressor.norm = SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16))
    compressor.rms_norm_eps = 1e-20
    slots = torch.arange(3, dtype=torch.int64)
    metadata = SimpleNamespace(slot_mapping=slots, compressed_kv_format=fmt)
    module = sys.modules[IluDeepseekCompressor.__module__]
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"owner": metadata}),
    )
    calls = []

    def produce(*args, **kwargs):
        calls.append((args, kwargs))
        args[8].fill_(5)
        kwargs["row_out"].fill_(0x3B)
        kwargs["scale_out"].fill_(2)

    name = "ixformer.inference.functions.deepseek.quantized_kv_cache"
    facade = ModuleType(name)
    facade.compress_norm_rope_quant_store = produce
    monkeypatch.setitem(sys.modules, name, facade)
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(8, 64))
    records = torch.zeros(capacity, fmt.payload_bytes(512), dtype=torch.uint8)
    scale_records = torch.zeros(capacity, fmt.scale_groups(512), dtype=torch.float16)
    latent = compressor(
        torch.zeros(3, 512),
        torch.arange(3),
        rotary_emb=rotary,
        row_out=records,
        scale_out=scale_records,
    )
    compressor.insert_cache(latent, torch.arange(3), rotary, row_out=records)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[6] is compressor.norm.weight and args[7] == 1e-20
    assert args[10] is layer.kv_cache and args[11] is layer.kv_cache_scales
    assert args[12] is slots and kwargs["format"] is fmt
    assert latent.dtype is torch.bfloat16 and (latent == 5).all()
    assert kwargs["row_out"].shape[0] == kwargs["scale_out"].shape[0] == 3
    assert (records[:3] == 0x3B).all() and (scale_records[:3] == 2).all()
    assert (records[3:] == 0).all() and (scale_records[3:] == 0).all()


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_pp_receives_and_relays_both_record_planes(fmt):
    model = IluDeepseekV41Model.__new__(IluDeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.config = SimpleNamespace(hidden_size=5120)
    model.pp_plan = SimpleNamespace(
        recvs_kv=True, recvs_topk=False, recvs_candidates=False, forwarder=0
    )
    model.main_kv_record = (fmt.payload_bytes(512), torch.uint8)
    model.indexer_kv_record = (132, torch.uint8)
    model.main_kv_scale_width = fmt.scale_groups(512)
    model.relays_kv = True
    model.main_kv_rows = torch.empty(3, fmt.payload_bytes(512), dtype=torch.uint8)
    model.main_kv_scale_rows = torch.empty(
        3, fmt.scale_groups(512), dtype=torch.float16
    )
    model.indexer_kv_rows = torch.empty(3, 132, dtype=torch.uint8)
    calls = []
    model.layers = [
        SimpleNamespace(
            attn=SimpleNamespace(
                accept_remote_kv=lambda *args, **kwargs: calls.append((args, kwargs))
            )
        )
    ]
    incoming = model.make_empty_intermediate_tensors(
        3, torch.bfloat16, torch.device("cpu")
    )
    assert isinstance(incoming, IntermediateTensors)
    incoming["main_kv"].fill_(0xCD)
    incoming["main_kv_scales"].fill_(3)
    model._accept_remote_kv(incoming, torch.arange(3))
    assert calls[0][1]["main_scale_rows"] is incoming["main_kv_scales"]
    assert torch.equal(model.main_kv_rows, incoming["main_kv"])
    assert torch.equal(model.main_kv_scale_rows, incoming["main_kv_scales"])
