"""Paired main-cache rows through native prefix geometry and producer export."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.worker.prefill_state.cache_rows import main_boundary_sources
from vllm_iluvatar.v1.worker.prefill_state.configuration import PrefixGeometry
from vllm_iluvatar.v1.worker.prefill_state.producer import ProducerPrefixState
from vllm_iluvatar.v1.worker.prefill_state.source_state import fetch_state, packet_bytes

from test_source_state import PacketTest


class CompressedPacketTest(PacketTest):
    def test_quantized_packets_validate_even_odd_and_1m_geometry(self):
        from vllm_iluvatar.v1.worker.prefill_state import source_state as pub

        outer = self

        class Socket:
            def send(self, value):
                import msgspec

                self.request = msgspec.msgpack.decode(value)

            def recv_multipart(self):
                import msgspec

                reply, raw = pub.serve_state(outer.wire, outer.engine, self.request)
                return [msgspec.msgpack.encode(reply), raw]

        for fmt in ("i8g64", "i4g32"):
            geometry = PrefixGeometry(main_kv_format=fmt)
            for n in (128, 129, 1048575, 1048576):
                with self.subTest(format=fmt, n=n):
                    rid = f"{fmt}-{n}"
                    self.shards(rid, n, main_format=fmt)
                    self.publisher().poll()
                    reply, raw = fetch_state(Socket(), self.engine, rid, 0, n,
                                             self.digest, geometry=geometry)
                    assert reply["status"] == "READY"
                    assert len(raw) == packet_bytes(n, geometry=geometry)
                    fields = reply["metadata"]["fields"]
                    assert sum(f["kind"] == "main_tail_scale" for f in fields) == n % 2
                    with pytest.raises(ValueError, match="geometry"):
                        fetch_state(Socket(), self.engine, rid, 0, n, self.digest,
                                    geometry=replace(geometry, main_kv_format="none"))
                    if n % 2:
                        with pytest.raises(ValueError):
                            pub.validate_fields(
                                [f for f in fields if f["kind"] != "main_tail_scale"],
                                len(raw), n, geometry=geometry)


@pytest.mark.parametrize("fmt,rows", [("none", 64), ("i8g64", 128), ("i4g32", 256)])
@pytest.mark.parametrize("n", [4096, 4097, 1048575, 1048576])
def test_native_producer_exports_paired_boundary_without_touching_page_padding(fmt, rows, n):
    geometry = PrefixGeometry(main_kv_format=fmt)
    owners, views = {}, {}
    dtypes = {"BF16": torch.bfloat16, "U8": torch.uint8, "FP16": torch.float16}
    for kind, (shape, dtype) in geometry.main_tail_fields().items():
        width = shape[-1]
        owner = torch.full((5, rows * width + 64), 9, dtype=dtypes[dtype])
        owners[kind] = owner
        views[kind] = owner.as_strided((5, rows, width), (rows * width + 64, width, 1))
    layer = SimpleNamespace(kv_cache=views["main_tail"],
                            kv_cache_scales=views.get("main_tail_scale"))
    specs = main_boundary_sources(layer, "model.layers.20.attn", 0, geometry)
    producer = ProducerPrefixState.__new__(ProducerPrefixState)
    producer.geometry, producer._src_specs = geometry, specs
    logical, row = divmod(n - 1, rows)
    blocks = [1] * (logical + 1)
    blocks[logical] = 3
    pool = {}
    before = {}
    for spec in specs:
        spec["cache"][3, row].fill_(3)
        before[spec["kind"]] = owners[spec["kind"]].clone()
        pool[spec["name"]] = torch.full((1, spec["cache"].shape[-1]), 7,
                                       dtype=spec["cache"].dtype)
    state = dict(N=n, H=n - n % 2, S=n - n % 2 - 128, blocks=[blocks], pool=pool)
    records = producer._copy_caches(state)
    assert len(records) == len(specs) * (n % 2)
    for spec in specs:
        assert torch.all(pool[spec["name"]] == (3 if n % 2 else 7))
        assert torch.equal(owners[spec["kind"]], before[spec["kind"]])
    if fmt != "none":
        layer.kv_cache_scales = None
        with pytest.raises(ValueError, match="paired"):
            main_boundary_sources(layer, "model.layers.20.attn", 0, geometry)
