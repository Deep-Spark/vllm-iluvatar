# SPDX-License-Identifier: Apache-2.0
"""Deferred source routing, cross-file pairing and upload session regressions."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.model_executor.layers.quantization.test_ilu_mxfp4_loader import (
    ExpertLayer,
    deliver,
    loader_for,
)
from vllm_iluvatar.model_executor.model_loader.ilu_default_loader import (
    IluDefaultModelLoader,
    Mxfp4LoadSession,
)
from vllm_iluvatar.model_executor.model_loader.ilu_paired_weights import (
    DeferredCheckpointTensor,
    SafetensorsReaders,
    checkpoint_references,
)


class MappedModel(torch.nn.Module):
    """Custom source names and existing routing, without a pairing/mapper API."""

    def __init__(self, count=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ExpertLayer() for _ in range(count)])
        self.loaders = [
            loader_for(layer, "ixinfer_w4a8", "TN") for layer in self.blocks
        ]
        for loader in self.loaders:
            loader._upload_buffer.close()
            loader._upload_buffer = None
        self.ordinary = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.routing = {}
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, tensor in weights:
            if name == "ordinary":
                self.ordinary.data.copy_(tensor.narrow(0, 1, 2))
            elif name in self.routing:
                layer, expert, shard, scale = self.routing[name]
                if scale:
                    tensor = tensor.view(torch.uint8)
                deliver(self.blocks[layer], shard, expert, tensor, scale)
        return {name for name, _ in self.named_parameters()}


def checkpoint(tmp_path, model, *, omit=(), split_scales=False):
    weights = {"ordinary": torch.tensor([1.0, 2.0, 3.0]), "unused": torch.ones(2)}
    scales = {}
    for layer in range(len(model.blocks)):
        for expert in (4, 5):
            for shard in ("w1", "w2", "w3"):
                for scale in (False, True):
                    name = f"source_{len(model.routing):03}"
                    model.routing[name] = (layer, expert, shard, scale)
                    if (layer, expert, shard, scale) in omit:
                        continue
                    shape = model.loaders[layer].source_shape(
                        "w2" if shard == "w2" else "w13", scale
                    )
                    tensor = torch.full(shape, 120 if scale else 0, dtype=torch.uint8)
                    if scale:
                        tensor = tensor.view(torch.float8_e8m0fnu)
                    (scales if scale else weights)[name] = tensor
    first = tmp_path / "a.safetensors"
    save_file(weights, first)
    files = [str(first)]
    groups = [{k: v} for k, v in scales.items()] if split_scales else [scales]
    for i, values in enumerate(groups):
        path = tmp_path / f"b{i:02}.safetensors"
        save_file(values, path)
        files.append(str(path))
    return files


def route(model, files, session):
    model.load_weights(
        (ref.source.name, DeferredCheckpointTensor(ref, session.readers))
        for ref in checkpoint_references(files)
    )


@pytest.mark.parametrize("split_scales", [False, True])
def test_actual_routing_builds_pairs_across_files_and_layers(tmp_path, split_scales):
    model = MappedModel()
    files = checkpoint(tmp_path, model, split_scales=split_scales)
    session = Mxfp4LoadSession(dict.fromkeys(model.loaders, "blocks"))
    session.readers = SafetensorsReaders(capacity=2)
    session.bind()
    try:
        route(model, files, session)
        assert model.calls == 1
        assert session.readers.reads == 1
        assert len(session.pairs) == 12
        assert not session.buffers
        torch.testing.assert_close(model.ordinary, torch.tensor([2.0, 3.0]))
        addresses = []
        for loader in model.loaders:
            convert = loader.convert_shard

            def observe(*args, loader=loader, convert=convert):
                slot = loader._upload_buffer
                addresses.append((slot.host.data_ptr(), slot.gpu.data_ptr()))
                assert len(session.readers.handles) <= 2
                return convert(*args)

            loader.convert_shard = observe
        session.finish()
        assert len(set(addresses)) == 1
        assert session.readers.reads == 25
        for loader in model.loaders:
            weights = loader.finalize()
            assert torch.count_nonzero(weights.w13_weight) == 0
            assert torch.count_nonzero(weights.w2_weight) == 0
    finally:
        session.close()
    assert not session.readers.handles
    assert all(x._load_session is x._upload_buffer is None for x in model.loaders)


@pytest.mark.parametrize("roles", [(True,), (False, True)])
def test_missing_half_or_whole_pair_fails_before_conversion(tmp_path, roles):
    model = MappedModel(1)
    files = checkpoint(tmp_path, model, omit=[(0, 4, "w1", role) for role in roles])
    session = Mxfp4LoadSession({model.loaders[0]: "block"})
    session.bind()
    try:
        route(model, files, session)
        with pytest.raises(ValueError, match="Missing MXFP4 pair"):
            session.finish()
        assert not session.buffers and not model.loaders[0]._loaded
    finally:
        session.close()
    assert model.loaders[0]._failed


def test_duplicate_sources_and_targets(tmp_path):
    model = MappedModel(1)
    files = checkpoint(tmp_path, model)
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        checkpoint_references(files + files[:1])
    session = Mxfp4LoadSession({model.loaders[0]: "block"})
    session.bind()
    try:
        route(model, files, session)
        with pytest.raises(ValueError, match="Duplicate MXFP4 target"):
            route(model, files, session)
    finally:
        session.close()


def test_deferred_views_and_unsupported_expert_data_transform(tmp_path):
    model = MappedModel(1)
    files = checkpoint(tmp_path, model)
    session = Mxfp4LoadSession({model.loaders[0]: "block"})
    session.bind()
    ref = next(r for r in checkpoint_references(files) if r.source.name == "source_000")
    tensor = DeferredCheckpointTensor(ref, session.readers)
    view = tensor.transpose(0, 1).narrow(0, 2, 4).detach()
    assert session.readers.reads == 0
    destination = torch.empty(view.shape, dtype=view.dtype)
    destination.copy_(view)
    assert session.readers.reads == 1 and torch.count_nonzero(destination) == 0
    try:
        with pytest.raises(ValueError, match="unsupported data transformation"):
            deliver(model.blocks[0], "w1", 4, tensor + 1)
        assert not session.buffers
    finally:
        session.close()
    with pytest.raises(RuntimeError, match="closed"):
        destination.copy_(view)


def test_registered_loader_discovers_callbacks_and_releases_session(
    tmp_path, monkeypatch
):
    from vllm.config.load import LoadConfig

    model = MappedModel(1)
    files = checkpoint(tmp_path, model)
    loader = IluDefaultModelLoader(LoadConfig(load_format="auto"))
    monkeypatch.setattr(loader, "_prepare_weights", lambda *a: ("", files, True, ""))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda *a: None)
    loader.load_weights(
        model,
        SimpleNamespace(
            model="checkpoint",
            revision=None,
            quantization="mxfp4",
            multimodal_config=None,
        ),
    )
    assert model.calls == 1 and loader._paired_session is None
    for quant in model.loaders:
        assert quant._load_session is quant._upload_buffer is None
        quant.finalize()


def test_unrelated_models_delegate_to_community(monkeypatch):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    calls = []
    monkeypatch.setattr(
        DefaultModelLoader,
        "load_weights",
        lambda self, model, config: calls.append(model),
    )
    model = torch.nn.Linear(2, 2)
    IluDefaultModelLoader(LoadConfig(load_format="auto")).load_weights(model, None)
    assert calls == [model]


def test_load_failure_releases_session(tmp_path, monkeypatch):
    from vllm.config.load import LoadConfig

    model = MappedModel(1)
    files = checkpoint(tmp_path, model, omit=[(0, 4, "w1", True)])
    loader = IluDefaultModelLoader(LoadConfig(load_format="auto"))
    monkeypatch.setattr(loader, "_prepare_weights", lambda *a: ("", files, True, ""))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda *a: None)
    with pytest.raises(ValueError, match="Missing MXFP4 pair"):
        loader.load_weights(
            model,
            SimpleNamespace(
                model="checkpoint",
                revision=None,
                quantization="mxfp4",
                multimodal_config=None,
            ),
        )
    assert loader._paired_session is None and model.loaders[0]._failed
    assert model.loaders[0]._load_session is model.loaders[0]._upload_buffer is None


def test_dspark_existing_loader_needs_no_pairing_interface(tmp_path, monkeypatch):
    from vllm_iluvatar.models.deepseek_v41 import dspark

    model = MappedModel(1)
    files = checkpoint(tmp_path, model)
    weights, scales = {}, {}
    for ref in checkpoint_references(files):
        if ref.source.name not in model.routing:
            continue
        _, expert, shard, scale = model.routing[ref.source.name]
        name = f"mtp.0.ffn.experts.{expert}.{shard}.{'scale' if scale else 'weight'}"
        reader = SafetensorsReaders()
        try:
            (scales if scale else weights)[name] = ref.read(reader)
        finally:
            reader.close()
    weights["mtp.0.attn.attn_sink"] = torch.tensor([3.0])
    save_file(weights, files[0])
    save_file(scales, files[1])
    mapping = [
        (
            f"experts.routed_experts.{'w2' if shard == 'w2' else 'w13'}_",
            f"experts.{expert}.{shard}.",
            expert,
            shard,
        )
        for expert in (4, 5)
        for shard in ("w1", "w2", "w3")
    ]
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: mapping
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    params = {
        f"model.layers.0.ffn.experts.routed_experts.{name}": p
        for name, p in model.blocks[0].named_parameters()
    }
    sink = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    params["model.layers.0.attn.attn_sink"] = sink
    draft = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=1, dspark_n_routed_experts=6, expert_dtype="fp4"
        ),
        named_parameters=lambda: params.items(),
        pad_shared_expert=False,
        load_own_embed_tokens=False,
    )
    session = Mxfp4LoadSession({model.loaders[0]: "draft"})
    session.bind()
    try:
        loaded = dspark.IluDSparkV41DraftModel.load_weights(
            draft,
            (
                (ref.source.name, DeferredCheckpointTensor(ref, session.readers))
                for ref in checkpoint_references(files)
            ),
        )
        assert loaded == set(params)
        assert sink.item() == 3 and session.readers.reads == 1
        assert len(session.pairs) == 6
        session.finish()
        model.loaders[0].finalize()
    finally:
        session.close()


def test_conversion_failure_releases_allocated_buffers(tmp_path):
    model = MappedModel(1)
    files = checkpoint(tmp_path, model)
    session = Mxfp4LoadSession({model.loaders[0]: "block"})
    session.bind()
    route(model, files, session)

    def fail(*args):
        raise ValueError("conversion failed")

    model.loaders[0].convert_shard = fail
    with pytest.raises(ValueError, match="conversion failed"):
        session.finish()
    buffers = list(session.buffers.values())
    assert buffers
    session.close()
    assert all(b.host is b.gpu is None for b in buffers)
    assert not session.readers.handles
    assert model.loaders[0]._failed
    assert model.loaders[0]._load_session is model.loaders[0]._upload_buffer is None


def test_reader_cleanup_attempts_all_handles_after_close_error():
    from unittest.mock import Mock

    readers = SafetensorsReaders()
    good, bad = Mock(), Mock()
    good.__exit__ = Mock()
    bad.__exit__ = Mock(side_effect=RuntimeError("close failed"))
    readers.handles["good"] = (good, None)
    readers.handles["bad"] = (bad, None)
    with pytest.raises(RuntimeError, match="close failed"):
        readers.close()
    good.__exit__.assert_called_once_with(None, None, None)
    assert readers.closed and not readers.handles


def test_cleanup_failure_invalidates_completed_session():
    from unittest.mock import Mock

    model = MappedModel(1)
    session = Mxfp4LoadSession({model.loaders[0]: "block"})
    session.bind()
    session.complete = True
    session.readers.close = Mock(side_effect=RuntimeError("close failed"))
    with pytest.raises(RuntimeError, match="close failed"):
        session.close()
    assert model.loaders[0]._failed
    assert model.loaders[0]._load_session is model.loaders[0]._upload_buffer is None
