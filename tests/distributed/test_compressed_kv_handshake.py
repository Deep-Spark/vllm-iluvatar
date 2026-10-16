"""Exercise the actual wire decoder without NIXL agents, sockets, or GPUs."""

from dataclasses import asdict
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    NixlConnectorWorker,
)


def agent(format_name):
    # Empty registrations occur on PP ranks too. Format identity cannot depend
    # on a scale pointer happening to be present in this rank's geometry.
    return NixlAgentMetadata(
        engine_id="peer",
        agent_metadata=b"native-agent",
        kv_caches_base_addr=[],
        device_id=0,
        num_blocks=3,
        kv_cache_layout="NIXL",
        block_size=64,
        ssm_sizes=(0, 0),
        attn_backend_name="ILU_FLASHMLA_SPARSE_DSV41",
        physical_blocks_per_logical_kv_block=1,
        physical_block_size=64,
        logical_num_blocks=3,
        layer_ptr_mapping={},
        layer_block_len_mapping={},
        layer_desc_mapping={},
        layer_num_blocks_mapping={},
        layer_group_mapping={},
        layer_kernel_blocks_mapping={},
        layer_block_tokens_mapping={},
        layer_backend_mapping={},
        global_rank=0,
        pp_rank=0,
        tp_rank=0,
        main_kv_format=format_name,
    )


def payload(format_name, *, inner_format=None, remote_hash="compatible"):
    return NixlHandshakePayload(
        compatibility_hash=remote_hash,
        agent_metadata_bytes=msgspec.msgpack.encode(
            agent(format_name if inner_format is None else inner_format)
        ),
        main_kv_format=format_name,
    )


def fetch(local_format, encoded, monkeypatch, *, enforce=False):
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", local_format)
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker.compat_hash = "compatible"
    worker.enforce_handshake_compat = enforce
    module = __import__(NixlConnectorWorker.__module__, fromlist=["zmq_comm2proxy"])
    monkeypatch.setattr(module, "zmq_comm2proxy", lambda *args, **kwargs: encoded)
    return worker._fetch_remote_agent_metadata(
        None,
        engine_id="peer",
        global_remote_rank=0,
        handshake_decoder=msgspec.msgpack.Decoder(NixlHandshakePayload),
        decoder=msgspec.msgpack.Decoder(NixlAgentMetadata),
    )


@pytest.mark.parametrize(
    "local_format,remote_format",
    [
        ("none", "i8g64"),
        ("none", "i4g32"),
        ("i8g64", "none"),
        ("i4g32", "none"),
        ("i8g64", "i4g32"),
        ("i4g32", "i8g64"),
    ],
)
@pytest.mark.parametrize("remote_hash", ["compatible", "wrong"])
def test_either_quantized_peer_requires_explicit_format_match(
    local_format, remote_format, remote_hash, monkeypatch
):
    encoded = msgspec.msgpack.encode(payload(remote_format, remote_hash=remote_hash))
    with pytest.raises(RuntimeError, match="main KV format mismatch"):
        fetch(local_format, encoded, monkeypatch)


@pytest.mark.parametrize("format_name", ["i8g64", "i4g32"])
def test_quantized_hash_check_precedes_agent_decoder_with_legacy_toggle_off(
    format_name, monkeypatch
):
    remote = NixlHandshakePayload(
        compatibility_hash="wrong",
        agent_metadata_bytes=b"not-msgpack",
        main_kv_format=format_name,
    )
    with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
        fetch(format_name, msgspec.msgpack.encode(remote), monkeypatch)


@pytest.mark.parametrize("format_name", ["none", "i8g64", "i4g32"])
def test_matching_format_and_hash_decode_the_explicit_agent_tag(
    format_name, monkeypatch
):
    metadata = fetch(
        format_name, msgspec.msgpack.encode(payload(format_name)), monkeypatch
    )
    assert metadata.main_kv_format == format_name


@pytest.mark.parametrize(
    "local_format,inner_format",
    [("none", "i8g64"), ("i8g64", "none"), ("i4g32", "i8g64")],
)
def test_envelope_and_agent_tags_must_agree(local_format, inner_format, monkeypatch):
    encoded = msgspec.msgpack.encode(payload(local_format, inner_format=inner_format))
    with pytest.raises(RuntimeError, match="format tag disagrees with agent metadata"):
        fetch(local_format, encoded, monkeypatch)


def test_unknown_remote_format_is_rejected_with_legacy_toggle_off(monkeypatch):
    with pytest.raises(RuntimeError, match="Unsupported remote NIXL main KV format"):
        fetch("none", msgspec.msgpack.encode(payload("new-format")), monkeypatch)


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("remote_hash", ["compatible", "wrong"])
def test_legacy_none_to_none_preserves_hash_toggle(enforce, remote_hash, monkeypatch):
    legacy_agent = asdict(agent("none"))
    legacy_agent.pop("main_kv_format")
    legacy_payload = {
        "compatibility_hash": remote_hash,
        "agent_metadata_bytes": msgspec.msgpack.encode(legacy_agent),
    }
    encoded = msgspec.msgpack.encode(legacy_payload)
    if enforce and remote_hash != "compatible":
        with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
            fetch("none", encoded, monkeypatch, enforce=enforce)
    else:
        assert (
            fetch("none", encoded, monkeypatch, enforce=enforce).main_kv_format
            == "none"
        )


@pytest.mark.parametrize("format_name", ["none", "i8g64", "i4g32"])
def test_public_registration_publishes_both_tags_on_empty_pp_rank(
    format_name, monkeypatch
):
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", format_name)
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    module = __import__(NixlConnectorWorker.__module__, fromlist=["NixlTpKVTopology"])
    monkeypatch.setattr(module, "host_kv_enabled", lambda: False)
    monkeypatch.setattr(module, "may_override_kv_cache", lambda caches: caches)
    monkeypatch.setattr(module, "NixlTpKVTopology", lambda **kwargs: None)
    monkeypatch.setattr(
        module, "compute_nixl_compatibility_hash", lambda config: "compatible"
    )
    metadata = agent(format_name)
    for name, value in asdict(metadata).items():
        setattr(worker, name, value)
    worker._has_mamba = False
    worker.use_mla = True
    worker.tp_world_size = 1
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 1)
    worker.vllm_config = SimpleNamespace()
    worker.attn_backend = None
    worker.backend_name = metadata.attn_backend_name
    worker._mamba_ssm_size = (0, 0)
    worker._logical_num_blocks = 3
    worker._skipped_layers = set()
    worker.nixl_wrapper = SimpleNamespace(get_agent_metadata=lambda: b"native-agent")
    received = []

    def register(caches, scales):
        received.append((caches, scales))
        return [], []

    worker._register_kv_views = register
    caches = {"swa": torch.empty(0, dtype=torch.bfloat16)}
    scale_views = {"main": torch.empty(0, dtype=torch.float16)}
    worker.register_kv_caches(caches, scale_views)
    assert received == [(caches, scale_views)]
    wire = worker.xfer_handshake_metadata
    inner = msgspec.msgpack.decode(wire.agent_metadata_bytes, type=NixlAgentMetadata)
    assert wire.main_kv_format == inner.main_kv_format == format_name
