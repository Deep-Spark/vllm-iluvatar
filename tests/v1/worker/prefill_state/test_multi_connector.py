"""E25 resolves one native endpoint inside MultiConnector without a router hook."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from test_lifecycle import config

from vllm_iluvatar.v1.worker.prefill_state.configuration import (
    SESSION_KEY,
    enabled,
    initialize_config,
    prefix_transfer_config,
    session_paths,
)
from vllm_iluvatar.v1.worker.prefill_state.runtime import attach_prefix_state


def multi_config(tmp_path):
    cfg = config(tmp_path, "kv_producer", 1, 8)
    cfg.kv_transfer_config.kv_connector = "MultiConnector"
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "connectors": [
            {
                "kv_connector": "OptNixlConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"prefix_state_root": str(tmp_path)},
            },
            {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"cpu_bytes_to_use": 512 * 1024**3},
            },
        ]
    }
    return cfg


def test_child_and_outer_worker_share_session_after_config_serialization(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    cfg = multi_config(tmp_path)
    assert enabled(cfg)
    initialize_config(cfg)
    transfer = prefix_transfer_config(cfg)
    children = cfg.kv_transfer_config.kv_connector_extra_config["connectors"]
    assert (
        transfer.kv_connector_extra_config is children[0]["kv_connector_extra_config"]
    )
    assert SESSION_KEY in children[0]["kv_connector_extra_config"]
    assert SESSION_KEY not in children[1]["kv_connector_extra_config"]
    child_cfg = deepcopy(cfg)
    child_cfg.kv_transfer_config = prefix_transfer_config(child_cfg)
    assert enabled(child_cfg)
    assert session_paths(cfg) == session_paths(child_cfg)
    assert prefix_transfer_config(cfg).engine_id == cfg.kv_transfer_config.engine_id


def test_child_optout_is_respected_and_multiple_endpoints_are_rejected(tmp_path):
    cfg = multi_config(tmp_path)
    child = cfg.kv_transfer_config.kv_connector_extra_config["connectors"][0]
    child["kv_connector_extra_config"]["enable_prefix_state_reuse"] = False
    assert not enabled(cfg)
    cfg.kv_transfer_config.kv_connector_extra_config["connectors"].append(
        deepcopy(child)
    )
    with pytest.raises(ValueError, match="unique NIXL"):
        enabled(cfg)


def test_attach_selects_only_native_child_and_rejects_ambiguity():
    seen = []
    nixl = NS(attach_prefix_state=seen.append)
    offload = NS()
    state = object()
    attach_prefix_state(NS(sub_connectors=[offload, nixl]), state)
    assert seen == [state]
    with pytest.raises(RuntimeError, match="exactly one"):
        attach_prefix_state(NS(sub_connectors=[nixl, nixl]), state)
