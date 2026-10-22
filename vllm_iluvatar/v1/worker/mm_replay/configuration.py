"""Opt-in P-only vision configuration carried by the native KV connector."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from vllm_iluvatar.v1.worker.prefill_state.configuration import prefix_transfer_config

SESSION_KEY = "_ilu_mm_replay_session"


@dataclass(frozen=True)
class ReplayOptions:
    enabled: bool = False
    prefix_reuse: bool = False
    host_cache_bytes: int = 4 * 1024**3

    @classmethod
    def from_config(cls, config):
        transfer = prefix_transfer_config(config)
        if transfer is None:
            return cls()
        enabled = transfer.get_from_extra_config("mm_encoder_on_prefill_only", False)
        reuse = transfer.get_from_extra_config("enable_mm_prefix_state_reuse", False)
        budget = transfer.get_from_extra_config("mm_host_cache_bytes", 4 * 1024**3)
        if type(enabled) is not bool or type(reuse) is not bool:
            raise ValueError("Multimodal replay options must be JSON booleans")
        if type(budget) is not int or budget <= 0:
            raise ValueError("mm_host_cache_bytes must be positive integer bytes")
        if reuse and not enabled:
            raise ValueError("MM prefix reuse requires P-only image encoding")
        if enabled:
            hf = config.model_config.hf_config
            if (
                getattr(hf, "model_type", "")
                not in ("deepseek_v41", "deepseek_v41_text")
                or not config.use_v2_model_runner
            ):
                raise ValueError(
                    "P-only image encoding requires DeepSeek V4.1 model runner V2"
                )
            if transfer.kv_role not in ("kv_producer", "kv_consumer"):
                raise ValueError("P-only image encoding requires an explicit P/D role")
            if (
                transfer.kv_role == "kv_consumer"
                and config.parallel_config.pipeline_parallel_size != 1
            ):
                raise ValueError("Remote image replay currently requires PP1 on D")
        return cls(enabled, reuse, budget)


def consumer_only(config):
    options = ReplayOptions.from_config(config)
    transfer = prefix_transfer_config(config)
    return options.enabled and transfer.kv_role == "kv_consumer"


def initialize_config(config):
    options = ReplayOptions.from_config(config)
    mm_config = getattr(config.model_config, "multimodal_config", None)
    if mm_config is not None:
        # Plugin-owned CPU parser mode; it does not alter admission limits or
        # enter the target graph/weight compilation hash.
        mm_config._ilu_mm_replay = options.enabled
    if not options.enabled:
        return
    transfer = prefix_transfer_config(config)
    transfer.kv_connector_extra_config.setdefault(SESSION_KEY, str(uuid.uuid4()))
    from .identity import model_identity

    digest = transfer.kv_connector_extra_config.setdefault(
        "_ilu_mm_model_identity", model_identity(config.model_config)
    )
    if mm_config is not None:
        mm_config._ilu_mm_model_identity = digest
    if transfer.kv_role == "kv_consumer":
        mm_config = config.model_config.multimodal_config
        if mm_config is None:
            raise ValueError(
                "Remote image replay requires the multimodal model wrapper"
            )
        mm_config.skip_mm_profiling = True


def paths(config):
    transfer = prefix_transfer_config(config)
    nonce = transfer.kv_connector_extra_config.get(SESSION_KEY)
    if not nonce:
        raise RuntimeError("Multimodal replay session was not initialized")
    default = "/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    root = Path(
        transfer.get_from_extra_config("mm_replay_root", default + "/vllm_mm_replay")
    )
    identity = hashlib.sha256(f"{transfer.engine_id}:{nonce}".encode()).hexdigest()
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory, nonce
