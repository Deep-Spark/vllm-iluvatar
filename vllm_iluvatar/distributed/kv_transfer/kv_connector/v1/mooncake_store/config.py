# SPDX-License-Identifier: Apache-2.0
"""CoreX-adapted Mooncake Store configuration and client setup."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
DEFAULT_MASTER_SERVER_ADDRESS = "127.0.0.1:50051"
DEFAULT_PROTOCOL = "rdma"
DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 0  # 0 B: Direct GPU/DRAM transfer via registered buffers
DEFAULT_TCP_LOCAL_BUFFER_SIZE = 64 * 1024 * 1024  # 64 MiB client staging for TCP + GPU
DEFAULT_OFFLOAD_LOCAL_BUFFER_SIZE = 512 * 1024 * 1024  # 512 MiB when disk offload enabled
DEFAULT_TENANT_ID = "default"
MOONCAKE_SLAB_SIZE_BYTES = 16 * 1024 * 1024  # 16 MiB (facebook::cachelib::Slab::kSize)

SUPPORTED_PROTOCOLS = frozenset({"rdma", "corex", "tcp", "cxl", "nvlink"})
MooncakeMode = Literal["embedded", "standalone-store"]

_SIZE_PATTERN = re.compile(r"^\s*([\d.]+)\s*(tb|tib|gb|gib|mb|mib|kb|kib|b)?\s*$")
_UNIT_MULTIPLIERS: dict[str, int] = {
    "tb": 1024**4,
    "tib": 1024**4,
    "gb": 1024**3,
    "gib": 1024**3,
    "mb": 1024**2,
    "mib": 1024**2,
    "kb": 1024,
    "kib": 1024,
    "b": 1,
}


def parse_byte_size(value: Any, *, field_name: str = "size") -> int:
    """Parse byte counts from int or unit-suffixed strings (e.g. '16GB', '512MB', '0')."""
    if isinstance(value, bool):
        raise TypeError(f"Unsupported boolean value for {field_name}: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")
        return value
    if not isinstance(value, str):
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Unsupported type for {field_name}: {type(value).__name__}"
            ) from exc
        if parsed < 0:
            raise ValueError(f"{field_name} must be >= 0, got {parsed}")
        return parsed

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    match = _SIZE_PATTERN.match(cleaned)
    if not match:
        raise ValueError(f"Invalid byte size format for {field_name}: {value!r}")

    number_str = match.group(1)
    unit = match.group(2) or "b"
    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value {number_str!r} for {field_name}: {value!r}"
        ) from exc
    if numeric_value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value!r}")
    return int(numeric_value * _UNIT_MULTIPLIERS[unit])


def normalize_tenant_id(value: Any) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise TypeError(
            f"tenant_id must be a string or null, got {type(value).__name__}: {value!r}"
        )
    cleaned = value.strip()
    return cleaned if cleaned else DEFAULT_TENANT_ID


def normalize_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off", ""):
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def get_requester_local_hostname(
    local_ip: str,
    extra_config: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the local hostname/IP advertised to Mooncake TransferEngine."""
    if extra_config is not None:
        explicit = normalize_optional_str(extra_config.get("local_hostname"))
        if explicit is not None:
            return explicit
    for env_key in ("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "MOONCAKE_LOCAL_HOSTNAME"):
        override = normalize_optional_str(os.getenv(env_key))
        if override is not None:
            return override
    return local_ip


def get_configured_preferred_segment(
    extra_config: Mapping[str, Any] | None,
    store_config: IluMooncakeStoreConfig | None = None,
) -> str | None:
    """Resolve optional preferred_segment string override."""
    if extra_config is not None and "preferred_segment" in extra_config:
        raw = extra_config.get("preferred_segment")
        if raw is not None and not isinstance(raw, bool):
            normalized = normalize_optional_str(raw)
            if normalized is None:
                raise ValueError(
                    "Mooncake preferred_segment override must be a non-empty string"
                )
            return normalized

    env_val = normalize_optional_str(os.getenv("MOONCAKE_PREFERRED_SEGMENT"))
    if env_val is not None:
        return env_val

    if store_config is not None and isinstance(store_config.preferred_segment, str):
        return normalize_optional_str(store_config.preferred_segment)
    return None


@dataclass
class IluMooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore on Iluvatar CoreX.

    Supports three layered configuration sources (highest precedence last):
      1. JSON file pointed to by ``MOONCAKE_CONFIG_PATH`` (or ``config_path``
         in ``kv_connector_extra_config``).
      2. Inline dictionary in ``kv_connector_extra_config["mooncake_config"]``
         or top-level Mooncake keys in ``kv_connector_extra_config``.
      3. ``MOONCAKE_*`` environment variables (``MOONCAKE_MASTER``,
         ``MOONCAKE_METADATA_SERVER``, ``MOONCAKE_PROTOCOL``,
         ``MOONCAKE_DEVICE_NAME``, ``MOONCAKE_GLOBAL_SEGMENT_SIZE``,
         ``MOONCAKE_LOCAL_BUFFER_SIZE``, ``MOONCAKE_PREFER_ALLOC_IN_SAME_NODE``).
    """

    metadata_server: str = DEFAULT_METADATA_SERVER
    master_server_address: str = DEFAULT_MASTER_SERVER_ADDRESS
    protocol: str = DEFAULT_PROTOCOL
    device_name: str = ""
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    prefer_alloc_in_same_node: bool = True
    preferred_segment: str | bool | None = None
    enable_offload: bool = False
    ssd_offload_path: str = ""
    tenant_id: str = DEFAULT_TENANT_ID

    def __post_init__(self) -> None:
        self.protocol = self.protocol.strip().lower()
        if not self.protocol:
            raise ValueError("Mooncake protocol cannot be empty.")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            logger.warning(
                "Using non-standard Mooncake protocol %r; supported CoreX "
                "protocols are rdma, corex, tcp.",
                self.protocol,
            )
        elif self.protocol == "corex":
            logger.warning(
                "MooncakeDistributedStore pools Host DRAM segments across "
                "processes/nodes, while CoreX GPU-IPC ('corex') cannot address "
                "cross-process Host DRAM directly. Prefer protocol='rdma' "
                "(RoCE + itr_peer_mem_drv) or 'tcp' for 2P/xP DRAM offloading."
            )
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"Unknown Mooncake mode: {self.mode!r}")
        if self.global_segment_size < 0:
            raise ValueError(
                f"global_segment_size must be >= 0, got {self.global_segment_size}"
            )
        if self.local_buffer_size < 0:
            raise ValueError(
                f"local_buffer_size must be >= 0, got {self.local_buffer_size}"
            )
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError(
                "embedded mode requires global_segment_size > 0 "
                "(use mode='standalone-store' for zero-segment remote clients)"
            )
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError(
                "standalone-store mode requires global_segment_size == 0"
            )
        if not self.master_server_address or not self.master_server_address.strip():
            raise ValueError("master_server_address must be non-empty.")
        if not self.metadata_server or not self.metadata_server.strip():
            raise ValueError("metadata_server must be non-empty.")
        if self.enable_offload and self.ssd_offload_path and not os.path.isabs(
            self.ssd_offload_path
        ):
            raise ValueError(
                f"ssd_offload_path must be an absolute path, got {self.ssd_offload_path!r}"
            )

    def resolve_rank_segment_size(
        self,
        *,
        participating_ranks: int,
        contribute_memory: bool,
        pool_dram_bytes: int | None = None,
    ) -> int:
        """Compute the DRAM segment bytes this worker rank mounts."""
        if not contribute_memory or self.mode == "standalone-store":
            return 0
        if pool_dram_bytes is not None:
            if pool_dram_bytes <= 0:
                raise ValueError(
                    f"pool_dram_bytes must be > 0, got {pool_dram_bytes}"
                )
            divisor = max(1, participating_ranks)
            per_rank = (
                (pool_dram_bytes // divisor)
                // MOONCAKE_SLAB_SIZE_BYTES
                * MOONCAKE_SLAB_SIZE_BYTES
            )
            if per_rank < MOONCAKE_SLAB_SIZE_BYTES:
                raise ValueError(
                    f"pool_dram_bytes ({pool_dram_bytes}) divided across "
                    f"{divisor} participating rank(s) is smaller than one "
                    f"Mooncake 16 MiB slab ({MOONCAKE_SLAB_SIZE_BYTES} bytes)."
                )
            return per_rank
        return self.global_segment_size

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> IluMooncakeStoreConfig:
        mode_raw = str(raw.get("mode", "embedded")).strip()
        if mode_raw not in ("embedded", "standalone-store"):
            raise ValueError(f"Unknown Mooncake mode: {mode_raw!r}")
        mode: MooncakeMode = mode_raw  # type: ignore[assignment]

        enable_offload = parse_bool(
            raw.get("enable_offload", raw.get("enable_ssd_offload", False)),
            default=False,
        )
        protocol_norm = str(raw.get("protocol") or DEFAULT_PROTOCOL).strip().lower()
        default_global = 0 if mode == "standalone-store" else DEFAULT_GLOBAL_SEGMENT_SIZE
        if enable_offload:
            default_local = DEFAULT_OFFLOAD_LOCAL_BUFFER_SIZE
        elif protocol_norm == "tcp":
            default_local = DEFAULT_TCP_LOCAL_BUFFER_SIZE
        else:
            default_local = DEFAULT_LOCAL_BUFFER_SIZE

        global_seg = parse_byte_size(
            raw.get("global_segment_size", default_global),
            field_name="global_segment_size",
        )
        local_buf = parse_byte_size(
            raw.get("local_buffer_size", default_local),
            field_name="local_buffer_size",
        )

        pref_seg = raw.get("preferred_segment")
        if isinstance(pref_seg, str):
            pref_seg = normalize_optional_str(pref_seg)

        return cls(
            metadata_server=str(
                raw.get("metadata_server") or DEFAULT_METADATA_SERVER
            ).strip(),
            master_server_address=str(
                raw.get("master_server_address") or DEFAULT_MASTER_SERVER_ADDRESS
            ).strip(),
            protocol=str(raw.get("protocol") or DEFAULT_PROTOCOL).strip().lower(),
            device_name=str(
                raw.get("device_name", raw.get("rdma_devices", "")) or ""
            ).strip(),
            mode=mode,
            global_segment_size=global_seg,
            local_buffer_size=local_buf,
            prefer_alloc_in_same_node=parse_bool(
                raw.get("prefer_alloc_in_same_node", True), default=True
            ),
            preferred_segment=pref_seg,
            enable_offload=enable_offload,
            ssd_offload_path=str(raw.get("ssd_offload_path", "") or "").strip(),
            tenant_id=normalize_tenant_id(raw.get("tenant_id", DEFAULT_TENANT_ID)),
        )

    @classmethod
    def from_file(cls, file_path: str) -> IluMooncakeStoreConfig:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Mooncake config file {file_path!r} must contain a JSON object."
            )
        return cls.load(extra_config={"mooncake_config": data}, ignore_env_path=True)

    @classmethod
    def load(
        cls,
        extra_config: Mapping[str, Any] | None = None,
        *,
        ignore_env_path: bool = False,
    ) -> IluMooncakeStoreConfig:
        """Load configuration from file + extra_config + environment variables."""
        merged: dict[str, Any] = {}
        extra = extra_config or {}

        config_path = normalize_optional_str(
            extra.get("mooncake_config_path") or extra.get("config_path")
        )
        if config_path is None and not ignore_env_path:
            config_path = normalize_optional_str(os.getenv("MOONCAKE_CONFIG_PATH"))

        if config_path is not None:
            with open(config_path, encoding="utf-8") as f:
                file_cfg = json.load(f)
            if not isinstance(file_cfg, dict):
                raise ValueError(
                    f"Mooncake config file {config_path!r} must contain a JSON object."
                )
            merged.update(file_cfg)

        inline_cfg = extra.get("mooncake_config")
        if inline_cfg is not None:
            if not isinstance(inline_cfg, Mapping):
                raise TypeError(
                    "kv_connector_extra_config['mooncake_config'] must be a dict"
                )
            merged.update(inline_cfg)

        for key in (
            "metadata_server",
            "master_server_address",
            "protocol",
            "device_name",
            "rdma_devices",
            "mode",
            "global_segment_size",
            "local_buffer_size",
            "prefer_alloc_in_same_node",
            "preferred_segment",
            "enable_offload",
            "enable_ssd_offload",
            "ssd_offload_path",
            "tenant_id",
        ):
            if key in extra and extra[key] is not None:
                merged[key] = extra[key]

        # Environment variable overrides (highest precedence for per-node / per-P launch)
        env_overrides = {
            "MOONCAKE_MASTER": "master_server_address",
            "MOONCAKE_METADATA_SERVER": "metadata_server",
            "MOONCAKE_PROTOCOL": "protocol",
            "MOONCAKE_DEVICE_NAME": "device_name",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "global_segment_size",
            "MOONCAKE_LOCAL_BUFFER_SIZE": "local_buffer_size",
            "MOONCAKE_MODE": "mode",
            "MOONCAKE_TENANT_ID": "tenant_id",
        }
        for env_key, field_key in env_overrides.items():
            env_val = os.getenv(env_key)
            if env_val is not None and env_val.strip() != "":
                merged[field_key] = env_val.strip()

        prefer_same_env = os.getenv("MOONCAKE_PREFER_ALLOC_IN_SAME_NODE")
        if prefer_same_env is not None and prefer_same_env.strip() != "":
            merged["prefer_alloc_in_same_node"] = parse_bool(prefer_same_env)

        if (
            not merged
            and config_path is None
            and "pool_dram_bytes" not in extra
            and "dram_bytes_to_use" not in extra
            and "cpu_bytes_to_use" not in extra
        ):
            raise ValueError(
                "MooncakeStoreConnector requires configuration via "
                "MOONCAKE_CONFIG_PATH, MOONCAKE_* environment variables, or "
                "kv_connector_extra_config (e.g. 'master_server_address' / "
                "'mooncake_config' / 'pool_dram_bytes' / 'cpu_bytes_to_use')."
            )

        return cls.from_dict(merged)


def resolve_pool_dram_bytes(extra_config: Mapping[str, Any] | None) -> int | None:
    """Resolve optional per-engine DRAM pool budget from extra_config.

    Accepts ``pool_dram_bytes``, ``dram_bytes_to_use``, and
    ``OffloadingConnector``-compatible ``cpu_bytes_to_use``.
    """
    if not extra_config:
        return None
    for key in ("pool_dram_bytes", "dram_bytes_to_use", "cpu_bytes_to_use"):
        if key in extra_config and extra_config[key] is not None:
            return parse_byte_size(extra_config[key], field_name=key)
    env_val = os.getenv("MOONCAKE_POOL_DRAM_BYTES")
    if env_val is not None and env_val.strip():
        return parse_byte_size(env_val, field_name="MOONCAKE_POOL_DRAM_BYTES")
    return None


def _store_setup_accepts_kwarg(store: Any, kwarg: str) -> bool:
    setup = getattr(store, "setup", None)
    if setup is None:
        return False
    try:
        sig = inspect.signature(setup)
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            return True
        return kwarg in sig.parameters
    except (TypeError, ValueError):
        doc = getattr(setup, "__doc__", "") or ""
        return kwarg in doc


def setup_ilu_mooncake_store(
    store: Any,
    config: IluMooncakeStoreConfig,
    local_hostname: str,
    *,
    global_segment_size: int | None = None,
    local_buffer_size: int | None = None,
    rank_ssd_path: str | None = None,
) -> None:
    """Initialize MooncakeDistributedStore for Iluvatar CoreX."""
    effective_global = (
        config.global_segment_size
        if global_segment_size is None
        else global_segment_size
    )
    effective_local = (
        config.local_buffer_size if local_buffer_size is None else local_buffer_size
    )

    setup_kwargs: dict[str, Any] = {}
    if config.tenant_id != DEFAULT_TENANT_ID and _store_setup_accepts_kwarg(
        store, "tenant_id"
    ):
        setup_kwargs["tenant_id"] = config.tenant_id

    if (
        config.enable_offload
        and effective_global > 0
        and _store_setup_accepts_kwarg(store, "enable_ssd_offload")
    ):
        setup_kwargs["enable_ssd_offload"] = True
        if rank_ssd_path or config.ssd_offload_path:
            setup_kwargs["ssd_offload_path"] = rank_ssd_path or config.ssd_offload_path

    ret = store.setup(
        local_hostname,
        config.metadata_server,
        effective_global,
        effective_local,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError(
            f"Initialize MooncakeDistributedStore failed (ret={ret}, "
            f"local_hostname={local_hostname!r}, "
            f"metadata_server={config.metadata_server!r}, "
            f"master={config.master_server_address!r}, "
            f"protocol={config.protocol!r}, device_name={config.device_name!r}, "
            f"global_segment_size={effective_global}, "
            f"local_buffer_size={effective_local})."
        )
