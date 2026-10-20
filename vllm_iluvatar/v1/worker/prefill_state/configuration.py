"""CPU configuration and topology for optional producer-prefix state reuse.

All ranks receive the same engine configuration before they are spawned. The
runtime identity belongs to that configuration, not the launcher environment
or the compilation hash.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SESSION_KEY = "_ilu_prefix_state_session"
_NIXL_CONNECTORS = ("OptNixlConnector", "IluNixlPushConnector")


@dataclass(frozen=True)
class _ChildTransfer:
    """A view of one MultiConnector child, retaining its mutable extra config."""

    kv_connector: str
    kv_role: str
    engine_id: str
    kv_connector_extra_config: dict

    def get_from_extra_config(self, key, default=None):
        return self.kv_connector_extra_config.get(key, default)


def prefix_transfer_config(config: Any) -> Any:
    """Resolve the unique NIXL endpoint without changing sibling connectors.

    The returned child shares its extra-config dictionary with the serialized
    engine configuration. Initializing a session therefore reaches the native
    MultiConnector child as well as the outer model worker.
    """
    transfer = getattr(config, "kv_transfer_config", None)
    if transfer is None:
        return None
    if transfer.kv_connector in _NIXL_CONNECTORS:
        return transfer
    if transfer.kv_connector != "MultiConnector":
        return None
    matches = []

    def visit(children, inherited_engine):
        for child in children:
            name = child.get("kv_connector")
            engine = child.get("engine_id", inherited_engine)
            if name in _NIXL_CONNECTORS:
                matches.append(
                    _ChildTransfer(
                        name,
                        child["kv_role"],
                        engine,
                        child.setdefault("kv_connector_extra_config", {}),
                    )
                )
            elif name == "MultiConnector":
                visit(
                    child.get("kv_connector_extra_config", {}).get("connectors", ()),
                    engine,
                )

    visit(transfer.kv_connector_extra_config.get("connectors", ()), transfer.engine_id)
    if len(matches) > 1:
        raise ValueError("Prefix-state reuse requires a unique NIXL connector")
    return matches[0] if matches else None


def text_config(config: Any) -> Any:
    model = config.model_config
    if model is None:
        return None
    hf = getattr(model, "hf_text_config", None) or model.hf_config
    return getattr(hf, "text_config", hf)


def enabled(config: Any) -> bool:
    transfer = prefix_transfer_config(config)
    if transfer is None:
        return False
    selected = transfer.get_from_extra_config("enable_prefix_state_reuse", True)
    if type(selected) is not bool:
        raise ValueError("enable_prefix_state_reuse must be a JSON boolean")
    if not selected or not getattr(config, "use_v2_model_runner", False):
        return False
    hf = text_config(config)
    if getattr(hf, "model_type", "") not in ("deepseek_v41", "deepseek_v41_text"):
        return False
    if transfer.kv_role not in ("kv_producer", "kv_consumer"):
        return False
    if (
        transfer.kv_role == "kv_consumer"
        and config.parallel_config.pipeline_parallel_size != 1
    ):
        return False
    if transfer.kv_role == "kv_producer":
        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
            producer_last_layer,
        )

        if (
            producer_last_layer(config) is None
            or config.parallel_config.data_parallel_size != 1
        ):
            return False
    if (
        getattr(hf, "sliding_window", 0) != 128
        or getattr(hf, "index_head_dim", 0) != 128
    ):
        return False
    if getattr(config.cache_config, "cache_dtype", "auto") not in ("auto", "bfloat16"):
        return False
    return (
        bool(getattr(hf, "kv_source_layer_ids", ()))
        and int(getattr(hf, "candidate_topk_blocks", 0)) > 0
    )


def initialize_config(config: Any) -> None:
    """Resolve producer cuts and one engine identity before config serialization."""
    transfer = getattr(config, "kv_transfer_config", None)
    hf = text_config(config)
    if (
        transfer is not None
        and transfer.kv_role == "kv_producer"
        and getattr(hf, "model_type", "") in ("deepseek_v41", "deepseek_v41_text")
    ):
        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
            producer_last_layer,
        )

        last = producer_last_layer(config)
        if last is not None:
            parts = layer_partitions(
                hf.num_hidden_layers,
                config.parallel_config.pipeline_parallel_size,
                last,
                os.getenv("VLLM_PP_LAYER_PARTITION"),
            )
            # Upstream layer-count helpers must see the same cut as the model
            # and the connector, including when state reuse is disabled.
            os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, parts))
    if enabled(config):
        transfer = prefix_transfer_config(config)
        transfer.kv_connector_extra_config.setdefault(SESSION_KEY, str(uuid.uuid4()))


def session_paths(config: Any) -> tuple[Path, str]:
    import hashlib

    transfer = prefix_transfer_config(config)
    if transfer is None:
        raise RuntimeError("Prefix-state reuse has no NIXL endpoint")
    nonce = transfer.kv_connector_extra_config.get(SESSION_KEY)
    if not nonce:
        raise RuntimeError(
            "Prefix-state session was not initialized by the Iluvatar platform"
        )
    default = "/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    root = Path(
        transfer.get_from_extra_config(
            "prefix_state_root", f"{default}/vllm_prefix_state"
        )
    )
    identity = hashlib.sha256(f"{transfer.engine_id}:{nonce}".encode()).hexdigest()
    wire = root / identity
    (wire / "source").mkdir(parents=True, exist_ok=True, mode=0o700)
    (wire / "decode").mkdir(exist_ok=True)
    return wire, nonce


def layer_partitions(
    num_layers: int, pp_size: int, last_layer: int | None, explicit: str | None = None
) -> tuple[int, ...]:
    """Resolve one full-depth partition and validate its executed prefix.

    The unused suffix belongs to the final logical stage. This preserves the
    checkpoint's full-depth numbering while giving every producer stage real
    layers. Explicit deployment partitions are never silently rewritten.
    """
    if type(pp_size) is not int or not 0 < pp_size <= num_layers:
        raise ValueError("PP size must be between one and the model depth")
    depth = num_layers if last_layer is None else min(num_layers, last_layer + 1)
    if explicit:
        try:
            parts = tuple(int(x.strip()) for x in explicit.split(","))
        except ValueError as error:
            raise ValueError(
                "VLLM_PP_LAYER_PARTITION must contain integer layer counts"
            ) from error
        if (
            len(parts) != pp_size
            or any(p <= 0 for p in parts)
            or sum(parts) != num_layers
        ):
            raise ValueError(
                f"PP={pp_size} requires {pp_size} positive partition counts summing to {num_layers}; got {parts}"
            )
    else:
        if pp_size > depth:
            raise ValueError(
                f"Cannot distribute {depth} executed producer layers over {pp_size} nonempty PP stages"
            )
        quotient, remainder = divmod(depth, pp_size)
        values = [quotient + (rank < remainder) for rank in range(pp_size)]
        values[-1] += num_layers - depth
        parts = tuple(values)
    start = 0
    for rank, count in enumerate(parts):
        if start >= depth:
            raise ValueError(
                f"Producer truncation at layer {depth - 1} leaves PP rank {rank} with no layer; adjust VLLM_PP_LAYER_PARTITION"
            )
        start += count
    return parts


def pipeline_bounds(
    num_layers: int, pp_size: int, last_layer: int | None, explicit: str | None = None
) -> tuple[tuple[int, int], ...]:
    parts = layer_partitions(num_layers, pp_size, last_layer, explicit)
    cursor = 0
    bounds = []
    for count in parts:
        bounds.append((cursor, cursor + count))
        cursor += count
    return tuple(bounds)


def config_pipeline_bounds(config: Any) -> tuple[tuple[int, int], ...]:
    """Shared by the model and both OptNixl roles; other models keep vLLM cuts."""
    hf = text_config(config)
    depth = hf.num_hidden_layers
    world = config.parallel_config.pipeline_parallel_size
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
        producer_last_layer,
    )

    last = producer_last_layer(config)
    if last is None:
        from vllm.distributed.utils import get_pp_indices

        return tuple(get_pp_indices(depth, rank, world) for rank in range(world))
    return pipeline_bounds(depth, world, last, os.getenv("VLLM_PP_LAYER_PARTITION"))


@dataclass(frozen=True)
class ProducerTopology:
    tp_size: int
    layer_bounds: tuple[tuple[int, int], ...]
    prefix_depth: int

    def __post_init__(self):
        if type(self.tp_size) is not int or self.tp_size <= 0 or not self.layer_bounds:
            raise ValueError("Invalid producer TP/PP topology")
        cursor = 0
        for start, end in self.layer_bounds:
            if (
                type(start) is not int
                or type(end) is not int
                or start != cursor
                or end <= start
            ):
                raise ValueError(
                    "Producer stages must cover a contiguous, nonempty prefix"
                )
            cursor = end
        if cursor != self.prefix_depth:
            raise ValueError("Producer layer coverage differs from the prefix depth")

    @property
    def pp_size(self) -> int:
        return len(self.layer_bounds)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def boundary_pp(self) -> int:
        return self.pp_size - 1

    def rank(self, pp: int, tp: int) -> int:
        if not 0 <= pp < self.pp_size or not 0 <= tp < self.tp_size:
            raise ValueError("Producer rank outside topology")
        return pp * self.tp_size + tp

    def owner(self, layer: int) -> int:
        for pp, (start, end) in enumerate(self.layer_bounds):
            if start <= layer < end:
                return pp
        raise ValueError("Layer outside producer prefix")

    def to_dict(self) -> dict:
        return {
            "tp_size": self.tp_size,
            "layer_bounds": [list(b) for b in self.layer_bounds],
            "prefix_depth": self.prefix_depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProducerTopology:
        return cls(
            data["tp_size"],
            tuple(tuple(b) for b in data["layer_bounds"]),
            data["prefix_depth"],
        )

    @classmethod
    def from_config(cls, config: Any) -> ProducerTopology:
        depth = max(text_config(config).kv_source_layer_ids) + 1
        bounds = tuple(
            (min(a, depth), min(b, depth)) for a, b in config_pipeline_bounds(config)
        )
        return cls(config.parallel_config.tensor_parallel_size, bounds, depth)


@dataclass(frozen=True)
class PrefixGeometry:
    """Full, TP-replicated boundary/state fields; cache row capacities are runtime data."""

    prefix_depth: int = 21
    window: int = 128
    head_dim: int = 512
    hidden_size: int = 5120
    hc_mult: int = 4
    topk: int = 512
    candidates: int = 2048
    ring_layers: tuple[int, ...] = (2, 8, 14)
    indexer_width: int = 132
    main_kv_format: str = "none"

    def __post_init__(self):
        if any(
            type(v) is not int or v <= 0
            for v in (
                self.prefix_depth,
                self.window,
                self.head_dim,
                self.hidden_size,
                self.hc_mult,
                self.topk,
                self.candidates,
                self.indexer_width,
            )
        ):
            raise ValueError("Invalid prefix geometry")
        if self.window != 128 or len(set(self.ring_layers)) != len(self.ring_layers):
            raise ValueError("Unsupported prefix replay geometry")
        if any(
            type(layer) is not int or not 0 <= layer < self.prefix_depth
            for layer in self.ring_layers
        ):
            raise ValueError("Invalid compressor source layer")
        self.main_tail_fields()

    def main_tail_fields(self) -> dict:
        """Boundary payload and optional scales, without requantization."""
        if self.main_kv_format == "none":
            return {"main_tail": ((1, self.head_dim), "BF16")}
        formats = {"i8g64": (8, 64), "i4g32": (4, 32)}
        if self.main_kv_format not in formats:
            raise ValueError("Unsupported prefix-state main KV format")
        bits, group = formats[self.main_kv_format]
        if self.head_dim % group:
            raise ValueError("Main KV head dimension must contain complete groups")
        return {
            "main_tail": ((1, self.head_dim * bits // 8), "U8"),
            "main_tail_scale": ((1, self.head_dim // group), "FP16"),
        }

    def to_dict(self) -> dict:
        return {**asdict(self), "ring_layers": list(self.ring_layers)}

    @classmethod
    def from_dict(cls, data: dict) -> PrefixGeometry:
        return cls(**{**data, "ring_layers": tuple(data["ring_layers"])})

    @classmethod
    def from_config(cls, config: Any) -> PrefixGeometry:
        hf = text_config(config)
        dtype = os.getenv("VLLM_ILU_DS4_INDEXER_KV_DTYPE", "bf16")
        width = (
            hf.index_head_dim + 4 if dtype in ("int8", "fp8") else hf.index_head_dim * 2
        )
        return cls(
            max(hf.kv_source_layer_ids) + 1,
            hf.sliding_window,
            hf.head_dim,
            hf.hidden_size,
            hf.hc_mult,
            hf.index_topk,
            hf.candidate_topk_blocks,
            tuple(i for i in hf.kv_source_layer_ids if hf.compress_ratios[i] > 1),
            width,
            os.getenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none"),
        )

    def fields(self, n: int, ring_rows: dict[int, int] | None = None) -> dict:
        rows = self.window + n % 2
        ring_rows = ring_rows or {layer: 16 for layer in self.ring_layers}
        if set(ring_rows) != set(self.ring_layers):
            raise ValueError("Incomplete compressor capacities")
        result = {
            ("SWA", layer): ((rows, self.head_dim), "BF16")
            for layer in range(self.prefix_depth)
        }
        result.update(
            {
                ("ring", layer): ((ring_rows[layer], 2 * self.head_dim), "FP32")
                for layer in self.ring_layers
            }
        )
        result.update(
            {
                ("boundary21", name): (shape, dtype)
                for name, shape, dtype in (
                    ("HC", (rows, self.hc_mult, self.hidden_size), "BF16"),
                    ("pre_mix", (rows, self.hc_mult), "FP32"),
                    ("topk20", (rows, self.topk), "I32"),
                    ("candidates20", (rows, self.candidates), "I32"),
                )
            }
        )
        if n % 2:
            result.update(
                {
                    ("indexer_tail", self.prefix_depth - 1): (
                        (self.indexer_width,),
                        "U8",
                    ),
                }
            )
        if n % 2:
            result.update(
                {
                    (kind, self.prefix_depth - 1): layout
                    for kind, layout in self.main_tail_fields().items()
                }
            )
        return result
