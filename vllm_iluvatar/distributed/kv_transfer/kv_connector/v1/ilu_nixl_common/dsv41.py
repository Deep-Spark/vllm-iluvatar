# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 topology facts and validation helpers shared across Ilu NIXL connectors."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm.distributed.utils import get_pp_indices
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from ..skip_layers import producer_last_layer
from .handshake_validation import (
    validate_indexer_kv_layout,
    validate_main_kv_format,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_DSV41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")


def _hf_text_config(vllm_config: Any) -> Any:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    hf = getattr(model_config, "hf_text_config", None)
    if hf is None:
        hf = getattr(model_config, "hf_config", None)
    return getattr(hf, "text_config", hf)


def is_dsv41_config(vllm_config: Any) -> bool:
    hf = _hf_text_config(vllm_config)
    model_type = getattr(hf, "model_type", None)
    # Reject MagicMock / non-string sentinels from unit-test doubles.
    return isinstance(model_type, str) and model_type in _DSV41_MODEL_TYPES


def resolve_swa_bounded_replay(vllm_config: Any) -> bool:
    """Mirror ``deepseek_v41.attention._resolve_swa_bounded_replay`` purely."""
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None or not getattr(cache_config, "swa_bounded_replay", True):
        return False
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return False
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return not (
        parallel_config is not None
        and getattr(parallel_config, "prefill_context_parallel_size", 1) > 1
    )


def resolve_quantized_kv_format(name: str):
    """Resolve ``i8g64`` / ``i4g32`` codec across pre/post-merge IxFormer paths."""
    try:
        from ixformer.inference.functions.models.deepseek_v41.kv.common.quantized_kv_format import (
            resolve_quant_format,
        )
    except ModuleNotFoundError:
        from ixformer.inference.functions.models.deepseek_v41.kv.float.quantized_kv_format import (
            resolve_quant_format,
        )
    return resolve_quant_format(name)


def resolve_main_kv_format_name() -> str:
    value = os.environ.get("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none")
    if value == "none":
        return "none"
    return resolve_quantized_kv_format(value).name


def resolve_indexer_kv_layout_name() -> str:
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import indexer_kv_layout

    return indexer_kv_layout()


@dataclass(frozen=True)
class Dsv41Layout:
    num_hidden_layers: int
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_ratios: tuple[int, ...]
    swa_bounded_replay: bool
    producer_layer_end: int | None
    main_kv_format: str = "none"
    indexer_kv_layout: str = "legacy"

    @property
    def kv_layer_end(self) -> int | None:
        """Half-open upper bound ``producer_layer_end + 1`` when truncated."""
        if self.producer_layer_end is None:
            return None
        return min(self.num_hidden_layers, self.producer_layer_end + 1)


def wire_dsv41_layout(
    layout: Dsv41Layout | None,
    *,
    host_staging_producer: bool = False,
    hetero_enabled: bool = False,
) -> Dsv41Layout | None:
    """Return the wire-visible DSv4.1 layout for handshake & format checks."""
    if layout is not None and host_staging_producer and hetero_enabled:
        return replace(
            layout,
            main_kv_format="none",
            indexer_kv_layout="legacy",
        )
    return layout


def resolve_dsv41_layout(vllm_config: VllmConfig | Any) -> Dsv41Layout | None:
    """Derive V4.1 topology facts from ``vllm_config``, or None for other models."""
    if not is_dsv41_config(vllm_config):
        return None
    hf = _hf_text_config(vllm_config)
    kv_sources = tuple(int(x) for x in (getattr(hf, "kv_source_layer_ids", None) or ()))
    index_sources = tuple(
        int(x) for x in (getattr(hf, "index_source_layer_ids", None) or ())
    )
    compress_ratios = tuple(
        int(x) for x in (getattr(hf, "compress_ratios", None) or ())
    )
    num_layers = int(getattr(hf, "num_hidden_layers", len(compress_ratios)))
    swa_replay = resolve_swa_bounded_replay(vllm_config)
    last_layer = producer_last_layer(vllm_config) if swa_replay else None
    return Dsv41Layout(
        num_hidden_layers=num_layers,
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        compress_ratios=compress_ratios,
        swa_bounded_replay=swa_replay,
        producer_layer_end=last_layer,
        main_kv_format=resolve_main_kv_format_name(),
        indexer_kv_layout=resolve_indexer_kv_layout_name(),
    )


def effective_pp_producers(
    consumer_pp_rank: int,
    consumer_pp_split: str,
    producer_pp_split: str,
    *,
    dsv41_layout: Dsv41Layout | None = None,
    kv_layer_end: int | None = None,
) -> int:
    """Count producer PP stages that own at least one transferable layer for ``consumer_pp_rank``.

    Under DeepSeek-V4.1 ``swa_bounded_replay``, every SWA, compressor ring, and
    PP forwarder replica is in ``skip_layers``, so ONLY ``kv_source_layers``
    own transferable KV regions. Excluding source-free producer stages (such as
    ``[3,6)`` in ``3,3,3,3,2,2,2,22``) lets empty producer ranks skip dummy
    NIXL notifications completely while keeping D's expected count exact.
    """
    c_counts = [int(x) for x in consumer_pp_split.split(",")]
    d_start = sum(c_counts[:consumer_pp_rank])
    d_end = d_start + c_counts[consumer_pp_rank]

    eff_layer_end = kv_layer_end
    if eff_layer_end is None and dsv41_layout is not None:
        eff_layer_end = dsv41_layout.kv_layer_end

    if (
        dsv41_layout is not None
        and dsv41_layout.swa_bounded_replay
        and dsv41_layout.kv_source_layers
    ):
        eff_d_end = min(d_end, eff_layer_end) if eff_layer_end is not None else d_end
        if d_start >= eff_d_end:
            return 0
        target_sources = [
            s for s in dsv41_layout.kv_source_layers if d_start <= s < eff_d_end
        ]
        if not target_sources:
            return 0
        p_counts = [int(x) for x in producer_pp_split.split(",")]
        count = 0
        p_start = 0
        for num_layers in p_counts:
            p_end = p_start + num_layers
            eff_p_end = (
                min(p_end, eff_layer_end) if eff_layer_end is not None else p_end
            )
            if p_start < eff_p_end and any(
                p_start <= s < eff_p_end for s in target_sources
            ):
                count += 1
            p_start = p_end
        return count

    if eff_layer_end is not None:
        d_end = min(d_end, eff_layer_end)
    if d_start >= d_end:
        return 0
    p_counts = [int(x) for x in producer_pp_split.split(",")]
    count = 0
    p_start = 0
    for num_layers in p_counts:
        p_end = p_start + num_layers
        eff_p_end = min(p_end, eff_layer_end) if eff_layer_end is not None else p_end
        if p_start < eff_p_end and max(d_start, p_start) < min(d_end, eff_p_end):
            count += 1
        p_start = p_end
    return count


def is_predictably_skipped_dsv41_stage(
    dsv41_layout: Dsv41Layout | None,
) -> bool:
    """Whether empty/inactive stages are excluded from ``effective_pp_producers``."""
    return bool(
        dsv41_layout is not None
        and dsv41_layout.swa_bounded_replay
        and dsv41_layout.kv_source_layers
    )


def legal_decode_cut_layers(layout: Dsv41Layout) -> tuple[int, ...]:
    """Valid stage start layers (rank >= 1) on a V4.1 consumer without forwarders."""
    valid: list[int] = []
    for start in range(1, layout.num_hidden_layers):
        first_compressed = next(
            (
                layer_id
                for layer_id in range(start, layout.num_hidden_layers)
                if layer_id < len(layout.compress_ratios)
                and layout.compress_ratios[layer_id] > 0
            ),
            None,
        )
        if first_compressed is None:
            valid.append(start)
            continue
        eligible = [s for s in layout.kv_source_layers if s <= first_compressed]
        kv_source = max(eligible) if eligible else None
        if kv_source is None or kv_source >= start:
            valid.append(start)
    return tuple(valid)


def _parse_pp_bounds(
    num_layers: int, pp_size: int, pp_split: str | None
) -> list[tuple[int, int]]:
    if pp_split:
        counts = [int(part.strip()) for part in pp_split.split(",") if part.strip()]
        if (
            len(counts) != pp_size
            or any(c <= 0 for c in counts)
            or sum(counts) != num_layers
        ):
            raise ValueError(
                f"Invalid PP partition {pp_split!r} for pp_size={pp_size}, "
                f"num_hidden_layers={num_layers}"
            )
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for count in counts:
            bounds.append((cursor, cursor + count))
            cursor += count
        return bounds
    return [get_pp_indices(num_layers, rank, pp_size) for rank in range(pp_size)]


def validate_decode_pp_split(
    vllm_config: Any,
    layout: Dsv41Layout | None,
    pp_split: str | None = None,
) -> None:
    """Fail fast on D-side PP cuts that break a V4.1 KV-sharing group."""
    if layout is None or not layout.kv_source_layers:
        return
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None or getattr(transfer, "kv_role", None) != "kv_consumer":
        return
    parallel_config = getattr(vllm_config, "parallel_config", None)
    pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1))
    if pp_size <= 1:
        return

    from vllm_iluvatar.models.deepseek_v41.pp_plan import _rank_needs

    effective_split = (
        pp_split if pp_split is not None else os.getenv("VLLM_PP_LAYER_PARTITION")
    )
    bounds = _parse_pp_bounds(layout.num_hidden_layers, pp_size, effective_split)
    bad_ranks: list[tuple[int, int, int, int, int]] = []
    for rank in range(1, pp_size):
        start, end = bounds[rank]
        forwarder, kv_source, _ = _rank_needs(
            start,
            end,
            layout.kv_source_layers,
            layout.index_source_layers,
            layout.compress_ratios,
        )
        if forwarder is not None and kv_source is not None:
            bad_ranks.append((rank, start, end, forwarder, kv_source))

    if bad_ranks:
        legal_cuts = legal_decode_cut_layers(layout)
        detail = ", ".join(
            f"rank {r} [{s}, {e}) strands layer {fwd} from kv_source {src}"
            for r, s, e, fwd, src in bad_ranks
        )
        raise ValueError(
            f"DeepSeek-V4.1 PD decode (pp_size={pp_size}) cannot cut inside a "
            f"KV-sharing group ({detail}). Valid non-zero stage start layers "
            f"for this model are {list(legal_cuts)}; set VLLM_PP_LAYER_PARTITION "
            "so every rank boundary lands on one of those layers."
        )


def validate_wire_format_tags(
    local_layout: Dsv41Layout | None,
    *,
    remote_engine_id: str,
    remote_main_kv_format: str = "none",
    remote_indexer_kv_layout: str = "legacy",
    local_main_kv_format: str | None = None,
    local_indexer_kv_layout: str | None = None,
) -> bool:
    """Validate main KV codec and indexer layout tags; return True if quantized.

    Callers that publish FP8 wire staging should pass the advertised wire tags
    (``none`` / ``legacy``) via the local_* overrides so P's compute codecs
    are not compared against D's native FP8 pages.
    """
    local_main_fmt = (
        local_main_kv_format
        if local_main_kv_format is not None
        else (
            local_layout.main_kv_format
            if local_layout is not None
            else resolve_main_kv_format_name()
        )
    )
    local_idx_layout = (
        local_indexer_kv_layout
        if local_indexer_kv_layout is not None
        else (
            local_layout.indexer_kv_layout
            if local_layout is not None
            else resolve_indexer_kv_layout_name()
        )
    )
    requires_compat = validate_main_kv_format(
        local_main_fmt,
        remote_main_kv_format,
        remote_engine_id=remote_engine_id,
    )
    validate_indexer_kv_layout(
        local_idx_layout,
        remote_indexer_kv_layout,
        remote_engine_id=remote_engine_id,
    )
    return requires_compat


def validate_dsv41_peer_handshake(
    local_layout: Dsv41Layout | None,
    *,
    remote_engine_id: str,
    remote_swa_bounded_replay: bool | None,
    remote_kv_source_layers: tuple[int, ...] | list[int] | None,
    remote_index_source_layers: tuple[int, ...] | list[int] | None,
    remote_compress_ratios: tuple[int, ...] | list[int] | None,
    remote_main_kv_format: str = "none",
    remote_indexer_kv_layout: str = "legacy",
    local_main_kv_format: str | None = None,
    local_indexer_kv_layout: str | None = None,
) -> bool:
    """Validate V4.1 P/D configuration symmetry."""
    requires_compat = validate_wire_format_tags(
        local_layout,
        remote_engine_id=remote_engine_id,
        remote_main_kv_format=remote_main_kv_format,
        remote_indexer_kv_layout=remote_indexer_kv_layout,
        local_main_kv_format=local_main_kv_format,
        local_indexer_kv_layout=local_indexer_kv_layout,
    )

    if local_layout is None:
        if remote_kv_source_layers is not None:
            raise RuntimeError(
                f"Remote engine {remote_engine_id} advertises DeepSeek-V4.1 "
                "metadata, but local model is not DeepSeek-V4.1."
            )
        return requires_compat

    if remote_swa_bounded_replay is None:
        raise RuntimeError(
            f"Remote engine {remote_engine_id} did not advertise "
            "swa_bounded_replay in handshake metadata; upgrade peer IluNixl."
        )
    if bool(remote_swa_bounded_replay) != local_layout.swa_bounded_replay:
        raise RuntimeError(
            f"DeepSeek-V4.1 swa_bounded_replay mismatch with remote engine "
            f"{remote_engine_id}: local={local_layout.swa_bounded_replay}, "
            f"remote={bool(remote_swa_bounded_replay)}"
        )

    if remote_kv_source_layers is not None and (
        tuple(remote_kv_source_layers) != local_layout.kv_source_layers
    ):
        raise RuntimeError(
            f"DeepSeek-V4.1 kv_source_layer_ids mismatch with remote engine "
            f"{remote_engine_id}: local={local_layout.kv_source_layers}, "
            f"remote={tuple(remote_kv_source_layers)}"
        )
    if remote_index_source_layers is not None and (
        tuple(remote_index_source_layers) != local_layout.index_source_layers
    ):
        raise RuntimeError(
            f"DeepSeek-V4.1 index_source_layer_ids mismatch with remote engine "
            f"{remote_engine_id}: local={local_layout.index_source_layers}, "
            f"remote={tuple(remote_index_source_layers)}"
        )
    if remote_compress_ratios is not None and (
        tuple(remote_compress_ratios) != local_layout.compress_ratios
    ):
        raise RuntimeError(
            f"DeepSeek-V4.1 compress_ratios mismatch with remote engine "
            f"{remote_engine_id}: local={local_layout.compress_ratios}, "
            f"remote={tuple(remote_compress_ratios)}"
        )
    return requires_compat


def compute_prefix_replay_alignment(
    vllm_config: Any,
    kv_cache_config: KVCacheConfig,
    *,
    is_consumer: bool,
    has_mamba: bool,
) -> int | None:
    """Return block-multiple alignment (2) for V4.1 D-side CR2 bounded replay."""
    if not is_consumer or has_mamba or not is_dsv41_config(vllm_config):
        return None
    all_specs: set[KVCacheSpec] = set()
    for group in getattr(kv_cache_config, "kv_cache_groups", ()):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            all_specs.update(spec.kv_cache_specs.values())
        else:
            all_specs.add(spec)
    replay_specs = [s for s in all_specs if getattr(s, "prefix_replay_tokens", 0)]
    if not replay_specs:
        return None
    align_enabled = os.getenv("VLLM_ILU_V41_REMOTE_REPLAY_ALIGN2", "1") != "0"
    if (
        align_enabled
        and all(
            isinstance(s, SlidingWindowMLASpec)
            and s.prefix_replay_tokens == 128
            and s.block_size == 64
            for s in replay_specs
        )
        and all(
            isinstance(
                s,
                (SlidingWindowMLASpec, MLAAttentionSpec, CircularBufferSpec),
            )
            and getattr(s, "tokens_per_state", 1) in (1, 2)
            for s in all_specs
        )
    ):
        return 2
    return None
