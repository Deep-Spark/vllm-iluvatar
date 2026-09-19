# SPDX-License-Identifier: Apache-2.0
"""Runtime Iluvatar quantization groups supplied through additional_config."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
)

from vllm_iluvatar.model_executor.layers.quantization.ilu_quant_registry import (
    IluvatarQuantRegistry,
)

logger = init_logger("vllm." + __name__)

_MAX_FILE_BYTES = 1024 * 1024
_CACHE_ATTR = "_ilu_runtime_quant_config_cache"
_ABSENT = object()


def _strict_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    where: str,
    *,
    required: set[str] = frozenset(),
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{where}: unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where}: missing fields: {sorted(missing)}")


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be an object")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where} must be a non-empty string")
    return value


def _strings(
    value: object, where: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise TypeError(
            f"{where} must be a {'possibly empty ' if allow_empty else ''}list"
        )
    result = tuple(_string(item, f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{where} contains duplicates")
    return result


def _targets(
    value: object, where: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = _strings(value, where, allow_empty=allow_empty)
    for target in result:
        if target.startswith("re:"):
            try:
                re.compile(target[3:])
            except re.error as exc:
                raise ValueError(f"{where}: invalid regex {target!r}") from exc
    return result


def _integer(value: object, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return value


def _optional_mapping(value: object, where: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, where)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class IluAppliesTo:
    model_types: tuple[str, ...]
    quant_methods: tuple[str, ...]


@dataclass(frozen=True)
class IluRequireConfig:
    original_disposition: str
    checkpoint_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class IluWeightEndpoint:
    format: str
    dtypes: tuple[str, ...] = ()
    dtype: str | None = None
    strategy: str | None = None
    extra: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class IluLoadingConfig:
    mode: str
    stage: str


@dataclass(frozen=True)
class IluWeightsConfig:
    source: IluWeightEndpoint
    target: IluWeightEndpoint
    loading: IluLoadingConfig


@dataclass(frozen=True)
class IluActivationConfig:
    num_bits: int
    type: str
    strategy: str
    dynamic: bool
    extra: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class IluValidationConfig:
    expected_selected: int
    expected_transformed: int
    expected_fallback: int


@dataclass(frozen=True)
class IluRuntimeQuantGroup:
    group_id: str
    priority: int
    targets: tuple[str, ...]
    require: IluRequireConfig
    weights: IluWeightsConfig
    input_activations: IluActivationConfig | None
    output_activations: IluActivationConfig | None
    method: str
    options: Mapping[str, object]
    validation: IluValidationConfig | None


@dataclass
class _GroupAudit:
    target_hits: dict[str, int]
    selected: int = 0
    transformed: int = 0
    fallback: int = 0
    summary_logged: bool = False


@dataclass(frozen=True)
class IluRuntimeQuantMethodContext:
    base_quant_config: QuantizationConfig
    original_method: QuantizeMethodBase | None
    original_disposition: str
    group: IluRuntimeQuantGroup
    prefix: str
    layer: torch.nn.Module
    runtime_config: IluRuntimeQuantConfig

    @property
    def options(self) -> Mapping[str, object]:
        return self.group.options

    def record_shape_result(self, *, transformed: bool) -> None:
        self.runtime_config.record_shape_result(
            self.group.group_id, transformed=transformed
        )

    def assert_validation_complete(self) -> None:
        self.runtime_config.assert_all_validation_complete()


@dataclass(frozen=True)
class IluRuntimeQuantConfig:
    schema_version: int
    enabled: bool
    applies_to: IluAppliesTo
    ignore: tuple[str, ...]
    config_groups: tuple[IluRuntimeQuantGroup, ...]
    canonical_digest: str
    source: str
    model_type: str
    quant_method: str
    _audits: dict[str, _GroupAudit] = field(compare=False, repr=False)
    _all_validated: list[bool] = field(compare=False, repr=False)

    def select_group(
        self,
        layer: torch.nn.Module,
        prefix: str,
        fused_mapping: Mapping[str, list[str]],
    ) -> IluRuntimeQuantGroup | None:
        if not self.enabled:
            return None
        if find_matched_target(prefix, layer, self.ignore, fused_mapping) is not None:
            return None

        matches: list[tuple[IluRuntimeQuantGroup, str]] = []
        for group in self.config_groups:
            target = find_matched_target(prefix, layer, group.targets, fused_mapping)
            if target is not None:
                matches.append((group, target))
        if not matches:
            return None

        max_priority = max(group.priority for group, _ in matches)
        winners = [
            (group, target)
            for group, target in matches
            if group.priority == max_priority
        ]
        if len(winners) != 1:
            raise ValueError(
                f"{prefix}: multiple ilu_quant groups have priority "
                f"{max_priority}: {[group.group_id for group, _ in winners]}"
            )

        group, target = winners[0]
        audit = self._audits[group.group_id]
        audit.selected += 1
        audit.target_hits[target] += 1
        validation = group.validation
        if validation and audit.selected > validation.expected_selected:
            raise ValueError(
                f"{group.group_id}: selected {audit.selected} layers, expected "
                f"{validation.expected_selected}"
            )
        return group

    def record_shape_result(self, group_id: str, *, transformed: bool) -> None:
        audit = self._audits[group_id]
        if transformed:
            audit.transformed += 1
        else:
            audit.fallback += 1
        group = next(g for g in self.config_groups if g.group_id == group_id)
        validation = group.validation
        if validation is None:
            return
        if audit.transformed > validation.expected_transformed:
            raise ValueError(f"{group_id}: transformed layer count exceeds expected")
        if audit.fallback > validation.expected_fallback:
            raise ValueError(f"{group_id}: fallback layer count exceeds expected")
        if (
            audit.selected == validation.expected_selected
            and audit.transformed + audit.fallback == audit.selected
        ):
            self.assert_validation_complete(group_id, log_summary=False)

    def assert_validation_complete(
        self, group_id: str, *, log_summary: bool = True
    ) -> None:
        group = next(g for g in self.config_groups if g.group_id == group_id)
        validation = group.validation
        audit = self._audits[group_id]
        if validation is not None:
            actual = (audit.selected, audit.transformed, audit.fallback)
            expected = (
                validation.expected_selected,
                validation.expected_transformed,
                validation.expected_fallback,
            )
            if actual != expected:
                raise RuntimeError(
                    f"{group_id}: runtime quant counts {actual} do not match {expected}"
                )
        missing_targets = [
            target for target, count in audit.target_hits.items() if count == 0
        ]
        if missing_targets:
            raise RuntimeError(
                f"{group_id}: targets matched no layers: {missing_targets}"
            )
        if log_summary and not audit.summary_logged:
            audit.summary_logged = True
            logger.info(
                "ILU runtime quant ready: source=%s schema=%d model_type=%s "
                "base_quant=%s group=%s method=%s loading=%s@%s selected=%d "
                "transformed=%d fallback=%d payload_sha256=%s",
                self.source,
                self.schema_version,
                self.model_type,
                self.quant_method,
                group_id,
                group.method,
                group.weights.loading.mode,
                group.weights.loading.stage,
                audit.selected,
                audit.transformed,
                audit.fallback,
                self.canonical_digest,
            )

    def assert_all_validation_complete(self) -> None:
        if self._all_validated[0]:
            return
        for group in self.config_groups:
            self.assert_validation_complete(group.group_id)
        self._all_validated[0] = True


def _parse_endpoint(value: object, where: str, *, source: bool) -> IluWeightEndpoint:
    data = _mapping(value, where)
    common = {"format", "dtypes"} if source else {"format", "dtype", "strategy"}
    required = {"format", "dtypes"} if source else {"format", "dtype", "strategy"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{where}: missing fields: {sorted(missing)}")
    extra = {key: item for key, item in data.items() if key not in common}
    return IluWeightEndpoint(
        format=_string(data["format"], f"{where}.format"),
        dtypes=_strings(data["dtypes"], f"{where}.dtypes") if source else (),
        dtype=None if source else _string(data["dtype"], f"{where}.dtype"),
        strategy=None if source else _string(data["strategy"], f"{where}.strategy"),
        extra=_freeze(extra),
    )


def _parse_activation(value: object, where: str) -> IluActivationConfig | None:
    data = _optional_mapping(value, where)
    if data is None:
        return None
    required = {"num_bits", "type", "strategy", "dynamic"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{where}: missing fields: {sorted(missing)}")
    if not isinstance(data["dynamic"], bool):
        raise TypeError(f"{where}.dynamic must be a boolean")
    return IluActivationConfig(
        num_bits=_integer(data["num_bits"], f"{where}.num_bits", minimum=1),
        type=_string(data["type"], f"{where}.type"),
        strategy=_string(data["strategy"], f"{where}.strategy"),
        dynamic=data["dynamic"],
        extra=_freeze({key: item for key, item in data.items() if key not in required}),
    )


def _parse_group(group_id: str, value: object) -> IluRuntimeQuantGroup:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", group_id) is None:
        raise ValueError(f"invalid ilu_quant group id: {group_id!r}")
    data = _mapping(value, f"config_groups.{group_id}")
    allowed = {
        "priority",
        "targets",
        "require",
        "weights",
        "input_activations",
        "output_activations",
        "iluvatar",
        "validation",
    }
    _strict_keys(
        data,
        allowed,
        f"config_groups.{group_id}",
        required=allowed - {"validation"},
    )

    require_data = _mapping(data["require"], f"{group_id}.require")
    _strict_keys(
        require_data,
        {"original_disposition", "checkpoint_dtypes"},
        f"{group_id}.require",
        required={"original_disposition", "checkpoint_dtypes"},
    )
    disposition = _string(
        require_data["original_disposition"],
        f"{group_id}.require.original_disposition",
    )
    if disposition not in {"ignored", "quantized", "any"}:
        raise ValueError(f"{group_id}: invalid original_disposition {disposition!r}")
    checkpoint_dtypes = _strings(
        require_data["checkpoint_dtypes"],
        f"{group_id}.require.checkpoint_dtypes",
    )

    weights_data = _mapping(data["weights"], f"{group_id}.weights")
    _strict_keys(
        weights_data,
        {"source", "target", "loading"},
        f"{group_id}.weights",
        required={"source", "target", "loading"},
    )
    source = _parse_endpoint(
        weights_data["source"], f"{group_id}.weights.source", source=True
    )
    if set(source.dtypes) != set(checkpoint_dtypes):
        raise ValueError(
            f"{group_id}: weights.source.dtypes must equal checkpoint_dtypes"
        )
    target = _parse_endpoint(
        weights_data["target"], f"{group_id}.weights.target", source=False
    )
    loading_data = _mapping(weights_data["loading"], f"{group_id}.weights.loading")
    _strict_keys(
        loading_data,
        {"mode", "stage"},
        f"{group_id}.weights.loading",
        required={"mode", "stage"},
    )
    mode = _string(loading_data["mode"], f"{group_id}.weights.loading.mode")
    stage = _string(loading_data["stage"], f"{group_id}.weights.loading.stage")
    if mode not in {"direct", "transform"}:
        raise ValueError(f"{group_id}: invalid loading mode {mode!r}")
    if stage not in {"none", "process_weights_after_loading", "weight_load"}:
        raise ValueError(f"{group_id}: invalid loading stage {stage!r}")
    if mode == "transform" and stage == "none":
        raise ValueError(f"{group_id}: transform + none is not supported")
    if mode == "direct" and stage == "weight_load":
        # weight_load means "the method converts each shard as it lands", which
        # only makes sense when source and target formats differ.
        raise ValueError(f"{group_id}: direct + weight_load is not supported")

    ilu_data = _mapping(data["iluvatar"], f"{group_id}.iluvatar")
    if "method" not in ilu_data:
        raise ValueError(f"{group_id}.iluvatar.method is required")
    method = _string(ilu_data["method"], f"{group_id}.iluvatar.method")
    options = _freeze({key: item for key, item in ilu_data.items() if key != "method"})

    validation = None
    if "validation" in data:
        validation_data = _mapping(data["validation"], f"{group_id}.validation")
        keys = {
            "expected_selected",
            "expected_transformed",
            "expected_fallback",
        }
        _strict_keys(
            validation_data,
            keys,
            f"{group_id}.validation",
            required=keys,
        )
        validation = IluValidationConfig(
            expected_selected=_integer(
                validation_data["expected_selected"],
                f"{group_id}.validation.expected_selected",
                minimum=1,
            ),
            expected_transformed=_integer(
                validation_data["expected_transformed"],
                f"{group_id}.validation.expected_transformed",
                minimum=0,
            ),
            expected_fallback=_integer(
                validation_data["expected_fallback"],
                f"{group_id}.validation.expected_fallback",
                minimum=0,
            ),
        )
        if validation.expected_selected != (
            validation.expected_transformed + validation.expected_fallback
        ):
            raise ValueError(
                f"{group_id}: expected_selected must equal transformed + fallback"
            )

    return IluRuntimeQuantGroup(
        group_id=group_id,
        priority=_integer(data["priority"], f"{group_id}.priority"),
        targets=_targets(data["targets"], f"{group_id}.targets"),
        require=IluRequireConfig(disposition, checkpoint_dtypes),
        weights=IluWeightsConfig(
            source=source,
            target=target,
            loading=IluLoadingConfig(mode=mode, stage=stage),
        ),
        input_activations=_parse_activation(
            data["input_activations"], f"{group_id}.input_activations"
        ),
        output_activations=_parse_activation(
            data["output_activations"], f"{group_id}.output_activations"
        ),
        method=method,
        options=options,
        validation=validation,
    )


def _load_payload(raw: object) -> tuple[Mapping[str, Any], str]:
    wrapper = _mapping(raw, "additional_config.ilu_quant")
    source = _string(wrapper.get("source"), "ilu_quant.source")
    if source == "inline":
        payload = dict(wrapper)
        payload.pop("source")
        return payload, source
    if source != "file":
        raise ValueError(f"ilu_quant.source must be inline or file, got {source!r}")

    _strict_keys(
        wrapper,
        {"source", "file", "sha256"},
        "additional_config.ilu_quant",
        required={"source", "file", "sha256"},
    )
    path_text = _string(wrapper["file"], "ilu_quant.file")
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError("ilu_quant.file must be absolute")
    if not path.is_file():
        raise ValueError(f"ilu_quant.file is not a regular file: {path}")
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(f"ilu_quant.file exceeds {_MAX_FILE_BYTES} bytes")
    raw_bytes = path.read_bytes()
    actual = hashlib.sha256(raw_bytes).hexdigest()
    expected = _string(wrapper["sha256"], "ilu_quant.sha256")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        raise ValueError("ilu_quant.sha256 must be sha256:<64 lowercase hex>")
    if expected != f"sha256:{actual}":
        raise ValueError(
            f"ilu_quant.file SHA256 mismatch: expected {expected}, got sha256:{actual}"
        )
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {path}") from exc
    return _mapping(payload, "ilu_quant file payload"), source


def parse_ilu_runtime_quant_config(
    raw: object,
    *,
    model_type: str,
    quant_method: str,
) -> IluRuntimeQuantConfig:
    payload, source = _load_payload(raw)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    allowed = {
        "schema_version",
        "enabled",
        "applies_to",
        "ignore",
        "config_groups",
    }
    _strict_keys(payload, allowed, "ilu_quant payload", required=allowed)
    schema_version = _integer(payload["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported ilu_quant schema_version: {schema_version}")
    enabled = payload["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError("ilu_quant.enabled must be a boolean")

    applies_data = _mapping(payload["applies_to"], "ilu_quant.applies_to")
    _strict_keys(
        applies_data,
        {"model_types", "quant_methods"},
        "ilu_quant.applies_to",
        required={"model_types", "quant_methods"},
    )
    applies_to = IluAppliesTo(
        model_types=_strings(applies_data["model_types"], "applies_to.model_types"),
        quant_methods=_strings(
            applies_data["quant_methods"], "applies_to.quant_methods"
        ),
    )
    if model_type not in applies_to.model_types:
        raise ValueError(
            f"ilu_quant does not apply to model_type {model_type!r}; "
            f"allowed={applies_to.model_types}"
        )
    if quant_method not in applies_to.quant_methods:
        raise ValueError(
            f"ilu_quant does not apply to quant_method {quant_method!r}; "
            f"allowed={applies_to.quant_methods}"
        )

    groups_data = _mapping(payload["config_groups"], "ilu_quant.config_groups")
    if not groups_data:
        raise ValueError("ilu_quant.config_groups must not be empty")
    groups = tuple(
        _parse_group(_string(group_id, "config group id"), value)
        for group_id, value in groups_data.items()
    )
    audits = {
        group.group_id: _GroupAudit(target_hits={target: 0 for target in group.targets})
        for group in groups
    }
    return IluRuntimeQuantConfig(
        schema_version=schema_version,
        enabled=enabled,
        applies_to=applies_to,
        ignore=_targets(payload["ignore"], "ilu_quant.ignore", allow_empty=True),
        config_groups=groups,
        canonical_digest=digest,
        source=source,
        model_type=model_type,
        quant_method=quant_method,
        _audits=audits,
        _all_validated=[False],
    )


def _get_cached_config(
    base_quant_config: QuantizationConfig,
) -> IluRuntimeQuantConfig | None:
    cached = getattr(base_quant_config, _CACHE_ATTR, _ABSENT)
    if cached is not _ABSENT:
        return None if cached is None else cached

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, Mapping):
        if additional_config:
            raise TypeError("VllmConfig.additional_config must be a mapping")
        setattr(base_quant_config, _CACHE_ATTR, None)
        return None
    raw = additional_config.get("ilu_quant")
    if raw is None:
        setattr(base_quant_config, _CACHE_ATTR, None)
        return None

    model_config = vllm_config.model_config
    if model_config is None:
        raise RuntimeError("ilu_quant requires VllmConfig.model_config")
    text_config = model_config.hf_text_config
    model_type = _string(
        getattr(text_config, "model_type", None), "hf_text_config.model_type"
    )
    quant_method = base_quant_config.get_name()
    resolved = parse_ilu_runtime_quant_config(
        raw, model_type=model_type, quant_method=quant_method
    )
    setattr(base_quant_config, _CACHE_ATTR, resolved)
    return resolved


def get_ilu_runtime_quant_config(
    base_quant_config: QuantizationConfig,
) -> IluRuntimeQuantConfig | None:
    """Parsed ``additional_config.ilu_quant`` for this base config, if any.

    Base configs that exist *only* to host ilu_quant need to tell "no payload"
    apart from "payload present but no group matched"; ``resolve_ilu_runtime_method``
    returns ``None`` for both.
    """
    return _get_cached_config(base_quant_config)


def resolve_ilu_runtime_method(
    *,
    base_quant_config: QuantizationConfig,
    original_method: QuantizeMethodBase | None,
    original_disposition: str,
    layer: torch.nn.Module,
    prefix: str,
) -> QuantizeMethodBase | None:
    config = _get_cached_config(base_quant_config)
    if config is None:
        return None
    group = config.select_group(layer, prefix, base_quant_config.packed_modules_mapping)
    if group is None:
        return None
    required = group.require.original_disposition
    if required != "any" and required != original_disposition:
        raise ValueError(
            f"{prefix}: ilu_quant group {group.group_id!r} requires original "
            f"disposition {required!r}, got {original_disposition!r}"
        )
    context = IluRuntimeQuantMethodContext(
        base_quant_config=base_quant_config,
        original_method=original_method,
        original_disposition=original_disposition,
        group=group,
        prefix=prefix,
        layer=layer,
        runtime_config=config,
    )
    return IluvatarQuantRegistry.create_runtime_method(group.method, context)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")
