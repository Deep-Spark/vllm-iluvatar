#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified, rule-driven checkpoint quantizer.

File A (configs) is a Compressed-Tensors ``quantization_config`` JSON document.
File B (rules) classifies checkpoint tensor names and selects offline
dequant/quant pipelines.  Configs may reference rule fields via ``$rules`` so
offline quant options remain the source of truth for matching runtime fields.
Runtime method selection remains in file A via each config group's
``iluvatar.method`` field and is handled by ``IluQuantConfig``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

if __package__:
    from .methods import DEQUANT_METHODS, QUANT_METHODS
else:  # Allow ``python examples/iluvatar-compressed-tensors/quant.py``.
    from methods import DEQUANT_METHODS, QUANT_METHODS

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_SAFETENSORS_DTYPE_NAMES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FN": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "F8_E8M0FNU": "float8_e8m0fnu",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "I64": "int64",
    "U64": "uint64",
    "F64": "float64",
}


class ProfileError(ValueError):
    """Raised when file A and file B are inconsistent."""


def _normalize_dtype(dtype: object | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).lower().removeprefix("torch.")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProfileError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str | None
    ndim: int | None
    shape: tuple[int, ...] | None = None

    @classmethod
    def from_tensor(cls, name: str, tensor: object) -> TensorInfo:
        dim_fn = getattr(tensor, "dim", None)
        shape = getattr(tensor, "shape", None)
        return cls(
            name=name,
            dtype=_normalize_dtype(getattr(tensor, "dtype", None)),
            ndim=dim_fn() if callable(dim_fn) else None,
            shape=tuple(shape) if shape is not None else None,
        )

    @classmethod
    def from_safetensors_slice(cls, name: str, tensor_slice: object) -> TensorInfo:
        get_dtype = tensor_slice.get_dtype
        get_shape = tensor_slice.get_shape
        shape = tuple(int(size) for size in get_shape())
        raw_dtype = str(get_dtype()).upper()
        return cls(
            name=name,
            dtype=_SAFETENSORS_DTYPE_NAMES.get(raw_dtype, raw_dtype.lower()),
            ndim=len(shape),
            shape=shape,
        )


@dataclass(frozen=True)
class MatchCondition:
    regex: tuple[re.Pattern[str], ...] = ()
    exclude_regex: tuple[re.Pattern[str], ...] = ()
    suffixes: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    ndims: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], field: str) -> MatchCondition:
        allowed = {
            "regex",
            "exclude_regex",
            "suffixes",
            "dtypes",
            "ndims",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ProfileError(f"{field} has unknown keys: {sorted(unknown)}")
        ndims = raw.get("ndims", [])
        if not isinstance(ndims, list) or not all(
            isinstance(ndim, int) for ndim in ndims
        ):
            raise ProfileError(f"{field}.ndims must be a list of integers")
        return cls(
            regex=tuple(
                re.compile(pattern)
                for pattern in _string_tuple(raw.get("regex"), f"{field}.regex")
            ),
            exclude_regex=tuple(
                re.compile(pattern)
                for pattern in _string_tuple(
                    raw.get("exclude_regex"), f"{field}.exclude_regex"
                )
            ),
            suffixes=_string_tuple(raw.get("suffixes"), f"{field}.suffixes"),
            dtypes=tuple(
                _normalize_dtype(dtype) or ""
                for dtype in _string_tuple(raw.get("dtypes"), f"{field}.dtypes")
            ),
            ndims=tuple(ndims),
        )

    def matches(self, tensor: TensorInfo) -> bool:
        name = tensor.name
        return (
            (not self.regex or any(pattern.search(name) for pattern in self.regex))
            and not any(pattern.search(name) for pattern in self.exclude_regex)
            and (not self.suffixes or name.endswith(self.suffixes))
            and (not self.dtypes or _normalize_dtype(tensor.dtype) in self.dtypes)
            and (not self.ndims or tensor.ndim in self.ndims)
        )


def _rules_value(rules_doc: Mapping[str, Any], path: str) -> Any:
    """Resolve ``rules.<rule_id>.<field...>`` against a rules document."""
    parts = path.split(".")
    if len(parts) < 2 or parts[0] != "rules":
        raise ProfileError(f"$rules path must start with 'rules.<id>', got {path!r}")
    rules_list = rules_doc.get("rules")
    if not isinstance(rules_list, list):
        raise ProfileError("rules document missing rules list")
    rule_id = parts[1]
    rule = next(
        (
            item
            for item in rules_list
            if isinstance(item, Mapping) and item.get("id") == rule_id
        ),
        None,
    )
    if rule is None:
        raise ProfileError(f"$rules references unknown rule id {rule_id!r}")
    value: Any = rule
    for part in parts[2:]:
        if not isinstance(value, Mapping) or part not in value:
            raise ProfileError(f"$rules references missing field {path!r}")
        value = value[part]
    return value


def _resolve_config(value: Any, rules_doc: Mapping[str, Any]) -> Any:
    """Resolve ``$rules`` references in configs.

    When ``none_as`` is set and the rules value equals that sentinel, emit
    ``null`` (e.g. offline ``group_size: -1`` → runtime ``group_size: null``).
    """
    if isinstance(value, list):
        return [_resolve_config(item, rules_doc) for item in value]
    if not isinstance(value, dict):
        return value
    if "$rules" in value:
        if set(value) - {"$rules", "none_as"}:
            raise ProfileError("$rules reference only supports optional none_as")
        result = _rules_value(rules_doc, str(value["$rules"]))
        if "none_as" in value and result == value["none_as"]:
            result = None
        return result
    if "$config" in value:
        raise ProfileError(
            "$config is reversed; put concrete values in rules and use $rules in configs"
        )
    return {key: _resolve_config(item, rules_doc) for key, item in value.items()}


def _contains_config_ref(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_config_ref(item) for item in value)
    if not isinstance(value, dict):
        return False
    if "$config" in value:
        return True
    return any(_contains_config_ref(item) for item in value.values())


@dataclass(frozen=True)
class PipelineRule:
    id: str
    priority: int
    condition: MatchCondition
    action: str
    scale: Mapping[str, str] | None
    dequant_method: str | None
    dequant_options: Mapping[str, Any]
    quant_method: str | None
    quant_options: Mapping[str, Any]
    outputs: Mapping[str, str]
    work_device: str

    def matches(self, name: str, tensor: torch.Tensor) -> bool:
        return self.condition.matches(TensorInfo.from_tensor(name, tensor))

    def matches_info(self, tensor: TensorInfo) -> bool:
        return self.condition.matches(tensor)


@dataclass(frozen=True)
class RuleSet:
    name: str
    default_action: str
    load_device: str
    model_config_patch: Mapping[str, Any]
    rules: tuple[PipelineRule, ...]

    @classmethod
    def load(cls, path: str | Path, quant_config: Mapping[str, Any]) -> RuleSet:
        with Path(path).open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if raw.get("schema_version") != 1:
            raise ProfileError("file B schema_version must be 1")
        default_action = raw.get("default_action", "keep")
        if default_action not in {"keep", "drop"}:
            raise ProfileError("default_action must be keep or drop")
        load_device = raw.get("load_device", "requested")
        if load_device not in {"cpu", "requested"}:
            raise ProfileError("load_device must be cpu or requested")
        parsed: list[tuple[int, PipelineRule]] = []
        for index, item in enumerate(raw.get("rules", [])):
            action = item.get("action")
            if action not in {"keep", "drop", "convert"}:
                raise ProfileError(f"rules[{index}].action is invalid")
            dequant = item.get("dequant", {})
            quant = item.get("quant", {})
            dequant_method = dequant.get("method")
            quant_method = quant.get("method")
            if action == "convert":
                if dequant_method not in DEQUANT_METHODS:
                    raise ProfileError(
                        f"rules[{index}] unknown dequant method {dequant_method!r}"
                    )
                if quant_method not in QUANT_METHODS:
                    raise ProfileError(
                        f"rules[{index}] unknown quant method {quant_method!r}"
                    )
            scale = item.get("scale")
            if scale is not None and (
                not isinstance(scale, dict)
                or set(scale) != {"from_suffix", "to_suffix"}
            ):
                raise ProfileError(f"rules[{index}].scale needs from_suffix/to_suffix")
            dequant_options = dequant.get("options", {})
            quant_options = quant.get("options", {})
            if not isinstance(dequant_options, dict) or not isinstance(
                quant_options, dict
            ):
                raise ProfileError(
                    f"rules[{index}] dequant/quant options must be objects"
                )
            if _contains_config_ref(dequant_options) or _contains_config_ref(
                quant_options
            ):
                raise ProfileError(
                    f"rules[{index}] must use concrete options; "
                    "configs should reference them via $rules"
                )
            parsed.append(
                (
                    index,
                    PipelineRule(
                        id=str(item["id"]),
                        priority=int(item.get("priority", 0)),
                        condition=MatchCondition.from_dict(
                            item.get("match", {}), f"rules[{index}].match"
                        ),
                        action=action,
                        scale=scale,
                        dequant_method=dequant_method,
                        dequant_options=dequant_options,
                        quant_method=quant_method,
                        quant_options=quant_options,
                        outputs=item.get("outputs", {}),
                        work_device=bool(item.get("work_device", "")),
                    ),
                )
            )
        rules = tuple(
            rule
            for _, rule in sorted(
                parsed, key=lambda indexed: (-indexed[1].priority, indexed[0])
            )
        )
        return cls(
            name=str(raw.get("name", Path(path).stem)),
            default_action=default_action,
            load_device=load_device,
            model_config_patch=raw.get("model_config_patch", {}),
            rules=rules,
        )

    def match(self, name: str, tensor: torch.Tensor) -> PipelineRule | None:
        return next((rule for rule in self.rules if rule.matches(name, tensor)), None)

    def match_info(self, tensor: TensorInfo) -> PipelineRule | None:
        return next((rule for rule in self.rules if rule.matches_info(tensor)), None)


class TensorStore:
    def __init__(
        self,
        input_path: Path,
        weight_map: Mapping[str, str],
        device: str,
        max_cached_shards: int = 2,
    ) -> None:
        self.input_path = input_path
        self.weight_map = weight_map
        self.device = device
        self.max_cached_shards = max_cached_shards
        self.loaded: dict[str, dict[str, torch.Tensor]] = {}

    def add(self, shard_name: str, tensors: dict[str, torch.Tensor]) -> None:
        self.loaded[shard_name] = tensors
        self._evict()

    def get(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            for tensors in self.loaded.values():
                if name in tensors:
                    return tensors[name]
            raise KeyError(f"companion tensor {name!r} is absent from index")
        if shard_name not in self.loaded:
            self.loaded[shard_name] = load_file(
                str(self.input_path / shard_name), device=self.device
            )
        tensors = self.loaded.pop(shard_name)
        self.loaded[shard_name] = tensors
        self._evict()
        return tensors[name]

    def _evict(self) -> None:
        while len(self.loaded) > self.max_cached_shards:
            oldest = next(iter(self.loaded))
            del self.loaded[oldest]
            if self.device == "cuda":
                torch.cuda.empty_cache()


def _replace_suffix(name: str, source: str, target: str) -> str:
    if not source:
        return name + target
    if not name.endswith(source):
        raise ValueError(f"{name!r} does not end with {source!r}")
    return name[: -len(source)] + target


def _output_key(weight_name: str, rule: PipelineRule, suffix: str) -> str:
    if rule.scale is not None:
        source_suffix = rule.scale["from_suffix"]
    elif weight_name.endswith(".weight"):
        source_suffix = ".weight"
    else:
        source_suffix = ""
    return _replace_suffix(weight_name, source_suffix, suffix)


def _convert_tensor(
    name: str,
    tensor: torch.Tensor,
    rule: PipelineRule,
    store: TensorStore,
    work_device: str,
) -> dict[str, torch.Tensor]:
    if rule.action == "drop":
        return {}
    if rule.action == "keep":
        return {name: tensor}

    scale = None
    if rule.scale is not None:
        scale_name = _replace_suffix(
            name, rule.scale["from_suffix"], rule.scale["to_suffix"]
        )
        scale = store.get(scale_name)
    value = DEQUANT_METHODS[rule.dequant_method](  # type: ignore[index]
        tensor, scale, rule.dequant_options
    )
    if rule.work_device:
        value = value.to(work_device)
    result = QUANT_METHODS[rule.quant_method](value, rule.quant_options)  # type: ignore[index]
    weight_name = (
        name
        if "weight" not in rule.outputs
        else _output_key(name, rule, rule.outputs["weight"])
    )
    output = {weight_name: result.weight.to(tensor.device).contiguous()}
    for field in ("weight_scale", "i8_weight_scale", "i8_weight_zero"):
        value = getattr(result, field)
        if value is None:
            continue
        if field not in rule.outputs:
            raise ProfileError(f"rule {rule.id!r} has no output suffix for {field}")
        output[_output_key(name, rule, rule.outputs[field])] = value.to(
            tensor.device
        ).contiguous()
    return output


def _rebuild_index(output_path: Path, source_index: Mapping[str, Any]) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_path in sorted(output_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt") as file:
            keys = file.keys()
            for key in keys:
                tensor_slice = file.get_slice(key)
                numel = 1
                for size in tensor_slice.get_shape():
                    numel *= size
                total_size += numel * _SAFETENSORS_DTYPE_BYTES.get(
                    tensor_slice.get_dtype(), 1
                )
                weight_map[key] = shard_path.name
    metadata = dict(source_index.get("metadata", {}))
    metadata["total_size"] = total_size
    with (output_path / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({"metadata": metadata, "weight_map": weight_map}, file, indent=2)


def _load_checkpoint_index(input_dir: Path) -> tuple[dict[str, Any], bool]:
    """Load checkpoint index from ``model.safetensors.index.json``, or from a lone ``model.safetensors``."""
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as file:
            source_index = json.load(file)
        if not isinstance(source_index, dict):
            raise ProfileError("model index must be an object")
        return source_index, True

    single_path = input_dir / "model.safetensors"
    if not single_path.is_file():
        raise FileNotFoundError(f"missing {index_path} and {single_path}")
    shards = sorted(input_dir.glob("*.safetensors"))
    if shards != [single_path]:
        raise FileNotFoundError(f"missing {index_path}")
    with safe_open(str(single_path), framework="pt") as file:
        weight_map = {key: "model.safetensors" for key in file.keys()}
    return {"metadata": {}, "weight_map": weight_map}, False


def _write_model_config(
    input_path: Path,
    output_path: Path,
    quant_config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> None:
    source = input_path / "config.json"
    if not source.is_file():
        return
    with source.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    for dotted_key, value in patch.items():
        parts = dotted_key.split(".")
        target = model_config
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        key = parts[-1]
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value
    model_config["quantization_config"] = quant_config
    with (output_path / "config.json").open("w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2)


def convert(
    input_path: str,
    output_path: str,
    configs: str,
    rules: str,
    *,
    device: str = "cuda",
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> dict[str, int]:
    input_dir = Path(input_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")

    with Path(rules).open("r", encoding="utf-8") as file:
        rules_doc = json.load(file)
    with Path(configs).open("r", encoding="utf-8") as file:
        quant_config = _resolve_config(json.load(file), rules_doc)
    rule_set = RuleSet.load(rules, quant_config)

    source_index, has_index = _load_checkpoint_index(input_dir)
    weight_map = source_index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ProfileError("model index weight_map must be an object")

    all_shards = sorted(input_dir.glob("*.safetensors"))
    shards = all_shards[slice(shard_start, shard_end)]
    if not shards:
        raise FileNotFoundError("no input safetensors shards selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_shard_names = {path.name for path in all_shards}
    stale_shards = sorted(
        path.name
        for path in output_dir.glob("*.safetensors")
        if path.name not in input_shard_names
    )
    if stale_shards:
        raise FileExistsError(
            "output directory contains safetensors shards absent from the input: "
            + ", ".join(stale_shards)
        )

    load_device = device if rule_set.load_device == "requested" else "cpu"
    store = TensorStore(input_dir, weight_map, load_device)
    stats: Counter[str] = Counter()
    for shard_path in tqdm(shards, desc="shards"):
        tensors = load_file(str(shard_path), device=load_device)
        store.add(shard_path.name, tensors)
        output: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            rule = rule_set.match(name, tensor)
            if rule is None:
                action = rule_set.default_action
                stats[f"default_{action}"] += 1
                if action == "keep":
                    output[name] = tensor
                continue
            output.update(_convert_tensor(name, tensor, rule, store, device))
            stats[rule.id] += 1
        save_file(output, str(output_dir / shard_path.name))

    for source in input_dir.iterdir():
        if (
            source.is_file()
            and source.suffix != ".safetensors"
            and source.name not in {"model.safetensors.index.json", "config.json"}
        ):
            shutil.copy(source, output_dir / source.name)

    # Rebuild only after a complete conversion. Partial jobs can be merged and
    # finalized by rerunning with the complete output directory.
    if len(shards) == len(all_shards):
        if has_index:
            _rebuild_index(output_dir, source_index)
        _write_model_config(
            input_dir, output_dir, quant_config, rule_set.model_config_patch
        )
    return dict(sorted(stats.items()))


def match_only(
    input_path: str,
    configs: str,
    rules: str,
) -> dict[str, Any]:
    """Match every indexed tensor using safetensors metadata only."""
    input_dir = Path(input_path)
    with Path(rules).open("r", encoding="utf-8") as file:
        rules_doc = json.load(file)
    with Path(configs).open("r", encoding="utf-8") as file:
        quant_config = _resolve_config(json.load(file), rules_doc)
    rule_set = RuleSet.load(rules, quant_config)

    source_index, _ = _load_checkpoint_index(input_dir)
    weight_map = source_index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ProfileError("model index weight_map must be an object")

    by_shard: dict[str, list[str]] = {}
    for name, shard_name in weight_map.items():
        by_shard.setdefault(str(shard_name), []).append(str(name))

    rule_stats = {
        rule.id: {
            "id": rule.id,
            "priority": rule.priority,
            "action": rule.action,
            "count": 0,
            "tensors": [],
        }
        for rule in rule_set.rules
    }
    default_key = f"default_{rule_set.default_action}"
    default_stats: dict[str, Any] = {
        "action": rule_set.default_action,
        "count": 0,
        "tensors": [],
    }

    for shard_name in sorted(by_shard):
        shard_path = input_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as file:
            shard_keys = set(file.keys())
            for name in sorted(by_shard[shard_name]):
                if name not in shard_keys:
                    raise KeyError(f"{name!r} is absent from {shard_path}")
                tensor = TensorInfo.from_safetensors_slice(name, file.get_slice(name))
                rule = rule_set.match_info(tensor)
                record = {
                    "name": name,
                    "shard": shard_name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape or ()),
                }
                target = default_stats if rule is None else rule_stats[rule.id]
                target["count"] += 1
                target["tensors"].append(record)

    return {
        "mode": "match_only",
        "input": str(input_dir),
        "configs": str(configs),
        "rules": str(rules),
        "total": len(weight_map),
        "rules_result": list(rule_stats.values()),
        default_key: default_stats,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--configs", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shard-start", type=int)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="match all indexed tensor metadata without loading or converting tensors",
    )
    parser.add_argument(
        "--match-report",
        help="write detailed per-rule match results as JSON (requires --match-only)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.match_only:
        if not args.match_report:
            parser.error("--match-only requires --match-report")
        stats = match_only(args.input, args.configs, args.rules)
        report_path = Path(args.match_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump({"stats": stats}, file, indent=2)
        print(
            json.dumps(
                {
                    "match_report": str(report_path),
                    "total": stats["total"],
                    "rules": {
                        item["id"]: item["count"] for item in stats["rules_result"]
                    },
                    "default": stats[
                        next(key for key in stats if key.startswith("default_"))
                    ]["count"],
                },
                indent=2,
            )
        )
        return
    if args.match_report:
        parser.error("--match-report requires --match-only")
    if not args.output:
        parser.error("--output is required unless --match-only is set")
    stats = convert(
        args.input,
        args.output,
        args.configs,
        args.rules,
        device=args.device,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
    )
    print(json.dumps({"stats": stats}, indent=2))


if __name__ == "__main__":
    main()
