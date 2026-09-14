#!/usr/bin/env python3
"""Create a derived checkpoint config for online FP8 dense linear weights.

The source checkpoint is never modified. All files except ``config.json`` are
symlinked into the output directory and a patched config is written there.

The Qwen3.8 preset is convenient for this model, while ``--preset none`` plus
``--remove-pattern``/``--set-quant`` makes the tool usable by other models and
by checkpoints that use different quantization/ignore key names.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, MutableMapping
from pathlib import Path
from typing import Any

QWEN38_REMOVE_PATTERNS = (
    r"(?:^|\.)self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$",
    r"(?:^|\.)self_attn\.indexer\.index_qk_proj$",
    r"(?:^|\.)linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$",
    r"(?:^|\.)mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)$",
    r"(?:^|\.)ple\.(?:key_proj|value_proj)$",
)
QWEN38_QUANT_UPDATES = {
    "none_moe_is_checkpoint_fp8_serialized": False,
    "none_moe_use_weight_block_size": False,
}
DEFAULT_IGNORE_KEYS = ("modules_to_not_convert", "ignored_layers")


def _parse_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError("configuration path must not be empty")
    return parts


def _get_mapping(root: MutableMapping[str, Any], path: str) -> MutableMapping[str, Any]:
    current: Any = root
    for key in _parse_path(path):
        if not isinstance(current, MutableMapping) or key not in current:
            raise KeyError(f"configuration path {path!r} does not exist")
        current = current[key]
    if not isinstance(current, MutableMapping):
        raise TypeError(f"configuration path {path!r} is not a JSON object")
    return current


def _walk_mappings(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], MutableMapping[str, Any]]]:
    if not isinstance(value, MutableMapping):
        return
    yield path, value
    for key, child in value.items():
        if isinstance(child, MutableMapping):
            yield from _walk_mappings(child, (*path, str(key)))


def find_quant_config(
    config: MutableMapping[str, Any],
    quant_config_path: str | None,
    ignore_key: str | None,
) -> tuple[str, MutableMapping[str, Any]]:
    """Resolve a quantization config without assuming its model-level key."""
    if quant_config_path:
        return quant_config_path, _get_mapping(config, quant_config_path)

    candidates = []
    ignore_keys = (ignore_key,) if ignore_key else DEFAULT_IGNORE_KEYS
    for path, mapping in _walk_mappings(config):
        if "quant_method" in mapping and any(key in mapping for key in ignore_keys):
            candidates.append((".".join(path), mapping))
    if len(candidates) != 1:
        paths = [path or "<root>" for path, _ in candidates]
        raise ValueError(
            "auto-detection requires exactly one quantization config containing "
            f"'quant_method' and an ignore list; found {paths}. "
            "Pass --quant-config-path explicitly."
        )
    return candidates[0]


def find_ignore_key(
    quant_config: MutableMapping[str, Any], ignore_key: str | None
) -> str:
    if ignore_key:
        if ignore_key not in quant_config:
            raise KeyError(f"ignore key {ignore_key!r} is absent")
        return ignore_key
    candidates = [key for key in DEFAULT_IGNORE_KEYS if key in quant_config]
    if len(candidates) != 1:
        raise ValueError(
            "auto-detection requires exactly one ignore-list key from "
            f"{DEFAULT_IGNORE_KEYS}; found {candidates}. Pass --ignore-key."
        )
    return candidates[0]


def patch_quant_config(
    config: MutableMapping[str, Any],
    *,
    quant_config_path: str | None,
    ignore_key: str | None,
    remove_patterns: Iterable[str],
    quant_updates: MutableMapping[str, Any],
) -> dict[str, Any]:
    resolved_path, quant_config = find_quant_config(
        config, quant_config_path, ignore_key
    )
    resolved_ignore_key = find_ignore_key(quant_config, ignore_key)
    ignored = quant_config[resolved_ignore_key]
    if not isinstance(ignored, list) or not all(
        isinstance(item, str) for item in ignored
    ):
        raise TypeError(f"{resolved_ignore_key!r} must be a list of strings")

    compiled = [re.compile(pattern) for pattern in remove_patterns]
    if not compiled:
        raise ValueError("at least one --remove-pattern or a preset is required")

    removed = [
        name for name in ignored if any(pattern.search(name) for pattern in compiled)
    ]
    if not removed:
        raise ValueError("no ignored layer matched the requested remove patterns")
    removed_set = set(removed)
    remaining = [name for name in ignored if name not in removed_set]
    quant_config[resolved_ignore_key] = remaining
    # IluFp8Config consumes these two conventional names. Preserve a vendor
    # source key for its own tooling, but also emit a normalized runtime key so
    # renaming the checkpoint field cannot silently quantize every Linear.
    runtime_ignore_key = resolved_ignore_key
    if resolved_ignore_key not in DEFAULT_IGNORE_KEYS:
        runtime_ignore_key = "ignored_layers"
        quant_config[runtime_ignore_key] = list(remaining)
    quant_config.update(quant_updates)

    return {
        "quant_config_path": resolved_path or "<root>",
        "ignore_key": resolved_ignore_key,
        "runtime_ignore_key": runtime_ignore_key,
        "original_count": len(ignored),
        "removed_count": len(removed),
        "remaining_count": len(ignored) - len(removed),
        "removed": removed,
        "pattern_counts": {
            pattern.pattern: sum(bool(pattern.search(name)) for name in ignored)
            for pattern in compiled
        },
        "updates": dict(quant_updates),
    }


def _parse_update(raw: str) -> tuple[str, Any]:
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("--set-quant expects KEY=JSON_VALUE")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"invalid JSON value in --set-quant {raw!r}: {error}"
        ) from error


def _ensure_output_is_safe(input_dir: Path, output_dir: Path) -> None:
    if input_dir == output_dir:
        raise ValueError("--input and --output must be different directories")
    if input_dir in output_dir.parents:
        raise ValueError("--output must not be nested inside --input")


def _link_checkpoint_files(
    input_dir: Path, output_dir: Path, config_relative_path: Path
) -> None:
    for source in input_dir.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(input_dir)
        if relative == config_relative_path:
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            destination.unlink()
        elif destination.exists():
            raise FileExistsError(
                f"refusing to replace non-symlink output file {destination}"
            )
        os.symlink(source.resolve(), destination)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--preset",
        choices=("qwen38-flash-next", "none"),
        default="qwen38-flash-next",
    )
    parser.add_argument(
        "--quant-config-path",
        help="Dotted path to quant config; default auto-detects it by content.",
    )
    parser.add_argument(
        "--ignore-key",
        help="Ignore-list key inside quant config; default auto-detects it.",
    )
    parser.add_argument(
        "--remove-pattern",
        action="append",
        default=[],
        help="Regex matched against checkpoint parameter/module names; repeatable.",
    )
    parser.add_argument(
        "--set-quant",
        action="append",
        default=[],
        type=_parse_update,
        metavar="KEY=JSON_VALUE",
        help="Set an arbitrary quant-config key; repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    config_relative_path = args.config_file
    if config_relative_path.is_absolute() or ".." in config_relative_path.parts:
        raise ValueError("--config-file must be a relative path inside the checkpoint")
    _ensure_output_is_safe(input_dir, output_dir)

    source_config = input_dir / config_relative_path
    with source_config.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, MutableMapping):
        raise TypeError("model config root must be a JSON object")

    patterns = list(args.remove_pattern)
    updates = dict(args.set_quant)
    if args.preset == "qwen38-flash-next":
        patterns = [*QWEN38_REMOVE_PATTERNS, *patterns]
        updates = {**QWEN38_QUANT_UPDATES, **updates}

    report = patch_quant_config(
        config,
        quant_config_path=args.quant_config_path,
        ignore_key=args.ignore_key,
        remove_patterns=patterns,
        quant_updates=updates,
    )
    print(
        f"quant_config={report['quant_config_path']} "
        f"ignore_key={report['ignore_key']} "
        f"runtime_ignore_key={report['runtime_ignore_key']}"
    )
    print(
        f"ignored layers: original={report['original_count']} "
        f"removed={report['removed_count']} remaining={report['remaining_count']}"
    )
    for pattern, count in report["pattern_counts"].items():
        print(f"pattern {pattern!r}: {count}")
    print(f"quant updates: {json.dumps(report['updates'], ensure_ascii=False)}")
    for name in report["removed"][:10]:
        print(f"removed sample: {name}")

    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _link_checkpoint_files(input_dir, output_dir, config_relative_path)
    destination_config = output_dir / config_relative_path
    destination_config.parent.mkdir(parents=True, exist_ok=True)
    destination_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"derived checkpoint written to {output_dir}")


if __name__ == "__main__":
    main()
