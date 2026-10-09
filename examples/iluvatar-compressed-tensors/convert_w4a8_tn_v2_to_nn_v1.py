#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rewrite a W4A8 MoE checkpoint from TN / pack_version=2 to NN / pack_version=1.

The conversion matches the runtime INT4 TN→NN helpers
``_relayout_int4_tn_to_nn`` / ``_pack_int4`` / ``_unpack_int4``:

* packed INT4 weights ``(n, k/2)`` → NN-packed ``(k, n/2)`` (32x32 swizzle)
* ``weight_scale *= 18`` so v1 ``alpha=1`` matches v2's hardcoded alpha
* drop ``i8_weight_scale`` / ``i8_weight_zero`` after folding them in

Shared experts, dense W8A8, and other tensors are copied unchanged.
Shards are converted one at a time: load → convert → store.
``--device auto`` uses CUDA when available.

Example::

    python examples/iluvatar-compressed-tensors/convert_w4a8_tn_v2_to_nn_v1.py \\
        --input /data/DeepSeek-V4-Flash-w4a8-v2-TN \\
        --output /data/DeepSeek-V4-Flash-w4a8-v1-NN
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


# ixformer ixinfer_gemm_helper uses alpha=18 for quant-v2 when no per-channel
# INT8->INT4 scale is supplied. Quant-v1 uses alpha=1 and expects that factor
# inside weight_scale.
_W4A8_V2_ALPHA = 18.0

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

# DeepSeek-V4 / GLM-style offline names: experts.<id>.w1|w2|w3.{weight,scale}
# Also accept fused w13 and stacked w13_weight / w2_weight.
_EXPERT_FIELD = re.compile(
    r"(?:^|\.)(?:experts(?:\.\d+)?\.(?:w13|w[123])\.(?P<field>weight(?:_scale)?|i8_weight_(?:scale|zero))"
    r"|w13_weight(?P<fused_scale>_scale)?)$"
)


def classify_expert_tensor(name: str) -> str | None:
    """Return ``weight``, ``weight_scale``, ``i8_weight_scale``, ``i8_weight_zero``,
    or ``None`` if ``name`` is not a routed-expert W4A8 tensor."""
    if "shared_experts" in name:
        return None
    match = _EXPERT_FIELD.search(name)
    if match is None:
        if name.endswith("w2_weight"):
            return "weight"
        if name.endswith("w2_weight_scale"):
            return "weight_scale"
        return None
    if match.group("field"):
        return match.group("field")
    return "weight_scale" if match.group("fused_scale") else "weight"


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """``(..., L // 2)`` packed INT4 -> ``(..., L)`` sign-extended values."""
    *lead, half = packed.shape
    if half % 16:
        raise ValueError(f"packed last dim {half} must be a multiple of 16")
    values = packed.reshape(*lead, half // 16, 16).to(torch.int16)
    low = (values & 0x0F) - ((values & 0x08) << 1)
    high = values >> 4
    return torch.cat([low, high], dim=-1).reshape(*lead, half * 2).to(torch.int8)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """``(..., L)`` INT4 values -> ``(..., L // 2)`` packed."""
    *lead, length = values.shape
    if length % 32:
        raise ValueError(f"value last dim {length} must be a multiple of 32")
    values = values.reshape(*lead, length // 32, 32).to(torch.int16)
    low, high = values[..., :16], values[..., 16:]
    packed = ((high & 0x0F) << 4) | (low & 0x0F)
    packed = torch.where(packed > 127, packed - 256, packed)
    return packed.reshape(*lead, length // 2).to(torch.int8)


def relayout_int4_tn_to_nn(weight_tn: torch.Tensor) -> torch.Tensor:
    """TN-packed INT4 ``(..., n, k/2)`` -> NN-packed ``(..., k, n/2)``."""
    if weight_tn.ndim < 2:
        raise ValueError(f"TN weight must be at least 2D, got {tuple(weight_tn.shape)}")
    *lead, n, half_k = weight_tn.shape
    k = half_k * 2
    if k % 32 or n % 32:
        raise ValueError(f"TN->NN relayout needs k={k}, n={n} multiples of 32")
    values = unpack_int4(weight_tn).transpose(-1, -2).contiguous()
    values = values.reshape(*lead, k // 32, 2, 16, n // 32, 2, 16)
    prefix = list(range(len(lead)))
    values = values.permute(
        *prefix,
        len(lead),
        len(lead) + 4,
        len(lead) + 2,
        len(lead) + 3,
        len(lead) + 1,
        len(lead) + 5,
    )
    return pack_int4(values.contiguous().reshape(*lead, k, n))


def fold_v2_scale(
    scale: torch.Tensor,
    i8_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fold the v2 INT8->INT4 factor into a v1 weight scale."""
    if i8_scale is None:
        return scale.float() * _W4A8_V2_ALPHA
    if i8_scale.numel() == 1:
        return scale.float() * i8_scale.float().reshape(())
    try:
        return scale.float() * i8_scale.float().reshape_as(scale)
    except RuntimeError as exc:
        raise ValueError(
            f"cannot fold i8_scale shape {tuple(i8_scale.shape)} into "
            f"weight_scale shape {tuple(scale.shape)}"
        ) from exc


def rewrite_quant_config(quant_config: Mapping[str, Any]) -> dict[str, Any]:
    """Set routed-expert W4A8 layout to NN / pack_version=1."""
    updated = deepcopy(dict(quant_config))
    if updated.get("quant_method") == "ilu_deepseek_v4_w4a8":
        if "version" in updated:
            updated["version"] = 1
        routed = updated.get("routed_experts")
        if isinstance(routed, dict):
            routed["gemm_format"] = "NN"
            routed["pack_version"] = 1
        return updated

    groups = updated.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            iluvatar = group.get("iluvatar")
            if isinstance(iluvatar, dict) and "pack_version" in iluvatar:
                iluvatar["gemm_format"] = "NN"
                iluvatar["pack_version"] = 1
    return updated


def source_w4a8_layout(quant_config: Mapping[str, Any]) -> tuple[str, int]:
    """Read the on-disk routed-expert ``gemm_format`` / ``pack_version``."""
    if quant_config.get("quant_method") == "ilu_deepseek_v4_w4a8":
        routed = quant_config.get("routed_experts") or {}
        return (
            str(routed.get("gemm_format", "")).upper(),
            int(routed.get("pack_version", quant_config.get("version", -1))),
        )
    groups = quant_config.get("config_groups") or {}
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        iluvatar = group.get("iluvatar") or {}
        if "pack_version" in iluvatar:
            return (
                str(iluvatar.get("gemm_format", "")).upper(),
                int(iluvatar["pack_version"]),
            )
    raise ValueError("quantization_config has no routed-expert pack_version")


def _paired_scale_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".weight_scale"
    if weight_name.endswith("_weight"):
        return weight_name + "_scale"
    raise ValueError(f"cannot derive scale name from {weight_name!r}")


def _paired_i8_scale_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".i8_weight_scale"
    return weight_name.replace("_weight", "_i8_weight_scale", 1)


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def relayout_int4_tn_to_nn_on_device(
    weight_tn: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Relayout on ``device``, returning a CPU tensor for safetensors."""
    if device.type == "cpu" or weight_tn.numel() == 0:
        return relayout_int4_tn_to_nn(weight_tn).contiguous()
    if weight_tn.ndim == 2 or weight_tn.shape[0] <= chunk_size:
        return _to_cpu(relayout_int4_tn_to_nn(weight_tn.to(device, non_blocking=True)))
    parts = [
        _to_cpu(
            relayout_int4_tn_to_nn(
                weight_tn[start : start + chunk_size].to(device, non_blocking=True)
            )
        )
        for start in range(0, weight_tn.shape[0], chunk_size)
    ]
    return torch.cat(parts, dim=0)


def convert_shard_tensors(
    tensors: Mapping[str, torch.Tensor],
    device: str | torch.device | None = "auto",
    *,
    chunk_size: int = 64,
) -> tuple[dict[str, torch.Tensor], Counter[str]]:
    """Convert one shard's tensors. Expert i8 aux tensors are dropped."""
    stats: Counter[str] = Counter()
    converted_scales: set[str] = set()
    output: dict[str, torch.Tensor] = {}
    grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
    work_device = resolve_device(device)

    def _fold_scale(name: str) -> None:
        scale_key = _paired_scale_name(name)
        if scale_key not in tensors or scale_key in converted_scales:
            return
        output[scale_key] = fold_v2_scale(
            tensors[scale_key], tensors.get(_paired_i8_scale_name(name))
        ).contiguous()
        converted_scales.add(scale_key)
        stats["scale"] += 1

    for name, tensor in tensors.items():
        if classify_expert_tensor(name) != "weight":
            continue
        if tensor.ndim == 2:
            grouped[tuple(tensor.shape)].append(name)
            continue
        output[name] = relayout_int4_tn_to_nn_on_device(
            tensor, work_device, chunk_size=chunk_size
        )
        stats["weight"] += 1
        _fold_scale(name)

    for names in grouped.values():
        stacked = torch.stack([tensors[name] for name in names])
        converted = relayout_int4_tn_to_nn_on_device(
            stacked, work_device, chunk_size=chunk_size
        )
        for index, name in enumerate(names):
            output[name] = converted[index].contiguous()
            stats["weight"] += 1
            _fold_scale(name)

    for name, tensor in tensors.items():
        kind = classify_expert_tensor(name)
        if kind is None:
            output[name] = tensor
            stats["kept"] += 1
            continue
        if kind in {"i8_weight_scale", "i8_weight_zero"}:
            stats["dropped_i8"] += 1
            continue
        if kind == "weight_scale" and name not in converted_scales:
            output[name] = fold_v2_scale(tensor).contiguous()
            converted_scales.add(name)
            stats["scale"] += 1

    return output, stats


def _replace_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def convert_one_shard(
    src: Path,
    dst: Path,
    device: torch.device,
    *,
    link_unchanged: bool = True,
) -> dict[str, Any]:
    print(f"[load]    {src.name}", flush=True)
    tensors = load_file(str(src), device="cpu")
    if not any(classify_expert_tensor(name) is not None for name in tensors):
        _replace_path(dst)
        if link_unchanged:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"[store]   {src.name} copied {len(tensors)} tensors", flush=True)
        return {"shard": src.name, "copied": True, "stats": {"kept": len(tensors)}}

    print(f"[convert] {src.name} device={device} tensors={len(tensors)}", flush=True)
    converted, stats = convert_shard_tensors(tensors, device=device)
    _replace_path(dst)
    print(f"[store]   {src.name}", flush=True)
    save_file(converted, str(dst))
    print(f"[done]    {src.name} {dict(stats)}", flush=True)
    return {"shard": src.name, "copied": False, "stats": dict(stats)}


def _rebuild_index(output_path: Path, source_index: Mapping[str, Any]) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_path in sorted(output_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt") as file:
            for key in file.keys():
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


def _write_model_config(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = input_path / "config.json"
    with source.open("r", encoding="utf-8") as file:
        model_config = json.load(file)
    quant_config = model_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError(f"{source} is missing quantization_config")
    model_config["quantization_config"] = rewrite_quant_config(quant_config)
    with (output_path / "config.json").open("w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2)
        file.write("\n")
    return model_config["quantization_config"]


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str | torch.device | None = "auto",
    shard_start: int | None = None,
    shard_end: int | None = None,
    link_unchanged: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")

    config_path = input_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)
    quant_config = model_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError(f"{config_path} is missing quantization_config")
    gemm_format, pack_version = source_w4a8_layout(quant_config)
    if not force and (gemm_format, pack_version) != ("TN", 2):
        raise ValueError(
            f"expected routed experts TN/v2, found {gemm_format}/v{pack_version}. "
            "Pass --force to rewrite anyway."
        )

    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")
    with index_path.open("r", encoding="utf-8") as file:
        source_index = json.load(file)

    all_shards = sorted(input_dir.glob("model-*-of-*.safetensors"))
    if not all_shards:
        all_shards = sorted(input_dir.glob("*.safetensors"))
    shards = all_shards[slice(shard_start, shard_end)]
    if not shards:
        raise FileNotFoundError("no input safetensors shards selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_device = resolve_device(device)
    print(f"converting {len(shards)} shards on {work_device}", flush=True)
    merged: Counter[str] = Counter()
    shard_results: list[dict[str, Any]] = []
    for shard in tqdm(shards, desc="shards"):
        result = convert_one_shard(
            shard,
            output_dir / shard.name,
            work_device,
            link_unchanged=link_unchanged,
        )
        merged.update(result["stats"])
        shard_results.append(result)

    for source in input_dir.iterdir():
        if (
            source.is_file()
            and source.suffix != ".safetensors"
            and source.name not in {"model.safetensors.index.json", "config.json"}
        ):
            shutil.copy2(source, output_dir / source.name)

    rewritten = None
    if len(shards) == len(all_shards):
        _rebuild_index(output_dir, source_index)
        rewritten = _write_model_config(input_dir, output_dir)
    return {
        "input": str(input_dir),
        "output": str(output_dir),
        "source_layout": f"{gemm_format}/v{pack_version}",
        "stats": dict(sorted(merged.items())),
        "shards": len(shard_results),
        "quantization_config": rewritten,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TN/v2 checkpoint directory")
    parser.add_argument("--output", required=True, help="NN/v1 output directory")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA if available), cpu, cuda, or cuda:N",
    )
    parser.add_argument("--shard-start", type=int)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="copy shards with no expert tensors instead of hardlinking",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even if quantization_config is not TN/v2",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = convert(
        args.input,
        args.output,
        device=args.device,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        link_unchanged=not args.copy_unchanged,
        force=args.force,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "quantization_config"}, indent=2
        )
    )
    if result["quantization_config"] is not None:
        layout = source_w4a8_layout(result["quantization_config"])
        print(f"wrote quantization_config routed experts {layout[0]}/v{layout[1]}")


if __name__ == "__main__":
    main()
