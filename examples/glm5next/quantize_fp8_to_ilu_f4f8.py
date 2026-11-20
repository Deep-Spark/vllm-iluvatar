#!/usr/bin/env python3
"""Convert a local GLM-5.3-Flash block-FP8 checkpoint to CQ ilu_f4f8 v1 on CPU."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROUTED_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)

DENSE_OR_SHARED_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\."
    r"(?:shared_experts\.)?(down_proj|gate_proj|up_proj)\.weight$"
)

QUANTIZATION_CONFIG = {
    "bf16_layers": [r"re:.*\.self_attn\..*"],
    "dense_linears": {
        "group_size": None,
        "input_activations": {
            "dynamic": True,
            "num_bits": 8,
            "strategy": "token",
            "symmetric": True,
            "type": "int",
        },
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
        "targets": [
            r"re:.*\.layers\.[0-2]\.mlp\.(gate_proj|up_proj|gate_up_proj|down_proj)$"
        ],
        "type": "int",
    },
    "format": "ilu-mixed-f4f8",
    "ignore": [
        r"re:^model\.visual\..*",
        r"re:.*embed_tokens.*",
        r"re:.*lm_head.*",
        r"re:.*norm.*",
        r"re:.*\.mlp\.gate$",
        r"re:.*\.hc_.*",
    ],
    "quant_method": "ilu_f4f8",
    "routed_experts": {
        "group_size": 256,
        "input_activations": {
            "dynamic": True,
            "num_bits": 8,
            "quant_dtype": "fp8_e4m3fn",
            "strategy": "token",
            "symmetric": True,
            "type": "float",
        },
        "num_bits": 4,
        "pack_factor": 2,
        "quant_dtype": "fp4_e2m1",
        "scale_dtype": "float32",
        "scale_form": "absolute",
        "storage_dtype": "uint8",
        "strategy": "group",
        "symmetric": True,
        "targets": [r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
        "type": "float",
        "weight_layout": "ix30_kmajor_pack32_v1",
    },
    "shared_experts": {
        "group_size": None,
        "input_activations": {
            "dynamic": True,
            "num_bits": 8,
            "strategy": "token",
            "symmetric": True,
            "type": "int",
        },
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
        "targets": [r"re:.*\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$"],
        "type": "int",
    },
    "version": 1,
}

BLOCK = 128
GROUP = 256
MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
PACK_ROWS = (
    4 * torch.arange(4)[:, None] + torch.tensor([0, 1, 2, 3, 16, 17, 18, 19])
).flatten()
PACK_SIGNS = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
INDEX = "model.safetensors.index.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def role(name: str) -> str:
    if ROUTED_RE.fullmatch(name):
        return "routed"
    if DENSE_OR_SHARED_RE.fullmatch(name):
        return "dense/shared"
    if ".self_attn." in name and name.endswith(".weight"):
        return "attention"
    raise ValueError(f"Unknown FP8 projection: {name}")


def inspect_source(source: Path):
    config = load_json(source / "config.json")
    quant = config.get("quantization_config", {})
    if (
        "Glm5NextForConditionalGeneration" not in (config.get("architectures") or [])
        or quant.get("quant_method") != "fp8"
        or quant.get("fmt") != "e4m3"
        or quant.get("weight_block_size") != [BLOCK, BLOCK]
    ):
        raise ValueError("Expected GLM-5.3-Flash with fp8/e4m3 128x128 block weights")
    weight_map = load_json(source / INDEX)["weight_map"]
    if not weight_map:
        raise ValueError("Empty safetensors index")
    shard_keys = defaultdict(list)
    for name, shard in weight_map.items():
        path = Path(shard)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            raise ValueError(f"Invalid indexed shard: {shard!r}")
        shard_keys[shard].append(name)

    fp8_roles = {}
    required_scales = {}
    stored_scales = {}
    for shard, keys in shard_keys.items():
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(keys):
                raise ValueError(f"{shard}: shard contents disagree with the index")
            for name in keys:
                part = reader.get_slice(name)
                dtype, shape = part.get_dtype(), part.get_shape()
                if name.endswith(".weight_scale_inv"):
                    if dtype != "F32":
                        raise ValueError(
                            f"{name}: expected an F32 block scale, got {dtype}"
                        )
                    stored_scales[name] = shape
                elif dtype == "F8_E4M3":
                    usage = role(name)
                    if len(shape) != 2 or (usage == "routed" and shape[1] % GROUP):
                        raise ValueError(
                            f"{name}: incompatible {usage} weight shape {shape}"
                        )
                    if name.removesuffix(".weight") + ".weight_scale" in weight_map:
                        raise ValueError(
                            f"{name}: output scale name already exists in the source"
                        )
                    fp8_roles[name] = usage
                    required_scales[name + "_scale_inv"] = [
                        (n + BLOCK - 1) // BLOCK for n in shape
                    ]
                elif ROUTED_RE.fullmatch(name) or DENSE_OR_SHARED_RE.fullmatch(name):
                    raise ValueError(
                        f"{name}: expected an FP8 expert or dense MLP weight, got {dtype}"
                    )
    if required_scales != stored_scales:
        problem = next(
            k
            for k in required_scales.keys() | stored_scales.keys()
            if required_scales.get(k) != stored_scales.get(k)
        )
        raise ValueError(
            f"{problem}: expected {required_scales.get(problem)}, got {stored_scales.get(problem)}"
        )
    if set(fp8_roles.values()) != {"routed", "dense/shared", "attention"}:
        raise ValueError(
            "The input lacks one or more GLM-5.3-Flash block-FP8 projection families"
        )
    return config, weight_map, shard_keys, fp8_roles


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    if rows % BLOCK or cols % BLOCK:
        expanded = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        return weight.float() * expanded[:rows, :cols]
    blocks = weight.reshape(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    return (blocks.float() * scale[:, None, :, None]).reshape(rows, cols)


def pack_routed(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed [K/2,N] E2M1 weights and absolute [K/256,N] FP32 scales."""
    n, k = weight.shape
    grouped = weight.T.contiguous().view(k // GROUP, GROUP, n)
    amax = grouped.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    normalized = (grouped / scale[:, None, :]).reshape(k, n)
    # The ix30 layout interleaves the sign and magnitude lanes independently.
    magnitude = torch.bucketize(normalized.abs(), MIDPOINTS, right=True).to(torch.uint8)
    nibbles = magnitude | ((normalized < 0).to(torch.uint8) << 3)
    rows = (
        nibbles.view(k // 32, 32, n).index_select(1, PACK_ROWS).view(k // 32, 4, 8, n)
    )
    lanes = (rows & 7) | (rows.index_select(2, PACK_SIGNS) & 8)
    packed = lanes[:, :, 0::2, :] | (lanes[:, :, 1::2, :] << 4)
    return packed.reshape(k // 2, n).contiguous(), scale.contiguous()


def quantize_projection(weight: torch.Tensor, kind: str):
    if not torch.isfinite(weight).all().item():
        raise ValueError("Non-finite dequantized weight")
    if kind == "routed":
        return pack_routed(weight)
    if kind == "dense/shared":
        amax = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        return (weight / scale).round().clamp_(-127, 127).to(torch.int8), scale
    return weight.to(torch.bfloat16), None


def convert_shard(source, shard, keys, weight_map, fp8_roles):
    weights = {}
    with ExitStack() as opened:
        readers = {}

        def reader(filename):
            if filename not in readers:
                readers[filename] = opened.enter_context(
                    safe_open(source / filename, framework="pt", device="cpu")
                )
            return readers[filename]

        for name in sorted(keys):
            if name.endswith(".weight_scale_inv"):
                continue
            tensor = reader(shard).get_tensor(name)
            if name not in fp8_roles:
                weights[name] = tensor
                continue
            scale_key = name + "_scale_inv"
            scale = reader(weight_map[scale_key]).get_tensor(scale_key)
            try:
                values, out_scale = quantize_projection(
                    dequantize(tensor, scale), fp8_roles[name]
                )
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            weights[name] = values
            if out_scale is not None:
                weights[name.removesuffix(".weight") + ".weight_scale"] = out_scale
    return weights


def copy_assets(source: Path, staging: Path) -> None:
    # Copy model/tokenizer/processor metadata, including HF chat templates.
    for item in source.iterdir():
        if item.name in {"config.json", INDEX} or item.name.startswith("."):
            continue
        if item.is_dir():
            if item.name == "chat_templates":
                shutil.copytree(item, staging / item.name)
        elif item.name.startswith(("LICENSE", "NOTICE", "README")) or item.suffix in {
            ".json",
            ".jinja",
            ".model",
            ".txt",
            ".tiktoken",
            ".py",
        }:
            shutil.copy2(item, staging / item.name)


@torch.inference_mode()
def convert(source, output, plan) -> None:
    config, weight_map, shard_keys, fp8_roles = plan
    if (
        output == source
        or source in output.parents
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError(f"Choose a new output directory outside the input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    out_map = {}
    total = 0
    # Publish the model only after every shard, the sidecars and the index are complete.
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temp:
        staging = Path(temp) / "model"
        staging.mkdir()
        for num, shard in enumerate(sorted(shard_keys), 1):
            tensors = convert_shard(
                source, shard, shard_keys[shard], weight_map, fp8_roles
            )
            if tensors:
                target = staging / shard
                target.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(target))
                out_map.update(dict.fromkeys(tensors, shard))
                total += sum(t.numel() * t.element_size() for t in tensors.values())
            print(f"[{num}/{len(shard_keys)}] {shard}", flush=True)
            del tensors
        copy_assets(source, staging)
        config = dict(config)
        config.pop("compression_config", None)
        config["quantization_config"] = QUANTIZATION_CONFIG
        write_json(staging / "config.json", config)
        write_json(
            staging / INDEX,
            {
                "metadata": {"total_size": total},
                "weight_map": dict(sorted(out_map.items())),
            },
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"{output}: {len(out_map)} tensors, {total:,} payload bytes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Original local GLM-5.3-Flash FP8 model directory",
    )
    ap.add_argument(
        "--output", type=Path, help="New output directory for the converted checkpoint"
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the source tensor index and shard headers only",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=min(16, torch.get_num_threads()),
        help="CPU threads (default: up to 16)",
    )
    args = ap.parse_args()
    if not args.check_only and args.output is None:
        ap.error("--output is required for conversion")
    if args.threads < 1:
        ap.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    source = args.input.resolve(strict=True)
    plan = inspect_source(source)
    print(
        f"{len(plan[2])} input shards; FP8 projections: {dict(Counter(plan[3].values()))}",
        flush=True,
    )
    if not args.check_only:
        # Resolve the parent while preserving an existing dangling destination symlink.
        dest = args.output.parent.resolve() / args.output.name
        convert(source, dest, plan)


if __name__ == "__main__":
    main()
