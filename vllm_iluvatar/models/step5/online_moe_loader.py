"""Bounded packed Step5 FP8 loading into the existing online FP4 method.

Read mates from the original safetensors index and complete gate/up/down for
one owned expert before advancing. CPU file mapping is checkpoint loading;
all resident parameters and converter staging are on the GPU.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)
_BANK = re.compile(
    "^(?:model\\.)?layers\\.(\\d+)\\.moe\\.(gate_proj|up_proj|down_proj)\\.(weight|weight_scale_inv)$"
)
_SHARDS = (
    ("gate_proj", "w1", "w13"),
    ("up_proj", "w3", "w13"),
    ("down_proj", "w2", "w2"),
)


class Step5OnlineExpertLoader:
    def __init__(self, checkpoint, layers, num_experts):
        self.root = Path(checkpoint)
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.layers = layers
        self.num_experts = num_experts
        self.loaded_names = set()
        self.seen = set()
        self.completed = set()
        self.staging_peak_bytes = 0
        self.expected = {}
        self.owned = {}
        for index, layer in layers.items():
            method = layer.quant_method
            h = method._hidden_size
            i = method._intermediate_size * layer.moe_config.moe_parallel_config.tp_size
            bn, bk = method.weight_block_size
            for projection, _, _ in _SHARDS:
                n, k = (h, i) if projection == "down_proj" else (i, h)
                base = f"model.layers.{index}.moe.{projection}"
                self.expected[base + ".weight"] = (num_experts, n, k)
                self.expected[base + ".weight_scale_inv"] = (
                    num_experts,
                    (n + bn - 1) // bn,
                    (k + bk - 1) // bk,
                )
            mapping = {
                expert: layer._map_global_expert_id_to_local_expert_id(expert)
                for expert in range(num_experts)
            }
            owned = [expert for expert, local in mapping.items() if local >= 0]
            if sorted(mapping[e] for e in owned) != list(
                range(method._num_local_experts)
            ):
                raise ValueError(
                    f"Step5 online layer {index}: invalid expert ownership"
                )
            self.owned[index] = owned

    @staticmethod
    def _canonical(name):
        return name if name.startswith("model.") else "model." + name

    def _check_tensor(self, name, tensor, *, sliced=False):
        expected = self.expected[name][1:] if sliced else self.expected[name]
        dtype = (
            torch.float32 if name.endswith("weight_scale_inv") else torch.float8_e4m3fn
        )
        if tuple(tensor.shape) != expected or tensor.dtype != dtype:
            raise ValueError(
                f"Step5 online {name}: expected {expected}/{dtype}, got {tuple(tensor.shape)}/{tensor.dtype}"
            )

    def _staging_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers.values()
            for tensor in layer.quant_method._staging.values()
        )

    def _load_layer(self, index, trigger_name, trigger):
        layer = self.layers[index]
        method = layer.quant_method
        names = [
            f"model.layers.{index}.moe.{projection}.{suffix}"
            for projection, _, _ in _SHARDS
            for suffix in ("weight", "weight_scale_inv")
        ]
        missing = set(names) - self.index.keys()
        if missing:
            raise ValueError(
                f"Step5 online missing checkpoint mates: {sorted(missing)}"
            )
        budget = 2 * method._intermediate_size * method._hidden_size
        layer_peak = 0
        with torch.device("cpu"), ExitStack() as stack:
            files = {
                filename: stack.enter_context(
                    safe_open(self.root / filename, framework="pt", device="cpu")
                )
                for filename in {self.index[name] for name in names}
            }
            readers = {name: files[self.index[name]].get_slice(name) for name in names}
            for name, reader in readers.items():
                if tuple(reader.get_shape()) != self.expected[name]:
                    raise ValueError(f"Step5 online checkpoint shape mismatch: {name}")
            for expert in self.owned[index]:
                for projection, shard, group in _SHARDS:
                    for suffix, param_suffix in (
                        ("weight", "weight"),
                        ("weight_scale_inv", "weight_scale_inv"),
                    ):
                        name = f"model.layers.{index}.moe.{projection}.{suffix}"
                        tensor = (
                            trigger[expert]
                            if name == trigger_name
                            else readers[name][expert]
                        )
                        if tensor.device.type != "cpu":
                            raise RuntimeError(
                                "Step5 checkpoint slices must remain on CPU until expert loading"
                            )
                        self._check_tensor(name, tensor, sliced=True)
                        param_name = f"{group}_{param_suffix}"
                        param = getattr(layer, param_name)
                        local_name = (
                            f"layers.{index}.moe.experts.routed_experts.{param_name}"
                        )
                        success = param.weight_loader(
                            param,
                            tensor,
                            local_name,
                            shard_id=shard,
                            expert_id=expert,
                            return_success=True,
                        )
                        if not success:
                            raise RuntimeError(
                                f"Step5 online owned expert {expert} was rejected"
                            )
                        self.loaded_names.add(local_name)
                        current = self._staging_bytes()
                        layer_peak = max(layer_peak, current)
                        self.staging_peak_bytes = max(self.staging_peak_bytes, current)
                        if current > budget:
                            raise RuntimeError(
                                f"Step5 online staging {current} exceeds one-expert budget {budget}"
                            )
                torch.cuda.synchronize(layer.w13_weight.device)
        torch.cuda.empty_cache()
        if method._staging or method._pending_units():
            raise RuntimeError(f"Step5 online layer {index}: staging was not released")
        if method._conversions != 3 * len(self.owned[index]):
            raise RuntimeError(
                f"Step5 online layer {index}: incomplete conversion count"
            )
        self.completed.add(index)
        method._step5_loading_audit = {
            "layer": index,
            "local_experts": len(self.owned[index]),
            "converted": method._conversions,
            "pending_peak": method._pending_peak,
            "staging_peak_bytes": layer_peak,
            "cross_layer_staging_peak_bytes": self.staging_peak_bytes,
            "staging_remaining_bytes": self._staging_bytes(),
            "source_scale_semantics": "multiplicative",
            "source_dtype": "float8_e4m3fn",
            "converter_backend": method.converter_backend or "auto",
            "gpu_allocated_after_layer": torch.cuda.memory_allocated(),
            "gpu_reserved_after_layer": torch.cuda.memory_reserved(),
            "gpu_max_allocated": torch.cuda.max_memory_allocated(),
        }
        logger.info("STEP5_ONLINE_LOAD %s", json.dumps(method._step5_loading_audit))

    def remaining_weights(self, weights, *, finalize=True):
        for name, tensor in weights:
            match = _BANK.fullmatch(name)
            if match is None or int(match[1]) not in self.layers:
                yield (name, tensor)
                continue
            index = int(match[1])
            canonical = self._canonical(name)
            if canonical in self.seen:
                raise ValueError(f"Step5 online duplicate source bank: {canonical}")
            self.seen.add(canonical)
            self._check_tensor(canonical, tensor)
            if match[3] == "weight" and index not in self.completed:
                self._load_layer(index, canonical, tensor)
        if finalize:
            self.finalize()

    def finalize(self):
        missing = self.expected.keys() - self.seen
        if missing or self.completed != self.layers.keys():
            raise ValueError(
                f"Step5 online incomplete source stream: {sorted(missing)}"
            )
