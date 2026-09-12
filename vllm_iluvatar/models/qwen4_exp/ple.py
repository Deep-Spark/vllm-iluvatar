# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from .config import Qwen4ExpTextConfig
from .ple_utils import copy_ple_embedding_shard_

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _require_ple_ops(*names: str):
    try:
        import ixformer._C as ixf_c_ops
    except Exception as exc:
        raise RuntimeError(
            "Qwen4Exp PLE requires the ixformer CUDA extension on CoreX30"
        ) from exc
    infer = getattr(ixf_c_ops, "infer", None)
    missing = [name for name in names if infer is None or not hasattr(infer, name)]
    if missing:
        raise RuntimeError(
            "Qwen4Exp PLE requires ixformer CUDA op(s): " + ", ".join(missing)
        )
    return infer


def _require_contiguous(t: torch.Tensor, name: str) -> None:
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_i32_contiguous(t: torch.Tensor, name: str) -> None:
    if t.dtype != torch.int32 or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous int32")


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


@dataclass(frozen=True)
class PLEVocabShard:
    """The half-open range of table rows owned by this TP rank."""

    org_vocab_start_index: int
    org_vocab_end_index: int


class Qwen4ExpNGramTable(nn.Module):
    """TP-sharded n-gram lookup table, FP8 / INT8 / unquantized.

    Deliberately *not* a ``VocabParallelEmbedding``. That layer zeroes its raw
    lookup output with an in-place ``masked_fill_``, which torch has no FP8
    kernel for, and it derives its quant method from the quant config, which
    returns ``None`` for embeddings. Here the rows are split evenly across TP
    ranks, dequantization happens before the mask so the mask runs in
    ``params_dtype``, and the partial results are summed -- every id lives on
    exactly one rank, so the sum reproduces the full table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        quant_kind: str | None,
        params_dtype: torch.dtype,
        divisible_by: int,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.params_dtype = params_dtype
        self.quant_kind = quant_kind

        # Pad so every rank owns the same row count while each rank's slice
        # keeps the checkpoint's own divisibility. Padding rows are never
        # indexed -- ids stay below org_vocab_size -- so they are left as is.
        stride = math.lcm(max(divisible_by, 1), self.tp_size)
        padded = ((num_embeddings + stride - 1) // stride) * stride
        rows = padded // self.tp_size
        start = get_tensor_model_parallel_rank() * rows
        self.shard_indices = PLEVocabShard(start, min(start + rows, num_embeddings))

        if quant_kind == "fp8":
            weight_dtype: torch.dtype = torch.float8_e4m3fn
            scale_shape: tuple[int, ...] | None = (1,)  # one global scale
        elif quant_kind == "int8":
            weight_dtype = torch.int8
            scale_shape = (rows, 1)  # one symmetric scale per row
        elif quant_kind is None:
            weight_dtype = params_dtype
            scale_shape = None
        else:
            raise ValueError(f"unknown PLE embedding quant kind: {quant_kind}")

        weight = torch.empty(rows, embedding_dim, dtype=weight_dtype)
        # zero_() has no FP8 kernel on every backend; go through the raw bytes.
        (weight.view(torch.uint8) if weight.element_size() == 1 else weight).zero_()
        self.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        if scale_shape is None:
            self.weight_scale = None
        else:
            self.register_parameter(
                "weight_scale",
                nn.Parameter(
                    torch.zeros(scale_shape, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            start = self.shard_indices.org_vocab_start_index
            owned = (ids >= start) & (ids < self.shard_indices.org_vocab_end_index)
            local_ids = torch.where(owned, ids - start, torch.zeros_like(ids))
        else:
            local_ids = ids
        rows = F.embedding(local_ids, self.weight)
        if self.quant_kind == "int8":
            scale = F.embedding(local_ids, self.weight_scale)
            output = rows.to(self.params_dtype) * scale.to(self.params_dtype)
        elif self.quant_kind == "fp8":
            output = rows.to(self.params_dtype) * self.weight_scale.to(
                self.params_dtype
            )
        else:
            output = rows
        if self.tp_size > 1:
            output = output.masked_fill(~owned.unsqueeze(-1), 0)
            output = tensor_model_parallel_all_reduce(output)
        return output


def _ple_ignored_layers(quant_config: QuantizationConfig) -> list[str]:
    ignored = getattr(quant_config, "ignored_layers", None)
    if ignored:
        return list(ignored)
    return list(getattr(quant_config, "ignore", None) or [])


def _is_compressed_tensors_quant(quant_config: QuantizationConfig) -> bool:
    name = str(quant_config.get_name())
    return name in ("compressed-tensors", "ilu_deepseek_v4_w4a8")


def _ple_embedding_quant_kind(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> str | None:
    """Storage format of the n-gram table: FP8, INT8 or unquantized."""

    if quant_config is None:
        return None

    ignored_layers = _ple_ignored_layers(quant_config)
    fused_mapping = getattr(quant_config, "packed_modules_mapping", {}) or {}
    if ignored_layers and is_layer_skipped(prefix, ignored_layers, fused_mapping):
        return None
    shard_prefix = f"{prefix}.shard_"
    if any(str(name).startswith(shard_prefix) for name in ignored_layers):
        return None

    if isinstance(quant_config, Fp8Config):
        return "fp8" if quant_config.is_checkpoint_fp8_serialized else None

    if _is_compressed_tensors_quant(quant_config):
        return "int8"
    return None


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        seed = int(getattr(config, "seed", 1234))
        base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        self.ngram_embedding = Qwen4ExpNGramTable(
            ((offset + divisor - 1) // divisor) * divisor,
            self.head_dim,
            quant_kind=_ple_embedding_quant_kind(
                quant_config, f"{prefix}.ngram_embedding"
            ),
            params_dtype=params_dtype or torch.get_default_dtype(),
            divisible_by=divisor,
        )

    def _cuda_ngram_hash(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        ops = _require_ple_ops("qwen_ple_ngram_hash")
        if not input_ids.is_cuda:
            raise RuntimeError("Qwen4Exp PLE n-gram hash requires a CUDA tensor")
        if self.ngram_heads != 16 or self.ngram_size != 3:
            raise ValueError(
                "ixformer PLE n-gram hash requires ngram_heads=16 and ngram_size=3"
            )
        ids = input_ids.reshape(-1)
        _require_contiguous(ids, "ple input_ids")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ple input_ids must be int32 or int64")
        _require_i32_contiguous(query_start_loc, "ple query_start_loc")
        _require_i32_contiguous(ngram_context, "ple ngram_context")
        ctx = ngram_context
        cu = query_start_loc
        if ctx.dim() != 2 or ctx.size(0) != cu.numel() - 1 or ctx.size(1) != 2:
            raise ValueError("ple ngram_context must have shape [num_requests, 2]")
        _require_contiguous(self.layer_multipliers, "ple layer_multipliers")
        _require_contiguous(self.ngram_heads_vocab_sizes, "ple vocab_sizes")
        _require_contiguous(self.ngram_heads_offsets, "ple offsets")
        out = torch.empty(
            (ids.shape[0], self.ngram_heads), device=ids.device, dtype=torch.int64
        )
        ops.qwen_ple_ngram_hash(
            ids,
            cu,
            ctx,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            out,
            self.eos_token_id,
        )
        return out

    def _cuda_ngram_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        table = self.ngram_embedding
        ops = _require_ple_ops("qwen_ple_ngram_lookup")
        if table.quant_kind != "fp8" or table.weight_scale is None:
            raise RuntimeError(
                "ixformer PLE n-gram lookup requires an FP8 table with a scale"
            )
        if not ngram_ids.is_cuda:
            raise RuntimeError("Qwen4Exp PLE n-gram lookup requires a CUDA tensor")
        if not table.weight.is_contiguous():
            raise ValueError("ple n-gram table weight must be contiguous")
        _require_contiguous(ngram_ids, "ple ngram_ids")
        _require_contiguous(table.weight_scale, "ple weight_scale")
        out = torch.empty(
            ngram_ids.shape[0],
            ngram_ids.shape[1] * table.embedding_dim,
            device=ngram_ids.device,
            dtype=torch.bfloat16,
        )
        ops.qwen_ple_ngram_lookup(
            ngram_ids,
            table.weight,
            table.weight_scale,
            out,
            table.shard_indices.org_vocab_start_index,
            table.shard_indices.org_vocab_end_index,
        )
        if table.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        hashed = self._cuda_ngram_hash(input_ids, query_start_loc, ngram_context)
        return self._cuda_ngram_lookup(hashed)

    def _copy_ngram_shard(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        suffix: str,
        destination: torch.Tensor,
        expected_width: int,
    ) -> bool:
        shard_prefix = "ngram_embedding.shard_"
        if not (name.startswith(shard_prefix) and name.endswith(suffix)):
            return False
        shard_text = name[len(shard_prefix) : -len(suffix)]
        if not shard_text.isdigit():
            return False
        shard_index = int(shard_text)
        if shard_index >= self.split_ngram_parts:
            raise ValueError(
                f"PLE embedding shard index {shard_index} exceeds "
                f"split_ngram_parts={self.split_ngram_parts}"
            )
        embedding = self.ngram_embedding
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        checkpoint_start = shard_index * shard_size
        expected_rows = max(
            0,
            min(shard_size, embedding.org_vocab_size - checkpoint_start),
        )
        if loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.unsqueeze(-1)
        expected_shape = (expected_rows, expected_width)
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for PLE embedding shard {shard_index} "
                f"{suffix}: expected {expected_shape}, got "
                f"{tuple(loaded_weight.shape)}"
            )
        copy_ple_embedding_shard_(
            destination,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=embedding.shard_indices.org_vocab_start_index,
            tp_end=embedding.shard_indices.org_vocab_end_index,
        )
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        embedding = self.ngram_embedding

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if self._copy_ngram_shard(
                name,
                loaded_weight,
                ".weight",
                embedding.weight.data,
                embedding.embedding_dim,
            ):
                loaded.add("ngram_embedding.weight")
                continue
            weight_scale = getattr(embedding, "weight_scale", None)
            if weight_scale is not None and self._copy_ngram_shard(
                name,
                loaded_weight,
                ".weight_scale",
                weight_scale.data,
                1,
            ):
                loaded.add("ngram_embedding.weight_scale")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module):
    """Per-Layer Embedding.

    The n-gram token history is derived by ``Qwen4ExpModelState``. The
    dilated-conv carry is stored in a host GDN layer's third mamba state so it
    follows the same block lifetime as the recurrent GDN state.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
            quant_config=quant_config,
            params_dtype=model_config.dtype,
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        # Holds the checkpoint weight only; the convolution itself runs through
        # F.conv1d in _dilated_short_conv, which prepends the per-request carry
        # instead of zero-padding.
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        # Filled by _install_ple_window_states after all decoder layers exist.
        self.state_host: nn.Module | None = None
        self.state_group_row: int | None = None

    def _resolve_conv_state(
        self,
        conv_state: torch.Tensor | None,
        state_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Select an explicit scratch state or this PLE layer's host GDN page."""

        if state_indices is None:
            return conv_state, None
        if conv_state is not None:
            indices = state_indices[0] if state_indices.ndim == 2 else state_indices
            return conv_state, indices
        if self.state_host is None or self.state_group_row is None:
            raise RuntimeError("PLE GDN-page state host was not initialized")
        return self.state_host.ple_conv_window, state_indices[self.state_group_row]

    def _cuda_gate_norm(
        self,
        key: torch.Tensor,
        hidden: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = _require_ple_ops("qwen_ple_gate_norm")
        if not key.is_cuda or key.dtype != torch.bfloat16:
            raise RuntimeError("ixformer PLE gate/norm requires CUDA bfloat16 tensors")
        token_count = key.shape[0]
        key_f = key.reshape(token_count, -1)
        hidden_f = hidden.reshape(token_count, -1)
        value_f = value.reshape(token_count, -1)
        _require_contiguous(key_f, "ple gate key")
        _require_contiguous(hidden_f, "ple gate hidden")
        _require_contiguous(value_f, "ple gate value")
        _require_contiguous(self.norm_key.weight, "ple norm_key")
        _require_contiguous(self.norm_query.weight, "ple norm_query")
        _require_contiguous(self.norm_conv.weight, "ple norm_conv")
        if key_f.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                f"PLE key width must be {self.hc_hidden_size}, got {key_f.shape[-1]}"
            )
        if value_f.shape[-1] != self.hidden_size:
            raise ValueError(
                f"PLE value width must be {self.hidden_size}, got {value_f.shape[-1]}"
            )
        gated = torch.empty_like(key_f)
        conv_in = torch.empty_like(key_f)
        ops.qwen_ple_gate_norm(
            key_f,
            hidden_f,
            value_f,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            gated,
            conv_in,
            float(self.norm_key.eps),
        )
        return gated, conv_in

    def _cuda_dilated_conv(
        self,
        conv_in: torch.Tensor,
        gated: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        ops = _require_ple_ops("qwen_ple_dilated_conv")
        if not conv_in.is_cuda or conv_in.dtype != torch.bfloat16:
            raise RuntimeError(
                "ixformer PLE dilated convolution requires CUDA bfloat16 tensors"
            )
        if (
            self.conv_kernel_size != 4
            or self.short_conv_dilation != 3
            or conv_in.shape[-1] % 2 != 0
            or conv_state.size(-1) < 9
        ):
            raise ValueError(
                "ixformer PLE dilated convolution requires kernel=4, dilation=3, "
                "an even channel width, and state length >= 9"
            )
        if conv_state.shape[-2] != self.hc_hidden_size:
            raise ValueError(
                "PLE conv_state channel dimension mismatch: "
                f"expected {self.hc_hidden_size}, got {conv_state.shape[-2]}"
            )
        x = conv_in
        residual = gated
        _require_contiguous(x, "ple conv_in")
        _require_contiguous(residual, "ple gated")
        weight = self.conv1d.weight.view(x.shape[-1], self.conv_kernel_size)
        if weight.dtype != x.dtype:
            raise ValueError("ple conv weight dtype must match conv_in")
        _require_contiguous(weight, "ple conv weight")
        _require_i32_contiguous(query_start_loc, "ple query_start_loc")
        _require_i32_contiguous(state_indices, "ple state_indices")
        if conv_state.stride(-1) != 1 or conv_state.stride(-2) != conv_state.size(-1):
            raise ValueError(
                "ple conv_state must be contiguous inside each [channels, state] block"
            )
        cu = query_start_loc
        idx = state_indices
        state = conv_state
        # The CUDA kernel skips NULL_BLOCK_ID rows used by FULL-graph padding.
        # Zero first so those rows remain a neutral PLE residual; active rows
        # are overwritten by the kernel.
        out = torch.zeros_like(x)
        ops.qwen_ple_dilated_conv(x, residual, weight, out, state, cu, idx, None, None)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        conv_state, state_indices = self._resolve_conv_state(conv_state, state_indices)
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        gated, conv_in = self._cuda_gate_norm(key, hidden_states, value)
        if conv_state is None or state_indices is None:
            num_reqs = query_start_loc.numel() - 1
            conv_state = conv_in.new_zeros(
                (num_reqs, self.hc_hidden_size, self.conv_state_len)
            )
            state_indices = torch.arange(
                num_reqs,
                device=conv_in.device,
                dtype=torch.int32,
            )
        return self._cuda_dilated_conv(
            conv_in, gated, conv_state, state_indices, query_start_loc
        )


__all__ = [
    "PLEVocabShard",
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpNGramTable",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
    "_ple_embedding_quant_kind",
]
