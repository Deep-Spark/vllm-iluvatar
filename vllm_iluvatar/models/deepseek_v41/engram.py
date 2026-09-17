# SPDX-License-Identifier: Apache-2.0
"""Engram: n-gram hash memory written into the residual stream (DeepSeek-V4.1).
"""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")


_FP8_BLOCK = 32


def _find_next_prime(start: int, seen: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def build_engram_primes(
    layer_ids: list[int], max_ngram_size: int, n_heads: int, engram_vocab_size: int
) -> list[tuple[tuple[int, ...], ...]]:
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], engram_vocab_size - 1
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return primes


def compute_hash_multipliers(
    layer_ids: list[int], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def _first_window_layer():
    try:
        from vllm.config import get_current_vllm_config

        ctx = get_current_vllm_config().compilation_config.static_forward_context
    except Exception:
        return None
    if not isinstance(ctx, dict):
        return None
    fallback = None
    for key, layer in ctx.items():
        name = str(key)
        if "window_kv" not in name:
            continue
        if ".0." in name or name.endswith("layers.0.self_attn.window_kv"):
            return layer
        if fallback is None:
            fallback = layer
    return fallback


def _window_metadata():
    try:
        from vllm.forward_context import get_forward_context

        md_by = get_forward_context().attn_metadata
    except Exception:
        return None
    if not isinstance(md_by, dict):
        return None
    fallback = None
    for key, md in md_by.items():
        if not hasattr(md, "block_table") or not hasattr(md, "slot_mapping"):
            continue
        name = str(key)
        if "window_kv" not in name:
            continue
        if ".0." in name or name.endswith("layers.0.self_attn.window_kv"):
            return md
        if fallback is None:
            fallback = md
    return fallback


def _window_nslots() -> int | None:
    layer = _first_window_layer()
    if layer is None:
        return None
    kv = getattr(layer, "kv_cache", None)
    if isinstance(kv, (list, tuple)):
        kv = kv[0] if kv else None
    if kv is None or not hasattr(kv, "numel") or kv.numel() == 0:
        return None
    return int(kv.shape[0]) * int(getattr(layer, "block_size", 0) or 0)


def _slots_at(md, req: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    bt = md.block_table
    bs = int(md.block_size)
    nblk = bt.size(1)
    # Clamp to the table we actually index -- num_reqs can exceed this row count
    # on a dummy / reshaped batch, and an OOB gather is a device assert with no
    # Python traceback (EngineCore just disappears).
    req = req.to(torch.int64).clamp(0, max(bt.size(0) - 1, 0))
    pos = positions.to(torch.int64)
    blk = pos // bs
    in_range = (pos >= 0) & (blk >= 0) & (blk < nblk)
    blocks = bt[req, blk.clamp(0, max(nblk - 1, 0))].to(torch.int64)
    slots = blocks * bs + pos % bs
    return torch.where(in_range & (blocks > 0), slots, slots.new_full((), -1))


class EngramTokenHistory(nn.Module):

    DEAD = -1

    def __init__(self, token_map: list[int], max_seq_len: int, device) -> None:
        super().__init__()
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "cache",
            torch.zeros(max_seq_len, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.max_seq_len = max_seq_len
        self._slots: torch.Tensor | None = None

    def _compress(self, ids: torch.Tensor) -> torch.Tensor:
        return self.token_map[ids.long().clamp(0, self.token_map.numel() - 1)]

    def update(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        self.cache[positions.long().clamp(0, self.max_seq_len - 1)] = self._compress(
            input_ids
        )

    def lookback(
        self, positions: torch.Tensor, max_ngram_size: int, pad_id: int
    ) -> torch.Tensor:
        """``[T, max_ngram_size]`` compressed ids ending at each position."""
        tokens = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(max_ngram_size):
            source = self.cache[(positions - shift).clamp_min(0)]
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, pad_id, source))
        return torch.stack(tokens, dim=-1)

    def ensure_slots(self, md) -> torch.Tensor | None:
        """Size the slot cache from the bound window KV, not from ``*.max()``."""
        need = _window_nslots()
        if need is None:
            sm = md.slot_mapping
            bt = md.block_table
            if sm is None or sm.numel() == 0 or bt.numel() == 0:
                return None
            cap = min(bt.numel() * max(int(md.block_size), 1), 1 << 20)
            sel = torch.nonzero(sm >= 0, as_tuple=False).reshape(-1)
            if sel.numel() == 0:
                return None
            need = min(int(sm.index_select(0, sel).amax().item()) + 1, cap)
        if need <= 0:
            return None
        if self._slots is None or self._slots.numel() < need:
            grown = torch.zeros(need, dtype=torch.int64, device=self.token_map.device)
            if self._slots is not None:
                grown[: self._slots.numel()] = self._slots
            self._slots = grown
        return self._slots

    def update_slots(self, input_ids: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write this chunk's compressed ids into window slots (official V1)."""
        assert self._slots is not None
        slots = slot_mapping[: input_ids.size(0)].to(torch.int64)
        valid = (slots >= 0) & (slots < self._slots.numel())
        sel = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if sel.numel():
            dst = slots.index_select(0, sel)
            src = self._compress(input_ids[: slots.size(0)].index_select(0, sel))
            self._slots.index_copy_(0, dst, src)

    def lookback_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        md,
        max_ngram_size: int,
        pad_id: int,
    ) -> torch.Tensor:
        """Official lookback: current chunk from ``input_ids``, else slot cache."""
        assert self._slots is not None
        t = min(
            input_ids.size(0),
            positions.size(0),
            md.token_to_req_indices.size(0),
        )
        if t == 0:
            return input_ids.new_empty(0, max_ngram_size)
        ids = input_ids[:t]
        pos = positions[:t].to(torch.int64)
        nreq = max(int(md.num_reqs), 1)
        qsl = md.query_start_loc
        req = md.token_to_req_indices[:t].to(torch.int64).clamp(
            0, min(nreq, qsl.numel()) - 1
        )
        t0 = qsl[req].to(torch.int64).clamp(0, t - 1)
        chunk_start = pos[t0]
        token_idx = torch.arange(t, device=pos.device)
        tokens = []
        blocked = torch.zeros(t, dtype=torch.bool, device=pos.device)
        for shift in range(max_ngram_size):
            look = pos - shift
            in_batch = look >= chunk_start
            batch_idx = (token_idx - shift).clamp_min(0)
            batch_src = self._compress(ids[batch_idx])
            slots = _slots_at(md, req, look)
            valid = (slots >= 0) & (slots < self._slots.numel())
            cache_src = look.new_full(look.shape, self.DEAD)
            sel = torch.nonzero(valid, as_tuple=False).reshape(-1)
            if sel.numel():
                cache_src.index_copy_(
                    0, sel, self._slots.index_select(0, slots.index_select(0, sel))
                )
            source = torch.where(in_batch, batch_src, cache_src)
            blocked = blocked | (pos < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, pad_id, source))
        return torch.stack(tokens, dim=-1)


class EngramEmbedding(nn.Module):
    """Row-sharded fp8 n-gram table on CPU, dequantized on lookup."""

    def __init__(self, num_embeddings: int, dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.part_num_embeddings = (num_embeddings + tp_size - 1) // tp_size
        self.start_idx = tp_rank * self.part_num_embeddings
        self.end_idx = min(self.start_idx + self.part_num_embeddings, num_embeddings)
        self.weight: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self._stage: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _staging(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pinned host buffers for ``n`` gathered rows. Grown, never shrunk."""
        buf = self._stage
        if buf is None or buf[0].shape[0] < n:
            buf = (
                torch.empty((n, self.dim), dtype=torch.uint8).pin_memory(),
                torch.empty((n, self.dim // _FP8_BLOCK), dtype=torch.uint8).pin_memory(),
                torch.empty((n,), dtype=torch.bool).pin_memory(),
            )
            self._stage = buf
        return buf[0][:n], buf[1][:n], buf[2][:n]

    def load_from_checkpoint(self, path: str, prefix: str) -> tuple[int, int]:
        """Copy this rank's row slice out of the safetensors mmap into anonymous RAM.

        The copy is not optional. A row slice is contiguous in the file, so
        ``contiguous()`` on it is a no-op and leaves the table as a file mapping.
        Mapped pages are reclaimable: the two engram tables are 183 GiB against
        473 GiB of page cache shared with the rest of the weights, so lookup rows
        get evicted and every gather faults them back in from disk. That showed up
        as single ``index_select`` calls of 100-200 ms on the prefill critical path.
        Anonymous pages can only leave via swap, which will not happen here.
        """
        import json
        import os

        from safetensors import safe_open

        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        out = []
        for suffix in ("weight", "scale"):
            name = f"{prefix}.{suffix}"
            shard = os.path.join(path, weight_map[name])
            with safe_open(shard, framework="pt", device="cpu") as f:
                tensor = f.get_slice(name)[self.start_idx : self.end_idx]
            setattr(self, suffix, tensor.clone())
            out.append(tensor.numel() * tensor.element_size())
        return out[0], out[1]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [T, n_hash_cols] on device -> [T, n_hash_cols, dim] bf16 on device."""
        assert self.weight is not None and self.scale is not None, (
            "engram table not loaded; call load_from_checkpoint first"
        )
        device = indices.device
        host_idx = indices.to("cpu")
        mask = (host_idx < self.start_idx) | (host_idx >= self.end_idx)
        local = (host_idx - self.start_idx).masked_fill_(mask, 0).view(-1)
        n = local.numel()
        w_buf, s_buf, m_buf = self._staging(n)
        torch.index_select(self.weight.view(torch.uint8), 0, local, out=w_buf)
        torch.index_select(self.scale.view(torch.uint8), 0, local, out=s_buf)
        m_buf.copy_(mask.view(-1))

        values = w_buf.to(device).view(self.weight.dtype).unflatten(0, indices.shape)
        scales = s_buf.to(device).view(self.scale.dtype).unflatten(0, indices.shape)
        mask = m_buf.to(device).unflatten(0, indices.shape)

        # Block dequantization: [..., dim] -> [..., dim // 32, 32] * [..., dim // 32, 1]
        values = values.float().unflatten(-1, (-1, _FP8_BLOCK)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        values = values.masked_fill_(mask.unsqueeze(-1), 0)

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values


def lookback_tokens(
    history: EngramTokenHistory,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    max_ngram_size: int,
    pad_id: int,
) -> torch.Tensor:
    """Compressed n-gram windows for this chunk. Shared by every engram layer."""
    n = min(input_ids.size(0), positions.size(0))
    ids = input_ids[:n]
    pos = positions[:n]
    md = _window_metadata()
    if md is not None and md.slot_mapping is not None and history.ensure_slots(md) is not None:
        history.update_slots(ids, md.slot_mapping)
        out = history.lookback_paged(ids, pos, md, max_ngram_size, pad_id)
        return out
    history.update(ids, pos)
    return history.lookback(pos, max_ngram_size, pad_id)


class IluEngram(nn.Module):
    """One engram layer: hash -> table lookup -> wkv -> gated add."""

    def __init__(
        self,
        config,
        layer_id: int,
        primes: tuple[tuple[int, ...], ...],
        multipliers: torch.Tensor,
        pad_id: int,
        device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self._qk_weight: torch.Tensor | None = None
        self.max_ngram_size = config.engram_max_ngram_size
        self.pad_id = pad_id
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads

        layer_index = list(config.engram_layer_ids).index(layer_id)
        self.embed = EngramEmbedding(
            config.engram_num_embeddings[layer_index], self.head_dim
        )

        flat = [p for per_ngram in primes for p in per_ngram]
        offsets = np.cumsum([0, *flat[:-1]])
        self.register_buffer(
            "primes",
            torch.tensor(primes, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "multipliers", multipliers.to(device), persistent=False
        )
        self.wkv_weight: torch.Tensor | None = None
        self.q_weight: torch.Tensor | None = None
        self.k_weight: torch.Tensor | None = None

    def load_from_checkpoint(self, path: str) -> None:
        """Load the table slice, plus wkv (dequantized to bf16) and q/k weights."""
        import json
        import os

        from safetensors import safe_open

        prefix = f"layers.{self.layer_id}.engram"
        w_bytes, s_bytes = self.embed.load_from_checkpoint(path, f"{prefix}.embed")
        logger.info(
            "engram layer %d: table rows [%d, %d) -> %.2f GiB weight + %.2f GiB scale on CPU",
            self.layer_id,
            self.embed.start_idx,
            self.embed.end_idx,
            w_bytes / 2**30,
            s_bytes / 2**30,
        )

        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        def read(name: str) -> torch.Tensor:
            shard = os.path.join(path, weight_map[f"{prefix}.{name}"])
            with safe_open(shard, framework="pt", device="cpu") as f:
                return f.get_tensor(f"{prefix}.{name}")

        device = self.primes.device
        wkv_q = read("wkv.weight")
        wkv_s = read("wkv.scale")
        self.wkv_weight = _dequant_block_fp8(wkv_q, wkv_s).to(device, torch.bfloat16)
        self.q_weight = read("q_weight").to(device)
        self.k_weight = read("k_weight").to(device)

    def hash_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[T, max_ngram_size]`` compressed ids -> ``[T, n_hash_cols]`` table rows."""
        products = tokens * self.multipliers
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def embed_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(self.hash_ids(tokens))

    def apply(self, hidden_states: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return self._gate(hidden_states, F.linear(rows.flatten(-2), self.wkv_weight))

    def forward(
        self,
        hidden_states: torch.Tensor,
        history: EngramTokenHistory,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = lookback_tokens(history, input_ids, positions, self.max_ngram_size, self.pad_id)
        return self.apply(hidden_states, self.embed_from_tokens(tokens))

    def _gate(self, hidden_states: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if self._qk_weight is None:
            self._qk_weight = (self.q_weight.float() * self.k_weight.float()).contiguous()

        return ixfops.ds4_engram_gate(
            hidden_states.contiguous(), kv, self._qk_weight, self.eps, self.clamp_value
        )


def _dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    w = weight.float()
    s = scale.float()
    s = s.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    return w * s[:out_features, :in_features]
