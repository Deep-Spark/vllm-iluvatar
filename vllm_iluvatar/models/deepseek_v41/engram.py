# SPDX-License-Identifier: Apache-2.0
"""Engram: n-gram hash memory written into the residual stream (DeepSeek-V4.1)."""

from __future__ import annotations

import contextlib
import os

import ixformer.inference.functions as ixfshared
import numpy as np
import torch
from ixformer.inference.functions.models import deepseek_v41 as ixfops
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_iluvatar.models.deepseek_v41.engram_storage import (
    check_shared_group,
    map_shared_checkpoint_rows,
    validate_engram_config,
)

logger = init_logger(f"vllm.{__name__}")


_FP8_BLOCK = 32
# Out-of-range hash ids (and DP padding) look up as zeros.
_DEAD_HASH = -1


def engram_dp_group(config):
    """DP replicas that must split one Engram table. ``None`` when DP size is 1."""
    if config is None or config.dp_shared_memory or not config.embedding_across_dp:
        return None
    try:
        group = get_dp_group()
    except AssertionError:
        return None
    if group is None or group.world_size <= 1:
        return None
    return group


def engram_dp_token_slot(group) -> int:
    """Common token count so every DP rank enters the Engram collective."""
    meta = get_forward_context().dp_metadata
    if meta is None:
        raise RuntimeError(
            "Engram DP lookup needs dp_metadata.num_tokens_across_dp_cpu"
        )
    counts = meta.num_tokens_across_dp_cpu
    start = 0
    stop = group.world_size
    return int(counts[start:stop].max())


# 表本身是 F8_E4M3，它的块缩放是 F8_E8M0。其余几个只是为了以后换量化格式时不用回来改。
_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
}


def _safetensors_entry(path: str, name: str) -> tuple[int, list[int], torch.dtype]:
    """``(该张量数据在文件中的绝对字节偏移, shape, dtype)``.

    safetensors 的头是 8 字节小端长度 + 一段 JSON，JSON 里每个张量的 ``data_offsets``
    是相对数据区起点的。自己解头是为了拿到绝对偏移好做 mmap —— ``safe_open`` 只肯给
    张量，不肯给偏移。
    """
    import json

    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    meta = header[name]
    dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
    if dtype is None:
        raise ValueError(f"{name}: 未支持的 safetensors dtype {meta['dtype']}")
    return 8 + header_len + meta["data_offsets"][0], meta["shape"], dtype


def _map_locked_rows(
    path: str, name: str, start: int, end: int
) -> tuple[torch.Tensor, object]:
    """Read-only shared mmap of checkpoint rows, locked for native UVA lookup.

    Retain the mapping for the lifetime of its device view. PROT_READ avoids
    private COW copies; mlock failure is explicit rather than a slow fallback.
    """
    import ctypes
    import mmap
    import resource

    abs_off, shape, dtype = _safetensors_entry(path, name)
    if not 0 <= start < end <= shape[0]:
        raise ValueError(f"{name}: invalid row interval {start}:{end} for {shape}")
    itemsize = torch.empty((), dtype=dtype).element_size()
    row_elems = 1
    for d in shape[1:]:
        row_elems *= d
    row_bytes = row_elems * itemsize
    nbytes = (end - start) * row_bytes
    abs_off += start * row_bytes

    # mmap 的 offset 必须页对齐，向下取整，差值交给 frombuffer 的 offset 抵掉。
    granularity = mmap.ALLOCATIONGRANULARITY
    map_off = abs_off - (abs_off % granularity)
    skew = abs_off - map_off

    fd = os.open(path, os.O_RDONLY)
    try:
        region = mmap.mmap(
            fd, nbytes + skew, mmap.MAP_SHARED, mmap.PROT_READ, offset=map_off
        )
    finally:
        os.close(fd)

    import warnings

    with warnings.catch_warnings():
        # 只读 buffer 会触发 "PyTorch 不支持不可写张量" 的告警。这里正是要不可写：
        # 表只被 index_select 读，写它反而是 bug。
        warnings.simplefilter("ignore")
        flat = torch.frombuffer(
            region, dtype=dtype, count=nbytes // itemsize, offset=skew
        )
    table = flat.view(end - start, *shape[1:])

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    # 提不动就照原样试，锁不上会走下面的回退
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
    addr = table.data_ptr()
    page = os.sysconf("SC_PAGESIZE")
    lo = addr - (addr % page)
    if libc.mlock(ctypes.c_void_p(lo), ctypes.c_size_t(nbytes + (addr - lo))) != 0:
        errno = ctypes.get_errno()
        raise OSError(
            errno, f"mlock {nbytes / 2**30:.2f} GiB 失败: {os.strerror(errno)}"
        )
    return table, region


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


def _max_batched_tokens():
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config().scheduler_config.max_num_batched_tokens


def _window_metadata():
    for md in (get_forward_context().attn_metadata or {}).values():
        if hasattr(md, "decode_swa_indices"):
            return md
    return None


class EngramEmbedding(nn.Module):
    """Own TP heads in shared host RAM, or TP x DP heads with across-DP lookup."""

    def __init__(self, num_embeddings, dim, head_sizes):
        super().__init__()
        vllm_config = get_current_vllm_config()
        validate_engram_config(vllm_config)
        self.dp_shared_memory = bool(vllm_config.engram_config.dp_shared_memory)
        self._shared_group = None
        if self.dp_shared_memory:
            self._shared_group = get_engram_dp_group()
            if (
                self._shared_group is None
                or self._shared_group.world_size
                != vllm_config.parallel_config.data_parallel_size
            ):
                raise RuntimeError(
                    "Engram dp_shared_memory requires all DP replicas on the same node"
                )
            check_shared_group(self._shared_group)
        self._dp_group = engram_dp_group(vllm_config.engram_config)
        dp_size = self._dp_group.world_size if self._dp_group else 1
        dp_rank = self._dp_group.rank_in_group if self._dp_group else 0
        tp_size = get_tensor_model_parallel_world_size()
        shards = tp_size * dp_size
        rank = get_tensor_model_parallel_rank() * dp_size + dp_rank
        self.n_hash_cols = len(head_sizes)
        self.local_cols = (len(head_sizes) + shards - 1) // shards
        assert (shards - 1) * self.local_cols < len(head_sizes), (
            "Engram ranks without hash heads"
        )
        self.head_start = rank * self.local_cols
        self.start_idx = sum(head_sizes[: self.head_start])
        self.end_idx = sum(head_sizes[: self.head_start + self.local_cols])
        assert self.end_idx <= num_embeddings
        self.dim = dim
        self._regions = []
        self.weight = self.scale = self._dev_weight = self._dev_scale = None
        max_tokens = _max_batched_tokens()
        device = torch.cuda.current_device()
        self._hash = torch.full(
            (max_tokens, self.n_hash_cols), -1, dtype=torch.int32, device=device
        )
        self._padded_hash = torch.full_like(self._hash, -1)
        self._lookup_rows = torch.zeros(
            max_tokens * dp_size,
            self.local_cols,
            dim,
            dtype=torch.bfloat16,
            device=device,
        )

    def load_from_checkpoint(self, path, prefix):
        import json

        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        result = []
        for suffix in ("weight", "scale"):
            name = f"{prefix}.{suffix}"
            checkpoint = os.path.join(path, weight_map[name])
            if self._shared_group is not None:
                table, view, region = map_shared_checkpoint_rows(
                    checkpoint,
                    name,
                    self.start_idx,
                    self.end_idx,
                    self._shared_group,
                    _safetensors_entry,
                )
                setattr(
                    self,
                    f"_dev_{suffix}",
                    view.view(torch.uint8) if suffix == "scale" else view,
                )
            else:
                table, region = _map_locked_rows(
                    checkpoint, name, self.start_idx, self.end_idx
                )
            setattr(self, suffix, table)
            self._regions.append(region)
            result.append(table.nbytes)
        # Register before warmup/capture. No CPU gather fallback in serving.
        if self._shared_group is None:
            self._dev_weight = ixfshared.get_cuda_view_from_cpu_tensor(self.weight)
            self._dev_scale = ixfshared.get_cuda_view_from_cpu_tensor(
                self.scale.view(torch.uint8)
            )
        logger.info(
            "Engram head shard %d:%d rows %d:%d mapped %.2f GiB (%s)",
            self.head_start,
            self.head_start + self.local_cols,
            self.start_idx,
            self.end_idx,
            sum(result) / 2**30,
            "/dev/shm shared across DP" if self.dp_shared_memory else "mmap across-DP",
        )
        return tuple(result)

    def ensure_device_view(self):
        return self._dev_weight is not None

    def lookup_local(self, indices):
        """Lookup local heads without TP communication, suitable for prefetch."""
        assert self._dp_group is None
        hashes = self._hash[: indices.shape[0]]
        hashes.copy_(indices)
        rows = self._lookup_rows[: indices.shape[0]]
        self._lookup(hashes, rows)
        return rows

    def _lookup(self, hashes, rows):
        ixfops.dsv41_engram_lookup_out(
            hashes,
            self._dev_weight,
            self._dev_scale,
            rows,
            self.start_idx,
            self.end_idx,
            self.head_start,
            self.local_cols,
        )

    def gather_tp_rows(self, rows):
        from vllm.distributed import tensor_model_parallel_all_gather

        if get_tensor_model_parallel_world_size() > 1:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.n_hash_cols]

    def forward(self, indices):
        n = indices.shape[0]
        hashes = self._hash[:n]
        hashes.copy_(indices)
        group = self._dp_group
        slot = n
        if group is not None:
            slot = engram_dp_token_slot(group)
            assert slot >= n and slot <= self._padded_hash.shape[0]
            padded = self._padded_hash[:slot]
            padded.fill_(-1)
            padded[:n].copy_(hashes)
            hashes = group.all_gather(padded, dim=0)
        rows = self._lookup_rows[: hashes.shape[0]]
        self._lookup(hashes, rows)
        # Collect heads in TP-major, DP-minor order, matching shard numbering.
        if group is not None:
            rows = group.all_gather(rows, dim=1)
        rows = self.gather_tp_rows(rows)
        start = group.rank_in_group * slot if group is not None else 0
        return rows[start : start + n, : self.n_hash_cols]


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
        prefetch_stream=None,
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
        self.layer_hash_index = layer_index
        self._rows: torch.Tensor | None = None
        self.embed = EngramEmbedding(
            config.engram_num_embeddings[layer_index],
            self.head_dim,
            tuple(p for group in primes for p in group),
        )
        self._prefetch_stream = None
        self._prefetch_done = None
        if self.embed._dp_group is None:
            self._prefetch_stream = prefetch_stream or torch.cuda.Stream(device=device)
            self._prefetch_done = torch.cuda.Event()

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
        self.register_buffer("multipliers", multipliers.to(device), persistent=False)
        self.wkv_weight: torch.Tensor | None = None
        self.wkv_scale: torch.Tensor | None = None
        self.q_weight: torch.Tensor | None = None
        self.k_weight: torch.Tensor | None = None

        # wkv emits hc_mult key streams plus one value stream, all hidden_size wide.
        # The key streams only ever feed sums over hidden, so each rank can keep just
        # its slice of them and let an all-reduce of two scalars per (token, stream)
        # finish the job. value has to stay replicated: it is added straight into the
        # full-width residual, and sharding it would cost an all-gather of the whole
        # hidden state instead. That takes wkv from (hc_mult + 1) * hidden rows down to
        # hc_mult * hidden / tp + hidden.
        tp_size = get_tensor_model_parallel_world_size()
        self.key_hidden = self.hidden_size // tp_size
        self.key_offset = get_tensor_model_parallel_rank() * self.key_hidden
        # The kernel reads the slice two elements at a time.
        self._shard_keys = (
            tp_size > 1 and self.hidden_size % tp_size == 0 and self.key_hidden % 2 == 0
        )
        if not self._shard_keys:
            self.key_hidden = self.hidden_size
            self.key_offset = 0

    def load_from_checkpoint(self, path: str) -> None:
        """Load the table slice, plus the wkv and q/k weights.

        ``wkv`` comes in one of two records and the checkpoint decides which:
        block-scaled FP8 (a companion ``wkv.scale``) or plain BF16. The
        Iluvatar-quantized V4.1 checkpoints leave engram out of the quantized
        set entirely -- ``re:.*\\.engram\\..*`` sits in the config's ``ignore``
        list -- so there the weight arrives BF16 with no scale at all, and
        :meth:`apply` must not reach for an FP8 kernel. Keyed off the file
        rather than off a runtime flag: the record is a property of the
        checkpoint, and a flag could disagree with it.
        """
        import json
        import os

        from safetensors import safe_open

        prefix = f"layers.{self.layer_id}.engram"
        w_bytes, s_bytes = self.embed.load_from_checkpoint(path, f"{prefix}.embed")
        if self.embed.ensure_device_view():
            logger.info(
                "engram layer %d: table registered for device lookup (rows [%d, %d))",
                self.layer_id,
                self.embed.start_idx,
                self.embed.end_idx,
            )
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
        wkv_s = read("wkv.scale") if f"{prefix}.wkv.scale" in weight_map else None
        if self._shard_keys:
            # Slice on the host so the device never holds the full-width weight.
            n = self.hc_mult * self.hidden_size
            keys_q = wkv_q[:n].unflatten(0, (self.hc_mult, self.hidden_size))[
                :, self.key_offset : self.key_offset + self.key_hidden
            ]
            wkv_q = torch.cat([keys_q.flatten(0, 1), wkv_q[n:]])
            if wkv_s is not None:
                n_s = n // _FP8_BLOCK
                keys_s = wkv_s[:n_s].unflatten(
                    0, (self.hc_mult, self.hidden_size // _FP8_BLOCK)
                )[
                    :,
                    self.key_offset // _FP8_BLOCK : (self.key_offset + self.key_hidden)
                    // _FP8_BLOCK,
                ]
                wkv_s = torch.cat([keys_s.flatten(0, 1), wkv_s[n_s:]])
        if wkv_s is None:
            self.wkv_weight = wkv_q.contiguous().to(device)
            self.wkv_scale = None
        else:
            self.wkv_weight, self.wkv_scale = (
                ixfshared.fp8_weight_block_to_channel_requant(
                    wkv_q.contiguous().to(device),
                    wkv_s.contiguous().to(device),
                    _FP8_BLOCK,
                    _FP8_BLOCK,
                )
            )
        self.q_weight = read("q_weight").to(device)
        self.k_weight = read("k_weight").to(device)
        self._qk_weight = (self.q_weight.float() * self.k_weight.float()).contiguous()
        self._kv = torch.zeros(
            _max_batched_tokens(),
            self.wkv_weight.shape[0],
            dtype=torch.bfloat16,
            device=device,
        )

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Prefetch this replica's heads; across-DP keeps synchronous collectives."""
        self._hash_ids = hash_ids
        if self._prefetch_stream is None:
            self._rows = self.embed(hash_ids)
        else:
            self._rows = self.embed._lookup_rows[: hash_ids.shape[0]]
            self._start_prefetch(hash_ids)

    @eager_break_during_capture
    def _start_prefetch(self, hash_ids):
        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        # Hashes alias NgramHashState's persistent output, retained by the model.
        with torch.cuda.stream(stream):
            self.embed.lookup_local(hash_ids)
            self._prefetch_done.record(stream)

    @eager_break_during_capture
    def _finish_prefetch(self):
        torch.cuda.current_stream().wait_event(self._prefetch_done)

    def apply(
        self, hidden_states: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The table lookup already hands back BF16 rows, so an unquantized wkv
        # needs no quantization step at all -- one plain GEMM and nothing FP8
        # anywhere on the path. See ``load_from_checkpoint`` for which record
        # is in play.
        if self.wkv_scale is None:
            x = rows.flatten(-2)
            if (
                os.getenv("VLLM_ILU_ENGRAM_LINEAR", "0") == "1"
                and 0 < x.shape[0] <= 32
                and x.dtype == torch.bfloat16
                and self.wkv_weight.dtype == torch.bfloat16
            ):
                kv = ixfshared.linear(x, self.wkv_weight, backend="ixinfer")
            else:
                kv = torch.matmul(x, self.wkv_weight.t())
            return self._gate(hidden_states, kv)

        x, x_scale = ixfshared.fp8_per_token_dynamic_scaled(rows.flatten(-2))
        kv = self._kv[: x.shape[0]]
        ixfshared.fp8_per_channel_mm(
            x, self.wkv_weight, x_scale, self.wkv_scale, kv, None, backend="ixmm"
        )
        return self._gate(hidden_states, kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rows = self._rows if self._rows is not None else self.embed(hash_ids)
        if self._rows is not None and self._prefetch_done is not None:
            self._finish_prefetch()
            rows = self.embed.gather_tp_rows(rows)
        self._rows = None
        if token_mask is not None:
            rows = rows.masked_fill(~token_mask.view(-1, 1, 1), 0)
        return self.apply(hidden_states, rows)

    def _gate(
        self, hidden_states: torch.Tensor, kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._qk_weight is None:
            self._qk_weight = (
                self.q_weight.float() * self.k_weight.float()
            ).contiguous()

        x = hidden_states.contiguous()
        if not self._shard_keys:
            return (
                ixfops.dsv41_engram_gate(
                    x, kv, self._qk_weight, self.eps, self.clamp_value
                ),
                None,
            )
        # qk_weight stays full width; the kernel offsets into it by key_offset.
        sums = ixfops.dsv41_engram_gate_reduce(x, kv, self._qk_weight, self.key_offset)
        sums = tensor_model_parallel_all_reduce(sums)
        if x.shape[0] <= 256:
            return ixfops.dsv41_engram_gate_apply_with_sqrsum(
                x, kv, sums, self.eps, self.clamp_value
            )
        return (
            ixfops.dsv41_engram_gate_apply(
                x, kv, sums, self.eps, self.clamp_value
            ),
            None,
        )


class EngramLayout:
    """Bucket layout of the n-gram hash tables. Same fields as upstream."""

    def __init__(self, config) -> None:
        self.layer_ids = tuple(config.engram_layer_ids)
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.pad_token_id = config.engram_pad_token_id
        self.primes = tuple(
            build_engram_primes(
                list(self.layer_ids),
                self.max_ngram_size,
                self.n_heads,
                config.engram_vocab_size,
            )
        )
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, config) -> EngramLayout | None:
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


class NgramHashState(nn.Module):
    def __init__(
        self, layout, swa_cache_module, token_map, multipliers, pad_id, device
    ):
        super().__init__()
        self.swa_cache_module = swa_cache_module
        self.block_size = int(swa_cache_module.block_size)
        self.pad_id = pad_id
        self.lookback_depth = layout.max_ngram_size - 1
        self.token_map = torch.tensor(token_map, dtype=torch.int32, device=device)
        self.multipliers = multipliers.to(device=device, dtype=torch.int64)
        flat = [[p for order in layer for p in order] for layer in layout.primes]
        self.primes = torch.tensor(flat, dtype=torch.int64, device=device)
        offsets = [list(np.cumsum([0, *row[:-1]])) for row in flat]
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        self._out = torch.full(
            (_max_batched_tokens(), len(flat), layout.n_hash_cols),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._cache = None
        self._kv_cache_ref = None

    def ensure_cache(self):
        import weakref

        kv = self.swa_cache_module.kv_cache
        if kv.numel() == 0:
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv:
            return True
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram hash cache must be initialized before capture")
        # Backend negotiation can change the page size after model construction.
        # Follow the bound SWA cache, rather than the provisional cache config.
        self.block_size = int(kv.shape[1])
        self._cache = torch.full(
            (kv.shape[0] * self.block_size,), -1, dtype=torch.int32, device=kv.device
        )
        self._kv_cache_ref = weakref.ref(kv)
        return True

    def forward(self, input_ids, positions, md, lookback_token_ids=None):
        if self.swa_cache_module.bounded_replay and lookback_token_ids is None:
            raise RuntimeError("SWA bounded replay requires runner Engram lookback")
        out = self._out[: input_ids.shape[0]]
        ixfops.dsv41_engram_hash_out(
            input_ids,
            self.token_map,
            positions,
            md.token_to_req_indices[: input_ids.shape[0]],
            md.query_start_loc,
            md.block_table,
            md.slot_mapping[: input_ids.shape[0]],
            self._cache,
            self.multipliers,
            self.primes,
            self.offsets,
            out,
            self.pad_id,
            -1,
            md.block_size,
            lookback_token_ids=lookback_token_ids,
        )
        return out
