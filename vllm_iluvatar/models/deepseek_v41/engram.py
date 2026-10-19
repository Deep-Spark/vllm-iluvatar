# SPDX-License-Identifier: Apache-2.0
"""Engram: n-gram hash memory written into the residual stream (DeepSeek-V4.1)."""

from __future__ import annotations

import contextlib
import os

import ixformer.inference.functions as ixfops
import numpy as np
import torch
from torch import nn
from vllm.distributed import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")


_FP8_BLOCK = 32
# Out-of-range hash ids (and DP padding) look up as zeros.
_DEAD_HASH = -1


def engram_dp_group():
    """DP replicas that must split one Engram table. ``None`` when DP size is 1."""
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
    """把 ``name`` 的行区间 ``[start, end)`` 以只读共享方式映射进来并锁住。

    返回 ``(张量, mmap 对象)``；mmap 对象必须被调用方持有，否则映射会随垃圾回收消失。

    为什么是这个组合，三点都是实测定下来的：
      * ``MAP_SHARED`` + ``PROT_READ``：所有映射同一文件区间的进程共享同一份物理页。
        表只按 TP 切，所以 DP 下同一段会被多个 rank 用到 —— 共享之后这些副本是白拿的
        （实测第二个进程锁同一段 11.44 GiB，系统可用内存只动了 0.02 GiB）。
      * ``mlock``：不锁的话页会被挤出去，每次 gather 都从磁盘换页，实测单次 9-240 ms，
        直接砸在 prefill 关键路径上。锁上之后是 0.12-0.16 ms。
      * 必须是只读映射。``safe_open`` 自己按 ``rw-p``（可写私有）映射文件，在那种映射
        上 ``mlock`` 会提前触发 COW，smaps 记成 Private_Dirty —— 等于又复制一份，
        共享一点没落着。
    """
    import ctypes
    import mmap
    import resource

    abs_off, shape, dtype = _safetensors_entry(path, name)
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


def _max_batched_tokens() -> int:
    """Token ceiling for one forward, for sizing host staging up front.

    Model construction runs inside ``set_current_vllm_config``, so the
    scheduler config is the real one here (unlike ``_first_window_layer``,
    which needs a registry that is only populated later). 0 if it is not.
    """
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)
    except Exception:
        return 0


def _first_window_layer():
    # The forward context is the only registry that is populated here. Reading
    # it off get_current_vllm_config() instead silently yields an empty dict:
    # outside a set_current_vllm_config scope that call hands back a fresh
    # default config, so every lookup missed and ensure_slots fell through to
    # its blocking *.max() path on each call.
    ctx = None
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context().no_compile_layers
    except Exception:
        ctx = None
    if not isinstance(ctx, dict) or not ctx:
        # No forward in flight (model init): the config still holds the registry.
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
        if not _is_window_key(name):
            continue
        if ".layers.0." in name:
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
        if not _is_window_key(name):
            continue
        if ".layers.0." in name:
            return md
        if fallback is None:
            fallback = md
    return fallback


def _is_window_key(name: str) -> bool:
    return "swa_cache" in name or "window_kv" in name


def _num_reqs(md) -> int:
    n = getattr(md, "num_reqs", None)
    if n is None:
        n = int(getattr(md, "num_decodes", 0)) + int(getattr(md, "num_prefills", 0))
    return max(int(n), 1)


def _window_nslots() -> int | None:
    """How many slots the window cache can address, from host-side shapes.

    A slot is ``block_id * block_size + offset``. That is bounded by the
    cache's page count, which lives on the module after ``bind_kv_cache``.
    Falling back to ``slot_mapping.max()`` syncs the device every prefill.
    """
    layer = _first_window_layer()
    if layer is None:
        return None
    kv = getattr(layer, "kv_cache", None)
    if isinstance(kv, (list, tuple)):
        kv = kv[0] if kv else None
    block_size = int(getattr(layer, "block_size", 0) or 0)
    if kv is not None and hasattr(kv, "numel") and kv.numel() > 0 and block_size > 0:
        return int(kv.shape[0]) * block_size
    return None


def _pool_nslots(block_size: int) -> int | None:
    """Page-pool bound when the window cache tensor is not bound yet."""
    if block_size <= 0:
        return None
    try:
        from vllm.config import get_current_vllm_config

        nblk = int(get_current_vllm_config().cache_config.num_gpu_blocks or 0)
    except Exception:
        return None
    if nblk <= 0:
        return None
    return nblk * block_size


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
        # Usable length of ``_slots``, which carries one extra trailing sink.
        self._nslots = 0

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
        need = _window_nslots()
        if need is None:
            sm = getattr(md, "slot_mapping", None)
            bt = getattr(md, "block_table", None)
            if sm is None or bt is None or sm.numel() == 0 or bt.numel() == 0:
                return None
            need = _pool_nslots(max(int(getattr(md, "block_size", 0) or 0), 1))
        if not need:
            return None
        # One trailing slot past ``need`` is a write sink: ``update_slots`` aims
        # the rows whose slot is out of range at it rather than selecting the
        # valid ones, which needs ``nonzero`` -- a host round trip, and rejected
        # outright while a graph captures. Nothing ever reads it.
        if self._slots is None or self._slots.numel() < need + 1:
            grown = torch.zeros(
                need + 1, dtype=torch.int64, device=self.token_map.device
            )
            if self._slots is not None:
                keep = min(self._nslots, need)
                grown[:keep] = self._slots[:keep]
            self._slots = grown
        self._nslots = need
        return self._slots

    def update_slots(self, input_ids: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write this chunk's compressed ids into window slots (official V1)."""
        assert self._slots is not None
        slots = slot_mapping[: input_ids.size(0)].to(torch.int64)
        # Write every row, sending the out-of-range ones to the sink
        # ``ensure_slots`` reserves at the end. The mirror of the read path's
        # clamp in ``lookback_paged``, and for the same reason.
        sink = self._nslots
        dst = torch.where(
            (slots >= 0) & (slots < sink), slots, slots.new_full((), sink)
        )
        src = self._compress(input_ids[: slots.size(0)])
        self._slots.index_copy_(0, dst, src)

    def write_slots_kernel(
        self, input_ids: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        assert self._slots is not None
        n = input_ids.size(0)
        torch.ops._C.engram_write_hash_cache(
            input_ids[:n],
            self.token_map,
            slot_mapping[:n],
            self._slots,
            int(self._nslots),
        )

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
        req = (
            md.token_to_req_indices[:t]
            .to(torch.int64)
            .clamp(0, min(nreq, qsl.numel()) - 1)
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
            nslots = self._nslots
            valid = (slots >= 0) & (slots < nslots)
            # Gather every row and mask afterwards. Selecting with nonzero would
            # read the count back to the host, stalling on the whole queued
            # stream once per shift. Clamping has to bound both ends: slots can
            # also run past the cache, and an out-of-range index_select is a
            # device assert rather than an exception.
            gathered = self._slots.index_select(0, slots.clamp(0, max(nslots - 1, 0)))
            cache_src = torch.where(valid, gathered, look.new_full((), self.DEAD))
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
        # DP=1 keeps the TP-only row split. DP>1 shards the same table across
        # TP x DP so each replica owns a disjoint row range; the lookup then
        # reduces each token's rows to its owner. TP-major rank matches the
        # official head-shard order.
        self._dp_group = engram_dp_group()
        dp_size = self._dp_group.world_size if self._dp_group is not None else 1
        dp_rank = self._dp_group.rank_in_group if self._dp_group is not None else 0
        num_shards = tp_size * dp_size
        shard_rank = tp_rank * dp_size + dp_rank
        self.part_num_embeddings = (num_embeddings + num_shards - 1) // num_shards
        self.start_idx = shard_rank * self.part_num_embeddings
        self.end_idx = min(self.start_idx + self.part_num_embeddings, num_embeddings)
        if self._dp_group is not None:
            logger.info(
                "engram table sharded across TP %d x DP %d, this rank rows [%d, %d)",
                tp_size,
                dp_size,
                self.start_idx,
                self.end_idx,
            )
        self.weight: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self._dev_weight: torch.Tensor | None = None
        self._dev_scale: torch.Tensor | None = None
        self._uva_failed = False
        self._stage: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # 映射对象要一直被持有：它一被回收，映射就没了，表底下的指针立刻失效。
        self._regions: list[object] = []

    def reserve_staging(self, n: int) -> None:
        """Pin the staging buffers at their ceiling, before any graph captures.

        Growing them mid-capture calls ``cudaHostRegister`` again, which the
        driver rejects (``ixErrorHostMemoryAlreadyRegistered`` at
        ``capture_end``), so the second capture size fails once the first has
        already sized the buffers.
        """
        if n > 0:
            self._staging(n)

    def _staging(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pinned host buffers for ``n`` gathered rows. Grown, never shrunk."""
        buf = self._stage
        if buf is None or buf[0].shape[0] < n:
            # ``device`` is explicit because model construction runs under a
            # CUDA default device, where an unqualified ``empty`` lands on the
            # GPU and cannot be pinned at all.
            buf = (
                torch.empty(
                    (n, self.dim), dtype=torch.uint8, device="cpu"
                ).pin_memory(),
                torch.empty(
                    (n, self.dim // _FP8_BLOCK), dtype=torch.uint8, device="cpu"
                ).pin_memory(),
                torch.empty((n,), dtype=torch.bool, device="cpu").pin_memory(),
            )
            self._stage = buf
        return buf[0][:n], buf[1][:n], buf[2][:n]

    def load_from_checkpoint(self, path: str, prefix: str) -> tuple[int, int]:
        """Map this rank's row slice read-only-shared and lock it resident.

        The rows have to stay resident: a plain file mapping gets its lookup rows
        evicted -- the two tables are 183 GiB against a page cache shared with the
        rest of the weights -- and every gather then faults them back from disk, at
        9-240 ms per ``index_select`` on the prefill critical path. ``mlock`` fixes
        that at 0.12-0.16 ms.
        ``safe_open`` cannot be used for this: it maps the file ``rw-p``, and
        ``mlock`` on a writable private mapping breaks COW up front, which lands
        right back on a private copy per process.
        """
        import json

        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        out = []
        for suffix in ("weight", "scale"):
            name = f"{prefix}.{suffix}"
            shard = os.path.join(path, weight_map[name])
            table, region = self._load_rows(shard, name)
            setattr(self, suffix, table)
            self._regions.append(region)
            out.append(table.numel() * table.element_size())
        return out[0], out[1]

    def _load_rows(self, shard: str, name: str) -> tuple[torch.Tensor, object]:
        """Shared locked mapping, or an anonymous copy when locking is not allowed.

        Locking needs ``CAP_IPC_LOCK`` or a raised ``RLIMIT_MEMLOCK``. Where that is
        refused, an unlocked mapping is not an option -- it is the 9-240 ms gather --
        so fall back to what this used to do and copy into anonymous memory, which
        only swap can reclaim. That keeps single-instance TP working anywhere and
        costs host RAM only under DP.
        """
        try:
            return _map_locked_rows(shard, name, self.start_idx, self.end_idx)
        except OSError as exc:
            from safetensors import safe_open

            logger.warning(
                "engram %s: 共享映射锁不住内存（%s），回退到匿名副本。"
                "单实例 TP 不受影响；DP 下每个副本各占一份表，注意主机内存。",
                name,
                exc,
            )
            with safe_open(shard, framework="pt", device="cpu") as f:
                tensor = f.get_slice(name)[self.start_idx : self.end_idx]
            return tensor.clone(), None

    def ensure_device_view(self) -> bool:
        """Register the mmap once so lookup reads it through UVA. No copy."""
        if self._dev_weight is not None:
            return True
        if self._uva_failed or self.weight is None or self.scale is None:
            return False
        try:
            weight = self.weight.view(torch.uint8)
            scale = self.scale.view(torch.uint8)
            if not weight.is_contiguous() or not scale.is_contiguous():
                raise RuntimeError("engram table is not contiguous")
            dev_weight = torch.ops._C.engram_register_host(weight)
            dev_scale = torch.ops._C.engram_register_host(scale)
        except Exception as exc:
            logger.warning(
                "engram 表 cudaHostRegister 失败，lookup 继续走 CPU gather：%s", exc
            )
            self._uva_failed = True
            self._dev_weight = None
            self._dev_scale = None
            return False
        self._dev_weight = dev_weight
        self._dev_scale = dev_scale
        return True

    def _gather_cpu(self, indices: torch.Tensor) -> torch.Tensor:
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
        values = values.float().unflatten(
            -1, (-1, _FP8_BLOCK)
        ) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        return values.masked_fill_(mask.unsqueeze(-1), 0)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(
                (*indices.shape, self.dim), dtype=torch.bfloat16, device=indices.device
            )
        if self.ensure_device_view():
            return torch.ops._C.engram_lookup_fp8(
                indices,
                self._dev_weight,
                self._dev_scale,
                int(self.start_idx),
                int(self.end_idx),
            )
        return self._gather_cpu(indices)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.weight is not None and self.scale is not None, (
            "engram table not loaded; call load_from_checkpoint first"
        )
        group = self._dp_group
        n_local = indices.shape[0]
        if group is None:
            values = self._lookup(indices)
            if get_tensor_model_parallel_world_size() > 1:
                values = tensor_model_parallel_all_reduce(values)
            return values

        slot = engram_dp_token_slot(group)
        if n_local > slot:
            raise ValueError(
                f"Engram token count {n_local} exceeds the DP token slot {slot}"
            )
        if n_local < slot:
            pad = indices.new_full((slot - n_local, indices.shape[1]), _DEAD_HASH)
            indices = torch.cat((indices, pad))
        gathered = group.all_gather(indices, dim=0)
        values = self._lookup(gathered)
        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_reduce(values)
        # Only this DP rank's token rows are consumed. Reduce directly to
        # their owner instead of materializing every rank's reduced rows.
        values = group.reduce_scatter(values, dim=0)
        return values[:n_local]


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
    if (
        md is not None
        and md.slot_mapping is not None
        and history.ensure_slots(md) is not None
    ):
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
        self.layer_hash_index = layer_index
        self._rows: torch.Tensor | None = None
        self._token_mask: torch.Tensor | None = None
        self.embed = EngramEmbedding(
            config.engram_num_embeddings[layer_index], self.head_dim
        )
        self.embed.reserve_staging(_max_batched_tokens() * self.n_hash_cols)

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
                ixfops.fp8_weight_block_to_channel_requant(
                    wkv_q.contiguous().to(device),
                    wkv_s.contiguous().to(device),
                    _FP8_BLOCK,
                    _FP8_BLOCK,
                )
            )
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

    def hash_from_metadata(
        self,
        history: EngramTokenHistory,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        md,
    ) -> torch.Tensor:
        """Paged lookback and n-gram hash in one kernel. Slot cache already written."""
        t = min(
            input_ids.size(0),
            positions.size(0),
            md.token_to_req_indices.size(0),
        )
        if t == 0:
            return torch.empty(
                (0, self.n_hash_cols), dtype=torch.int64, device=input_ids.device
            )
        return torch.ops._C.engram_ngram_hash(
            input_ids[:t],
            positions[:t],
            history.token_map,
            history._slots,
            md.block_table,
            md.token_to_req_indices[:t],
            md.query_start_loc,
            self.multipliers,
            self.primes,
            self.offsets,
            int(self.pad_id),
            int(history._nslots),
            int(md.block_size),
            _num_reqs(md),
            True,
        )

    def embed_from_metadata(
        self,
        history: EngramTokenHistory,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        md,
    ) -> torch.Tensor:
        return self.embed(self.hash_from_metadata(history, input_ids, positions, md))

    def prepare_embeddings(
        self, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> None:
        """Look up this layer's rows before the decoder loop, as official does.

        ``token_mask`` (True = keep) rides along to the decoder layer, which
        hands it back to :meth:`forward`; image-span rows are zeroed there.
        """
        self._hash_ids = hash_ids
        self._token_mask = token_mask
        self._rows = self.embed(hash_ids)

    def apply(self, hidden_states: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
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
                kv = ixfops.linear(x, self.wkv_weight, backend="ixinfer")
            else:
                kv = torch.matmul(x, self.wkv_weight.t())
            return self._gate(hidden_states, kv)

        x, x_scale = ixfops.fp8_per_token_dynamic_scaled(rows.flatten(-2))
        kv = torch.empty(
            (x.shape[0], self.wkv_weight.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        ixfops.fp8_per_channel_mm(
            x, self.wkv_weight, x_scale, self.wkv_scale, kv, None, backend="ixmm"
        )
        return self._gate(hidden_states, kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = self._rows if self._rows is not None else self.embed(hash_ids)
        self._rows = None
        if token_mask is not None:
            # Community zeroes the gate inside its post-wkv kernel; the
            # ixformer gate takes no mask, so zero the table rows instead.
            # ``wkv`` has no bias, so those rows give kv == 0, and the gate
            # adds ``gate * value`` with ``value == 0`` -- the position passes
            # through untouched, same as community.
            rows = rows.masked_fill(~token_mask.view(-1, 1, 1), 0)
        return self.apply(hidden_states, rows)

    def forward_from_history(
        self,
        hidden_states: torch.Tensor,
        history: EngramTokenHistory,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        md = _window_metadata()
        if (
            md is not None
            and md.slot_mapping is not None
            and history.ensure_slots(md) is not None
        ):
            n = min(input_ids.size(0), positions.size(0))
            history.write_slots_kernel(input_ids[:n], md.slot_mapping)
            rows = self.embed_from_metadata(history, input_ids, positions, md)
        else:
            tokens = lookback_tokens(
                history, input_ids, positions, self.max_ngram_size, self.pad_id
            )
            rows = self.embed_from_tokens(tokens)
        return self.apply(hidden_states, rows)

    def _gate(self, hidden_states: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if self._qk_weight is None:
            self._qk_weight = (
                self.q_weight.float() * self.k_weight.float()
            ).contiguous()

        x = hidden_states.contiguous()
        if not self._shard_keys:
            return ixfops.ds4_engram_gate(
                x, kv, self._qk_weight, self.eps, self.clamp_value
            )
        # qk_weight stays full width; the kernel offsets into it by key_offset.
        sums = ixfops.ds4_engram_gate_reduce(x, kv, self._qk_weight, self.key_offset)
        sums = tensor_model_parallel_all_reduce(sums)
        return ixfops.ds4_engram_gate_apply(x, kv, sums, self.eps, self.clamp_value)


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
    """One hash for the batch, shared by every engram layer.

    ``ensure_cache`` sizes the slot store from the bound SWA KV cache.
    There is no device reduction and no host sync.
    """

    def __init__(
        self,
        layout: EngramLayout,
        swa_cache_module,
        token_map,
        multipliers,
        pad_id,
        device,
        *,
        lookback_is_prompt_only: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.block_size = int(swa_cache_module.block_size)
        self.lookback_depth = layout.max_ngram_size - 1
        self.lookback_is_prompt_only = lookback_is_prompt_only
        self.pad_id = int(pad_id)
        self._cache: torch.Tensor | None = None
        self._nslots = 0
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            multipliers.to(device=device, dtype=torch.int64),
            persistent=False,
        )
        prime_rows = []
        offset_rows = []
        for primes in layout.primes:
            flat = [p for group in primes for p in group]
            prime_rows.append(torch.tensor(primes, dtype=torch.int64))
            offset_rows.append(
                torch.tensor(np.cumsum([0, *flat[:-1]]), dtype=torch.int64)
            )
        self.register_buffer(
            "primes", torch.stack(prime_rows).to(device), persistent=False
        )
        self.register_buffer(
            "offsets", torch.stack(offset_rows).to(device), persistent=False
        )

    def ensure_cache(self) -> bool:
        kv = self.swa_cache_module.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0] if kv else None
        if kv is None or kv.numel() == 0:
            self._cache = None
            self._nslots = 0
            return False
        # The KV coordinator can enlarge a group's pages to equalize page
        # sizes. Use the bound cache geometry, matching its slot mapping and
        # block table, rather than the layer's original KVCacheSpec.
        self.block_size = int(kv.shape[1])
        nslots = int(kv.shape[0]) * self.block_size
        if self._cache is None or self._cache.numel() < nslots + 1:
            self._cache = torch.zeros(nslots + 1, dtype=torch.int64, device=kv.device)
        self._nslots = nslots
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        token_to_req: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        # The dead masks are community's way of telling the hash kernels which
        # positions take no part in an n-gram (the image spans). Here that is
        # routed through ``token_map`` instead: the image token maps to the
        # kernels' dead id, so ``engram_write_hash_cache`` stores it in the
        # token cache and ``engram_ngram_hash`` blocks every n-gram reaching
        # one, whether the position comes from this batch, the lookback window
        # or a page written by an earlier step. The masks are therefore
        # redundant, and the signature keeps them only to match community's.
        del dead_mask, lookback_dead_mask
        assert self._cache is not None and slot_mapping is not None
        assert block_table is not None
        n = min(input_ids.size(0), positions.size(0), token_to_req.size(0))
        torch.ops._C.engram_write_hash_cache(
            input_ids[:n],
            self.token_map,
            slot_mapping[:n],
            self._cache,
            int(self._nslots),
        )
        cols = []
        for i in range(self.multipliers.shape[0]):
            cols.append(
                torch.ops._C.engram_ngram_hash(
                    input_ids[:n],
                    positions[:n],
                    self.token_map,
                    self._cache,
                    block_table,
                    token_to_req[:n],
                    query_start_loc,
                    self.multipliers[i],
                    self.primes[i],
                    self.offsets[i],
                    self.pad_id,
                    int(self._nslots),
                    self.block_size,
                    int(num_reqs),
                    True,
                    lookback_token_ids,
                    self.lookback_is_prompt_only,
                )
            )
        return torch.stack(cols, dim=1)
