"""Tensor-parallel topology and rank mapping for push transfers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology


@dataclass(frozen=True)
class TensorParallelMapping:
    """一个 local TP worker 连接 remote engine 时的目标 rank 集合。"""

    # 当前 worker 的本地 TP rank / world size。
    local_tp_rank: int
    local_tp_size: int
    # 对端 TP world size。
    remote_tp_size: int
    # 当前 local rank 负责的第一个 remote TP rank。
    first_remote_tp_rank: int
    # 当前 local rank 需要连接的连续 remote rank 数。
    # remote TP <= local TP 时为 1；remote TP 更大时可能大于 1。
    remote_rank_count: int
    # 两端 TP world size 的无符号整数倍。
    ratio: int

    @property
    def remote_tp_ranks(self) -> range:
        """当前 local worker 应建立 handshake/WRITE 的 remote TP ranks。"""

        return range(
            self.first_remote_tp_rank,
            self.first_remote_tp_rank + self.remote_rank_count,
        )


@dataclass
class NixlTpKVTopology(TransferTopology):
    """Iluvatar Push 使用的 TP 拓扑。

    父类保存 ``tp_rank/tp_size/block_size`` 等本地信息；本类补充 Push
    所需的“本地写到哪些远端 rank”。P/D 的 block size 在握手时强制相等，
    这里不再处理 block size 比例。
    """

    def tp_ratio(self, remote_tp_size: int) -> int:
        """返回两端 TP size 的整数倍，不表达方向。

        例：local TP=8、remote TP=2 和 local TP=2、remote TP=8
        都返回 4；具体方向由 ``self.tp_size < remote_tp_size`` 判断。
        """

        if remote_tp_size <= 0:
            raise ValueError("remote_tp_size must be positive")
        larger: int = max(self.tp_size, remote_tp_size)
        smaller: int = min(self.tp_size, remote_tp_size)
        if larger % smaller:
            raise ValueError(
                "P/D tensor parallel sizes must have an integral ratio, got "
                f"local={self.tp_size}, remote={remote_tp_size}"
            )
        return larger // smaller

    def get_target_remote_rank(self, remote_tp_size: int) -> int:
        """返回当前 local rank 对应的第一个 remote rank。

        - local TP < remote TP：一个 local writer 覆盖连续多个 remote ranks；
        - local TP >= remote TP：多个 local writers 映射到同一 remote rank。
        """

        ratio: int = self.tp_ratio(remote_tp_size)
        if self.tp_size < remote_tp_size:
            return self.tp_rank * ratio
        return self.tp_rank // ratio

    def build_mapping(self, remote_tp_size: int) -> TensorParallelMapping:
        """一次性生成 handshake 循环需要的完整 TP mapping。"""

        ratio: int = self.tp_ratio(remote_tp_size)
        first_remote: int = self.get_target_remote_rank(remote_tp_size)
        remote_count: int = max(1, remote_tp_size // self.tp_size)
        return TensorParallelMapping(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_size=remote_tp_size,
            first_remote_tp_rank=first_remote,
            remote_rank_count=remote_count,
            ratio=ratio,
        )

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        """Return whether registered KV tensors use block as dim 0."""

        return True

    def get_transfer_cache_regions(
        self,
        cache_or_caches: Any,
        _layer_spec: Any,
    ) -> list[Any]:
        """Return the tensor regions that should be registered for a layer.

        vLLM's current attention layouts pack K/V into the content dimension,
        so attention layers register as one region. Hybrid Mamba passes tuple
        state, where the first item is the device state tensor used here.
        """

        if isinstance(cache_or_caches, (list, tuple)):
            return [cache_or_caches[0]]
        return [cache_or_caches]
