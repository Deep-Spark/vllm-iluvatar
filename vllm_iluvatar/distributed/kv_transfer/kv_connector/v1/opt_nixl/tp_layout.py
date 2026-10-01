"""Pure tensor-parallel KV transfer layout adjustment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

# 一个 block 内的 NIXL 片段：(相对 block 起点的字节偏移, payload 字节数)。
Descriptor: TypeAlias = tuple[int, int]
# 外层按已注册 pointer 排列，内层是该 pointer 的一个或多个传输片段。
LayerDescriptors: TypeAlias = list[list[Descriptor]]


@dataclass(frozen=True)
class AdjustedXferSide:
    """一侧调整后的 pointer / descriptor / stride。"""

    ptr_ids: list[int]
    desc: LayerDescriptors
    # inter-block stride，不是 descriptor payload length。
    block_len: list[int]


@dataclass(frozen=True)
class AdjustedXferLayout:
    """TP 调整后的 local/remote 布局。"""

    # 逐 pointer 的 writer 选择；False 的 pointer 不进入 dlist。
    pointer_write: list[bool]
    local: AdjustedXferSide
    remote: AdjustedXferSide


@dataclass(frozen=True)
class TpLayoutAdjuster:
    """按 TP head 分片和 KV 复制调整双方 descriptor。

    两端注册的都是调度 block，block size 与每块 kernel 数在握手时逐 pointer
    强制相等，所以这里只处理 TP：

    1. Mamba state：conv/ssm segment 按 TP ratio 做字节切片；
    2. MLA 或 P/D 两侧都已复制 KV：整段复制，只要求 payload 长度一致；
    3. 普通 GQA/MHA：按 KV-head 所在字节区间切 local/remote descriptor。

    offset 与 stride 始终各用各端的值（packed slab 或不同 PP 切分下两端可以
    不同）。本类只计算布局，不注册内存、不创建 NIXL handle，也不修改输入。
    """

    # 当前 writer 的本地 TP rank / world size。
    local_tp_rank: int
    local_tp_size: int
    # 当前正在建立 dlist 的远端 TP rank / world size。
    remote_tp_rank: int
    remote_tp_size: int
    # 数据并行 rank 参与 replicated-KV writer 去重，避免多个副本重复 WRITE。
    dp_rank: int
    # 模型全局 KV head 数，用来判断 TP rank 上的 KV 是否已经复制。
    total_num_kv_heads: int
    # bool 表示所有 pointer 相同；tuple 表示逐 pointer 是否按 MLA/复制语义处理。
    use_mla: bool | tuple[bool, ...]
    # Mamba state 的 page stride 可能因 padding 不同，但 conv/ssm segment
    # 按 TP ratio 做字节切片。
    is_mamba: bool | tuple[bool, ...]

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def adjust(
        self,
        *,
        local_ptr_ids: list[int],
        remote_ptr_ids: list[int],
        local_desc: LayerDescriptors,
        remote_desc: LayerDescriptors,
        local_block_len: list[int],
        remote_block_len: list[int],
    ) -> AdjustedXferLayout:
        """返回当前 remote TP rank 对应的 descriptor 切片与 writer 选择。"""

        (
            use_mla_by_layer,
            is_mamba_by_layer,
        ) = self._validate_inputs(
            local_ptr_ids=local_ptr_ids,
            remote_ptr_ids=remote_ptr_ids,
            local_desc=local_desc,
            remote_desc=remote_desc,
            local_block_len=local_block_len,
            remote_block_len=remote_block_len,
        )

        # 无符号 TP 倍数；方向始终由 local_tp_size 与 remote_tp_size 比较。
        tp_ratio: int = max(self.local_tp_size, self.remote_tp_size) // min(
            self.local_tp_size, self.remote_tp_size
        )
        local_splits: bool = self.local_tp_size >= self.remote_tp_size
        # 当 TP rank 数不少于 KV head 数时，至少有多个 rank 持有相同 KV head。
        is_local_kv_replicated: bool = self.local_tp_size >= self.total_num_kv_heads
        is_remote_kv_replicated: bool = self.remote_tp_size >= self.total_num_kv_heads

        pointer_write: list[bool] = [True] * len(local_desc)
        new_local_desc: LayerDescriptors = [[] for _ in local_desc]
        new_remote_desc: LayerDescriptors = [[] for _ in local_desc]

        for index, (l_desc, r_desc, local_block_bytes, remote_block_bytes) in enumerate(
            zip(local_desc, remote_desc, local_block_len, remote_block_len)
        ):
            new_l_desc: list[Descriptor] = []
            new_r_desc: list[Descriptor] = []

            if is_mamba_by_layer[index]:
                for segment_index, ((l_start, l_len), (r_start, r_len)) in enumerate(
                    zip(l_desc, r_desc)
                ):
                    if local_splits:
                        # 多个 local TP writers 将各自的 state slice 写入
                        # 同一 remote rank 的不同字节区间。
                        self._require(
                            l_len == r_len // tp_ratio
                            and l_start == r_start // tp_ratio,
                            "Mamba segment geometry mismatch for local TP "
                            f"split at pointer {index}, segment {segment_index}: "
                            f"local=({l_start}, {l_len}), "
                            f"remote=({r_start}, {r_len}), "
                            f"tp_ratio={tp_ratio}",
                        )
                        new_l_desc.append((l_start, l_len))
                        new_r_desc.append(
                            (r_start + (self.local_tp_rank % tp_ratio) * l_len, l_len)
                        )
                    else:
                        # 一个 local TP writer 覆盖多个 remote ranks；每个
                        # remote rank 从 local state 中读取自己的 slice。
                        self._require(
                            l_len // tp_ratio == r_len
                            and l_start // tp_ratio == r_start,
                            "Mamba segment geometry mismatch for remote TP "
                            f"split at pointer {index}, segment {segment_index}: "
                            f"local=({l_start}, {l_len}), "
                            f"remote=({r_start}, {r_len}), "
                            f"tp_ratio={tp_ratio}",
                        )
                        new_l_desc.append(
                            (l_start + (self.remote_tp_rank % tp_ratio) * r_len, r_len)
                        )
                        new_r_desc.append((r_start, r_len))

            # MLA 的 latent KV，或两侧都因 TP>=KV heads 而复制的 KV，不按
            # head 再切片；每个有效 writer 搬运完整 payload。
            elif use_mla_by_layer[index] or (
                is_local_kv_replicated and is_remote_kv_replicated
            ):
                for segment_index, ((l_start, l_len), (r_start, r_len)) in enumerate(
                    zip(l_desc, r_desc)
                ):
                    self._require(
                        l_len == r_len,
                        "replicated / MLA payload length mismatch at pointer "
                        f"{index}, segment {segment_index}: "
                        f"local=({l_start}, {l_len}), "
                        f"remote=({r_start}, {r_len})",
                    )
                    new_l_desc.append((l_start, l_len))
                    new_r_desc.append((r_start, r_len))
                # 复制语义下 local TP 更多时，并非每个副本都要 WRITE。
                # 使用 (tp_rank + dp_rank) 选出唯一 writer。
                if local_splits:
                    pointer_write[index] = (
                        self.local_tp_rank + self.dp_rank
                    ) % tp_ratio == 0

            elif local_splits:
                # 普通 GQA/MHA，多个 local TP writers 指向一个 remote rank：
                # 每个 local writer 写 remote block 中不同的 head slice。
                self._require(
                    not is_remote_kv_replicated,
                    "remote KV heads must not be replicated in this GQA "
                    "local-TP-to-remote-TP branch",
                )
                local_repeat_times: int = max(
                    self.local_tp_size // self.total_num_kv_heads, 1
                )
                if is_local_kv_replicated:
                    pointer_write[index] = (
                        self.local_tp_rank + self.dp_rank
                    ) % local_repeat_times == 0
                # 去掉本地重复 KV rank 后，真正参与 head 切分的倍数。
                reduced_tp_ratio: int = tp_ratio // local_repeat_times
                slice_index: int = (
                    self.local_tp_rank // local_repeat_times % reduced_tp_ratio
                )
                if pointer_write[index]:
                    for segment_index, (
                        (l_start, l_len),
                        (r_start, r_len),
                    ) in enumerate(zip(l_desc, r_desc)):
                        self._require(
                            l_len == r_len // reduced_tp_ratio
                            and l_start == r_start // reduced_tp_ratio,
                            "GQA descriptor geometry mismatch at pointer "
                            f"{index}, segment {segment_index}: "
                            f"local=({l_start}, {l_len}), "
                            f"remote=({r_start}, {r_len}), "
                            f"tp_ratio={reduced_tp_ratio}",
                        )
                        new_l_desc.append((l_start, l_len))
                        new_r_desc.append((r_start + slice_index * l_len, l_len))
                self._require(
                    remote_block_bytes == local_block_bytes * reduced_tp_ratio,
                    "Remote P worker KV layer cache must be of shape "
                    "[2, N, local_kv_heads*tp_ratio, block_size, head_dim] "
                    "and same dtype: "
                    f"local_block_len[{index}]={local_block_bytes}, "
                    f"remote_block_len[{index}]={remote_block_bytes}",
                )

            else:
                # 普通 GQA/MHA，一个 local writer 覆盖多个 remote ranks：
                # 每个 remote rank 从 local block 中取自己的 head slice。
                self._require(
                    not is_local_kv_replicated,
                    "local KV heads must not be replicated in this GQA "
                    "local-TP-to-remote-TP branch",
                )
                remote_repeat_times: int = max(
                    self.remote_tp_size // self.total_num_kv_heads, 1
                )
                reduced_tp_ratio = tp_ratio // remote_repeat_times
                slice_index = (
                    self.remote_tp_rank // remote_repeat_times % reduced_tp_ratio
                )
                for segment_index, ((l_start, l_len), (r_start, r_len)) in enumerate(
                    zip(l_desc, r_desc)
                ):
                    self._require(
                        l_len // tp_ratio == r_len and l_start // tp_ratio == r_start,
                        "GQA descriptor geometry mismatch at pointer "
                        f"{index}, segment {segment_index}: "
                        f"local=({l_start}, {l_len}), "
                        f"remote=({r_start}, {r_len}), "
                        f"tp_ratio={tp_ratio}",
                    )
                    new_l_desc.append((l_start + slice_index * r_len, r_len))
                    new_r_desc.append((r_start, r_len))
                self._require(
                    remote_block_bytes == local_block_bytes // reduced_tp_ratio,
                    "Remote P worker KV layer cache must be of shape "
                    "[2, N, local_kv_heads//tp_ratio, block_size, head_dim] "
                    "and same dtype: "
                    f"local_block_len[{index}]={local_block_bytes}, "
                    f"remote_block_len[{index}]={remote_block_bytes}",
                )

            new_local_desc[index] = new_l_desc
            new_remote_desc[index] = new_r_desc

        return AdjustedXferLayout(
            pointer_write=pointer_write,
            local=AdjustedXferSide(
                ptr_ids=list(local_ptr_ids),
                desc=new_local_desc,
                block_len=list(local_block_len),
            ),
            remote=AdjustedXferSide(
                ptr_ids=list(remote_ptr_ids),
                desc=new_remote_desc,
                block_len=list(remote_block_len),
            ),
        )

    def _validate_inputs(
        self,
        *,
        local_ptr_ids: list[int],
        remote_ptr_ids: list[int],
        local_desc: LayerDescriptors,
        remote_desc: LayerDescriptors,
        local_block_len: list[int],
        remote_block_len: list[int],
    ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """校验 rank、数组形状，并规范化逐 pointer 的 MLA / Mamba 标记。"""

        for name, value in (
            ("local_tp_size", self.local_tp_size),
            ("remote_tp_size", self.remote_tp_size),
            ("total_num_kv_heads", self.total_num_kv_heads),
        ):
            self._require(value > 0, f"{name} must be positive, got {value}")

        self._require(
            0 <= self.local_tp_rank < self.local_tp_size,
            "local_tp_rank must be within local TP size, got "
            f"rank={self.local_tp_rank}, size={self.local_tp_size}",
        )
        self._require(
            0 <= self.remote_tp_rank < self.remote_tp_size,
            "remote_tp_rank must be within remote TP size, got "
            f"rank={self.remote_tp_rank}, size={self.remote_tp_size}",
        )
        larger_tp: int = max(self.local_tp_size, self.remote_tp_size)
        smaller_tp: int = min(self.local_tp_size, self.remote_tp_size)
        self._require(
            larger_tp % smaller_tp == 0,
            "P/D tensor parallel sizes must have an integral ratio, got "
            f"local={self.local_tp_size}, remote={self.remote_tp_size}",
        )

        use_mla_by_layer: tuple[bool, ...] = (
            (self.use_mla,) * len(local_desc)
            if isinstance(self.use_mla, bool)
            else tuple(self.use_mla)
        )
        is_mamba_by_layer: tuple[bool, ...] = (
            (self.is_mamba,) * len(local_desc)
            if isinstance(self.is_mamba, bool)
            else tuple(self.is_mamba)
        )
        # local/remote descriptor、stride 和 per-pointer 标记必须一一对应。
        lengths: set[int] = {
            len(local_ptr_ids),
            len(remote_ptr_ids),
            len(local_desc),
            len(remote_desc),
            len(local_block_len),
            len(remote_block_len),
            len(use_mla_by_layer),
            len(is_mamba_by_layer),
        }
        self._require(
            len(lengths) == 1 and bool(local_desc),
            "TP layout input length mismatch or empty inputs: "
            f"local_ptr={len(local_ptr_ids)}, remote_ptr={len(remote_ptr_ids)}, "
            f"local_desc={len(local_desc)}, remote_desc={len(remote_desc)}, "
            f"local_block_len={len(local_block_len)}, "
            f"remote_block_len={len(remote_block_len)}, "
            f"use_mla={len(use_mla_by_layer)}, "
            f"is_mamba={len(is_mamba_by_layer)}",
        )
        self._require(
            not any(
                use_mla and is_mamba
                for use_mla, is_mamba in zip(use_mla_by_layer, is_mamba_by_layer)
            ),
            "a pointer cannot be both MLA attention and Mamba state",
        )
        for index, (l_desc, r_desc) in enumerate(zip(local_desc, remote_desc)):
            self._require(
                len(l_desc) == len(r_desc) and bool(l_desc),
                f"descriptor segment mismatch at pointer {index}: "
                f"local={len(l_desc)}, remote={len(r_desc)}",
            )
        return use_mla_by_layer, is_mamba_by_layer
