"""Pipeline-parallel mapping helpers used during the NIXL handshake."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .handshake_validation import validate_pp_splits


@dataclass(frozen=True)
class PipelineSlice:
    """一个 local PP stage 与一个 remote PP stage 的层区间交集。

    这里同时保存两套相对坐标：
    - ``remote_start/end``：相对于 remote PP stage 起点；
    - ``local_start/end``：相对于当前 local PP stage 起点。
    两边均采用左闭右开区间。
    """

    # 与本地 stage 相交的远端 PP rank。
    remote_pp_rank: int
    # 交集在远端 stage 内的相对层区间 [remote_start, remote_end)。
    remote_start: int
    remote_end: int
    # 同一交集在本地 stage 内的相对层区间 [local_start, local_end)。
    local_start: int
    local_end: int


class PipelineMappingPlanner:
    """计算 P/D 使用不同 PP 切分时的层映射和通知 fanout。

    ``*_pp_split`` 只保存每个 stage 的起始层号，例如 12 层、PP=3
    可表示为 ``[0, 4, 8]``；模型总层数作为最后一个隐式边界。
    """

    def __init__(
        self,
        *,
        local_stage_start: int,
        local_stage_end: int,
        local_pp_split: Sequence[int],
        num_hidden_layers: int,
    ) -> None:
        if not 0 <= local_stage_start < local_stage_end <= num_hidden_layers:
            raise ValueError(
                "invalid local PP layer range "
                f"[{local_stage_start}, {local_stage_end}) for "
                f"{num_hidden_layers} layers"
            )
        validate_pp_splits(local_pp_split, num_hidden_layers)
        # 当前 worker 所持有的模型全局层区间。
        self.local_stage_start: int = local_stage_start
        self.local_stage_end: int = local_stage_end
        # 所有 local PP stage 的起始层号，用于计算 completion fanout。
        self.local_pp_split: tuple[int, ...] = tuple(local_pp_split)
        self.num_hidden_layers: int = num_hidden_layers

    def target_slices(self, remote_pp_split: Sequence[int]) -> list[PipelineSlice]:
        """返回与当前 local stage 相交的所有 remote PP stage。

        例如 local=[2, 8)，remote stages=[0, 4), [4, 9)：
        返回 remote rank 0 的 local [0, 2) 和 rank 1 的 local [2, 6)。
        ``local_offset`` 连续累加，确保结果完整覆盖当前 local stage。
        """

        validate_pp_splits(remote_pp_split, self.num_hidden_layers)
        remote_bounds: list[int] = [*remote_pp_split, self.num_hidden_layers]
        result: list[PipelineSlice] = []
        local_offset: int = 0

        for remote_rank, (remote_stage_start, remote_stage_end) in enumerate(
            zip(remote_bounds, remote_bounds[1:])
        ):
            # 先在模型全局层坐标中求交集，再分别换算成两端 stage 相对坐标。
            overlap_start: int = max(self.local_stage_start, remote_stage_start)
            overlap_end: int = min(self.local_stage_end, remote_stage_end)
            if overlap_start >= overlap_end:
                continue
            width: int = overlap_end - overlap_start
            result.append(
                PipelineSlice(
                    remote_pp_rank=remote_rank,
                    remote_start=overlap_start - remote_stage_start,
                    remote_end=overlap_end - remote_stage_start,
                    local_start=local_offset,
                    local_end=local_offset + width,
                )
            )
            local_offset += width

        if not result or local_offset != self.local_stage_end - self.local_stage_start:
            raise RuntimeError(
                "remote PP partition does not cover local layer range "
                f"[{self.local_stage_start}, {self.local_stage_end})"
            )
        return result

    def remote_notification_ranges(
        self, remote_pp_split: Sequence[int]
    ) -> dict[int, tuple[int, int]]:
        """映射 ``remote_pp_rank -> [local_pp_start, local_pp_end)``。

        一个 remote stage 可能覆盖多个 local stage。远端完成通知中携带这个
        local PP rank 半开区间，接收端据此知道需要等待多少个 PP writer。
        """

        validate_pp_splits(remote_pp_split, self.num_hidden_layers)
        local_bounds: list[int] = [*self.local_pp_split, self.num_hidden_layers]
        remote_bounds: list[int] = [*remote_pp_split, self.num_hidden_layers]
        result: dict[int, tuple[int, int]] = {}

        for remote_rank, (remote_start, remote_end) in enumerate(
            zip(remote_bounds, remote_bounds[1:])
        ):
            # 使用模型全局坐标判断两个 PP stage 是否有非空交集。
            overlapping: list[int] = [
                local_rank
                for local_rank, (local_start, local_end) in enumerate(
                    zip(local_bounds, local_bounds[1:])
                )
                if max(local_start, remote_start) < min(local_end, remote_end)
            ]
            if not overlapping:
                raise RuntimeError(
                    f"remote PP rank {remote_rank} has no local PP overlap"
                )
            result[remote_rank] = (min(overlapping), max(overlapping) + 1)

        return result
