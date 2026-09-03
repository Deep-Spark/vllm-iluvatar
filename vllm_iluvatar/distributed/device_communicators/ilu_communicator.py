# SPDX-License-Identifier: Apache-2.0
"""
IluCommunicator — Iluvatar GPU communicator for vLLM.

Inherits from the community CudaCommunicator and overrides communication
primitives to use ixformer.distributed (ixfd) instead of NCCL/pynccl.

The ixfd communication group is created internally from the standard torch
ProcessGroup passed in as device_group, so parallel_state.py does not need
any ixfd awareness.

Registered via IluPlatform.get_device_communicator_cls().
"""

import os

import ixformer.distributed as ixfd
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _world
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)
from vllm.distributed.utils import StatelessProcessGroup



class IluCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            tcp_store_group,
            use_all2all=use_all2all,
        )
        self.use_vllm_comm = os.environ.get("VLLM_FORCE_NCCL_COMM", None) not in [
            "1",
            "Y",
            "y",
        ]
        self.tcp_store_group = tcp_store_group
        self.is_stateless_group = _world.pg_map.get(cpu_group, None) is None
        self.use_vllm_comm = self.use_vllm_comm and not self.is_stateless_group

        if (
            self.use_vllm_comm
            and device_group is not None
            and not self.is_stateless_group
        ):
            self.ixfd_group = ixfd.init_comm_with_store(device_group)
        else:
            self.ixfd_group = None

        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from vllm_iluvatar.distributed.device_communicators.all2all import (
                    AgRsAll2AllManager,
                )

                self.all2all_manager = AgRsAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_high_throughput":
                from vllm_iluvatar.distributed.device_communicators.all2all import (
                    DeepEPHTAll2AllManager,
                )

                self.all2all_manager = DeepEPHTAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_low_latency":
                from vllm_iluvatar.distributed.device_communicators.all2all import (
                    DeepEPLLAll2AllManager,
                )

                self.all2all_manager = DeepEPLLAll2AllManager(
                    self.cpu_group, tcp_store_group
                )

    # ------------------------------------------------------------------
    # Communication primitives — override with ixfd
    # ------------------------------------------------------------------

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        if self.use_vllm_comm:
            ixfd.all_reduce(input_, group=self.ixfd_group, async_op=True)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        if self.use_vllm_comm:
            ixfd.all_gather_into_tensor(
                output_tensor,
                input_,
                group=self.ixfd_group,
                async_op=True,
            )
        else:
            torch.distributed.all_gather_into_tensor(
                output_tensor, input_, group=self.device_group
            )
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        if self.use_vllm_comm:
            ixfd.gather(
                input_,
                gather_list,
                dst=self.ranks[dst],
                group=self.ixfd_group,
                async_op=True,
            )
        else:
            torch.distributed.gather(
                input_,
                gather_list,
                dst=self.ranks[dst],
                group=self.device_group,
            )

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gatherv(
        comm,
        input_: "torch.Tensor | list[torch.Tensor]",
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        """Device-group all-gather(v). ``comm`` is the owning ``CudaCommunicator``."""
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = comm.world_size

        # 'sizes' is unneeded when every rank contributes the same number of rows.
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        if sizes is None:

            def _all_gather_dev(inp: torch.Tensor) -> torch.Tensor:
                inp_c = inp.contiguous()
                input_size = inp_c.size()
                output_size = (input_size[0] * world_size,) + tuple(input_size[1:])
                output_tensor = torch.empty(
                    output_size, dtype=inp_c.dtype, device=inp_c.device
                )
                if comm.use_vllm_comm:
                    ixfd.all_gather_into_tensor(
                        output_tensor, inp_c, group=comm.ixfd_group, async_op=True
                    )
                else:
                    torch.distributed.all_gather_into_tensor(
                        output_tensor, inp_c, group=comm.device_group
                    )
                return output_tensor

            if isinstance(input_, torch.Tensor):
                return _all_gather_dev(input_)
            return [_all_gather_dev(inp) for inp in input_]

        # Variable-length all-gather (DP ranks carry different token counts):
        # pad every contribution to max(sizes), fixed-size all-gather, then slice
        # each rank back to its real length and concat.
        assert len(sizes) == world_size
        max_size = max(sizes)

        def _all_gatherv_dev(inp: torch.Tensor) -> torch.Tensor:
            inp_c = inp.contiguous()
            s_self = sizes[comm.rank_in_group]
            assert inp_c.shape[0] == s_self, f"{inp_c.shape[0]} != {s_self}"
            rest = tuple(inp_c.shape[1:])
            padded = torch.zeros(
                (max_size,) + rest, dtype=inp_c.dtype, device=inp_c.device
            )
            if s_self > 0:
                padded[:s_self] = inp_c
            gathered = torch.empty(
                (world_size * max_size,) + rest,
                dtype=inp_c.dtype,
                device=inp_c.device,
            )
            if comm.use_vllm_comm:
                ixfd.all_gather_into_tensor(
                    gathered, padded, group=comm.ixfd_group, async_op=True
                )
            else:
                torch.distributed.all_gather_into_tensor(
                    gathered, padded, group=comm.device_group
                )
            gathered = gathered.view((world_size, max_size) + rest)
            parts = [gathered[r, : sizes[r]] for r in range(world_size)]
            return torch.cat(parts, dim=0)

        if isinstance(input_, torch.Tensor):
            return _all_gatherv_dev(input_)
        return [_all_gatherv_dev(inp) for inp in input_]

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if self.use_vllm_comm:
            ixfd.reduce_scatter_tensor(
                output, input_tensor, group=self.ixfd_group, async_op=True
            )
        else:
            torch.distributed.reduce_scatter_tensor(
                output, input_tensor, group=self.device_group
            )

        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        comm,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Device-group reduce-scatter(v)."""
        world_size = comm.world_size
        if dim < 0:
            dim += input_.dim()

        # Note: input must be contiguous, possible bug in reduce_scatter_tensor.
        input_tensor = input_.movedim(0, dim).contiguous()
        sizes_equal = sizes is None or sizes.count(sizes[0]) == len(sizes)

        if sizes_equal:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
            output_shape = (chunk_size,) + tuple(input_tensor.shape[1:])
            output = torch.empty(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            if comm.use_vllm_comm:
                ixfd.reduce_scatter_tensor(
                    output, input_tensor, group=comm.ixfd_group, async_op=True
                )
            else:
                torch.distributed.reduce_scatter_tensor(
                    output, input_tensor, group=comm.device_group
                )
            return output.movedim(0, dim).contiguous()

        # Variable-length reduce-scatter (DP ranks carry different token counts):
        # pad every segment to max(sizes), fixed-size reduce-scatter, slice off the
        # garbage padded rows.
        assert sizes is not None and len(sizes) == world_size
        assert input_tensor.shape[0] == sum(sizes)
        max_size = max(sizes)
        rest = tuple(input_tensor.shape[1:])
        padded = torch.zeros(
            (world_size * max_size,) + rest,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        offset = 0
        for r in range(world_size):
            s = sizes[r]
            if s > 0:
                padded[r * max_size : r * max_size + s] = input_tensor[
                    offset : offset + s
                ]
            offset += s
        out = torch.empty(
            (max_size,) + rest, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if comm.use_vllm_comm:
            ixfd.reduce_scatter_tensor(out, padded, group=comm.ixfd_group, async_op=True)
        else:
            torch.distributed.reduce_scatter_tensor(out, padded, group=comm.device_group)
        out = out[: sizes[comm.rank_in_group]]
        return out.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        if self.use_vllm_comm:
            ixfd.send(tensor, self.ranks[dst], self.ixfd_group)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        if self.use_vllm_comm:
            ixfd.recv(tensor, self.ranks[src], self.ixfd_group)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        if self.is_stateless_group:
            assert self.tcp_store_group is not None
            tensor.copy_(self.tcp_store_group.broadcast(tensor, src))
            return tensor
        if self.use_vllm_comm:
            ixfd.broadcast(
                tensor,
                src=self.ranks[src],
                group=self.ixfd_group,
                async_op=True,
            )
        else:
            torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast_device_tensor(
        self,
        tensor: torch.Tensor,
        src_global_rank: int,
        group=None,
    ):
        """Async broadcast for device tensors, returning a handle.

        Used by GroupCoordinator.broadcast_tensor_dict() to delegate
        GPU tensor broadcast to the communicator.
        """
        if self.is_stateless_group:
            assert self.tcp_store_group is not None
            src = self.ranks.index(src_global_rank)
            tensor.copy_(self.tcp_store_group.broadcast(tensor, src))
            return None
        if self.use_vllm_comm:
            g = group if group is not None else self.ixfd_group
            return ixfd.broadcast(tensor, src=src_global_rank, group=g, async_op=True)
        g = group if group is not None else self.device_group
        return torch.distributed.broadcast(
            tensor, src=src_global_rank, group=g, async_op=True
        )

    def destroy(self):
        if self.ixfd_group is not None and not self.is_stateless_group:
            ixfd.destroy_process_group(self.ixfd_group)
            self.ixfd_group = None
        super().destroy()


# ------------------------------------------------------------------
# Startup-time signature validation (编程规则 §3)
# ------------------------------------------------------------------

from vllm.distributed.device_communicators.base_device_communicator import (  # noqa: E402
    DeviceCommunicatorBase,
)

from vllm_iluvatar.ilu_util.signature import validate_patch_signatures  # noqa: E402

_ILU_OVERRIDES_FROM_CUDA = [
    "all_reduce",
    "reduce_scatter",
    "send",
    "recv",
    "broadcast",
]
_ILU_OVERRIDES_FROM_BASE = ["all_gather", "gather"]

validate_patch_signatures(
    CudaCommunicator,
    IluCommunicator,
    _ILU_OVERRIDES_FROM_CUDA,
    strict=True,
)
validate_patch_signatures(
    DeviceCommunicatorBase,
    IluCommunicator,
    _ILU_OVERRIDES_FROM_BASE,
    strict=True,
)
