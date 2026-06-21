# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar-specific All2All manager implementations.

NaiveAll2AllManager: broadcast-based naive all2all for testing and debugging.
"""

import torch
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def naive_multicast(
        self,
        x: torch.Tensor,
        cu_tokens_across_sp_cpu: torch.Tensor,
        is_sequence_parallel: bool,
    ) -> torch.Tensor:
        input_was_1d = x.dim() == 1
        if input_was_1d:
            x = x.unsqueeze(-1)

        buffer = torch.empty(
            (cu_tokens_across_sp_cpu[-1], *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )

        rank = self.rank if is_sequence_parallel else self.dp_rank
        world_size = self.world_size if is_sequence_parallel else self.dp_world_size
        comm_group = get_ep_group() if is_sequence_parallel else get_dp_group()

        start = 0 if rank == 0 else cu_tokens_across_sp_cpu[rank - 1]
        end = cu_tokens_across_sp_cpu[rank]
        buffer[start:end].copy_(x)
        for idx in range(world_size):
            start = 0 if idx == 0 else cu_tokens_across_sp_cpu[idx - 1]
            end = cu_tokens_across_sp_cpu[idx]
            comm_group.broadcast(buffer[start:end], idx)

        if input_was_1d:
            return buffer.squeeze(-1)
        return buffer

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        router_logits = self.naive_multicast(
            router_logits, cu_tokens_across_sp_cpu, is_sequence_parallel
        )

        if extra_tensors is not None:
            extra_tensors = [
                self.naive_multicast(
                    tensor, cu_tokens_across_sp_cpu, is_sequence_parallel
                )
                for tensor in extra_tensors
            ]
            return hidden_states, router_logits, extra_tensors
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_weights = self.naive_multicast(
            topk_weights, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_ids = self.naive_multicast(
            topk_ids, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank
        comm_group = get_ep_group() if is_sequence_parallel else get_dp_group()

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = comm_group.all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def destroy(self):
        pass
