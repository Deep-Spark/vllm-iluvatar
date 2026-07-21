# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar-specific All2All manager implementations.

NaiveAll2AllManager: broadcast-based naive all2all for testing and debugging.
"""

from typing import Any

import torch
import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    Cache,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.import_utils import has_deep_ep


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


DISABLE_DEEPEP = envs.VLLM_ILU_DISABLE_DEEPEP
logger = init_logger(__name__)


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        self._install_prepare_finalize()

    @staticmethod
    def _install_prepare_finalize() -> None:
        # Community all2all_utils constructs the HT prepare/finalize class from
        # module globals. Bind it to the plugin implementation so HT routing
        # uses the migrated Iluvatar prepare/finalize path.
        import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils

        from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize import (
            ilu_deepep_ht,
        )

        all2all_utils.DeepEPHTPrepareAndFinalize = (
            ilu_deepep_ht.DeepEPHTPrepareAndFinalize
        )

    def _compute_deepep_buffer_bytes(self):
        num_ranks = self.world_size
        int64_t = 8

        def sizeof(t):
            if t in [float, int]:
                return 4

            if isinstance(t, int):
                return t

            raise ValueError(t)

        # ref DeepSeekV3 args
        import deep_ep

        num_channels = deep_ep.Buffer.num_sms // 2
        hidden = 7168
        itemsize = 2
        num_topk = 8
        num_scales = hidden // 128

        # inconsistent impl for config and deep_ep.cpp
        # config = deep_ep.Buffer.get_dispatch_config(num_ranks)
        # return config.get_nvl_buffer_size_hint(hidden * 2, num_ranks), config.get_rdma_buffer_size_hint(hidden * 2, num_ranks)

        # num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens
        config_map = {
            2:   (deep_ep.Buffer.num_sms, 16, 256, 6, 128),
            4:   (deep_ep.Buffer.num_sms, 16, 256, 6, 128),
            8:   (deep_ep.Buffer.num_sms, 6, 256, 6, 128),
            16:  (deep_ep.Buffer.num_sms, 16, 288, 20, 128),
            24:  (deep_ep.Buffer.num_sms, 8, 288, 32, 128),
            32:  (deep_ep.Buffer.num_sms, 8, 288, 32, 128),
            64:  (deep_ep.Buffer.num_sms, 20, 288, 28, 128),
            128: (deep_ep.Buffer.num_sms, 20, 560, 32, 128),
            144: (deep_ep.Buffer.num_sms, 32, 720, 12, 128),
            160: (deep_ep.Buffer.num_sms, 28, 720, 12, 128),
        }
        num_max_nvl_chunked_recv_tokens = config_map[num_ranks][2]

        nb = 0
        nb += num_ranks * num_ranks * sizeof(int)                                                     # Size prefix matrix
        nb += num_channels * num_ranks * sizeof(int)                                                  # Channel start offset
        nb += num_channels * num_ranks * sizeof(int)                                                  # Channel end offset
        nb += num_channels * num_ranks * sizeof(int) * 2                                              # Queue head and tail
        nb += num_channels * num_ranks * num_max_nvl_chunked_recv_tokens * hidden * itemsize          # Data buffer
        nb += num_channels * num_ranks * num_max_nvl_chunked_recv_tokens * sizeof(int)                # Source index buffer
        nb += num_channels * num_ranks * num_max_nvl_chunked_recv_tokens * num_topk * sizeof(int64_t) # Top-k index buffer
        nb += num_channels * num_ranks * num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)   # Top-k weight buffer
        nb += num_channels * num_ranks * num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales # FP8 scale buffer
        return nb

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_nvl_bytes = self._compute_deepep_buffer_bytes()
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = num_nvl_bytes
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            # explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself."
        )

        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep  # type: ignore[import-not-found]

        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class BufferWrapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.buffer = None


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        self._install_prepare_finalize()

    @staticmethod
    def _install_prepare_finalize() -> None:
        # Community all2all_utils constructs the LL prepare/finalize class from
        # module globals. Bind those globals from the manager path so the
        # BufferWrapper handle is paired with the plugin implementation.
        import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils

        from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize import (
            ilu_deepep_ll,
        )

        all2all_utils.DeepEPLLPrepareAndFinalize = (
            ilu_deepep_ll.DeepEPLLPrepareAndFinalize
        )
        all2all_utils.DEEPEP_QUANT_BLOCK_SHAPE = ilu_deepep_ll.DEEPEP_QUANT_BLOCK_SHAPE

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            # allow_nvlink_for_low_latency_mode=True,
            # allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            # explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer if not DISABLE_DEEPEP else BufferWrapper
        )
        return handle

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0
