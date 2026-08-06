# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

from vllm_iluvatar.distributed.device_communicators.all2all import (
    DISABLE_DEEPEP,
    BufferWrapper,
)

logger = init_logger(__name__)

DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]

VLLM_ILU_DEEPEP_USE_SCATTER = envs.VLLM_ILU_DEEPEP_USE_SCATTER
VLLM_ILU_LL_FUSED_AG_AR = envs.VLLM_ILU_LL_FUSED_AG_AR and envs.VLLM_ILU_DISABLE_DEEPEP


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP low-latency kernels.
    """

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends
    # on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up hidden size to the closest supported hidden size.
        _supported_hs = DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
        # Check sorted
        num_supported_hs = len(_supported_hs)
        assert all(
            [
                _supported_hs[i] < _supported_hs[i + 1]
                for i in range(num_supported_hs - 1)
            ]
        )

        for x in _supported_hs:
            if x >= hidden_size:
                return x

        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        buffer: deep_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()

        if DISABLE_DEEPEP and torch.cuda.current_device() == 0:
            logger.warning_once(">> DISABLE_DEEPEP")

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = False
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handles: list[tuple | None] = [None, None]
        self.num_dispatchers_ = num_dispatchers

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

        self.use_int8_dispatch = envs.VLLM_ILU_DEEPEP_USE_INT8

        # We don't have enough information to determine if we should dispatch
        # activation scales in a packed ue8m0 format during object construction
        # time. This setting is handled by post_init_setup.
        self.use_ue8m0_dispatch = False

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        if not fused_experts.supports_packed_ue8m0_act_scales():
            # Early exit.
            return

        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update DeepEPLLPrepareFinalize to do packed ue8m0 scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "DeepEPLLPrepareAndFinalize is setup to dispatch raw/unquantized "
                f"activations despite ({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
            )

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        if DISABLE_DEEPEP:
            return torch.int32
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _do_quant(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(x, tuple) and self.use_int8_dispatch:
            return x

        if torch.is_tensor(x) and x.dtype in [torch.bfloat16, torch.half]:
            return x, None

        raise RuntimeError("invalid input")

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if DISABLE_DEEPEP:
            from ixformer.distributed import nvsm_ep

            if isinstance(self.buffer, BufferWrapper):
                if self.buffer.buffer is None:
                    tp_size = get_tensor_model_parallel_world_size()
                    max_tokens_per_rank = self.max_tokens_per_rank
                    if VLLM_ILU_LL_FUSED_AG_AR or VLLM_ILU_DEEPEP_USE_SCATTER:
                        max_tokens_per_rank //= tp_size

                    self.buffer.buffer = nvsm_ep.nvsm_init_ep_comm(
                        max_num_tokens=max_tokens_per_rank,
                        hidden_size=hidden_size,
                        num_experts=num_experts,
                        topk=topk_ids.shape[1],
                        dtype=torch.bfloat16,
                        use_int8=self.use_int8_dispatch,
                        group=self.buffer.group,
                        tp_size=tp_size,
                    )
                self.buffer = self.buffer.buffer

            recv_x, num_tokens_per_local_experts_gpu, recv_scales, *_ = (
                nvsm_ep.nvsm_low_latency_dispatch(
                    self.buffer, a1, topk_ids.int(), topk_weights
                )
            )

            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=num_tokens_per_local_experts_gpu,
                expert_num_tokens_cpu=None,
            )
            return (
                lambda: 1,
                lambda: (recv_x, recv_scales, expert_tokens_meta, None, None),
            )

        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids.long(),
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            use_int8=self.use_int8_dispatch,
            # round_scale needs to be set to dispatch in ue8m0
            # round_scale=self.use_ue8m0_dispatch,
            # use_ue8m0=self.use_ue8m0_dispatch,
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            ),
        )

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        hook()
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        if DISABLE_DEEPEP:
            from ixformer.distributed import nvsm_ep

            nvsm_ep.nvsm_low_latency_combine(
                self.buffer, fused_expert_output, topk_ids, topk_weights, out=output
            )
            return lambda: output, lambda: None

        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        _, _, recv_hook = self.buffer.low_latency_combine(
            fused_expert_output,
            combine_topk_ids.long(),
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )

        return recv_hook, lambda: None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
