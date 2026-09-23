# SPDX-License-Identifier: Apache-2.0

# EP implementation notes:
# 1. Simplified MoE EP flow: enabled by this file by default, centered on
#    Dispatch -> MoE -> Combine.
# 2. DeepEP backend flows:
#    2.1. HT: enabled by all2all_backend=deepep_high_throughput and uses
#         Scatter -> Dispatch -> MoE -> Combine -> All Gather.
#    2.2. LL: enabled by all2all_backend=deepep_low_latency, or by decode
#         override below, and uses the same flow.
#         Scatter -> Dispatch -> MoE -> Combine -> All Gather.
#    2.3. VLLM_ILU_DEEPEP_USE_SCATTER applies to both HT and LL. It defaults to
#         enabled and uses TP scatter before Dispatch to avoid redundant
#         Dispatch communication when TP is enabled. Set it to 0 to disable.
#    2.4. VLLM_ILU_LL_FUSED_AG_AR is LL-only. It defaults to enabled, but only
#         takes effect with VLLM_ILU_DISABLE_DEEPEP=1; set it to 0 to disable
#         fused AG/AR.
# 3. Decode EP override: disabled by default. Set
#    VLLM_ILU_DECODE_EP_BACKEND to deepep_low_latency to use LL for decode
#    while prefill keeps the
#    configured all2all_backend.
# 4. Decode override safety: always gated by all-DP-ranks decode-only metadata,
#    and does not have a separate switch. This keeps DP ranks on the same
#    collective communication path.


from contextlib import nullcontext
from types import MethodType

import ixformer.distributed as ixfd
import torch
from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize.ilu_deepep_ll import (
    VLLM_ILU_DEEPEP_USE_SCATTER,
    VLLM_ILU_LL_FUSED_AG_AR,
    nvsm_dispatch_quant_kwargs,
)

logger = init_logger(__name__)

_DS4_DECODE_EP_BACKEND = envs.VLLM_ILU_DECODE_EP_BACKEND
_USE_FUSED_MOE = envs.VLLM_ILU_MOE_EP_LL_MOE_ALGO == "FUSED"
_MAX_NUM_SEQS: int | None = None


def set_max_num_seqs(max_num_seqs: int) -> None:
    global _MAX_NUM_SEQS
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    _MAX_NUM_SEQS = max_num_seqs


def set_max_num_seqs_from_config(vllm_config) -> None:
    # Keep the cached fallback identical to the value _get_max_num_seqs()
    # derives when the vllm_config context is still active, so DeepEP LL
    # buffers are sized the same way during CUDA graph capture.
    set_max_num_seqs(_decode_max_tokens_per_rank_from_config(vllm_config))


def _decode_max_tokens_per_rank_from_config(vllm_config) -> int:
    # Uniform decode query length is (1 + num_speculative_tokens). With MTP
    # enabled, each seq can contribute multiple tokens per decode step.
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    num_spec_tokens = 0
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        num_spec_tokens = int(
            getattr(speculative_config, "num_speculative_tokens", 0) or 0
        )
    return max_num_seqs * (1 + num_spec_tokens)


def _decode_ll_max_tokens_per_rank() -> int:
    """Token slots each rank can receive through the low-latency dispatch.

    The LL path only carries decode steps (prefill goes through AG-RS / HT), so
    the bound is the uniform decode batch rather than ``max_num_batched_tokens``.
    After TP scatter / fused AG-AR each rank only holds a shard; ceil division
    matches the uneven chunking in flatten().
    """
    max_tokens_per_rank = _get_max_num_seqs()
    if VLLM_ILU_LL_FUSED_AG_AR or VLLM_ILU_DEEPEP_USE_SCATTER:
        tp_size = get_tensor_model_parallel_world_size()
        max_tokens_per_rank = (max_tokens_per_rank + tp_size - 1) // tp_size
    return max_tokens_per_rank


def _get_max_num_seqs() -> int:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return _decode_max_tokens_per_rank_from_config(vllm_config)
    if _MAX_NUM_SEQS is not None:
        return _MAX_NUM_SEQS
    raise RuntimeError(
        "max_num_seqs is unavailable; call set_max_num_seqs() during "
        "model initialization"
    )


def _check_shared_gate_up(
    hidden: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.dtype],
) -> None:
    q, scales, out_dtype = packed
    assert hidden.dtype == torch.bfloat16
    assert out_dtype == hidden.dtype
    assert q.dtype == torch.int8 and q.shape == hidden.shape
    assert scales.dtype == torch.float32
    assert scales.numel() == hidden.numel() // hidden.shape[-1]


class IluFusedMoEEP:
    _dp_ep_ll_all2all_manager = None
    _dp_ep_ll_all2all_handle = None
    _dp_ep_ll_all2all_handle_args = None
    _dp_ep_ll_preinit_done = False

    def init_ep(self):
        if self._should_preinit_dp_ep_ll_handle():
            runner = self.experts
            # moe_quant_config is None during model construction because it needs
            # post-processed weight scales. Defer NVSM initialization until after
            # weight processing, but before CUDA graph capture; patch only once.
            original_init = runner.maybe_init_modular_kernel

            def init_modular_kernel_and_nvsm(runner_self):
                original_init()
                runner_self.routed_experts._ensure_moe_quant_config_init()
                self._get_forward_dp_ep_ll_all2all()
                logger.info_once(
                    "Initialized Hybrid EP low-latency decode backend "
                    "(configured_backend=%s, decode_override=%s).",
                    runner_self.moe_config.moe_parallel_config.all2all_backend,
                    _DS4_DECODE_EP_BACKEND,
                )

            runner.maybe_init_modular_kernel = MethodType(
                init_modular_kernel_and_nvsm,
                runner,
            )
            IluFusedMoEEP._dp_ep_ll_preinit_done = True

    def _should_use_forward_dp_ep(self) -> bool:
        moe_parallel_config = self.experts.moe_config.moe_parallel_config
        all2all_backend = moe_parallel_config.all2all_backend
        return (
            moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
            and all2all_backend
            in (
                "naive",
                "allgather_reducescatter",
                "ag-rs",
                "ag_rs",
                "deepep_low_latency",
                "deepep_high_throughput",
            )
            and (
                all2all_backend in ("deepep_low_latency", "deepep_high_throughput")
                or not self.experts._quant_method.supports_internal_mk
            )
            and not self.experts._quant_method.is_monolithic
        )

    def _is_all_ranks_decode_only(self) -> bool:
        ctx = get_forward_context()
        return bool(
            ctx.dp_metadata is not None and ctx.dp_metadata.all_ranks_decode_only
        )

    def _should_preinit_dp_ep_ll_handle(self) -> bool:
        # Process-wide handle; first EP MoE layer wins (not always layer 0).
        if IluFusedMoEEP._dp_ep_ll_preinit_done:
            return False
        configured_backend = self.experts.moe_config.moe_parallel_config.all2all_backend
        return (
            _DS4_DECODE_EP_BACKEND == "deepep_low_latency"
            and configured_backend != "deepep_low_latency"
            and self._should_use_forward_dp_ep()
        )

    def _nvsm_dispatch_quant_kwargs(self) -> dict[str, object]:
        quant_method = self.experts._quant_method
        base_quant_method = getattr(quant_method, "old_quant_method", quant_method)
        quant_config = base_quant_method.moe_quant_config
        if quant_config is not None:
            return nvsm_dispatch_quant_kwargs(quant_config)
        return {
            "use_int8": bool(envs.VLLM_ILU_DEEPEP_USE_INT8),
            "use_fp8": False,
        }

    def _get_forward_dp_ep_ll_all2all(self):
        from vllm_iluvatar.distributed.device_communicators.all2all import (
            DeepEPLLAll2AllManager,
        )

        ep_comm = get_ep_group().device_communicator
        assert ep_comm is not None
        ll_manager = IluFusedMoEEP._dp_ep_ll_all2all_manager
        handle = IluFusedMoEEP._dp_ep_ll_all2all_handle
        if ll_manager is not None and handle is not None:
            return ll_manager, handle

        if ll_manager is None:
            ll_manager = DeepEPLLAll2AllManager(
                ep_comm.cpu_group,
                getattr(ep_comm, "tcp_store_group", None),
            )
            IluFusedMoEEP._dp_ep_ll_all2all_manager = ll_manager

        moe_config = self.experts.moe_config
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe_config.max_num_tokens,
            token_hidden_size=moe_config.hidden_dim,
            num_ep_ranks=ll_manager.world_size,
            num_global_experts=moe_config.num_experts,
            num_local_experts=moe_config.num_experts // ll_manager.world_size,
        )
        if (
            handle is None
            or IluFusedMoEEP._dp_ep_ll_all2all_handle_args != all_to_all_args
        ):
            handle = ll_manager.get_handle(all_to_all_args)
            IluFusedMoEEP._dp_ep_ll_all2all_handle = handle
            IluFusedMoEEP._dp_ep_ll_all2all_handle_args = all_to_all_args

        if hasattr(handle, "buffer") and handle.buffer is None:
            from ixformer.distributed import nvsm_ep

            max_tokens_per_rank = _decode_ll_max_tokens_per_rank()
            tp_size = get_tensor_model_parallel_world_size()

            dispatch_quant_kwargs = self._nvsm_dispatch_quant_kwargs()
            handle.buffer = nvsm_ep.nvsm_init_ep_comm(
                max_num_tokens=max_tokens_per_rank,
                hidden_size=moe_config.hidden_dim,
                num_experts=moe_config.num_experts,
                topk=moe_config.experts_per_token,
                dtype=torch.bfloat16,
                group=handle.group,
                tp_size=tp_size,
                **dispatch_quant_kwargs,
            )
            handle.ilu_max_tokens_per_rank = max_tokens_per_rank
        return ll_manager, handle

    def _get_forward_dp_ep_ll_moe_kernel(self):
        runner = self.experts
        quant_method = runner._quant_method
        configured_backend = runner.moe_config.moe_parallel_config.all2all_backend
        if configured_backend == "deepep_low_latency":
            return quant_method.moe_kernel

        cached_kernel = getattr(self, "_dp_ep_ll_moe_kernel", None)
        if cached_kernel is not None:
            return cached_kernel

        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEKernel,
        )

        from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize.ilu_deepep_ll import (  # noqa: E501
            DeepEPLLPrepareAndFinalize,
        )

        base_quant_method = getattr(quant_method, "old_quant_method", quant_method)
        global_to_physical = physical_to_global = local_expert_global_ids = None
        routing_tables = runner._expert_routing_tables()
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables

        ll_manager, handle = self._get_forward_dp_ep_ll_all2all()

        use_fp8_dispatch = bool(self._nvsm_dispatch_quant_kwargs()["use_fp8"])
        # Must match the NVSHMEM buffer built in
        # ``_get_forward_dp_ep_ll_all2all``. ``moe_config.max_num_tokens``
        # is the prefill bound (max_num_batched_tokens), and the batched
        # expert layout is sized as max_tokens_per_rank * num_dispatchers
        # per expert -- overstating it here costs GiBs of workspace and
        # output per captured graph.
        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=_decode_ll_max_tokens_per_rank(),
            num_dispatchers=ll_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )
        cached_kernel = FusedMoEKernel(
            prepare_finalize,
            base_quant_method.select_gemm_impl(
                prepare_finalize,
                runner.routed_experts,
            ),
        )
        self._dp_ep_ll_moe_kernel = cached_kernel
        return cached_kernel

    def _flatten_input_ids_for_moe(
        self,
        input_ids: torch.Tensor | None,
        org_shape: torch.Size,
    ) -> torch.Tensor | None:
        if input_ids is None:
            return None
        if tuple(input_ids.shape) == tuple(org_shape[:-1]):
            return input_ids.reshape(-1)
        if input_ids.dim() == 1 and input_ids.shape[0] == org_shape[0]:
            view_shape = (org_shape[0],) + (1,) * (len(org_shape) - 2)
            return input_ids.view(view_shape).expand(org_shape[:-1]).reshape(-1)
        return input_ids.reshape(-1)

    def forward_dp_ep(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_gate_up_input: tuple[torch.Tensor, torch.Tensor, torch.dtype]
        | None = None,
    ) -> torch.Tensor:
        all2all_backend = self.experts.moe_config.moe_parallel_config.all2all_backend
        all_ranks_decode_only = self._is_all_ranks_decode_only()

        if _DS4_DECODE_EP_BACKEND is not None and all_ranks_decode_only:
            if _DS4_DECODE_EP_BACKEND == "deepep_low_latency":
                return self.forward_dp_ep_deep_ep(
                    hidden_states,
                    input_ids,
                    shared_gate_up_input=shared_gate_up_input,
                )
            raise ValueError(
                f"Invalid VLLM_ILU_DECODE_EP_BACKEND: {_DS4_DECODE_EP_BACKEND!r}, "
                f"allowed value is 'deepep_low_latency'."
            )

        if all2all_backend == "deepep_low_latency":
            return self.forward_dp_ep_deep_ep(
                hidden_states,
                input_ids,
                shared_gate_up_input=shared_gate_up_input,
            )

        if all2all_backend == "deepep_high_throughput":
            return self.forward_dp_ep_deep_ep(
                hidden_states,
                input_ids,
                enable_fused_ag_ar=False,
                moe_kernel=self.experts._quant_method.moe_kernel,
                shared_gate_up_input=shared_gate_up_input,
            )

        if all2all_backend in (
            "naive",
            "allgather_reducescatter",
            "ag-rs",
            "ag_rs",
        ):
            return self.forward_dp_ep_naive(
                hidden_states,
                input_ids,
                shared_gate_up_input=shared_gate_up_input,
            )
        assert shared_gate_up_input is None
        return self.forward_dp_ep_modular(hidden_states, input_ids)

    def forward_dp_ep_modular(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        return final_hidden_states.view(org_shape)

    def forward_dp_ep_deep_ep(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        enable_fused_ag_ar: bool = True,
        moe_kernel=None,
        shared_gate_up_input: tuple[torch.Tensor, torch.Tensor, torch.dtype]
        | None = None,
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, org_shape[-1])
        input_ids = self._flatten_input_ids_for_moe(input_ids, org_shape)

        runner = self.experts
        moe_config = runner.moe_config
        if moe_config.pcp_size > 1:
            raise NotImplementedError(
                "IluDeepseekV4MoE.forward_dp_ep_deep_ep does not support PCP "
                f"(pcp_size={moe_config.pcp_size})."
            )
        quant_method = runner._quant_method
        if moe_kernel is None:
            moe_kernel = self._get_forward_dp_ep_ll_moe_kernel()
            if moe_kernel is None:
                raise NotImplementedError(
                    "IluDeepseekV4MoE.forward_dp_ep_deep_ep requires the "
                    "DeepEP LL-style modular kernel path. "
                    f"Got quant method {quant_method.__class__.__name__}."
                )
        shared_experts_input = hidden_states

        hidden_states, routed_shared_input = runner.apply_routed_input_transform(
            hidden_states
        )
        if routed_shared_input is not None:
            shared_experts_input = routed_shared_input

        hidden_dim = hidden_states.shape[-1]
        pre_xform_trunc_size = None
        post_xform_trunc_size = None
        if (
            not quant_method.skip_forward_padding
            and moe_config.hidden_dim != hidden_dim
        ):
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, moe_config.hidden_dim - hidden_dim),
                mode="constant",
                value=0.0,
            )
            post_xform_trunc_size = hidden_dim

        runner.routed_experts._ensure_moe_quant_config_init()

        if runner.gate is not None:
            router_logits, _ = runner.gate(hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states)

        ctx = get_forward_context()
        sequence_parallel_ctx = (
            ctx.dp_metadata.sp_local_sizes(moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sequence_parallel_ctx:
            router_input_ids = (
                input_ids
                if getattr(runner.router, "_hash_indices_table", None) is not None
                else None
            )
            topk_indices_dtype = moe_kernel.prepare_finalize.topk_indices_dtype()
            topk_weights, topk_ids = runner.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=topk_indices_dtype,
                input_ids=router_input_ids,
            )

            final_hidden_states = self._forward_dp_ep_flatten(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
                shared_gate_up_input=shared_gate_up_input,
                pre_xform_trunc_size=pre_xform_trunc_size,
                enable_fused_ag_ar=enable_fused_ag_ar,
                moe_kernel=moe_kernel,
            )

        assert runner.routed_output_transform is None, (
            "not support routed_output_transform"
        )

        if post_xform_trunc_size is not None:
            final_hidden_states = final_hidden_states[..., :post_xform_trunc_size]

        return runner._maybe_add_zero_expert_output(final_hidden_states).view(org_shape)

    def forward_dp_ep_naive(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_gate_up_input: tuple[torch.Tensor, torch.Tensor, torch.dtype]
        | None = None,
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, org_shape[-1])
        input_ids = self._flatten_input_ids_for_moe(input_ids, org_shape)

        runner = self.experts
        moe_config = runner.moe_config
        if moe_config.pcp_size > 1:
            raise NotImplementedError(
                "IluDeepseekV4MoE.forward_dp_ep does not support PCP "
                f"(pcp_size={moe_config.pcp_size})."
            )
        quant_method = runner._quant_method
        shared_experts = runner.shared_experts
        shared_experts_input = hidden_states
        hidden_states, routed_shared_input = runner.apply_routed_input_transform(
            hidden_states
        )
        if routed_shared_input is not None:
            shared_experts_input = routed_shared_input
        hidden_dim = hidden_states.shape[-1]
        pre_xform_trunc_size = None
        post_xform_trunc_size = None
        if (
            not quant_method.skip_forward_padding
            and moe_config.hidden_dim != hidden_dim
        ):
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, moe_config.hidden_dim - hidden_dim),
                mode="constant",
                value=0.0,
            )
            post_xform_trunc_size = hidden_dim
        runner.routed_experts._ensure_moe_quant_config_init()

        if shared_experts is not None:
            assert shared_experts_input is not None
            if shared_gate_up_input is not None:
                _check_shared_gate_up(shared_experts_input, shared_gate_up_input)
            shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)
        ctx = get_forward_context()
        sequence_parallel_ctx = (
            ctx.dp_metadata.sp_local_sizes(moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sequence_parallel_ctx:
            dp_metadata = ctx.dp_metadata
            assert dp_metadata is not None
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            assert sizes is not None
            dist_group = (
                get_ep_group() if moe_config.is_sequence_parallel else get_dp_group()
            )
            assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

            router_input_ids = (
                input_ids
                if getattr(runner.router, "_hash_indices_table", None) is not None
                else None
            )
            tensors_to_gather = [hidden_states]
            if router_input_ids is not None:
                tensors_to_gather.append(router_input_ids)

            gathered_tensors = dist_group.all_gatherv(
                tensors_to_gather,
                dim=0,
                sizes=sizes,
            )
            hidden_states = gathered_tensors[0]
            if router_input_ids is not None:
                router_input_ids = gathered_tensors[1]
            if runner.gate is not None:
                router_logits, _ = runner.gate(hidden_states)
            else:
                router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = runner.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=quant_method.topk_indices_dtype,
                input_ids=router_input_ids,
            )
            shared_output = None
            apply_fused_moe = (
                getattr(quant_method, "apply_fused_moe", None)
                if _USE_FUSED_MOE
                else None
            )
            if apply_fused_moe is None:
                fused_output = runner.routed_experts.forward_modular(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                )
            else:
                fused_output = apply_fused_moe(
                    layer=runner.routed_experts,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
            if shared_experts is not None:
                assert shared_experts_input is not None
                shared_layer_input = (
                    shared_experts_input
                    if shared_gate_up_input is None
                    else shared_gate_up_input
                )
                shared_experts(
                    shared_layer_input,
                    shared_experts._determine_shared_experts_order(
                        shared_experts_input
                    ),
                )
                shared_output = shared_experts.output
        if pre_xform_trunc_size is not None:
            fused_output = fused_output[..., :pre_xform_trunc_size]

        if runner._fused_output_is_reduced:
            raise NotImplementedError(
                "IluDeepseekV4MoE.forward_dp_ep expects unreduced fused output "
                "so shared and routed outputs can be added before one all-reduce."
            )

        if shared_output is not None and fused_output.dtype == torch.float16:
            shared_output *= runner.shared_experts_scaling_factor
        fused_output = runner.apply_routed_output_transform(fused_output)
        final_hidden_states = fused_output
        if shared_output is not None:
            shard_rank = (
                get_ep_group().rank_in_group
                if moe_config.is_sequence_parallel
                else get_dp_group().rank_in_group
            )
            assert shard_rank < len(sizes)
            shard_start = 0 if shard_rank == 0 else sum(sizes[:shard_rank])
            shard_end = shard_start + sizes[shard_rank]
            # final_hidden_states = final_hidden_states.clone()
            final_hidden_states[shard_start:shard_end] += shared_output
        ep_comm = get_ep_group().device_communicator
        assert ep_comm is not None and ep_comm.ixfd_group is not None
        ixfd.all_reduce(final_hidden_states, group=ep_comm.ixfd_group, async_op=True)
        shard_rank = (
            get_ep_group().rank_in_group
            if moe_config.is_sequence_parallel
            else get_dp_group().rank_in_group
        )
        assert shard_rank < len(sizes)
        shard_start = 0 if shard_rank == 0 else sum(sizes[:shard_rank])
        shard_end = shard_start + sizes[shard_rank]
        final_hidden_states = final_hidden_states[shard_start:shard_end]
        if post_xform_trunc_size is not None:
            final_hidden_states = final_hidden_states[..., :post_xform_trunc_size]

        final_hidden_states = runner._maybe_add_zero_expert_output(
            final_hidden_states
        ).view(org_shape)
        return final_hidden_states

    # The LL batched layout is sized for the worst case (every route landing on
    # one expert), so the routed-expert output is the same shape for every
    # decode step and every layer. Allocating it per call puts one of these
    # multi-GiB tensors into each captured graph's private pool; keeping one per
    # (shape, dtype, ubatch) instead lets every layer and every graph share a
    # single address. Layers run one at a time and finalize consumes the buffer
    # before the next layer dispatches, so sharing is safe; the padding rows are
    # garbage either way because ``apply`` only writes the active ranges.
    _ll_expert_output_cache: dict[tuple, torch.Tensor] = {}

    def _ll_expert_output_buffer(
        self,
        a1q: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (dbo_current_ubatch_id(), tuple(a1q.shape), dtype, a1q.device)
        buf = IluFusedMoEEP._ll_expert_output_cache.get(key)
        if buf is None:
            buf = torch.empty(a1q.shape, dtype=dtype, device=a1q.device)
            IluFusedMoEEP._ll_expert_output_cache[key] = buf
        return buf

    def _forward_dp_ep_flatten(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        shared_gate_up_input: tuple[torch.Tensor, torch.Tensor, torch.dtype]
        | None = None,
        pre_xform_trunc_size: int = None,
        enable_fused_ag_ar: bool = True,
        moe_kernel=None,
    ) -> torch.Tensor:
        runner = self.experts
        routed_experts = runner.routed_experts
        moe_kernel = moe_kernel or self._get_forward_dp_ep_ll_moe_kernel()
        if moe_kernel is None:
            raise NotImplementedError(
                "IluDeepseekV4MoE.forward_dp_ep requires an initialized "
                "modular MoE kernel."
            )

        # init props
        prepare_finalize = moe_kernel.prepare_finalize
        fused_experts = moe_kernel.fused_experts
        w1 = routed_experts.w13_weight
        w2 = routed_experts.w2_weight
        global_num_experts = routed_experts.global_num_experts
        expert_map = routed_experts.expert_map
        apply_router_weight_on_input = routed_experts.apply_router_weight_on_input

        # scatter
        orig_num_tokens = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        tp_chunk_sizes = None
        max_tp_chunk_size = None
        use_fused_ag_ar = enable_fused_ag_ar and VLLM_ILU_LL_FUSED_AG_AR
        if tp_size > 1 and (VLLM_ILU_DEEPEP_USE_SCATTER or use_fused_ag_ar):
            base_chunk_size = orig_num_tokens // tp_size
            extra_chunks = orig_num_tokens % tp_size
            tp_chunk_sizes = [
                base_chunk_size + (1 if rank < extra_chunks else 0)
                for rank in range(tp_size)
            ]
            max_tp_chunk_size = max(tp_chunk_sizes)
            shard_start = sum(tp_chunk_sizes[:tp_rank])
            shard_end = shard_start + tp_chunk_sizes[tp_rank]
            hidden_states = hidden_states[shard_start:shard_end]
            topk_weights = topk_weights[shard_start:shard_end]
            topk_ids = topk_ids[shard_start:shard_end]

        # dispatch
        assert prepare_finalize.supports_async()
        prepare_ret = prepare_finalize.prepare_async(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            fused_experts.quant_config,
            defer_input_quant=fused_experts.expects_unquantized_inputs,
        )
        prepare_hook, prepare_receiver = (
            prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
        )
        if prepare_hook is not None:
            prepare_hook()

        if shared_gate_up_input is not None:
            assert shared_experts_input is not None
            _check_shared_gate_up(shared_experts_input, shared_gate_up_input)

        shared_output = None
        if runner.shared_experts is not None:
            assert shared_experts_input is not None
            # This is the accepted dispatch-window schedule: dispatch is
            # launched first, shared experts run before the receiver waits for
            # dispatch completion, and routed experts run afterward. Backends
            # with independent communication progress can overlap that window.
            # Do not move shared experts to their auxiliary stream here: the
            # extra CUDA Graph fork/join was substantially slower per layer.
            shared_layer_input = (
                shared_experts_input
                if shared_gate_up_input is None
                else shared_gate_up_input
            )
            shared_output = runner.shared_experts._layer(shared_layer_input)

        (
            a1q,
            a1q_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        ) = prepare_receiver()

        # moe forward
        if expert_topk_ids is not None:
            topk_ids = expert_topk_ids
        if expert_topk_weights is not None:
            topk_weights = expert_topk_weights

        if hasattr(fused_experts, "local_num_experts"):
            fused_experts.local_num_experts = routed_experts.local_num_experts

        fused_output = self._ll_expert_output_buffer(
            a1q,
            fused_experts.workspace_dtype(hidden_states.dtype),
        )
        fused_experts.apply(
            output=fused_output,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=routed_experts.activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=fused_experts.a2_scale,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        if pre_xform_trunc_size is not None:
            fused_output = fused_output[..., :pre_xform_trunc_size]

        assert runner.moe_config.in_dtype == fused_output.dtype
        shared_experts_scaling_factor = 1.0
        if shared_output is not None and fused_output.dtype == torch.float16:
            shared_experts_scaling_factor = runner.shared_experts_scaling_factor

        # The standard experts fold the runner's routed output scale into
        # moe_output_reduce_sum, but the batched experts leave it to the caller:
        # their topk weighting and reduction happen in the combine kernel below.
        routed_experts_scaling_factor = 1.0
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            routed_experts_scaling_factor = float(
                getattr(fused_experts, "runner_scaling_factor", 1.0)
            )

        # fused combine + all gather + all reduce
        if tp_size > 1 and use_fused_ag_ar:
            from ixformer.distributed import nvsm_ep

            comm = prepare_finalize.buffer
            return nvsm_ep.nvsm_low_latency_fused_combine_ag_ar(
                comm=comm,
                x=fused_output,
                topk_ids=topk_ids.int(),
                topk_weights=topk_weights,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_num_tokens_per_rank=tp_chunk_sizes,
                shared_x=shared_output,
                shared_experts_scaling_factor=shared_experts_scaling_factor,
                router_experts_scaling_factor=routed_experts_scaling_factor,
            )

        # create all gather input tensor
        ag_buffer = None
        if max_tp_chunk_size is not None:
            ag_buffer = hidden_states.new_empty(
                (max_tp_chunk_size, *hidden_states.shape[1:])
            )
            output = ag_buffer[: hidden_states.shape[0]]
        else:
            output = torch.empty_like(hidden_states)

        # do combine
        weight_and_reduce_impl = fused_experts.finalize_weight_and_reduce_impl()
        finalize_ret = prepare_finalize.finalize_async(
            output,
            fused_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        finalize_hook, finalize_receiver = (
            finalize_ret if isinstance(finalize_ret, tuple) else (None, finalize_ret)
        )
        if finalize_hook is not None:
            finalize_hook()
        finalize_receiver()

        # do all gather
        if tp_chunk_sizes is not None:
            assert max_tp_chunk_size is not None and ag_buffer is not None
            max_chunk_size = max_tp_chunk_size
            if max_chunk_size == 0:
                return output.new_empty((0, *output.shape[1:]))

            gathered_output = torch.empty(
                (tp_size * max_chunk_size, *output.shape[1:]),
                device=output.device,
                dtype=output.dtype,
            )
            tp_group = get_tp_group()
            tp_comm = tp_group.device_communicator
            if tp_comm is not None and tp_comm.ixfd_group is not None:
                ixfd.all_gather_into_tensor(
                    gathered_output,
                    ag_buffer,
                    group=tp_comm.ixfd_group,
                    async_op=True,
                )
            else:
                torch.distributed.all_gather_into_tensor(
                    gathered_output,
                    ag_buffer,
                    group=tp_group.device_group,
                )

            gathered_output = gathered_output.view(
                (tp_size, max_chunk_size, *output.shape[1:])
            )
            output = torch.cat(
                [
                    gathered_output[rank, :chunk_size]
                    for rank, chunk_size in enumerate(tp_chunk_sizes)
                    if chunk_size > 0
                ],
                dim=0,
            )
            output = output[:orig_num_tokens]

        # do all reduce + add
        if shared_output is not None and fused_output.dtype == torch.float16:
            shared_output *= runner.shared_experts_scaling_factor

        # nvsm_low_latency_combine has no scaling argument, so the routed scale
        # that the fused combine takes as router_experts_scaling_factor is
        # applied here instead.
        if routed_experts_scaling_factor != 1.0:
            output.mul_(routed_experts_scaling_factor)

        fused_output = output
        if runner._fused_output_is_reduced:
            shared_output = runner._maybe_reduce_shared_expert_output(shared_output)

        return fused_output if shared_output is None else shared_output + fused_output
