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
import os
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
from vllm.utils.torch_utils import aux_stream, current_stream
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_moe_padding import (
    _gathered_router_padding,
)
from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize.ilu_deepep_ll import (
    VLLM_ILU_DEEPEP_USE_SCATTER,
    VLLM_ILU_LL_FUSED_AG_AR,
    nvsm_dispatch_quant_kwargs,
)

logger = init_logger(__name__)

# Named for DS4 originally; the override now also serves GLM-5.2 (``naive``).
_DECODE_EP_BACKEND = envs.VLLM_ILU_DECODE_EP_BACKEND
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
    # Shared DeepEP LL manager; NVSM communicators are keyed by MoE shape so
    # target (e.g. 384/topk6) and DSpark draft (128/topk3) do not collide.
    _dp_ep_ll_all2all_manager = None
    _dp_ep_ll_handles: dict = {}
    _dp_ep_ll_preinit_done = False
    _dp_ep_ll_preinit_shapes: set[tuple[int | None, int | None]] = set()

    def init_ep(self):
        self._validate_decode_ep_backend()
        if not self._should_preinit_dp_ep_ll_handle():
            return
        runner = self.experts
        if getattr(runner, "_ilu_nvsm_init_patched", False):
            return
        # Seed the module-global max-num-seqs consumed by ``_get_max_num_seqs``
        # (called later from ``_get_forward_dp_ep_ll_all2all``). deepseek_v4 —
        # unlike step3p5 / minimax_m2 — does not call ``set_max_num_seqs`` in its
        # own ``__init__``, and the NVSM init wrapper below runs on the first
        # forward (``profile_run`` under ``determine_available_memory``), where
        # ``set_current_vllm_config`` is no longer active so
        # ``get_current_vllm_config_or_none()`` returns None. Model construction
        # runs inside ``set_current_vllm_config``, so capture the value here and
        # stash it in the module global for the later forward.
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and _MAX_NUM_SEQS is None:
            set_max_num_seqs(_decode_max_tokens_per_rank_from_config(vllm_config))

        # moe_quant_config is None during model construction because it needs
        # post-processed weight scales. The runner lazily initializes it on
        # the first forward — ``_forward_impl`` calls
        # ``routed_experts._ensure_moe_quant_config_init`` (introduced by the
        # MoE modular-kernel refactor, replacing the removed
        # ``runner.maybe_init_modular_kernel``) — which runs after weight
        # processing and before CUDA graph capture. Wrap that hook to also
        # bring up the DeepEP low-latency all2all handle + NVSM dispatch
        # state at the same point; patch once per MoE shape.
        routed_experts = runner.routed_experts

        original_init = routed_experts._ensure_moe_quant_config_init

        def ensure_moe_quant_config_and_nvsm(re_self):
            original_init()
            self._get_forward_dp_ep_ll_all2all()
            logger.info_once(
                "Initialized Hybrid EP low-latency decode backend "
                "(configured_backend=%s, decode_override=%s).",
                runner.moe_config.moe_parallel_config.all2all_backend,
                _DECODE_EP_BACKEND,
            )

        routed_experts._ensure_moe_quant_config_init = MethodType(
            ensure_moe_quant_config_and_nvsm,
            routed_experts,
        )
        runner._ilu_nvsm_init_patched = True
        moe_config = runner.moe_config
        IluFusedMoEEP._dp_ep_ll_preinit_done = True
        IluFusedMoEEP._dp_ep_ll_preinit_shapes.add(
            (
                getattr(moe_config, "num_experts", None),
                getattr(moe_config, "experts_per_token", None),
            )
        )

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
                or _DECODE_EP_BACKEND == "deepep_low_latency"
                or not self.experts._quant_method.supports_internal_mk
            )
            and not self.experts._quant_method.is_monolithic
        )

    def _validate_decode_ep_backend(self) -> None:
        if _DECODE_EP_BACKEND is None:
            return
        if not self._should_use_forward_dp_ep():
            raise ValueError(
                "VLLM_ILU_DECODE_EP_BACKEND requires a supported DP+EP layer."
            )
        backend = self.experts.moe_config.moe_parallel_config.all2all_backend
        if _DECODE_EP_BACKEND == "naive" and backend not in (
            "allgather_reducescatter",
            "ag-rs",
            "ag_rs",
        ):
            raise ValueError(
                "VLLM_ILU_DECODE_EP_BACKEND=naive requires an AG-RS "
                f"all2all_backend; got {backend!r}."
            )

    def _use_decode_ep_override(self) -> bool:
        return _DECODE_EP_BACKEND is not None and self._is_all_ranks_decode_only()

    def _is_all_ranks_decode_only(self) -> bool:
        ctx = get_forward_context()
        return bool(
            ctx.dp_metadata is not None and ctx.dp_metadata.all_ranks_decode_only
        )

    def _should_preinit_dp_ep_ll_handle(self) -> bool:
        moe_config = self.experts.moe_config
        shape = (
            getattr(moe_config, "num_experts", None),
            getattr(moe_config, "experts_per_token", None),
        )
        if shape == (None, None):
            if IluFusedMoEEP._dp_ep_ll_preinit_done:
                return False
        elif shape in IluFusedMoEEP._dp_ep_ll_preinit_shapes:
            return False
        configured_backend = moe_config.moe_parallel_config.all2all_backend
        return (
            _DECODE_EP_BACKEND == "deepep_low_latency"
            and configured_backend != "deepep_low_latency"
            and self._should_use_forward_dp_ep()
        )

    def _nvsm_dispatch_quant_kwargs(self) -> dict[str, object]:
        quant_method = self.experts._quant_method
        base_quant_method = getattr(quant_method, "old_quant_method", quant_method)
        if base_quant_method.moe_quant_config is None and hasattr(
            self.experts, "routed_experts"
        ):
            self.experts.routed_experts._ensure_moe_quant_config_init()
        quant_config = base_quant_method.moe_quant_config
        if quant_config is not None:
            return nvsm_dispatch_quant_kwargs(quant_config)
        return {
            "use_int8": bool(envs.VLLM_ILU_DEEPEP_USE_INT8),
            "use_fp8": False,
        }

    def _dp_ep_ll_handle_key(
        self,
        all_to_all_args: dict,
        max_tokens_per_rank: int,
        topk: int,
        dispatch_quant_kwargs: dict[str, object],
    ) -> tuple:
        quant_items = tuple(sorted(dispatch_quant_kwargs.items(), key=lambda kv: kv[0]))
        return (
            all_to_all_args["num_global_experts"],
            topk,
            all_to_all_args["token_hidden_size"],
            all_to_all_args["num_ep_ranks"],
            all_to_all_args["max_num_tokens_per_dp_rank"],
            max_tokens_per_rank,
            quant_items,
        )

    def _get_forward_dp_ep_ll_all2all(self):
        from vllm_iluvatar.distributed.device_communicators.all2all import (
            DeepEPLLAll2AllManager,
        )

        ep_comm = get_ep_group().device_communicator
        assert ep_comm is not None
        if IluFusedMoEEP._dp_ep_ll_all2all_manager is None:
            IluFusedMoEEP._dp_ep_ll_all2all_manager = DeepEPLLAll2AllManager(
                ep_comm.cpu_group,
                getattr(ep_comm, "tcp_store_group", None),
            )
        ll_manager = IluFusedMoEEP._dp_ep_ll_all2all_manager

        moe_config = self.experts.moe_config
        max_tokens_per_rank = _decode_ll_max_tokens_per_rank()
        # Size the BufferWrapper cache key by the NVSM token budget, not
        # moe_config.max_num_tokens (prefill-scale), so handle.buffer matches
        # the compiled EPConfig.
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=max_tokens_per_rank,
            token_hidden_size=moe_config.hidden_dim,
            num_ep_ranks=ll_manager.world_size,
            num_global_experts=moe_config.num_experts,
            num_local_experts=moe_config.num_experts // ll_manager.world_size,
        )

        dispatch_quant_kwargs = self._nvsm_dispatch_quant_kwargs()
        cache_key = self._dp_ep_ll_handle_key(
            all_to_all_args,
            max_tokens_per_rank,
            moe_config.experts_per_token,
            dispatch_quant_kwargs,
        )
        handle = IluFusedMoEEP._dp_ep_ll_handles.get(cache_key)
        if handle is None:
            handle = ll_manager.get_handle(all_to_all_args)
            IluFusedMoEEP._dp_ep_ll_handles[cache_key] = handle

        if hasattr(handle, "buffer") and handle.buffer is None:
            from ixformer.distributed import nvsm_ep

            tp_size = get_tensor_model_parallel_world_size()
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

    @staticmethod
    def _router_input_ids(runner, input_ids: torch.Tensor | None):
        """Token ids the router needs: hash routing and/or vision ``bias_vl``."""
        if input_ids is None:
            return None
        router = runner.router
        if getattr(router, "_hash_indices_table", None) is not None:
            return input_ids
        if getattr(router, "bias_vl", None) is not None:
            return input_ids
        return None

    def forward_dp_ep(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_gate_up_input: (
            tuple[torch.Tensor, torch.Tensor, torch.dtype] | None
        ) = None,
    ) -> torch.Tensor:
        all2all_backend = self.experts.moe_config.moe_parallel_config.all2all_backend
        if self._use_decode_ep_override():
            forward = (
                self.forward_dp_ep_naive
                if _DECODE_EP_BACKEND == "naive"
                else self.forward_dp_ep_deep_ep
            )
            return forward(
                hidden_states, input_ids, shared_gate_up_input=shared_gate_up_input
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
        # Adapted to vLLM v0.31.0 (db9527a468): the runner always holds the gate (it was
        # constructed via ``FusedMoEFactory(gate=gate, ...)``), so it computes
        # ``router_logits`` internally. ``router_logits=hidden_states`` is the
        # community ``_forward_fused_moe`` placeholder convention; the
        # ``is_internal_router`` branch was dropped in the community.
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )
        return final_hidden_states.view(org_shape)

    def forward_dp_ep_deep_ep(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        enable_fused_ag_ar: bool = True,
        moe_kernel=None,
        shared_gate_up_input: (
            tuple[torch.Tensor, torch.Tensor, torch.dtype] | None
        ) = None,
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
            router_input_ids = self._router_input_ids(runner, input_ids)
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

    @staticmethod
    def _combine_ag_rs_output(
        fused_output: torch.Tensor,
        shared_output: torch.Tensor | None,
        dist_group,
        sizes: list[int],
        *,
        sequence_parallel: bool,
        use_tp1_rs: bool = False,
    ) -> torch.Tensor:
        """Return this rank's slice of the gathered routed output.

        Shared experts are added into the local slice first. The default then
        reduce-scatters over ``dist_group`` and, outside sequence parallel,
        all-reduces that slice across TP. ``VLLM_ILU_FUSE_EP_COMBINE`` folds
        those two collectives into one EP all-reduce. Sequence parallel keeps
        the reduce-scatter: its ``dist_group`` is already the EP group.
        ``VLLM_ILU_MOE_EP_RS_TP_AG`` uses EP reduce-scatter + TP all-gather
        for equal DP chunks with 18–162 local tokens.
        """
        rank = dist_group.rank_in_group
        start = sum(sizes[:rank])
        stop = start + sizes[rank]
        if shared_output is not None:
            fused_output[start:stop] += shared_output
        tp_group = get_tp_group()
        local_tokens = sizes[rank]
        use_rs_ag = (
            envs.VLLM_ILU_MOE_EP_RS_TP_AG
            and not sequence_parallel
            and tp_group.world_size > 1
            and 18 <= local_tokens <= 162
            and local_tokens % tp_group.world_size == 0
            and all(size == local_tokens for size in sizes)
            and get_ep_group().world_size == len(sizes) * tp_group.world_size
        )
        if use_rs_ag:
            logger.info_once(
                "MoE EP reduce-scatter + TP all-gather: local_tokens=%d",
                local_tokens,
            )
            ep_comm = get_ep_group().device_communicator
            assert ep_comm is not None and ep_comm.ixfd_group is not None
            tp_comm = tp_group.device_communicator
            assert tp_comm is not None and tp_comm.ixfd_group is not None
            shard = fused_output.new_empty(
                (local_tokens // tp_group.world_size, *fused_output.shape[1:])
            )
            ixfd.reduce_scatter_tensor(
                shard, fused_output, group=ep_comm.ixfd_group, async_op=True
            )
            gathered = fused_output.new_empty((local_tokens, *fused_output.shape[1:]))
            ixfd.all_gather_into_tensor(
                gathered, shard, group=tp_comm.ixfd_group, async_op=True
            )
            return gathered
        fuse = (not sequence_parallel) and envs.VLLM_ILU_FUSE_EP_COMBINE
        if fuse:
            ep_group = get_ep_group()
            if (
                use_tp1_rs
                and tp_group.world_size == 1
                and local_tokens > 0
                and all(size == local_tokens for size in sizes)
                and dist_group.ranks == ep_group.ranks
            ):
                return dist_group.reduce_scatterv(fused_output, dim=0, sizes=sizes)
            ep_comm = ep_group.device_communicator
            assert ep_comm is not None and ep_comm.ixfd_group is not None
            # Stream-ordered: the slice below runs on the same stream, so the
            # dropped handle needs no explicit wait.
            ixfd.all_reduce(fused_output, group=ep_comm.ixfd_group, async_op=True)
            return fused_output[start:stop]
        local = dist_group.reduce_scatterv(fused_output, dim=0, sizes=sizes)
        if not sequence_parallel and get_tensor_model_parallel_world_size() > 1:
            local = get_tp_group().all_reduce(local)
        return local

    def forward_dp_ep_naive(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_gate_up_input: (
            tuple[torch.Tensor, torch.Tensor, torch.dtype] | None
        ) = None,
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
        # vLLM v0.31.0 (db9527a468) removed ``SharedExperts.maybe_sync_shared_experts_stream``
        # (the pre-dispatch aux-stream sync) and replaced it with ``maybe_forward_async`` +
        # ``wait``. This naive DP+EP path does NOT use pre-dispatch async shared experts:
        # shared experts run synchronously AFTER the routed experts below (see the
        # ``shared_experts(...)`` call after ``forward_modular``/``apply_fused_moe``), so
        # there is nothing to sync here. Drop the stale call (same rationale as
        # ``IluMoERunner._forward_impl``).
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
            shared_output = None

            if os.environ.get("VLLM_ILU_V41_PACKED_FP8_DISPATCH", "0") == "1":
                from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fp8_packed_dispatch import (
                    maybe_packed_experts,
                )
                packed_output = maybe_packed_experts(
                    self, hidden_states, input_ids, ctx, dist_group, sizes
                )
                if packed_output is not None:
                    fused_output = packed_output
                    router_input_ids = self._router_input_ids(runner, input_ids)
                    hidden_states = None
                else:
                    fused_output = None
            else:
                fused_output = None

            if fused_output is not None:
                pass
            else:

                router_input_ids = self._router_input_ids(runner, input_ids)
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
                with _gathered_router_padding(ctx, dist_group, sizes):
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
                "so shared and routed outputs can be added before the combine."
            )

        if shared_output is not None and fused_output.dtype == torch.float16:
            shared_output *= runner.shared_experts_scaling_factor
        fused_output = runner.apply_routed_output_transform(fused_output)
        final_hidden_states = self._combine_ag_rs_output(
            fused_output,
            shared_output,
            dist_group,
            sizes,
            sequence_parallel=moe_config.is_sequence_parallel,
            use_tp1_rs=(
                getattr(self, "_supports_agrs_tp1_rs", False)
                and envs.VLLM_ILU_V41_AGRS_TP1_RS
            ),
        )
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
        shared_gate_up_input: (
            tuple[torch.Tensor, torch.Tensor, torch.dtype] | None
        ) = None,
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

        if shared_gate_up_input is not None:
            assert shared_experts_input is not None
            _check_shared_gate_up(shared_experts_input, shared_gate_up_input)

        shared_output = None
        shared_stream = None
        if runner.shared_experts is not None:
            assert shared_experts_input is not None
            shared_layer_input = (
                shared_experts_input
                if shared_gate_up_input is None
                else shared_gate_up_input
            )
            if (
                getattr(self, "_ll_shared_dispatch_overlap_enabled", False)
                and orig_num_tokens > 0
            ):
                # Fork BEFORE prepare_async: NVSM dispatch itself is queued on
                # the main stream. Forking afterward would serialize shared
                # compute behind dispatch, including during CUDA Graph replay.
                shared_stream = aux_stream()
                assert shared_stream is not None
                shared_stream.wait_stream(current_stream())
                shared_inputs = (
                    shared_layer_input
                    if isinstance(shared_layer_input, tuple)
                    else (shared_layer_input,)
                )
                for tensor in shared_inputs:
                    if isinstance(tensor, torch.Tensor):
                        tensor.record_stream(shared_stream)
                with torch.cuda.stream(shared_stream):
                    shared_output = runner.shared_experts._layer(shared_layer_input)

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

        if runner.shared_experts is not None and shared_stream is None:
            # Preserve the serial dispatch-window schedule when disabled.
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

        # Join after routed GEMMs: shared compute on aux_stream overlaps both
        # NVSM dispatch and routed expert GEMMs before combine/AG/AR consumes
        # shared_output.
        if shared_stream is not None:
            current_stream().wait_stream(shared_stream)
            assert shared_output is not None
            shared_output.record_stream(current_stream())

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
        output_is_reduced = getattr(moe_kernel, "output_is_reduced", None)
        active_output_is_reduced = (
            output_is_reduced()
            if callable(output_is_reduced)
            else getattr(runner, "_fused_output_is_reduced", False)
        )
        if active_output_is_reduced:
            shared_output = runner._maybe_reduce_shared_expert_output(
                shared_output, fused_output_is_reduced=active_output_is_reduced
            )

        return fused_output if shared_output is None else shared_output + fused_output
