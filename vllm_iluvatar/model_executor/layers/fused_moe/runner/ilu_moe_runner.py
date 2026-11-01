# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MoERunner — inherits community MoERunner, overrides only the
delta methods required for corex / Iluvatar hardware.

Baseline : community ``runner/moe_runner.py`` @ v0.31.0 (db9527a468)
Delta    : ~250 lines vs. the original 1010-line full-class copy

Override summary (each marked with ``# [ILU override]`` in-line):

  __init__                          elastic-EP install + FP16 scaling /
                                    RoutedExperts scale placement
  _select_forward                   bypass torch.compile custom ops
  forward                           tuple hidden_states (fused norm-quant)
  _apply_quant_method               fused shared output path
  _forward_impl                     pass input_ids; gather before replicated gate
  _maybe_dispatch                   input_ids + _kernel_handles_own_dispatch
  _maybe_combine                    _kernel_handles_own_dispatch guard
  _maybe_apply_routed_scale_to_output  FP16 shared_experts_scaling_factor
  do_naive_dispatch_combine         drop supports_internal_mk check
  _kernel_handles_own_dispatch      new property (monolithic W16A16 guard)
  _ilu_fused_ep_combine + helpers   5 collectives per layer -> 2 (PCP/tuple safe)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward_shared,
    _unpack,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_moe_padding import (
    _gathered_router_padding,
)

logger = init_logger(__name__)

# Interface contract for the gather-before-gate path (``_dispatch_before_gate``).
# A model opts in by setting ``gather_before_gate = True`` on its gate and may
# override ``gather_before_gate_max_tokens`` to tune the decode-batch bound; see
# IluGateLinear in vllm_iluvatar/models/glm5next/model.py.
#
# The defaults live here rather than on a base class because every ILU gate
# derives straight from the upstream
# ``vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear`` (they are
# siblings, not a chain), and ``self.gate`` may be any model's gate -- including
# gates from models we do not override at all. Declaring the defaults upstream is
# not an option, so both attributes are read through ``getattr`` with these
# fallbacks: a gate that sets the flag but forgets the bound still works.
GATHER_BEFORE_GATE_DEFAULT = False
GATHER_BEFORE_GATE_MAX_TOKENS_DEFAULT = 32


class IluMoERunner(MoERunner):
    """Iluvatar override of the community MoERunner.

    Inherits all community behaviour and overrides only the methods whose
    logic differs on corex / Iluvatar hardware.
    """

    # [ILU override] __init__ — delta: elastic-EP patch install + private
    # Iluvatar output scale placement for kernels that scale after reduction.
    def __init__(self, *args, routed_scaling_factor: float = 1.0, **kwargs):
        super().__init__(*args, routed_scaling_factor=routed_scaling_factor, **kwargs)
        logger.info_once("Use Iluvatar MoERunner.")

        # Performance-path toggle: gather hidden states before the replicated
        # gate on eligible models (see ``_dispatch_before_gate``). Default on —
        # the benchmarks and serving were measured with it enabled.
        self._gather_before_gate_enabled = envs.VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE

        # Read here rather than in forward: the thread-local VllmConfig is not
        # guaranteed to be set during CUDA graph capture.
        self._ilu_elastic_ep = bool(
            get_current_vllm_config().parallel_config.enable_elastic_ep
        )
        if self._ilu_elastic_ep:
            from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep import (  # noqa: E501
                install_ilu_elastic_ep_patches,
            )

            install_ilu_elastic_ep_patches()

        self.shared_experts_scaling_factor = 1.0
        # Keep RoutedExperts.routed_scaling_factor router-aligned per upstream.
        # Iluvatar kernels read this private field as their output-side scale.
        self.routed_experts._ilu_output_scale = routed_scaling_factor
        if (
            self._shared_experts is not None
            and self.moe_config.in_dtype == torch.float16
        ):
            self.routed_scaling_factor = 1.0
            self.shared_experts_scaling_factor = 1.0 / routed_scaling_factor
            self.routed_experts._ilu_output_scale = 1.0

    # [ILU override] _select_forward — delta: corex does not support
    # torch.compile; always use the Python wrapper.
    def _select_forward(self) -> Callable:
        return _moe_forward_shared

    # [ILU override] forward — delta: support tuple hidden_states from fused
    # norm-quant path (quantized_output, output_scales, fp_output). Does NOT
    # override apply_routed_input_transform — uses the community version and
    # discards the second return value in the tuple case.
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._ilu_fused_ep_combine:
            if isinstance(hidden_states, tuple):
                assert len(hidden_states) == 3, (
                    "Expected hidden_states tuple of "
                    "(quantized_output, output_scales, fp_output)"
                )
                shared_experts_input = (
                    *hidden_states[:2],
                    hidden_states[-1].dtype,
                )
                hidden_states = hidden_states[-1]
                hidden_states, _ = super().apply_routed_input_transform(hidden_states)
                return self._forward_fused_ep_combine(
                    hidden_states, router_logits, input_ids, shared_experts_input
                )
            return self._forward_fused_ep_combine(
                hidden_states, router_logits, input_ids, None
            )

        if not isinstance(hidden_states, tuple):
            return super().forward(hidden_states, router_logits, input_ids)

        assert len(hidden_states) == 3, (
            "Expected hidden_states tuple of "
            "(quantized_output, output_scales, fp_output)"
        )
        shared_experts_input = (
            *hidden_states[:2],
            hidden_states[-1].dtype,
        )
        hidden_states = hidden_states[-1]
        hidden_states, _ = super().apply_routed_input_transform(hidden_states)
        # Pass None so community _maybe_pad_hidden_states skips
        # .shape[-1] on the tuple; safe because routed_output_transform
        # is None on Iluvatar.
        hidden_states, og_pre, og_post = self._maybe_pad_hidden_states(
            None, hidden_states
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = _unpack(result)

        if og_pre is not None:
            fused_output = fused_output[..., :og_pre]

        shared_output = self._maybe_reduce_shared_expert_output(shared_output)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(result, og_post)
        return self._maybe_add_zero_expert_output(result)

    # [ILU override] _apply_quant_method — delta: fused shared output
    # optimisation; unquantized W16A16 stays unfused so runner-side routed
    # scaling applies only to the routed output, not the shared residual.
    def _is_unquantized_fused_moe_method(self) -> bool:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        return isinstance(self._quant_method, UnquantizedFusedMoEMethod)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        experts_order = SharedExpertsOrder.NONE
        if self._shared_experts is not None:
            experts_order = self._shared_experts._determine_shared_experts_order(
                shared_experts_input
            )

        can_fused_shared_output = (
            experts_order == SharedExpertsOrder.NO_OVERLAP
            and self.moe_config.in_dtype != torch.float16
            and not self.do_naive_dispatch_combine
            # Routed tokens are gathered across PCP; shared experts stay local.
            and self.moe_config.pcp_size == 1
        )
        use_mk_internal_shared_experts = (
            experts_order == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        )

        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        if self.routed_experts.quant_method.is_monolithic:
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            fused_shared_output = (
                self._shared_experts.output if can_fused_shared_output else None
            )

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=(
                    self._shared_experts
                    if can_fused_shared_output or use_mk_internal_shared_experts
                    else None
                ),
                shared_experts_input=(
                    shared_experts_input
                    if use_mk_internal_shared_experts
                    else fused_shared_output
                ),
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return_shared_output = experts_order in (
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        ) or (
            experts_order == SharedExpertsOrder.NO_OVERLAP
            and (
                self.moe_config.in_dtype == torch.float16
                or self.do_naive_dispatch_combine
                or self.moe_config.pcp_size > 1
            )
        )

        return (
            self._shared_experts.output
            if (
                self._shared_experts is not None
                and return_shared_output
                and not can_fused_shared_output
            )
            else None,
            fused_out,
        )

    def _dispatch_before_gate(self) -> bool:
        # Gather hidden rows before the gate when the model opts in via its gate
        # (``gather_before_gate``): the router is replicated row-wise, so this
        # avoids a second collective for logits and exposes a larger GEMM to the
        # gate kernel. Eligibility is a model property, not a shape fingerprint;
        # a false positive only changes the collective path, never the result.
        gate = self.gate
        if not (
            self._gather_before_gate_enabled
            and getattr(gate, "gather_before_gate", GATHER_BEFORE_GATE_DEFAULT)
            and self.do_naive_dispatch_combine
            and not self._kernel_handles_own_dispatch
            and self.moe_config.moe_parallel_config.all2all_backend
            == "allgather_reducescatter"
            and self.moe_config.pcp_size == 1
            and not self._fse_fuse_gate
        ):
            return False
        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return False
        # Every EP rank must choose the same collective sequence, including
        # mixed prefill/decode and idle DP ranks. This metadata is already CPU
        # resident and shared across DP; no extra collective or GPU sync.
        max_tokens = int(metadata.num_tokens_across_dp_cpu.max())
        max_per_shard = getattr(
            gate,
            "gather_before_gate_max_tokens",
            GATHER_BEFORE_GATE_MAX_TOKENS_DEFAULT,
        )
        return 0 < max_tokens <= max_per_shard * self.moe_config.sp_size

    def _gather_before_gate(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Called inside _sequence_parallel_context, as with normal dispatch.
        metadata = get_forward_context().dp_metadata
        assert metadata is not None
        sizes = metadata.get_chunk_sizes_across_dp_rank()
        group = (
            get_ep_group() if self.moe_config.is_sequence_parallel else get_dp_group()
        )
        assert sizes[group.rank_in_group] == hidden_states.shape[0]
        tensors = [hidden_states]
        if input_ids is not None:
            tensors.append(input_ids)
        gathered = group.all_gatherv(tensors, dim=0, sizes=sizes)
        return gathered[0], gathered[1] if input_ids is not None else None

    # [ILU override] _forward_impl — pass input_ids through dispatch and
    # gather hidden states before the replicated GLM decode gate when eligible.
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.routed_experts._ensure_moe_quant_config_init()
        if getattr(self, "_ilu_fused_ep_combine", False):
            return self._forward_impl_fused_ep_combine(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        dispatch_before_gate = self._dispatch_before_gate()
        # vLLM v0.31.0 (db9527a468) removed ``_maybe_sync_shared_experts_stream`` (the
        # pre-dispatch aux-stream sync) and replaced it with
        # ``SharedExperts.maybe_forward_async`` + ``wait``. ILU does NOT use
        # pre-dispatch async shared experts — ``_apply_quant_method`` below
        # orchestrates shared experts synchronously via ``_maybe_apply_shared_experts``
        # (NO_OVERLAP before routed experts, MULTI_STREAM_OVERLAPPED after), so
        # there is nothing to sync here. Drop the stale call.
        if self.gate is not None and not dispatch_before_gate:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            if dispatch_before_gate:
                hidden_states, input_ids = self._gather_before_gate(
                    hidden_states, input_ids
                )
                router_logits, _ = self.gate(hidden_states)
            else:
                hidden_states, router_logits, input_ids = self._maybe_dispatch(
                    hidden_states, router_logits, input_ids
                )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    @property
    def _needs_pcp_dispatch_combine(self) -> bool:
        return self.moe_config.pcp_size > 1 and (
            not self.moe_config.moe_parallel_config.use_all2all_kernels
            or not self._kernel_handles_own_dispatch
        )

    # [ILU override] _maybe_dispatch — delta: propagate input_ids through
    # DP/EP dispatch + PCP all_gather, guard with _kernel_handles_own_dispatch.
    # Signature differs from community (adds input_ids param + return); safe
    # because _forward_impl is also overridden.
    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine and not self._kernel_handles_own_dispatch:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
                extra_tensors=[input_ids] if input_ids is not None else None,
            )
            if input_ids is not None:
                assert len(result) == 3
                hidden_states, router_logits, extra_tensors = result
                assert len(extra_tensors) == 1
                input_ids = extra_tensors[0]
            else:
                assert len(result) == 2
                hidden_states, router_logits = result

        if self._needs_pcp_dispatch_combine:
            # EP alone does not install a dispatch kernel. Native W4A8 still
            # needs all PCP tokens on every expert shard before local GEMMs.
            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)
            if input_ids is not None:
                input_ids = get_pcp_group().all_gather(input_ids, dim=0)

        return hidden_states, router_logits, input_ids

    # [ILU override] _maybe_combine — delta: guard with
    # _kernel_handles_own_dispatch.
    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor]:
        if self.do_naive_dispatch_combine and not self._kernel_handles_own_dispatch:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self._needs_pcp_dispatch_combine:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)

        if shared_output is not None:
            return shared_output, hidden_states
        else:
            return hidden_states

    # [ILU override] _maybe_apply_routed_scale_to_output — delta: FP16 uses
    # shared_experts_scaling_factor instead of dividing fused_output (avoids
    # FP16 overflow).
    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        assert self.moe_config.in_dtype == fused_output.dtype
        if shared_output is not None and fused_output.dtype == torch.float16:
            shared_output = shared_output * self.shared_experts_scaling_factor
        if self._is_unquantized_fused_moe_method():
            fused_output = fused_output * self.routed_experts._ilu_output_scale
        return shared_output, fused_output

    # [ILU override] do_naive_dispatch_combine — delta: drop the
    # supports_internal_mk check; Iluvatar kernels do not yet implement
    # internal MK dispatch.
    @property
    def do_naive_dispatch_combine(self) -> bool:
        return self.moe_config.dp_size > 1

    # [ILU override] _kernel_handles_own_dispatch (new property) — delta:
    # after elastic-EP scale-up, the quant method may be wrapped into a
    # FusedMoEModularMethod whose kernel does its own DP/EP dispatch/combine.
    # Monolithic Iluvatar W16A16 layers are excluded.
    @property
    def _kernel_handles_own_dispatch(self) -> bool:
        from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep import (  # noqa: E501
            ilu_kernel_handles_own_dispatch,
        )

        return ilu_kernel_handles_own_dispatch(self._quant_method)

    # ------------------------------------------------------------------
    # [ILU addition] fused EP combine — collective launch reduction
    # ------------------------------------------------------------------
    # On the AG/RS naive DP+EP path the community schedule costs five
    # collective launches per MoE layer. Feat's runner still owns PCP
    # all-gather and the fused-norm-quant tuple path, so this fast path
    # is rewritten against those contracts rather than the 0.30 runner:
    #
    #   * all-gather only hidden_states (and input_ids if hash routing)
    #   * recompute the gate after the gather
    #   * fold this rank's shared-expert partial into its DP rows
    #   * one EP all-reduce replaces DP reduce-scatter + TP all-reduce
    #
    # PCP, SP, elastic EP, EPLB and kernels that already reduce stay on
    # the community schedule. Kill switch: ``VLLM_ILU_FUSE_EP_COMBINE=0``.
    @property
    def _ilu_fused_ep_combine(self) -> bool:
        cached = self.__dict__.get("_ilu_fused_ep_combine_cached")
        if cached is None:
            cached = self._ilu_compute_fused_ep_combine()
            self.__dict__["_ilu_fused_ep_combine_cached"] = cached
        return cached

    def _ilu_compute_fused_ep_combine(self) -> bool:
        """Config-only eligibility check, evaluated once on first forward."""
        if not envs.VLLM_ILU_FUSE_EP_COMBINE:
            return False

        moe = self.moe_config
        reasons: list[str] = []
        if getattr(self, "_ilu_elastic_ep", False):
            reasons.append("elastic_ep")
        if getattr(getattr(moe, "moe_parallel_config", None), "enable_eplb", False):
            reasons.append("eplb")
        if not self.do_naive_dispatch_combine:
            reasons.append("not_naive_dispatch_combine")
        if self._kernel_handles_own_dispatch:
            reasons.append("kernel_owns_dispatch")
        if getattr(self, "_fused_output_is_reduced", False):
            reasons.append("fused_output_already_reduced")
        if getattr(moe, "ep_size", 1) <= 1:
            reasons.append(f"ep_size={getattr(moe, 'ep_size', 1)}")
        if getattr(moe, "is_sequence_parallel", False):
            reasons.append("sequence_parallel")
        if getattr(moe, "pcp_size", 1) != 1:
            reasons.append(f"pcp_size={getattr(moe, 'pcp_size', 1)}")
        if getattr(self, "gate", None) is None:
            reasons.append("no_gate")
        if moe.in_dtype == torch.float16:
            reasons.append("fp16_shared_scaling")
        if self._is_unquantized_fused_moe_method():
            reasons.append("unquantized_output_scale")
        if getattr(self, "routed_output_transform", None) is not None:
            reasons.append("routed_output_transform")
        if getattr(moe, "skip_final_all_reduce", False):
            reasons.append("skip_final_all_reduce")
        if isinstance(getattr(self, "router", None), ZeroExpertRouter):
            reasons.append("zero_expert_router")
        if getattr(self, "is_monolithic", False):
            reasons.append("monolithic_quant_method")

        enabled = not reasons
        logger.debug(
            "ILU fused EP combine: %s%s",
            "enabled" if enabled else "disabled",
            "" if enabled else " (" + ", ".join(reasons) + ")",
        )
        return enabled

    def _ilu_dp_chunk_sizes(self) -> list[int]:
        """Per-DP-rank token counts, readable outside the SP context.

        Sequence parallelism is rejected by the eligibility check, so
        ``num_tokens_across_dp_cpu`` is the chunk list all-gatherv uses.
        """
        # Eligibility already rejects sequence parallelism.
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        return dp_metadata.num_tokens_across_dp_cpu.tolist()

    @property
    def _ilu_router_reads_input_ids(self) -> bool:
        """Whether ``input_ids`` still has to be put on the wire."""
        router = getattr(self, "router", None)
        return (
            getattr(router, "_hash_indices_table", None) is not None
            or getattr(router, "bias_vl", None) is not None
        )

    def _forward_impl_fused_ep_combine(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``_forward_impl`` with a 1-tensor DP gather and no combine."""
        dp_group = get_dp_group()
        sizes = self._ilu_dp_chunk_sizes()
        assert sizes[dp_group.rank_in_group] == hidden_states.shape[0], (
            f"{sizes[dp_group.rank_in_group]} != {hidden_states.shape[0]}"
        )

        with self._sequence_parallel_context():
            if input_ids is not None and self._ilu_router_reads_input_ids:
                hidden_states, input_ids = dp_group.all_gatherv(
                    [hidden_states, input_ids], dim=0, sizes=sizes
                )
            else:
                hidden_states = dp_group.all_gatherv(
                    hidden_states, dim=0, sizes=sizes
                )
                input_ids = None

            assert self.gate is not None
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

            with _gathered_router_padding(get_forward_context(), dp_group, sizes):
                shared_output, fused_output = self._apply_quant_method(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                )

        assert not isinstance(fused_output, UnfinalizedMoEOutput), (
            "fused EP combine requires a materialized routed output"
        )
        return shared_output, fused_output

    def _forward_fused_ep_combine(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """``forward`` tail with RS(DP) + AR(TP) folded into AR(EP)."""
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        # Tuple shared input from fused norm-quant has no ``.shape[-1]``.
        pad_input = (
            None if isinstance(shared_experts_input, tuple) else shared_experts_input
        )
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                pad_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = _unpack(result)
        fused_output = cast(torch.Tensor, fused_output)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]
        fused_output = fused_output.contiguous()

        sizes = self._ilu_dp_chunk_sizes()
        dp_rank = get_dp_group().rank_in_group
        start = sum(sizes[:dp_rank])
        end = start + sizes[dp_rank]
        assert fused_output.shape[0] == sum(sizes), (
            f"{fused_output.shape[0]} != {sum(sizes)}"
        )

        if shared_output is not None:
            fused_output[start:end] += shared_output

        fused_output = get_ep_group().all_reduce(fused_output)
        result = fused_output[start:end]

        if og_hidden_dim_post_xform is not None:
            result = result[..., :og_hidden_dim_post_xform]
        return result

