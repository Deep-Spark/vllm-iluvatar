# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MoERunner — inherits community MoERunner, overrides only the
delta methods required for corex / Iluvatar hardware.

Baseline : community ``runner/moe_runner.py`` @ v0.30.1rc0

Override summary (each marked with ``# [ILU override]`` in-line):

  __init__                          elastic-EP install + FP16 scaling /
                                    RoutedExperts scale placement
  _select_forward                   bypass torch.compile custom ops
  forward                           tuple hidden_states (fused norm-quant)
  _apply_quant_method               fused shared output path
  _forward_impl                     pass input_ids through dispatch
  _maybe_dispatch                   input_ids + _kernel_handles_own_dispatch
  _maybe_combine                    _kernel_handles_own_dispatch guard
  _maybe_apply_routed_scale_to_output  FP16 shared_experts_scaling_factor
  do_naive_dispatch_combine         drop supports_internal_mk / SP checks
  _is_unquantized_fused_moe_method  new helper (W16A16 detection)
  _kernel_handles_own_dispatch      new property (monolithic W16A16 guard)
  _ilu_fused_ep_combine + helpers   new: 5 collectives per layer -> 2

0.30.1 shared-experts rework absorbed here: ``SharedExperts`` no longer has
``maybe_sync_shared_experts_stream`` / ``_run_in_aux_stream``.  The aux-stream
launch now happens in ``maybe_forward_async`` (called from ``_forward_impl``),
which returns whether it enqueued anything; that bool is threaded into
``_apply_quant_method``, which joins with ``wait()`` instead of re-invoking the
module under ``SharedExpertsOrder.MULTI_STREAM_OVERLAPPED``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward_shared,
    _unpack,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_moe_padding import (
    _gathered_router_padding,
)

logger = init_logger("vllm." + __name__)


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
    # discards the second return value in the tuple case. Everything after
    # ``_forward_entry`` mirrors community forward().
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(hidden_states, tuple):
            if self._ilu_fused_ep_combine:
                return self._forward_fused_ep_combine(
                    hidden_states, router_logits, input_ids, shared_experts_input
                )
            return super().forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )

        assert len(hidden_states) == 3, (
            "Expected hidden_states tuple of "
            "(quantized_output, output_scales, fp_output)"
        )
        assert shared_experts_input is None, (
            "Fused norm-quant hidden_states carries its own shared-experts "
            "input; a pre-applied routed input transform is not supported."
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
        fused_output = cast(torch.Tensor, fused_output)

        if og_pre is not None:
            fused_output = fused_output[..., :og_pre]

        # fused_output_is_reduced = self._fused_output_is_reduced

        # fused_output, fused_output_is_reduced = (
        #     self._maybe_reduce_routed_output_before_transform(
        #         fused_output,
        #         fused_output_is_reduced,
        #     )
        # )

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

    def _is_unquantized_fused_moe_method(self) -> bool:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        return isinstance(self._quant_method, UnquantizedFusedMoEMethod)

    # [ILU override] _apply_quant_method — delta: fused shared output
    # optimisation. When the shared branch runs inline (NO_OVERLAP) and the
    # kernel can take a residual, its output is handed to the routed kernel as
    # ``shared_experts_input`` and added there, so nothing is returned to the
    # runner. Unquantized W16A16 and FP16 stay unfused so the runner-side
    # routed scaling applies only to the routed output, not the shared
    # residual.
    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
        experts_order = SharedExpertsOrder.NONE
        if self._shared_experts is not None:
            experts_order = self._shared_experts._determine_shared_experts_order(
                shared_experts_input
            )

        can_fused_shared_output = (
            experts_order == SharedExpertsOrder.NO_OVERLAP
            and self.moe_config.in_dtype != torch.float16
            and not self.do_naive_dispatch_combine
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

            # Pops ``_shared_experts.output``, so the tail of this method must
            # not read it again in the fused case.
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

        if shared_experts_overlapping:
            assert self._shared_experts is not None
            self._shared_experts.wait()

        return_shared_output = experts_order in (
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        ) or (
            experts_order == SharedExpertsOrder.NO_OVERLAP
            and (
                self.moe_config.in_dtype == torch.float16
                or self.do_naive_dispatch_combine
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

    # [ILU override] _forward_impl — delta: pass input_ids through
    # _maybe_dispatch (3-tuple return).
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput]
    ):
        self.routed_experts._ensure_moe_quant_config_init()

        if self._ilu_fused_ep_combine:
            return self._forward_impl_fused_ep_combine(
                hidden_states, router_logits, shared_experts_input, input_ids
            )

        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                shared_experts_input
            )

        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = dispatch_unquantized_gemm()(
                    self, hidden_states, self._combined_gate_weight, None
                )
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            hidden_states, router_logits, input_ids = self._maybe_dispatch(
                hidden_states,
                router_logits,
                input_ids,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                shared_experts_overlapping=shared_experts_overlapping,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
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

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)
            if input_ids is not None:
                input_ids = get_pcp_group().all_gather(input_ids, dim=0)

        return hidden_states, router_logits, input_ids

    # [ILU override] _maybe_combine — delta: guard the EP combine with
    # _kernel_handles_own_dispatch, and key the paired return off
    # ``shared_output`` rather than ``self.shared_experts``. The latter is
    # required by the fused shared output path: the shared branch exists but
    # its result was folded into the routed kernel, so _apply_quant_method
    # returns None and community's ``assert shared_output is not None`` would
    # fire.
    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor | UnfinalizedMoEOutput,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]
    ):
        if self.do_naive_dispatch_combine and not self._kernel_handles_own_dispatch:
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "Naive expert-parallel combine cannot consume a deferred "
                    "MoE output."
                )
            hidden_states = get_ep_group().combine(
                hidden_states,
                self.moe_config.is_sequence_parallel,
            )

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "PCP reduce-scatter cannot consume a deferred MoE output."
                )
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)

        if shared_output is not None:
            return shared_output, hidden_states
        else:
            return hidden_states

    # [ILU override] _maybe_apply_routed_scale_to_output — delta: FP16 uses
    # shared_experts_scaling_factor instead of dividing fused_output (avoids
    # FP16 overflow); every other dtype has the routed scale folded into
    # topk_weights by the router, except the unquantized W16A16 kernel which
    # scales after reduction via ``_ilu_output_scale``.
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
    # supports_internal_mk check (Iluvatar kernels do not implement internal
    # MK dispatch) and the is_sequence_parallel term (SP is not supported on
    # corex, and the Iluvatar kernels own the SP scatter themselves).
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
    # collective launches per MoE layer:
    #
    #   _maybe_dispatch   all_gatherv([hidden, router_logits, input_ids])
    #                     -> ilu_communicator issues one all_gather_into_tensor
    #                        per list entry, i.e. 3 launches
    #   _maybe_combine    reduce_scatterv over the DP group            -> 1
    #   _maybe_reduce_final_output  all_reduce over the TP group       -> 1
    #
    # At batch 1 each one moves a few KB, so the layer pays five launch
    # latencies (times 40 layers, inside the decode CUDA graph) to move
    # almost nothing. Three of the five are removable:
    #
    #   * router_logits is a per-token GEMM of hidden_states. Running the
    #     gate *after* the gather reproduces exactly the same logits on
    #     every rank without putting them on the wire.
    #   * input_ids is read only by hash routing and the vision sentinel bias
    #     (see ``_ilu_router_reads_input_ids``).
    #   * RS(DP) + AR(TP) == AR(EP) once the shared-expert partial is folded
    #     into this rank's own shard: with EP, tp_size is flattened into
    #     ep_size = dp_size * tp_size and every expert lives on exactly one
    #     EP rank, so summing the routed partials over the EP group is the
    #     complete routed sum; and the only ranks holding a nonzero shared
    #     term for row ``t`` are the TP peers of the DP rank that owns ``t``,
    #     which are all inside the same EP group.
    #
    # This is the reduction ``IluFusedMoEEP.forward_dp_ep_naive`` already
    # performs. It is unreachable for DeepSeek-V4.1, whose FFN drives the
    # runner directly instead of going through ``IluFusedMoEEP.forward_dp_ep``.
    # Note the accumulation order changes (one bf16 reduction tree instead of
    # RS-then-AR), so this is not bit-identical to the community path.
    #
    # Kill switch: ``VLLM_ILU_FUSE_EP_COMBINE=0``.
    @property
    def _ilu_fused_ep_combine(self) -> bool:
        cached = self.__dict__.get("_ilu_fused_ep_combine_cached")
        if cached is None:
            cached = self._ilu_compute_fused_ep_combine()
            self.__dict__["_ilu_fused_ep_combine_cached"] = cached
        return cached

    def _ilu_compute_fused_ep_combine(self) -> bool:
        """Config-only eligibility check, evaluated once on first forward.

        Deferred past ``__init__`` because ``_fused_output_is_reduced``
        depends on the modular kernel, which is only attached in
        ``process_weights_after_loading``. Everything read here is static
        afterwards, with the two exceptions (elastic EP, EPLB) rejected up
        front, so the result is safe to cache.
        """
        if not envs.VLLM_ILU_FUSE_EP_COMBINE:
            return False

        moe = self.moe_config
        reasons: list[str] = []
        # Both can swap the quant method / expert map at runtime, which would
        # invalidate the cached decision below.
        if self._ilu_elastic_ep:
            reasons.append("elastic_ep")
        if moe.moe_parallel_config.enable_eplb:
            reasons.append("eplb")
        # Must be the naive AG/RS DP+EP schedule this fast path replaces.
        if not self.do_naive_dispatch_combine:
            reasons.append("not_naive_dispatch_combine")
        if self._kernel_handles_own_dispatch:
            reasons.append("kernel_owns_dispatch")
        if self._fused_output_is_reduced:
            reasons.append("fused_output_already_reduced")
        if moe.ep_size <= 1:
            reasons.append(f"ep_size={moe.ep_size}")
        # SP redistributes tokens itself; PCP adds a second gather domain.
        if moe.is_sequence_parallel:
            reasons.append("sequence_parallel")
        if moe.pcp_size != 1:
            reasons.append(f"pcp_size={moe.pcp_size}")
        # The gate must be local, otherwise router_logits has to be gathered.
        if self.gate is None:
            reasons.append("no_gate")
        # These all need work between the routed output and the reduction,
        # which the single fused all-reduce has no place for.
        if moe.in_dtype == torch.float16:
            reasons.append("fp16_shared_scaling")
        if self._is_unquantized_fused_moe_method():
            reasons.append("unquantized_output_scale")
        if self.routed_output_transform is not None:
            reasons.append("routed_output_transform")
        if moe.skip_final_all_reduce:
            reasons.append("skip_final_all_reduce")
        if isinstance(self.router, ZeroExpertRouter):
            reasons.append("zero_expert_router")
        if self.is_monolithic:
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

        ``DPMetadata.local_sizes`` is only filled inside
        ``_sequence_parallel_context``, and the slice in
        ``_forward_fused_ep_combine`` runs after that context has exited.
        With sequence parallelism rejected by the eligibility check,
        ``sp_size == 1`` and ``_compute_sp_num_tokens`` is the identity, so
        ``num_tokens_across_dp_cpu`` is the chunk list the community
        all-gather uses.
        """
        assert self.moe_config.sp_size == 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        return dp_metadata.num_tokens_across_dp_cpu.tolist()

    @property
    def _ilu_router_reads_input_ids(self) -> bool:
        """Whether ``input_ids`` still has to be put on the wire.

        Across every community router, ``input_ids`` is read in exactly two
        cases: hash routing (``_hash_indices_table``, DeepSeek-V4 hash layers)
        and the vision sentinel bias (``bias_vl`` in FusedTopKBiasRouter).
        Text-only DeepSeek-V4.1 has neither, so its dispatch drops to a single
        tensor; the VL checkpoint brings ``bias_vl`` back.
        The monolithic path, which also forwards ``input_ids``, is excluded by
        ``_ilu_compute_fused_ep_combine``.
        """
        return (
            getattr(self.router, "_hash_indices_table", None) is not None
            or getattr(self.router, "bias_vl", None) is not None
        )

    def _forward_impl_fused_ep_combine(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``_forward_impl`` with a 1-tensor dispatch and no combine.

        Returns the routed output still spanning *all* DP rows and still
        unreduced; ``_forward_fused_ep_combine`` finishes it with one EP
        all-reduce. The incoming ``router_logits`` (computed by the caller on
        this rank's rows only) is discarded and recomputed after the gather.
        """
        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                shared_experts_input
            )

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
                hidden_states = dp_group.all_gatherv(hidden_states, dim=0, sizes=sizes)
                input_ids = None

            assert self.gate is not None
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = dispatch_unquantized_gemm()(
                    self, hidden_states, self._combined_gate_weight, None
                )
            else:
                router_logits, _ = self.gate(hidden_states)

            with _gathered_router_padding(get_forward_context(), dp_group, sizes):
                shared_output, fused_output = self._apply_quant_method(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                    shared_experts_overlapping=shared_experts_overlapping,
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
        """``MoERunner.forward`` tail with RS(DP) + AR(TP) folded into AR(EP).

        ``_ilu_compute_fused_ep_combine`` has already ruled out every step the
        community tail would do between the two outputs and the reduction
        (fp16 shared scaling, unquantized output scale, routed output
        transform, zero experts, pre-reduced fused output), so the only work
        left here is the shared add and the reduction itself.
        """
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
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
        # The trim above can leave a strided view, which the collective below
        # cannot consume; no-op when nothing was actually trimmed.
        fused_output = fused_output.contiguous()

        # fused_output still covers every DP rank's rows, in the gather order
        # all_gatherv used, so this rank owns [start, end).
        sizes = self._ilu_dp_chunk_sizes()
        dp_rank = get_dp_group().rank_in_group
        start = sum(sizes[:dp_rank])
        end = start + sizes[dp_rank]
        assert fused_output.shape[0] == sum(sizes), (
            f"{fused_output.shape[0]} != {sum(sizes)}"
        )

        # Fold this rank's TP-partial shared output into its own rows so the
        # single all-reduce below sums it together with the routed partials.
        if shared_output is not None:
            fused_output[start:end] += shared_output

        # One reduction over DP x TP replaces reduce_scatter(DP) +
        # all_reduce(TP). IluCommunicator.all_reduce is a single in-place
        # ixfd.all_reduce.
        fused_output = get_ep_group().all_reduce(fused_output)
        result = fused_output[start:end]

        if og_hidden_dim_post_xform is not None:
            result = result[..., :og_hidden_dim_post_xform]
        return result
