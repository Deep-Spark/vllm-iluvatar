# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MoERunner — inherits community MoERunner, overrides only the
delta methods required for corex / Iluvatar hardware.

Baseline : community ``runner/moe_runner.py`` @ main_e0871ad2259
Delta    : ~250 lines vs. the original 1010-line full-class copy

Override summary (each marked with ``# [ILU override]`` in-line):

  __init__                          elastic-EP install + FP16 scaling
  _select_forward                   bypass torch.compile custom ops
  forward                           tuple hidden_states (fused norm-quant)
  _apply_quant_method               fused shared output path
  _forward_impl                     pass input_ids through dispatch
  _maybe_dispatch                   input_ids + _kernel_handles_own_dispatch
  _maybe_combine                    _kernel_handles_own_dispatch guard
  _maybe_apply_routed_scale_to_output  FP16 shared_experts_scaling_factor
  do_naive_dispatch_combine         drop supports_internal_mk check
  _kernel_handles_own_dispatch      new property (monolithic W16A16 guard)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_pcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward_shared,
    _unpack,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)

logger = init_logger(__name__)


class IluMoERunner(MoERunner):
    """Iluvatar override of the community MoERunner.

    Inherits all community behaviour and overrides only the methods whose
    logic differs on corex / Iluvatar hardware.
    """

    # [ILU override] __init__ — delta: elastic-EP patch install + FP16
    # scaling redistribution (shared_experts_scaling_factor).
    def __init__(self, *args, routed_scaling_factor: float = 1.0, **kwargs):
        super().__init__(*args, routed_scaling_factor=routed_scaling_factor, **kwargs)
        logger.info_once("Use Iluvatar MoERunner.")

        if get_current_vllm_config().parallel_config.enable_elastic_ep:
            from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep import (  # noqa: E501
                install_ilu_elastic_ep_patches,
            )

            install_ilu_elastic_ep_patches()

        self.shared_experts_scaling_factor = 1.0
        if (
            self._shared_experts is not None
            and self.moe_config.in_dtype == torch.float16
        ):
            self.routed_scaling_factor = 1.0
            self.shared_experts_scaling_factor = 1.0 / routed_scaling_factor

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
            # Pass None so community _maybe_pad_hidden_states skips
            # .shape[-1] on the tuple; safe because routed_output_transform
            # is None on Iluvatar.
            hidden_states, og_pre, og_post = self._maybe_pad_hidden_states(
                None, hidden_states
            )
        else:
            hidden_states, shared_experts_input = super().apply_routed_input_transform(
                hidden_states
            )
            hidden_states, og_pre, og_post = self._maybe_pad_hidden_states(
                shared_experts_input, hidden_states
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
    # optimisation; when NO_OVERLAP and not FP16 and not naive dispatch,
    # fuse the shared-expert add into the MoE kernel.
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
                shared_experts=self._shared_experts,
                shared_experts_input=fused_shared_output,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return_shared_output = (
            experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
            or (
                experts_order == SharedExpertsOrder.NO_OVERLAP
                and (
                    self.moe_config.in_dtype == torch.float16
                    or self.do_naive_dispatch_combine
                )
            )
        )

        return (
            self._shared_experts.output if return_shared_output else None,
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.routed_experts._ensure_moe_quant_config_init()
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        if self.gate is not None:
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

        if self.moe_config.pcp_size > 1:
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

        if self.moe_config.pcp_size > 1:
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
