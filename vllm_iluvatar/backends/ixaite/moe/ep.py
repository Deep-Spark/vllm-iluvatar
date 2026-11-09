# Source: ilu_fused_moe_ep.py at 8932464; ixaite owns the corrected EP collective.
# SPDX-License-Identifier: Apache-2.0
"""Keep the naive EP reduction on the selected ixaite communicator."""

from contextlib import nullcontext

import torch
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP as _BaselineFusedMoEEP,
    _check_shared_gate_up,
    _USE_FUSED_MOE,
)


# Override only the vendor collective; other unchanged orchestration remains shared.
class IluFusedMoEEP(_BaselineFusedMoEEP):
    # Gather, run experts, reduce through ixaite, and return this rank's token slice.
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
        assert ep_comm is not None
        ep_comm.all_reduce(final_hidden_states)
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
