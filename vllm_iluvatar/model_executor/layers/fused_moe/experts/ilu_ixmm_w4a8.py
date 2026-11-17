# SPDX-License-Identifier: Apache-2.0
"""IxMM INT4/INT8 experts; counts and expert ranges stay on device."""

import ixformer.inference.functions as ops
import torch

from .ilu_compute_backend import MoEComputeBackend, MoEWeightSpec, register_moe_backend


@register_moe_backend
class IxMMW4A8Backend(MoEComputeBackend):
    name = "ixmm_w4a8"
    priority = 30
    supported_layouts = {"ivcore30": ("NN",)}
    weight_specs = {"NN": MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8)}

    def gemm(self, weights, x, scale, prefix, layout, output=None, mode=None):
        return ops.moe_w4a8_group_gemm_ixmm(
            a=x,
            scale_a=scale.reshape(-1).contiguous(),
            weight=getattr(weights, prefix + "_weight"),
            weight_scale=getattr(weights, prefix + "_weight_scale").flatten(1),
            group_layout=layout,
            output=output,
            backend="ixmm",
        )

    def run_standard(
        self,
        x,
        weights,
        topk_weights,
        topk_ids,
        expert_map,
        *,
        workspace,
        output=None,
        residual=None,
        w13_gemm_mode=None,
    ):
        if x.shape[0] == 0:
            return self.empty_standard_output(x, output, residual)
        k = topk_ids.shape[-1]
        start, end = self.local_range(expert_map)
        layout = workspace.get("layout")
        mapping = workspace.get("mapping")
        inverse, _, routing_weights, offsets, _, valid = (
            ops.moe_compute_token_index_ep_packed(
                topk_ids=topk_ids.to(torch.int32),
                num_experts=self.global_num_experts,
                start_expert_id=start,
                end_expert_id=end,
                topk_weights=topk_weights,
                graph_safe=True,
                group_layout=layout,
                packed_to_expert_row=mapping,
            )
        )
        q = workspace.get("input")
        scale = workspace.get("input_scale").reshape(-1)
        ops.moe_expand_input_ep_dynamic_scaled_int8(x, inverse, q, scale, topk=k)
        intermediate = workspace.get("gemm1")
        self.gemm(weights, q, scale, "w13", layout, intermediate)
        del q, scale
        workspace.release("input", "input_scale")
        q, scale = ops.activation_dynamic_scaled_int8(
            input=intermediate,
            act_type="swiglu",
            output_format=0,
            expand_tokens_gpu=valid,
            limit=self.limit,
        )
        del intermediate
        workspace.release("gemm1")
        out = workspace.get("gemm2")
        self.gemm(weights, q, scale, "w2", layout, out)
        del q, scale
        workspace.release("input", "input_scale")
        return ops.moe_output_reduce_sum_ep(
            input=out,
            token_row_offsets=offsets,
            weights=routing_weights,
            scaling_factor=self.output_scale,
            output=output,
            extra_residual=residual,
            packed_to_expert_row=mapping,
        )

    def run_batched(self, x, weights, scale, counts, *, output, workspace):
        x = x.reshape(-1, x.shape[-1])
        if not counts.is_cuda:
            raise ValueError("IxMM batched experts require device token counts")
        counts = counts.to(torch.int32).reshape(-1).contiguous()
        capacity = x.shape[0] // self.local_num_experts
        if x.dtype != torch.int8:
            x, scale = ops.scaled_int8_quant(x)
        elif scale is None:
            raise ValueError("INT8 dispatched activations require scales")
        layout = ops.moe_make_padded_dense_tokens_per_experts_fp8(
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
            group_layout=workspace.get("layout"),
        )
        intermediate = workspace.get("gemm1")
        self.gemm(weights, x, scale, "w13", layout, intermediate)
        q, scale = ops.moe_group_gemm_swiglu_quant(
            input=intermediate,
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
            limit=self.limit,
        )
        del intermediate
        workspace.release("gemm1")
        self.gemm(weights, q, scale, "w2", layout, output.reshape(-1, output.shape[-1]))
        return output
