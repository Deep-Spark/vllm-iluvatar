# SPDX-License-Identifier: Apache-2.0
"""FP4/FP8 local experts using IxFormer primitives, with device-only routing."""

import ixformer.inference.functions as ops
import torch

from .ilu_compute_backend import MoEComputeBackend, MoEWeightSpec, register_moe_backend


@register_moe_backend
class IxMMFP4FP8Backend(MoEComputeBackend):
    name = "ixmm_fp4fp8"
    priority = 10
    supported_layouts = {"ivcore30": ("fp4fp8",)}
    weight_specs = {
        "fp4fp8": MoEWeightSpec(
            "ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256
        )
    }
    activation_dtype = torch.float8_e4m3fn
    standard_layout_columns = 3
    requires_contiguous_experts = False
    prefer_hybrid_ep = False

    def gemm(self, weights, x, scale, prefix, layout, output=None, mode=None):
        return ops.group_gemm_fp4(
            a=x,
            scale_a=scale,
            weight=getattr(weights, prefix + "_weight"),
            scale_weight=getattr(weights, prefix + "_weight_scale"),
            group_layout=layout,
            output=output,
            backend="ixmm",
            mode=mode,
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
        layout = workspace.get("layout")
        inverse, order, layout = ops.moe_compute_token_index_ep_group_layout(
            topk_ids,
            0,
            self.local_num_experts,
            expert_map=expert_map,
            group_layout=layout,
        )
        q, scale = ops.moe_expand_input_dynamic_scaled_fp8(x, order, topk_ids.shape[-1])
        intermediate = workspace.get("gemm1")
        self.gemm(weights, q, scale, "w13", layout, intermediate, mode=w13_gemm_mode)
        del q, scale
        workspace.release("input", "input_scale")
        q, scale = ops.swiglu_limit_dynamic_scaled_fp8(
            intermediate, order, float(self.limit)
        )
        padded_k = weights.w2_weight.shape[1] * 2
        if q.shape[-1] != padded_k:
            q = torch.nn.functional.pad(q, (0, padded_k - q.shape[-1]))
        del intermediate
        workspace.release("gemm1")
        out = workspace.get("gemm2")
        self.gemm(weights, q, scale, "w2", layout, out)
        del q, scale
        workspace.release("input", "input_scale")
        return ops.moe_gather_output_reduce_sum(
            out,
            topk_weights.float().contiguous(),
            inverse,
            output=output,
            extra_residual=residual,
            scaling_factor=self.output_scale,
        )

    def run_batched(self, x, weights, scale, counts, *, output, workspace):
        experts = self.local_num_experts
        x = x.reshape(-1, x.shape[-1])
        capacity = x.shape[0] // experts
        counts = counts.to(device=x.device, dtype=torch.int32).reshape(-1).contiguous()
        if x.dtype != torch.float8_e4m3fn:
            quant = workspace.get("input")
            scales = workspace.get("input_scale")
            x, scale = ops.dynamic_scaled_fp8_quant(quant, x, scales)
        elif scale is None:
            raise ValueError("FP8 dispatched activations require scales")
        layout = ops.moe_make_padded_dense_tokens_per_experts_fp8(
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
            group_layout=workspace.get("layout"),
        )
        intermediate = workspace.get("gemm1")
        self.gemm(weights, x, scale.reshape(-1), "w13", layout, intermediate)
        q, scale = ops.moe_group_gemv_swiglu_quant_fp8(
            input=intermediate,
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
            act_type="swiglu",
            limit=self.limit,
        )
        del intermediate
        workspace.release("gemm1")
        padded_k = weights.w2_weight.shape[1] * 2
        if q.shape[-1] != padded_k:
            q = torch.nn.functional.pad(q, (0, padded_k - q.shape[-1]))
        self.gemm(
            weights,
            q,
            scale.reshape(-1),
            "w2",
            layout,
            output.reshape(-1, output.shape[-1]),
        )
        return output
