# SPDX-License-Identifier: Apache-2.0
"""IxInfer TN/NN v2 experts, independently composed from operator primitives."""

import ixformer.inference.functions as ops
import torch
from ixformer.inference.backend_registry import registry
from vllm.forward_context import get_forward_context

from vllm_iluvatar import envs
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)

from .ilu_compute_backend import MoEComputeBackend, MoEWeightSpec, register_moe_backend


@register_moe_backend
class IxInferW4A8Backend(MoEComputeBackend):
    name = "ixinfer_w4a8"
    priority = 20
    supported_layouts = {"ivcore11": ("NN", "TN"), "ivcore30": ("TN",)}
    weight_specs = {
        layout: MoEWeightSpec(
            f"ixinfer_w4a8_{layout.lower()}_v2", layout, torch.int8, code_multiplier=18
        )
        for layout in ("TN", "NN")
    }

    def __init__(self, config, selection):
        super().__init__(config, selection)
        # Freeze NN GEMV policy per instance before graph capture. TN keeps
        # IxFormer facade selection, as in the existing W4A8 implementation.
        self._nn_perf = envs.VLLM_ILU_MOE_W4A8_GEMV_PERF
        self._gemv_backends = (
            frozenset(
                registry.available_backends("moe_w4a8_group_gemv", selection.arch)
            )
            if selection.layout == "NN"
            else frozenset()
        )

    def _gemv(self, **args):
        backend = None
        if self.selection.layout == "NN":
            if (
                self._nn_perf
                and "ixformer_perf" in self._gemv_backends
                and ops.moe_w4a8_group_gemv_perf_supported(**args)
            ):
                backend = "ixformer_perf"
            elif "ixinfer" in self._gemv_backends:
                backend = "ixinfer"
            else:
                raise RuntimeError("No available W4A8 NN GEMV implementation")
        return ops.moe_w4a8_group_gemv(**args, backend=backend)

    def gemm(
        self,
        weights,
        x,
        scale,
        prefix,
        counts,
        dtype,
        *,
        decode=False,
        order=None,
        dense=None,
        output=None,
    ):
        args = dict(
            input=x,
            i_scales=scale,
            weight=getattr(weights, prefix + "_weight"),
            w_scales=getattr(weights, prefix + "_weight_scale"),
            output_dtype=dtype,
            tokens_per_experts=counts,
            dst_to_src=order,
            format=0 if self.selection.layout == "NN" else 2,
            group_size=-1,
            output=output,
        )
        if decode:
            return self._gemv(**args, dense_tokens_per_experts=dense)
        return ops.moe_w4a8_group_gemm(**args, version=2)

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
        ctx = get_forward_context()
        metadata = getattr(ctx, "dp_metadata", None)
        decode = _is_only_decode(ctx.attn_metadata)
        if metadata is not None and not torch.cuda.is_current_stream_capturing():
            decode = bool(metadata.all_ranks_decode_only)
        _ensure_cuda_graph_safe_dispatch(decode, ctx.attn_metadata)
        start, end = self.local_range(expert_map)
        result = ops.moe_compute_token_index_ep(
            topk_ids=topk_ids,
            num_experts=self.global_num_experts,
            start_expert_id=start,
            end_expert_id=end,
            graph_safe=decode,
            return_dense_expert_sizes=decode,
        )
        inverse, order, counts, host_counts, rows = result[:5]
        dense = result[5] if decode else None
        if decode:
            rows = topk_ids.numel()
        elif host_counts.sum() == 0:
            return self.empty_standard_output(x, output, residual)
        q, scale = ops.moe_expand_input_dynamic_scaled_int8(
            hidden_states=x,
            dst_to_src=order,
            dst_tokens=rows,
            topk=topk_ids.shape[-1],
            src_to_dst=inverse,
            output_format=int(decode),
            check_valid=decode,
        )
        selected_counts = counts if decode else host_counts
        intermediate = self.gemm(
            weights,
            q,
            scale,
            "w13",
            selected_counts,
            x.dtype,
            decode=decode,
            dense=dense,
        )
        del q, scale
        workspace.release("input", "input_scale")
        q, scale = ops.swiglu_limit_dynamic_scaled_int8(
            input=intermediate,
            limit=float(self.limit),
            output_format=int(decode),
            dst_to_src=order,
            check_valid=decode,
        )
        del intermediate
        workspace.release("gemm1")
        out = workspace.get("gemm2")
        self.gemm(
            weights,
            q,
            scale,
            "w2",
            selected_counts,
            x.dtype,
            decode=decode,
            dense=dense,
            order=order,
            output=out,
        )
        del q, scale
        workspace.release("input", "input_scale")
        return ops.moe_output_reduce_sum(
            input=out.view(x.shape[0], topk_ids.shape[-1], self.hidden),
            topk_weight=topk_weights,
            mask=inverse == -1,
            scaling_factor=self.output_scale,
            output=output,
            extra_residual=residual,
        )

    def run_batched(self, x, weights, scale, counts, *, output, workspace):
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] == 0:
            return output
        counts = counts.to(
            device=x.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        capacity = x.shape[0] // self.local_num_experts
        if x.dtype != torch.int8:
            x, scale = ops.scaled_int8_quant(x)
        elif scale is None:
            raise ValueError("INT8 dispatched activations require scales")
        q, scale, dense = ops.moe_group_gemv_reformat_input_fused_dense(
            input=x,
            input_scales=scale.reshape(-1).contiguous(),
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
        )
        intermediate = self.gemm(
            weights, q, scale, "w13", counts, output.dtype, decode=True, dense=dense
        )
        del q, scale
        workspace.release("input", "input_scale")
        q, scale = ops.moe_group_gemv_swiglu_quant(
            input=intermediate,
            expert_num_tokens=counts,
            max_num_tokens_per_expert=capacity,
            limit=self.limit,
        )
        del intermediate
        workspace.release("gemm1")
        self.gemm(
            weights,
            q,
            scale,
            "w2",
            counts,
            output.dtype,
            decode=True,
            dense=dense,
            output=output.reshape(-1, output.shape[-1]),
        )
        return output
