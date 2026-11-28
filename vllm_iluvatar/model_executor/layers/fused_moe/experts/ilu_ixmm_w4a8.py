# SPDX-License-Identifier: Apache-2.0
"""IxMM INT4/INT8 experts; counts and expert ranges stay on device."""

from dataclasses import replace

import ixformer.inference.functions as ops
import torch

from vllm_iluvatar.model_executor.layers.fused_moe.activation import swiglustep_limit

from .ilu_compute_backend import MoEComputeBackend, MoEWeightSpec, register_moe_backend


@register_moe_backend
class IxMMW4A8Backend(MoEComputeBackend):
    name = "ixmm_w4a8"
    priority = 30
    supports_prequantized_int8 = True
    # activation_dynamic_scaled_int8 preserves plain SwiGLU when limit is None.
    requires_activation_limit = False
    supported_activations = ("silu", "swiglu", "swiglustep", "swigluoai_uninterleave")
    _N_ALIGNMENT = 128
    _K_ALIGNMENT = 64
    supported_layouts = {"ivcore11": ("NN",), "ivcore30": ("NN",)}
    weight_specs = {"NN": MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8)}

    @classmethod
    def padded_config(cls, config):
        # Gate/up is fused along N; each half shares its width with W2 K.
        i = (config.intermediate + cls._K_ALIGNMENT - 1) // cls._K_ALIGNMENT
        w13_k = (config.hidden + cls._K_ALIGNMENT - 1) // cls._K_ALIGNMENT
        w2_n = (config.hidden + cls._N_ALIGNMENT - 1) // cls._N_ALIGNMENT
        return replace(
            config,
            intermediate=i * cls._K_ALIGNMENT,
            w13_k_padded=w13_k * cls._K_ALIGNMENT,
            w2_n_padded=w2_n * cls._N_ALIGNMENT,
        )

    @classmethod
    def dimension_error(cls, config, layout):
        if min(config.w13_k, config.w2_n) < config.hidden:
            return "ixmm_w4a8 runtime hidden dimensions are smaller than logical hidden"
        for projection, n, k in (
            ("w13", 2 * config.intermediate, config.w13_k),
            ("w2", config.w2_n, config.w2_k),
        ):
            if n % cls._N_ALIGNMENT or k % cls._K_ALIGNMENT:
                return (
                    f"ixmm_w4a8 requires {projection} N % {cls._N_ALIGNMENT} == 0 "
                    f"and K % {cls._K_ALIGNMENT} == 0; got N={n}, K={k}"
                )
        return None

    @classmethod
    def support_error(cls, config, selection):
        error = super().support_error(config, selection)
        if error is not None or config.activation != "swigluoai_uninterleave":
            return error
        for name, expected in (("limit", 7.0), ("alpha", 1.702), ("beta", 1.0)):
            value = getattr(config, name)
            if value is not None and float(value) != expected:
                return f"IxMM SwiGLU-OAI INT8 requires {name}={expected}, got {value}"
        return None

    def __init__(self, config, selection):
        super().__init__(config, selection)
        if config.activation == "swiglustep":
            self.limit = swiglustep_limit(self.limit)

    def _swiglustep_quant(self, intermediate):
        activated = ops.swiglustep_and_mul(input=intermediate, limit=self.limit)
        return ops.scaled_int8_quant(activated)

    def _swigluoai_quant(self, intermediate):
        # IxMM W13 produces two contiguous halves, while the INT8 op
        # consumes interleaved gate/up pairs.
        gate, up = intermediate.chunk(2, dim=-1)
        interleaved = torch.stack((gate, up), dim=-1).flatten(1)
        return ops.activation_swigluoai_dynamic_scaled_int8(
            input=interleaved, output_format=0
        )

    @property
    def supports_batched_experts(self) -> bool:
        # The padded expert layout builder is provided by the ivcore30 backend.
        return self.selection.arch == "ivcore30"

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
        quantized_input=None,
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
        padded_hidden = self.config.w13_k != self.hidden
        q = (
            torch.empty(
                (topk_ids.numel(), self.hidden), device=x.device, dtype=torch.int8
            )
            if padded_hidden
            else workspace.get("input")
        )
        scale = workspace.get("input_scale").reshape(-1)
        # BI150 uses the separate quantize and EP expand kernels.
        if self.selection.arch == "ivcore11" and quantized_input is None:
            quantized_input = ops.scaled_int8_quant(x)
        if quantized_input is None:
            ops.moe_expand_input_ep_dynamic_scaled_int8(x, inverse, q, scale, topk=k)
        else:
            source, source_scale = quantized_input
            packed = type(source_scale) is int
            ops.moe_expand_input_ep(
                hidden_states=source,
                src_to_expert_row=inverse,
                expand_tokens=topk_ids.numel(),
                topk=k,
                output=q,
                output_scale=scale,
                input_scale=None if packed else source_scale,
                packed_scale_offset=source_scale if packed else None,
            )
        if padded_hidden:
            q = torch.nn.functional.pad(q, (0, self.config.w13_k - self.hidden))
        intermediate = workspace.get("gemm1")
        self.gemm(weights, q, scale, "w13", layout, intermediate)
        del q, scale
        workspace.release("input", "input_scale")
        if self.config.activation == "swiglustep":
            q, scale = self._swiglustep_quant(intermediate)
        elif self.config.activation == "swigluoai_uninterleave":
            q, scale = self._swigluoai_quant(intermediate)
        else:
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
        if self.config.w2_n != self.hidden:
            out = out[:, : self.hidden].contiguous()
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
        if x.shape[-1] != self.config.w13_k:
            x = torch.nn.functional.pad(x, (0, self.config.w13_k - x.shape[-1]))
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
        if self.config.activation == "swiglustep":
            q, scale = self._swiglustep_quant(intermediate)
        elif self.config.activation == "swigluoai_uninterleave":
            q, scale = self._swigluoai_quant(intermediate)
        else:
            q, scale = ops.moe_group_gemm_swiglu_quant(
                input=intermediate,
                expert_num_tokens=counts,
                max_num_tokens_per_expert=capacity,
                limit=self.limit,
            )
        del intermediate
        workspace.release("gemm1")
        output_rows = output.reshape(-1, output.shape[-1])
        needs_crop = self.config.w2_n != self.hidden
        gemm_output = workspace.get("gemm2") if needs_crop else output_rows
        self.gemm(weights, q, scale, "w2", layout, gemm_output)
        if needs_crop:
            output_rows.copy_(gemm_output[:, : self.hidden])
        return output
