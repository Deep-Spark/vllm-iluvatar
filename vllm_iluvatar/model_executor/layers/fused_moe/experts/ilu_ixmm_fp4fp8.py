# SPDX-License-Identifier: Apache-2.0
"""FP4/FP8 local experts using IxFormer primitives, with device-only routing."""

import ixformer.inference.functions as ops
import torch

from .ilu_compute_backend import MoEComputeBackend, MoEWeightSpec, register_moe_backend

# MiniMax-M3 stores gate and up as packed halves and names the activation after
# that layout; the ixformer op reads the same packing, so no interleave step
# stands between GEMM1 and it. Unlike ivcore11's INT8 counterpart it takes the
# three coefficients as arguments, so any checkpoint's values are runnable.
_SWIGLUOAI = "swigluoai_uninterleave"
_SWIGLUOAI_DEFAULTS = {"alpha": 1.702, "beta": 1.0}

# Above this many expanded rows the sort pays for itself: ixmm schedules heavy
# experts first, and the imbalance only matters once the GEMMs are large
# enough. The bound is the one the pre-backend MXFP4 path measured.
_SKIP_SORT_MAX_ROWS = 256

# Grouped-GEMM tiling for DeepSeek-V4.1's decode step. Mode 1 is a win only for
# that model's shape, so the guard is the shape itself rather than a model name.
_V41_DECODE = {"max_rows": 768, "hidden": 5120, "top_k": 6}


@register_moe_backend
class IxMMFP4FP8Backend(MoEComputeBackend):
    name = "ixmm_fp4fp8"
    priority = 10
    supported_layouts = {"ivcore30": ("fp4fp8",)}
    weight_specs = {
        "fp4fp8": MoEWeightSpec(
            "ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256, alignment=32
        )
    }
    activation_dtype = torch.float8_e4m3fn
    standard_layout_columns = 3
    requires_contiguous_experts = False
    prefer_hybrid_ep = False
    supported_activations = ("silu", "swiglu", "swiglustep", _SWIGLUOAI)
    requires_activation_limit = False
    supports_prequantized_fp8 = True

    # [ILU override] MoEComputeBackend.standard_layout_width -- delta: a third
    # column means ``[start, end, expert_id]`` sorted by descending token
    # count; two means ``[start, end]`` in natural expert order, which skips
    # the sort kernel. Only a model that opted in may ask, and only for a
    # batch small enough that the scheduling gain cannot repay the sort.
    def standard_layout_width(self, rows, skip_sort):
        return 2 if skip_sort and rows <= _SKIP_SORT_MAX_ROWS else 3

    @classmethod
    def support_error(cls, config, selection):
        error = super().support_error(config, selection)
        if error is None and config.activation == _SWIGLUOAI and config.limit is None:
            # The other activations treat a missing limit as "no clamp"; this
            # one has no unclamped form.
            return f"{cls.name} requires a clamp limit for {_SWIGLUOAI}"
        return error

    @staticmethod
    def _decode_mode(rows, hidden, top_k):
        """Grouped-GEMM tile mode for this step, or ``None`` to auto-select."""
        from vllm_iluvatar import envs

        matches = rows <= _V41_DECODE["max_rows"] and (hidden, top_k) == (
            _V41_DECODE["hidden"],
            _V41_DECODE["top_k"],
        )
        return 1 if envs.VLLM_ILU_V41_DECODE_MOE_MODE1 and matches else None

    def _activation_quant(self, intermediate, order, fp8_output=None):
        """GEMM1 output -> activation -> per-token FP8 for GEMM2."""
        if self.config.activation == _SWIGLUOAI:
            return ops.swigluoai_uninterleave_dynamic_scaled_fp8(
                intermediate,
                order,
                float(self.limit),
                **{
                    name: float(getattr(self.config, name) or default)
                    for name, default in _SWIGLUOAI_DEFAULTS.items()
                },
                fp8_output=fp8_output,
            )
        if self.config.activation == "swiglustep" or self.limit is None:
            return ops.swiglustep_dynamic_scaled_fp8(
                intermediate, order, float(self.limit or 0.0), fp8_output=fp8_output
            )
        return ops.swiglu_limit_dynamic_scaled_fp8(
            intermediate, order, float(self.limit), fp8_output=fp8_output
        )

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
        quantized_input=None,
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
        if quantized_input is None:
            q, scale = ops.moe_expand_input_dynamic_scaled_fp8(
                x, order, topk_ids.shape[-1]
            )
        elif type(quantized_input[1]) is int or (
            quantized_input[0].is_contiguous() and quantized_input[1].is_contiguous()
        ):
            # The native byte copy visits only local routes, preserving either
            # the routed packet's inline scale or separate gathered FP8 scales.
            source, source_scale = quantized_input
            packed = type(source_scale) is int
            q = workspace.get("input")
            scale = workspace.get("input_scale").reshape(-1)
            ops.moe_expand_input_ep(
                hidden_states=source.view(torch.int8),
                src_to_expert_row=inverse,
                expand_tokens=topk_ids.numel(),
                topk=topk_ids.shape[-1],
                output=q.view(torch.int8),
                output_scale=scale,
                input_scale=None if packed else source_scale,
                packed_scale_offset=source_scale if packed else None,
            )
        else:
            from ..ilu_fp8_packed_dispatch import expand_prequantized_input

            q, scale = expand_prequantized_input(
                quantized_input[0], quantized_input[1], order, topk_ids.shape[-1]
            )
        mode = self._decode_mode(topk_ids.numel(), self.hidden, topk_ids.shape[-1])
        intermediate = workspace.get("gemm1")
        self.gemm(
            weights,
            q,
            scale,
            "w13",
            layout,
            intermediate,
            mode=w13_gemm_mode if w13_gemm_mode is not None else mode,
        )
        del q, scale
        workspace.release("input", "input_scale")
        padded_k = weights.w2_weight.shape[1] * 2
        # Only clamped SwiGLU accepts a wider output and zeroes its tail.
        # SwiGLUStep and SwiGLU-OAI require the exact logical output width.
        fp8_output = None
        if (
            padded_k > self.intermediate
            and self.limit is not None
            and self.config.activation not in ("swiglustep", _SWIGLUOAI)
        ):
            fp8_output = torch.empty(
                (intermediate.shape[0], padded_k),
                dtype=self.activation_dtype,
                device=intermediate.device,
            )
        q, scale = self._activation_quant(intermediate, order, fp8_output)
        del intermediate
        workspace.release("gemm1")
        if q.shape[-1] < padded_k:
            q = torch.nn.functional.pad(q, (0, padded_k - q.shape[-1]))
        out = workspace.get("gemm2")
        self.gemm(weights, q, scale, "w2", layout, out, mode=mode)
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
        if self.config.activation == _SWIGLUOAI:
            # The batched GEMV fuses its own activation and offers only
            # swiglu / swiglustep, so there is nowhere to put the OAI form.
            raise NotImplementedError(
                f"{self.name} batched experts do not support {_SWIGLUOAI}"
            )
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
            act_type="swiglustep"
            if self.config.activation == "swiglustep"
            else "swiglu",
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
