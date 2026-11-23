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

from .ilu_compute_backend import (
    MoEComputeBackend,
    MoEWeightSpec,
    MoEWorkspaceSpec,
    register_moe_backend,
)

# SwiGLU-OAI, in the two layouts a checkpoint can declare. The activation op is
# interleaved-only, so the packed-halves variant is converted before the call.
_SWIGLUOAI = "swigluoai"
_SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
_SWIGLUOAI_ACTIVATIONS = (_SWIGLUOAI, _SWIGLUOAI_UNINTERLEAVE)

# ``activation_swigluoai_dynamic_scaled_int8`` takes no coefficients: it applies
# ``ref_swigluoai_and_mul``, which clamps at 7.0, scales the swish gate by 1.702
# and adds 1.0 to the up branch. A checkpoint asking for anything else has to be
# rejected rather than silently run as this activation.
_SWIGLUOAI_LIMIT, _SWIGLUOAI_ALPHA, _SWIGLUOAI_BETA = 7.0, 1.702, 1.0

# Interleaved GEMM1 output, held only for SwiGLU-OAI.
_INTERLEAVED = "gemm1_interleaved"


@register_moe_backend
class IxInferW4A8Backend(MoEComputeBackend):
    name = "ixinfer_w4a8"
    pack_version = 2
    priority = 20
    supported_layouts = {"ivcore11": ("NN", "TN"), "ivcore30": ("TN",)}
    supported_activations = ("silu", "swiglu", *_SWIGLUOAI_ACTIVATIONS)
    weight_specs = {
        layout: MoEWeightSpec(
            f"ixinfer_w4a8_{layout.lower()}_v2", layout, torch.int8, code_multiplier=18
        )
        for layout in ("TN", "NN")
    }

    # ``moe_w4a8_group_gemm`` / ``moe_w4a8_group_gemv`` shape rules per weight
    # layout, as ``(N multiple, K multiple)``; both also need ``K >= 256``. The
    # kernels enforce exactly this and say so when they refuse -- the same
    # bounds ``compressed_tensors/ilu_utils.py`` W4A8GroupGemmInfo pads to.
    _SHAPE_RULES = {"NN": (128, 64), "TN": (16, 128)}
    _MIN_K = 256

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

    # [ILU override] MoEComputeBackend.dimension_error -- delta: the rule the
    # W4A8 kernels actually enforce, instead of the weight spec's single 256.
    # That scalar is a proxy: it is sufficient but not necessary, and it turns
    # away shapes the kernels run, e.g. MiniMax-M3 at TP8, whose intermediate
    # is 384 per rank. N and K differ per projection, so each is checked.
    @classmethod
    def dimension_error(cls, config, layout):
        n_multiple, k_multiple = cls._SHAPE_RULES[layout]
        for projection, n, k in (
            ("w13", 2 * config.intermediate, config.hidden),
            ("w2", config.hidden, config.intermediate),
        ):
            if n % n_multiple or k % k_multiple or k < cls._MIN_K:
                return (
                    f"{cls.name} {layout} requires {projection} N % "
                    f"{n_multiple} == 0 and K % {k_multiple} == 0 with K >= "
                    f"{cls._MIN_K}; got N={n}, K={k}"
                )
        return None

    # [ILU override] MoEComputeBackend.support_error -- delta: SwiGLU-OAI
    # coefficients are baked into the activation op, so they are part of what
    # this backend can and cannot consume.
    @classmethod
    def support_error(cls, config, selection):
        error = super().support_error(config, selection)
        if error is not None or config.activation not in _SWIGLUOAI_ACTIVATIONS:
            return error
        for name, value, baked in (
            ("limit", config.limit, _SWIGLUOAI_LIMIT),
            ("alpha", config.alpha, _SWIGLUOAI_ALPHA),
            ("beta", config.beta, _SWIGLUOAI_BETA),
        ):
            # None means the layer left it at the activation's own default,
            # which is the baked value.
            if value is not None and float(value) != baked:
                return (
                    f"{cls.name} runs SwiGLU-OAI through an op with {name}="
                    f"{baked} compiled in; this layer asks for {value}"
                )
        return None

    # [ILU override] MoEComputeBackend.workspace_spec -- delta: SwiGLU-OAI needs
    # a second [rows, 2I] buffer, because its op reads interleaved gate/up pairs
    # while the grouped GEMM writes packed halves.
    def workspace_spec(self, rows, dtype, *, batched=False, skip_sort=False):
        spec = super().workspace_spec(rows, dtype, batched=batched, skip_sort=skip_sort)
        if self.config.activation not in _SWIGLUOAI_ACTIVATIONS:
            return spec
        return MoEWorkspaceSpec(
            {
                **spec.buffers,
                _INTERLEAVED: ((rows, 2 * self.intermediate), dtype),
            }
        )

    def _activation_quant(self, intermediate, order, decode, workspace):
        """GEMM1 output -> activation -> per-row INT8 for GEMM2."""
        if self.config.activation not in _SWIGLUOAI_ACTIVATIONS:
            return ops.swiglu_limit_dynamic_scaled_int8(
                input=intermediate,
                limit=float(self.limit),
                output_format=int(decode and self.pack_version == 2),
                dst_to_src=order,
                check_valid=decode,
            )
        if self.config.activation == _SWIGLUOAI_UNINTERLEAVE:
            rows, columns = intermediate.shape
            interleaved = workspace.get(_INTERLEAVED, rows)
            half = columns // 2
            interleaved[:, 0::2].copy_(intermediate[:, :half])
            interleaved[:, 1::2].copy_(intermediate[:, half:])
            intermediate = interleaved
        return ops.activation_swigluoai_dynamic_scaled_int8(
            input=intermediate,
            dst_to_src=order,
            output_format=int(decode and self.pack_version == 2),
            check_valid=decode,
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
        if decode and self.pack_version == 1:
            if dtype != torch.bfloat16:
                raise ValueError("IxInfer NN/v1 Graph GEMM requires BF16 output")
            if output is None:
                output = torch.empty(
                    (x.shape[0], args["w_scales"].shape[-1]),
                    device=x.device,
                    dtype=dtype,
                )
            args.update(i_scales=scale.reshape(-1).contiguous(), output=output)
            ops.moe_w4a8_group_gemm(**args, version=1, enable_cuda_graph=True)
            return output
        if decode:
            return self._gemv(**args, dense_tokens_per_experts=dense)
        return ops.moe_w4a8_group_gemm(**args, version=self.pack_version)

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
        gemv = decode and self.pack_version == 2
        start, end = self.local_range(expert_map)
        result = ops.moe_compute_token_index_ep(
            topk_ids=topk_ids,
            num_experts=self.global_num_experts,
            start_expert_id=start,
            end_expert_id=end,
            graph_safe=decode,
            return_dense_expert_sizes=gemv,
        )
        inverse, order, counts, host_counts, rows = result[:5]
        dense = result[5] if gemv else None
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
            output_format=int(gemv),
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
        q, scale = self._activation_quant(intermediate, order, decode, workspace)
        del intermediate
        workspace.release("gemm1", _INTERLEAVED)
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
        if self.config.activation in _SWIGLUOAI_ACTIVATIONS:
            # ``moe_group_gemv_swiglu_quant`` takes no activation argument; it
            # is SwiGLU only. Computing it here would quietly be the wrong
            # activation, so the batched (DeepEP LL) path refuses SwiGLU-OAI.
            raise NotImplementedError(
                f"{self.name} batched experts have no SwiGLU-OAI activation op"
            )
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


@register_moe_backend
class IxInferW4A8NNV1Backend(IxInferW4A8Backend):
    """BI150 NN/v1: row scales include the INT4 divisor and Graph GEMM consumes device counts."""

    name = "ixinfer_w4a8_nn_v1"
    priority = 40
    pack_version = 1
    supports_batched_experts = False
    supported_layouts = {"ivcore11": ("NN",)}
    weight_specs = {"NN": MoEWeightSpec("ixinfer_w4a8_nn_v1", "NN", torch.int8)}

    @classmethod
    def support_error(cls, config, selection):
        if config.local_num_experts < 2:
            return "IxInfer NN/v1 Graph GEMM requires at least two local experts"
        return super().support_error(config, selection)

    def run_batched(self, x, weights, scale, counts, *, output, workspace):
        raise NotImplementedError("IxInfer NN/v1 requires Standard expert activations")
