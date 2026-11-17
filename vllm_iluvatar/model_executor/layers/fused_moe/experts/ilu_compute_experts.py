# SPDX-License-Identifier: Apache-2.0
"""Bind local compute to the community prepare/experts/finalize protocol."""

from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExpertsModular
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)


class _IluComputeExperts(FusedMoEExpertsModular):
    """Shared backend binding, without choosing an activation/reduction protocol."""

    def __init__(
        self,
        backend,
        weights,
        moe_config,
        quant_config,
        max_num_tokens,
        num_dispatchers,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self.backend = backend
        self.weights = weights
        self.local_num_experts = backend.local_num_experts

    @staticmethod
    def _supports_current_device():
        return True  # Construction is guarded by the architecture selector.

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        # Explicit selector only; do not advertise unvalidated Oracle matches.
        return False

    @staticmethod
    def _supports_activation(activation):
        return activation.value in ("silu", "swiglu")

    @staticmethod
    def _supports_parallel_config(config):
        return True

    def supports_expert_map(self):
        return True

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return (
            w1.shape[0],
            a1.numel() // a1.shape[-1],
            self.backend.intermediate * 2,
            a1.shape[-1],
            topk_ids.shape[-1],
        )


class IluBatchedComputeExperts(_IluComputeExperts):
    """Expert-major compute; Finalize owns TopK weighting and output scaling."""

    @staticmethod
    def activation_format():
        return FusedMoEActivationFormat.BatchedExperts

    @property
    def runner_scaling_factor(self):
        # run_batched returns unscaled expert values for communication combine.
        return self.backend.output_scale

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0, N), (0, K), (local_num_experts, M // local_num_experts, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    ):
        if expert_tokens_meta is None:
            raise ValueError("Batched expert compute requires valid token counts")
        if apply_router_weight_on_input:
            raise NotImplementedError("Compute requires output TopK weighting")
        workspace = self.backend.workspace_spec(
            hidden_states.numel() // hidden_states.shape[-1], output.dtype, batched=True
        ).allocate(hidden_states.device)
        return self.backend.run_batched(
            hidden_states,
            self.weights,
            a1q_scale,
            expert_tokens_meta.expert_num_tokens,
            output=output,
            workspace=workspace,
        )


class IluStandardComputeExperts(_IluComputeExperts):
    """Token-major compute; the backend owns TopK reduction and output scaling."""

    @property
    def runner_scaling_factor(self):
        # run_standard has already applied the layer's output scale.
        return 1.0

    @staticmethod
    def activation_format():
        return FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self):
        return True

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0, N), (0, K), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    ):
        if apply_router_weight_on_input:
            raise NotImplementedError("Original MXFP4 computes TopK weights on output")
        workspace = self.backend.workspace_spec(
            topk_ids.numel(), output.dtype
        ).allocate(hidden_states.device)
        self.backend.run_standard(
            hidden_states,
            self.weights,
            topk_weights,
            topk_ids,
            expert_map,
            output=output,
            workspace=workspace,
        )
        return output


def bind_compute_experts(backend, weights, moe_config, quant_config, prepare_finalize):
    """Bind only supported communication formats to the same per-layer backend."""
    fmt = prepare_finalize.activation_format
    if fmt == FusedMoEActivationFormat.BatchedExperts:
        cls = IluBatchedComputeExperts
        capacity = prepare_finalize.max_num_tokens_per_rank()
        if capacity is None:
            raise ValueError("Batched experts require max_num_tokens_per_rank")
        dispatchers = prepare_finalize.num_dispatchers()
    elif fmt == FusedMoEActivationFormat.Standard:
        cls, capacity, dispatchers = IluStandardComputeExperts, None, None
    else:
        raise NotImplementedError(f"Unsupported communication activation format {fmt}")
    return cls(backend, weights, moe_config, quant_config, capacity, dispatchers)
