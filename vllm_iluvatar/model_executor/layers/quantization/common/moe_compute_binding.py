# SPDX-License-Identifier: Apache-2.0
"""The lifecycle every converted routed-expert source shares after loading.

Backend selection, compute binding, communication adapters and ``apply`` do not
depend on the checkpoint format: once a Method has produced ``MoEWeights`` in
the selected backend's contract, everything downstream is identical. The source
formats differ only in how they reach that point, so they carry this mixin and
implement ``create_weights`` / ``process_weights_after_loading`` themselves.

``compute_source`` names the source in selection errors.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a8_moe_quant_config

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.layers.fused_moe.activation import swiglustep_limit

logger = init_logger(__name__)


def shrink_ll_max_tokens_to_decode(prepare_finalize):
    """Size the batched expert layout for decode instead of prefill.

    The community all2all factory builds DeepEP LL with
    ``moe_config.max_num_tokens`` -- the prefill bound
    (``max_num_batched_tokens``). When LL only carries decode (a
    ``VLLM_ILU_DECODE_EP_BACKEND`` override on top of a non-LL configured
    backend) the real bound is the uniform decode batch, which is also what the
    NVSHMEM buffer in ``ilu_fused_moe_ep`` is built for. The batched layout
    holds ``max_tokens_per_rank * num_dispatchers`` rows per expert, so keeping
    the prefill bound costs GiBs of workspace and output per captured graph.
    """
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
        _DECODE_EP_BACKEND,
        _decode_ll_max_tokens_per_rank,
    )

    if _DECODE_EP_BACKEND != "deepep_low_latency":
        return

    current = getattr(prepare_finalize, "max_tokens_per_rank", None)
    decode_max = _decode_ll_max_tokens_per_rank()
    if current is None or decode_max >= current:
        return

    prepare_finalize.max_tokens_per_rank = decode_max
    logger.info_once(
        "Sized DeepEP LL batched experts for decode: max_tokens_per_rank "
        "%d -> %d, matching the NVSHMEM dispatch buffer.",
        current,
        decode_max,
    )


def get_ilu_output_scale(layer):
    """The runner-side output scale, ``1.0`` unless a kernel opted in.

    ``IluMoERunner`` writes ``_ilu_output_scale`` from the runner's
    ``routed_scaling_factor`` (pinned to ``1.0`` unless
    ``apply_routed_scale_to_output`` is set). Do **not** use
    ``layer.routed_scaling_factor``; the router already folded that into
    ``topk_weights``.
    """
    return getattr(layer, "_ilu_output_scale", 1.0)


def decode_skip_sort(layer) -> bool:
    """Whether this step may route without sorting experts by token count.

    Opt-in per model (DeepSeek-V4.1 sets ``_ilu_v41_skip_sort_supported`` on
    its routed experts) and per deployment, and only for a step that carries
    no prefill: the sort exists to schedule heavy experts first, which is
    worth its kernel only once the GEMMs are large. The backend applies its
    own row bound on top; this answers the question the layer owns.
    """
    from vllm_iluvatar import envs

    if not (
        getattr(layer, "_ilu_v41_skip_sort_supported", False)
        and envs.VLLM_ILU_V41_MOE_SKIP_SORT
    ):
        return False
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    if ctx.dp_metadata is not None:
        return ctx.dp_metadata.all_ranks_decode_only
    metadata = ctx.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    return all(
        getattr(value, "num_prefill_tokens", None) == 0
        and getattr(value, "num_decode_tokens", 0) > 0
        for value in metadata.values()
    )


class MoEComputeBindingMixin:
    """Select a backend, bind the converted weights, run them."""

    compute_source = "routed experts"

    @property
    def is_monolithic(self):
        return False

    @property
    def supports_eplb(self):
        return False

    def _select_compute(self):
        if self.selection is None:
            from vllm_iluvatar.config.ilu_compute import (
                compute_options,
                select_four_bit_expert_compute,
            )
            from vllm_iluvatar.ilu_util.arch import cuda_arch

            self.selection = select_four_bit_expert_compute(
                cuda_arch(),
                compute_options("moe").get("mode", "auto"),
                source=self.compute_source,
            )
        return self.selection

    def _compute_config(self, layer):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEComputeConfig,
        )

        activation = getattr(layer.activation, "value", layer.activation)
        limit = getattr(layer, "swiglu_limit", None)
        if activation == "swiglustep":
            limit = swiglustep_limit(limit)
        return MoEComputeConfig(
            hidden=layer.moe_config.hidden_dim,
            intermediate=layer.moe_config.intermediate_size_per_partition,
            local_num_experts=layer.local_num_experts,
            global_num_experts=layer.global_num_experts,
            ep_rank=layer.moe_config.ep_rank,
            activation=activation,
            limit=limit,
            output_scale=float(get_ilu_output_scale(layer)),
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            alpha=getattr(layer, "swiglu_alpha", None),
            beta=getattr(layer, "swiglu_beta", None),
        )

    def _bind_compute(self, layer, weights=None):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEWeights,
            make_moe_compute,
        )

        if self._compute is not None:
            raise RuntimeError(f"{self.compute_source} compute has already been bound")
        compute = make_moe_compute(self._compute_config(layer), self._select_compute())
        if weights is None:
            weights = MoEWeights.from_layer(layer, compute.weight_spec)
        compute.validate_weights(weights)
        self._runtime_weights, self._compute = weights, compute
        layer._ilu_mxfp4_ixmm = self.selection.name == "ixmm_fp4fp8"

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return FusedMoEMethodBase.maybe_make_prepare_finalize(self, routing_tables)

    def get_fused_moe_quant_config(self, layer):
        if self._select_compute().name == "ixmm_fp4fp8":
            return mxfp4_w4a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                gemm1_clamp_limit=self._compute_config(layer).limit,
            )
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4",
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
            bind_compute_experts,
        )

        if self._compute is None:
            raise RuntimeError(
                f"{self.compute_source} weights must be finalized before "
                "binding communication"
            )
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            shrink_ll_max_tokens_to_decode(prepare_finalize)
        return bind_compute_experts(
            self._compute,
            self._runtime_weights,
            self.moe,
            self.moe_quant_config,
            prepare_finalize,
            layer,
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self._compute is None:
            raise RuntimeError(f"{self.compute_source} weights have not been finalized")
        compute_kwargs = {}
        output_dtype = x.dtype if isinstance(x, torch.Tensor) else x[2]
        if isinstance(x, tuple):
            q, scale, _ = x
            packed = type(scale) is int
            valid_dtype = (
                q.dtype == torch.float8_e4m3fn
                and self._compute.supports_prequantized_fp8
            ) or (q.dtype == torch.int8 and self._compute.supports_prequantized_int8)
            if (
                not valid_dtype
                or q.ndim != 2
                or output_dtype not in (torch.float16, torch.bfloat16)
            ):
                raise ValueError(
                    "Selected compute cannot consume this prequantized input"
                )
            if packed:
                if (
                    scale != self._compute.hidden
                    or q.shape[1] < scale + 4
                    or not q.is_contiguous()
                ):
                    raise ValueError("Invalid inline activation scale")
            elif (
                q.shape[1] != self._compute.hidden
                or scale.dtype != torch.float32
                or scale.device != q.device
                or scale.numel() != q.shape[0]
            ):
                raise ValueError("Prequantized input requires one FP32 scale per token")
            else:
                scale = scale.reshape(-1)
            x = q
            compute_kwargs["quantized_input"] = (q, scale)
            compute_kwargs["output"] = torch.empty(
                (q.shape[0], self._compute.hidden), device=q.device, dtype=output_dtype
            )
        residual = shared_experts_input
        if residual is not None and residual.shape[0] != x.shape[0]:
            residual = None
        # Preserve the measured Step5 W13 tile choice without selecting a
        # different weight format, quantization lifecycle or compute backend.
        mode = (
            1
            if getattr(layer, "_ilu_step5_native_activation", False)
            and ilu_envs.VLLM_ILU_STEP5_FP4_SMALL_TILES
            and topk_ids.shape[1] == 8
            and 16 < topk_ids.shape[0] <= 64
            else None
        )
        workspace = self._compute.workspace_spec(
            topk_ids.numel(), output_dtype, skip_sort=decode_skip_sort(layer)
        ).allocate(x.device)
        return self._compute.run_standard(
            x,
            self._runtime_weights,
            topk_weights,
            topk_ids,
            layer.expert_map,
            workspace=workspace,
            residual=residual,
            w13_gemm_mode=mode,
            **compute_kwargs,
        )
