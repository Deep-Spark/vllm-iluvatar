# SPDX-License-Identifier: Apache-2.0
"""Dense FFN and MoE blocks for HY V4 on Iluvatar.

Corresponds to: vllm/models/hy_v4/nvidia/moe.py (upstream #54160)

Both blocks are built entirely out of community infrastructure
(`MergedColumnParallelLinear` / `RowParallelLinear` / `FusedMoEFactory` /
`GateLinear`), all of which exist unchanged in the Iluvatar baseline. The
Iluvatar delta for the routed experts lives in the quantization method
(`IluModelOptMxFp8Config` -> `IluMxFp8FusedMoE`), which `FusedMoEFactory`
reaches through ``quant_config``, so nothing model-side has to change.

The FP32 sigmoid-top8 path uses BF16 operands with FP32 accumulation/output
when its FP32 router weights are exactly representable as BF16. The derived
weight is prepared after loading; other weights retain the FP32 fallback.
The legacy grouped-topk path keeps its BF16/FP16 input contract.

``swiglu_limit`` is threaded through `FusedMoEFactory` so the clamp travels via
the quant config to the expert backend; `IluMxFp8FusedMoE.apply` reads it off
the layer and selects ixformer's ``dsv4_clamp`` SwiGLU variant.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_iluvatar.ilu_util.arch import IVCORE11, IVCORE30, cuda_arch
from vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_gate_linear import (
    IluGateLinear,
)

logger = init_logger(__name__)


class _HYV4RouterLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.prepare_router_weight()


class HYV4RouterGate(IluGateLinear):
    """HY4 router: preserve FP32 parameters/logits and cache lossless BF16 W.

    The standard loader postprocessing hook owns the derived buffer. Checkpoint
    loads invalidate it; changed/replaced parameters cannot consume stale data.
    Explicit architecture dispatch leaves other models' gate policy unchanged.
    """

    def __init__(self, input_size: int, output_size: int, prefix: str):
        super().__init__(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=prefix,
        )
        self.quant_method = _HYV4RouterLinearMethod()
        self.router_backend = {IVCORE11: "ixinfer", IVCORE30: "ixmm"}.get(cuda_arch())
        self.register_buffer("_router_weight_bf16", None, persistent=False)
        self._router_weight_version = -1
        self._router_weight_ptr = 0

    def weight_loader(self, param, loaded_weight):
        self._router_weight_bf16 = None
        super().weight_loader(param, loaded_weight)

    def prepare_router_weight(self) -> None:
        self._router_weight_bf16 = None
        if self.router_backend is None:
            return
        from ixformer.inference.backend_registry import registry

        if self.router_backend not in registry.available_backends(
            "gemm_bf16_bf16_fp32"
        ):
            raise RuntimeError(f"HYV4 router requires backend {self.router_backend!r}")
        weight = self.weight.detach()
        bf16 = weight.to(torch.bfloat16).contiguous()
        if not torch.equal(bf16.float(), weight):
            logger.warning(
                "HYV4 router %s retains FP32 GEMM: weight is not lossless BF16",
                self.prefix,
            )
            return
        self._router_weight_bf16 = bf16
        self._router_weight_version = (
            None if self.weight.is_inference() else self.weight._version
        )
        self._router_weight_ptr = self.weight.data_ptr()
        logger.info(
            "HYV4 router BF16 operands ready: %s backend=%s logits=FP32",
            self.prefix,
            self.router_backend,
        )

    def forward(self, x: torch.Tensor):
        weight = self._router_weight_bf16
        if (
            weight is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.device == weight.device
            and self._router_weight_version
            == (None if self.weight.is_inference() else self.weight._version)
            and self.weight.data_ptr() == self._router_weight_ptr
        ):
            import ixformer.inference.functions as ixf_ops

            return ixf_ops.gemm_bf16_bf16_fp32(
                x,
                weight,
                format="TN",
                backend=self.router_backend,
            ), None
        return super().forward(x)


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU feed-forward block.

    Used both for the dense decoder layers and for the MoE shared experts.
    The routed-expert SwiGLU clamp does not apply here, matching the reference
    implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class HYV4MoEFused(nn.Module):
    """HY V4 MoE layer with optional SwiGLU clamp support.

    When ``config.swiglu_limit > 0`` the routed experts use a clamped SwiGLU::

        gate = clamp(gate, max=limit)
        up = clamp(up, -limit, limit)
        output = silu(gate) * up

    Dense layers and shared experts are NOT clamped. The clamp is forwarded to
    ``FusedMoEFactory`` as ``swiglu_limit`` so it travels through the quant
    config to every expert backend instead of mutating global state.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # The legacy grouped-topk kernel only accepts BF16/FP16 logits, but
        # the existing sigmoid-top8 kernel supports FP32 logits and bias.
        # With its exact HY4 contract enabled, preserve reference FP32 router
        # accumulation rather than rounding logits before sigmoid/top-k.
        from vllm import envs

        fp32_router = (
            envs.VLLM_ILU_GLM_MOE_PREPARE_FUSION
            and config.num_experts == 256
            and top_k == 8
            and config.route_norm
        )
        if fp32_router:
            self.gate = HYV4RouterGate(
                config.hidden_size, config.num_experts, prefix=f"{prefix}.gate"
            )
        else:
            self.gate = GateLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                prefix=f"{prefix}.gate",
            )
        logger.info_once("HYV4 router FP32 compute: %s", fp32_router)

        self.shared_experts: HYV4FeedForward | None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )

        raw_swiglu_limit = getattr(config, "swiglu_limit", 0)
        use_swiglu_clamp = bool(raw_swiglu_limit) and float(raw_swiglu_limit) > 0
        moe_swiglu_limit = float(raw_swiglu_limit) if use_swiglu_clamp else None

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=router_scaling_factor,
            e_score_correction_bias=self.expert_bias,
            shared_experts=self.shared_experts,
            swiglu_limit=moe_swiglu_limit,
        )
        self.prefix = prefix
        if use_swiglu_clamp:
            logger.debug_once(
                "HYV4MoEFused: swiglu_limit=%.1f enabled for routed experts",
                moe_swiglu_limit,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts), FP32 on sigmoid-top8 path.
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)
