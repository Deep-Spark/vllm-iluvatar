# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4-Flash model for the *raw FP8* checkpoint.

Three-layer module tree (``ForCausalLM`` / ``Model`` / ``DecoderLayer``), the
MoE FFN (``IluDeepseekV4Fp8MoE``), the ``WeightsMapper`` that renames the
checkpoint keys, weight-loading rules, Load_Verifier, and arch registration.
This module registers no torch custom op.

Architecture
------------
Self-written model rather than subclassing community ``DeepseekV4Model`` /
``DeepseekV4DecoderLayer`` because the community forwards are wired to the
tilelang MHC kernels and the model allocates three aux streams at build time.

* MHC (``hc_pre`` / ``hc_post`` / ``hc_head``) runs on the ixformer kernels
  ``ixfops.ds4_mhc_pre_v2`` / ``ds4_mhc_post`` / ``ds4_hc_head``.  The
  orchestration matches the W4A8 fork (``deepseek_v4/model.py:561-635``,
  ``:757-772``): each sub-layer does its own ``hc_post`` and ``attn_norm`` /
  ``ffn_norm`` are called explicitly, rather than NVIDIA's cross-sub-layer
  ``mhc_fused_post_pre`` with the norm folded into the pre.
* ``IluDeepseekV4Fp8MoE.forward``: the whole FFN goes through the community
  ``FusedMoE`` runner like the W4A8 fork -- ``IluGateLinear`` for the gate
  GEMM, ``ops.topk_hash_softplus_sqrt`` for the top-k, and
  ``IluMxfp4MoEMethod`` -> ``ilu_mxfp4_fused_experts()`` for the routed MXFP4
  experts.
* The shared-expert activation is the community ``SiluAndMulWithClamp``, which
  resolves to the plugin's OOT ``IluSiluAndMulWithClamp``.

Weight loading
--------------
``IluDeepseekV4Fp8Model.load_weights`` holds the stacked / experts /
``attn_sink`` main loop.  ``IluDeepseekV4Fp8ForCausalLM.load_weights`` wraps it
with the Load_Verifier (statistics, unconsumed-tensor report, strict reverse
check).  Tensor classification for the statistics lives in ``fp8_load_stats``.

Quantization-config constraint
------------------------------
``IluDeepseekV4Fp8MoE`` builds a community ``FusedMoE``, and which MoE method
its routed experts get is decided entirely by ``vllm_config.quant_config``.  It
has to be ``IluDeepseekV4FP8Config``
(``layers/quantization/ilu_deepseek_v4_fp8.py``, registered under the community
name ``deepseek_v4_fp8``), which returns ``IluMxfp4MoEMethod``.  In production
that is automatic: the checkpoint's ``quant_method="fp8"`` plus
``model_type="deepseek_v4"`` resolve to ``deepseek_v4_fp8``, and the plugin's
registration shadows the community class.

Coexistence
-----------
``ilu_register._iter_package_modules`` imports every top-level module of this
package at start-up, so this module coexists with the W4A8 ``model.py`` /
``attention.py``.  It therefore registers no torch custom op and no quantization
config.  Arch names override the community pair
(``DeepseekV4ForCausalLM`` / ``DeepSeekV4MTPModel``) and do not collide with the
W4A8 pair (``IluDeepseekV4ForCausalLM`` / ``IluDeepSeekV4MTPModel``).
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import ixformer.inference.functions as ixfops  # type: ignore[import]
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from vllm_iluvatar.models.deepseek_v4 import fp8_load_stats
from vllm_iluvatar.models.deepseek_v4.fp8_attention import (
    IluDeepseekV4Fp8Attention,
)

# Use a logger inside the ``vllm.`` namespace so that vLLM's configured handler
# picks up INFO-level messages (load statistics).  A ``vllm_iluvatar.…`` logger
# would only hit ``logging.lastResort`` (WARNING), dropping the stats lines.
logger = init_logger(f"vllm.{__name__}")

# Default for ``IluDeepseekV4Fp8ForCausalLM._LOAD_SKIP_SUBSTRS``; see that attribute.
_LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("mtp.",)

# Cap reported names in load diagnostics.
_MAX_REPORTED_NAMES = 20

# The attribute name ``IluDeepseekV4Fp8Model`` is registered under on
# ``IluDeepseekV4Fp8ForCausalLM``, i.e. the single prefix level
# ``AutoWeightsLoader._groupby_prefix`` strips off (``name.split(".", 1)[1]``)
# before handing a weight to that child. The Load_Verifier has to put it back to
# compare the child's bookkeeping against the checkpoint-side names.
_MODEL_PREFIX = "model."


def _is_load_skipped(name: str, skip_substrs: tuple[str, ...] = _LOAD_SKIP_SUBSTRS) -> bool:
    """Whether ``AutoWeightsLoader`` would skip this (mapped) name.

    Mirrors ``AutoWeightsLoader._can_skip`` for the substrings this model passes:
    the caller's list plus the rotary leftovers the loader appends to every
    ``skip_substrs`` list. Used by both Load_Verifier directions so the whitelist
    is identical on the parameter and the checkpoint side.

    ``skip_substrs`` is a parameter rather than the module constant so a subclass can widen the
    list via ``_LOAD_SKIP_SUBSTRS`` and have both directions of the check follow.
    """
    return any(substr in name for substr in skip_substrs) or any(
        substr in name for substr in AutoWeightsLoader.ROTARY_EMBEDS_UNUSED_WEIGHTS
    )


def _make_fp8_weights_mapper() -> WeightsMapper:
    """Checkpoint-name -> parameter-name map for the raw FP8 checkpoint.

    Equivalent to community ``_make_deepseek_v4_weights_mapper("fp4")``
    (``vllm/models/deepseek_v4/nvidia/model.py:1311-1350``).

    ``orig_to_new_regex`` **must stay ordered**. ``WeightsMapper``
    (``vllm/model_executor/models/utils.py``) walks the regex mapping in
    insertion order and does not stop at the first hit, so the MXFP4 experts
    rule has to run first: routed experts live on ``Mxfp4MoEMethod``, which
    registers ``w{1,2,3}_weight_scale`` (no ``_inv``), while every other FP8
    block scale is ``Fp8LinearMethod``'s ``weight_scale_inv``. Reversed, the
    generic rule would rewrite ``.scale`` first and the experts rule could never
    match. (After the experts rule fires the key ends in ``_scale``, not
    ``.scale``, so the generic rule correctly misses it -- which is also why the
    ``hc_*_scale`` plain tensors are left alone.)
    """
    return WeightsMapper(
        orig_to_new_regex={
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        },
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            # No trailing dot: the parameters are ``hc_head_fn`` /
            # ``hc_head_base`` / ``hc_head_scale``, so a dot would produce
            # ``model.hc_head._fn``.  Community is dot-less too
            # (``nvidia/model.py:1330``).
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            # Only the shared expert's ``w2`` is renamed; ``w1`` / ``w3`` are
            # fused into ``gate_up_proj`` by the loader's stacked mapping,
            # and routed ``w2`` is untouched because the pattern is anchored
            # on ``.shared_experts.``.
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class IluDeepseekV4Fp8MLP(nn.Module):
    """Shared-expert MLP: block-FP8 ``gate_up_proj`` + ``down_proj``.

    Community counterpart: ``DeepseekV4MLP``
    (``vllm/models/deepseek_v4/nvidia/model.py:82-139``).  Re-implemented rather
    than imported, but kept byte-compatible: the parameter names and the
    intermediate-size round-up match the checkpoint mapping.

    Intermediate round-up: block-FP8 shards in whole ``block_size[0]`` rows, so
    the per-rank block count is rounded up and multiplied back out, keeping the
    linear's even TP split block-aligned.  On this checkpoint the round-up is a
    no-op, but it must stay so ``_pad_shared_expert_weight`` agrees.

    ``forward`` runs ``gate_up_proj`` -> SwiGLU(clamp) -> ``down_proj``.  The
    two projections are block-FP8 and dispatch to
    ``IluFp8BlockScaledMMKernel``; ``act_fn`` is the community
    ``SiluAndMulWithClamp`` (-> OOT ``IluSiluAndMulWithClamp``).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()

        block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is not None:
            tp_size = get_tensor_model_parallel_world_size()
            n_local = cdiv(intermediate_size // block_size[0], tp_size)
            intermediate_size = n_local * block_size[0] * tp_size
        self.intermediate_size = intermediate_size

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
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported "
                "for now."
            )
        # DS-V4 always ships ``swiglu_limit``, so there is no
        # plain-``SiluAndMul`` branch.  ``SiluAndMulWithClamp`` resolves to the
        # plugin's OOT ``IluSiluAndMulWithClamp``.
        assert swiglu_limit is not None, (
            "DeepSeek-V4 FP8 shared-expert MLP requires config.swiglu_limit"
        )
        self.act_fn = SiluAndMulWithClamp(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``down_proj`` has ``reduce_results=False`` -- the MoE runner owns the
        # single EP/TP all-reduce.
        gate_up, _ = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        out, _ = self.down_proj(act)
        return out


class IluDeepseekV4Fp8MoE(nn.Module):
    """DS-V4-Flash FFN: routing gate + shared expert MLP + routed MXFP4 experts.

    Community counterpart: ``DeepseekV4MoE.__init__`` /
    ``_init_fused_moe_experts`` (``vllm/models/deepseek_v4/nvidia/model.py``
    ``:513-603`` and ``:647-691``), minus the MegaMoE branch
    (``kernel_config.moe_backend == "deep_gemm_mega_moe"`` needs a DeepGEMM
    kernel Iluvatar does not have; never enabled here).

    Owned parameters (the ``experts`` ones are created by community
    ``Mxfp4MoEMethod.create_weights``, ``quantization/mxfp4.py:530-622``):

    * ``gate.weight``                                 ``[256, 4096]``   bf16
    * ``gate.e_score_correction_bias``                ``[256]``         fp32
    * ``gate.tid2eid``                                ``[129280, 6]``   int32
    * ``shared_experts.gate_up_proj.weight``           ``[512, 4096]``  fp8
    * ``shared_experts.gate_up_proj.weight_scale_inv`` ``[4, 32]``      e8m0
    * ``shared_experts.down_proj.weight``              ``[4096, 256]``  fp8
    * ``shared_experts.down_proj.weight_scale_inv``    ``[32, 2]``      e8m0
    * ``experts.routed_experts.w13_weight``       ``[32, 4096, 2048]``  uint8
    * ``experts.routed_experts.w13_weight_scale`` ``[32, 4096, 128]``   uint8
    * ``experts.routed_experts.w2_weight``       ``[32, 4096, 1024]``  uint8
    * ``experts.routed_experts.w2_weight_scale`` ``[32, 4096, 64]``    uint8

    ``gate.tid2eid`` and ``gate.e_score_correction_bias`` are mutually exclusive:
    hash layers (``layer_id < config.num_hash_layers``, i.e. 0/1/2) route by
    table lookup and get ``tid2eid``; the remaining 40 layers use
    ``topk_method == "noaux_tc"`` and get the bias.

    The routed shapes above are for EP8 (32 local experts holding the *full*
    intermediate size).  Nothing here forces that split: ``FusedMoE`` derives it
    from ``FusedMoEParallelConfig.make``, so a TP-only launch would give
    different per-rank shapes.  The shared-expert MLP is TP-sharded either way.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        # --- routing gate (community nvidia/model.py:554-583) ----------------
        # ``out_dtype=float32``: the router runs in fp32 regardless of the
        # activation dtype. ``quant_config`` is not forwarded -- GateLinear
        # hard-codes ``quant_config=None`` on its ReplicatedLinear base, and the
        # checkpoint stores ``ffn.gate.weight`` unquantised (bf16).
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # Both attributes always exist; exactly one of them is a Parameter. The
        # None-initialisation matters because FusedMoE / the forward read them
        # unconditionally.
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None

        self.layer_id = extract_layer_index(prefix)
        is_hash_moe = self.layer_id < config.num_hash_layers
        # int32 on the FusedMoE path (community picks int64 only for MegaMoE).
        # The checkpoint tensor is I64, so the loader converts on copy.
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # ``randint`` rather than ``empty``: with --load-format=dummy the
            # table is used as an index, and garbage would fault
            # (community nvidia/model.py:566-577).
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        # --- shared expert (community nvidia/model.py:585-603) ---------------
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = IluDeepseekV4Fp8MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                # False: the routed path owns the final all-reduce, so the
                # shared expert must not reduce its partial result twice
                # (community passes ``self.use_mega_moe``, False here).
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # --- routed experts (community nvidia/model.py:647-691) -------------
        # The real expert split is owned by ``FusedMoE``
        # (``FusedMoEParallelConfig.make`` -> ``routed_experts.expert_map`` /
        # ``.local_num_experts``, which is what ``ilu_mxfp4_fused_experts()``
        # reads); only the divisibility precondition is checked here.
        eplb_config = parallel_config.eplb_config
        n_physical_experts = self.n_routed_experts + eplb_config.num_redundant_experts
        assert n_physical_experts % self.tp_size == 0, (
            f"n_physical_experts={n_physical_experts} must be divisible by "
            f"tp_size={self.tp_size}. Adjust num_redundant_experts."
        )

        # ``FusedMoE`` is a factory returning a ``MoERunner``; the four routed
        # parameters live on its ``routed_experts`` child, so their qualified
        # names are ``{prefix}.experts.routed_experts.w*``. ``gate`` and
        # ``shared_experts`` are handed in as well (community does the same) so
        # the runner can fuse routing and the shared branch; they are the *same*
        # module objects registered above, so ``named_parameters()`` reports
        # them once, under the ``{prefix}.gate`` / ``{prefix}.shared_experts``
        # names -- ``named_modules(remove_duplicate=True)`` visits the direct
        # children first and then skips the aliased subtrees under ``experts``.
        #
        # ``quant_config.get_quant_method`` runs inside this call and decides the
        # routed-expert method; it must be ``IluDeepseekV4FP8Config`` so we get
        # ``IluMxfp4MoEMethod``. See the module docstring.
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hand the whole FFN to the community runner.

        Same shape as community ``DeepseekV4MoE.forward``.  There is **no**
        ``tensor_model_parallel_all_reduce`` here: the runner owns the single
        EP/TP all-reduce.  ``routed_scaling_factor`` is likewise handled by the
        router (multiplied into ``topk_weights``).

        ``is_internal_router`` is true because ``FusedMoE`` was constructed with
        ``gate=self.gate``, so ``router_logits`` is a placeholder and the real
        gate GEMM runs inside the runner.
        """
        if self.gate.tid2eid is not None:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
            input_ids = input_ids.to(dtype=self.hash_indices_dtype)

        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        return final_hidden_states.view(org_shape)


class IluDeepseekV4Fp8DecoderLayer(nn.Module):
    """One DS-V4-Flash decoder layer: MLA attention + MoE FFN + MHC parameters.

    Community counterpart: ``DeepseekV4DecoderLayer``
    (``vllm/models/deepseek_v4/nvidia/model.py:794-867``).  The MHC parameter
    shapes are derived, not hard-coded, so the mini config used by the unit
    tests stays self-consistent:
    ``mix_hc = (2 + hc_mult) * hc_mult`` (24 at ``hc_mult == 4``) and
    ``hc_dim = hc_mult * hidden_size`` (16384 at the real ``hidden_size``).
    All six are fp32 plain tensors loaded by ``default_weight_loader``.
    """

    attention_cls: type[nn.Module] = IluDeepseekV4Fp8Attention

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.attn = self.attention_cls(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        self.ffn = IluDeepseekV4Fp8MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        # MHC (multi-head channel) scalars kept under community names, so the
        # ixformer kernels take them straight (``hc_pre`` / ``hc_post`` below).
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        # Community keeps this at a literal 3 (per-layer alpha / beta / gamma),
        # independent of hc_mult: nvidia/model.py:853-866.
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MHC pre: ``(T, hc_mult, H)`` -> sub-layer input + the two mixes.

        Same call as the W4A8 fork (``deepseek_v4/model.py:561-587``), pinned to
        the default operator version ``ds4_mhc_pre_v2`` (bf16 GEMM + a single
        fp32 epilogue). ``layer_input`` comes back **un-normed** -- the kernel has
        no ``norm_weight`` parameter, so ``attn_norm`` / ``ffn_norm`` are applied
        explicitly in ``forward``.

        Returns ``(layer_input (T,H), post_mix (T,hc_mult,1) fp32,
        comb_mix (T,hc_mult,hc_mult) fp32)`` -- note the kernel itself returns
        ``layer_input`` last; the order here follows W4A8.
        """
        post_mix, comb_mix, layer_input = ixfops.ds4_mhc_pre_v2(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, comb_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """MHC post: fold the sub-layer output back into the ``hc_mult`` streams.

        ``out[t,o,h] = post[t,o,0] * x[t,h] + sum_i comb[t,i,o] * residual[t,i,h]``
        (W4A8 ``deepseek_v4/model.py:589-596``). ``comb`` is
        ``[token, in_stream, out_stream]``; transposing it silently changes the
        result, so it is passed through untouched.
        """
        return ixfops.ds4_mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """MHC pre/post around attention and MoE.

        Structure follows the W4A8 fork (``deepseek_v4/model.py:597-635``): each
        sub-layer runs ``hc_pre -> norm -> sub-layer -> hc_post`` and carries no
        state across the layer boundary -- ``x`` is always the full
        ``(T, hc_mult, H)`` residual stream.

        NVIDIA instead defers each ``mhc_post`` and fuses it into the next
        sub-layer's ``mhc_pre`` (``nvidia/model.py:868-957``), with the norm
        folded in as well.  That is the same operator sequence in the same order
        with the same bf16 rounding points, so the two forms are exactly
        value-preserving; ixformer has no fused post+pre op, so nothing would be
        gained by keeping the deferred shape here.
        """
        # ``ds4_mhc_pre_v2`` feeds ``fn`` straight into a same-dtype GEMM, so the
        # fp32 checkpoint weights are cast once on the first forward and cached
        # in place (W4A8 ``model.py:605-612``). ``hc_head_fn`` must stay fp32 --
        # ``ds4_hc_head`` validates it as float32.
        if self.hc_attn_fn.dtype != x.dtype:
            self.hc_attn_fn.data = self.hc_attn_fn.data.to(x.dtype)
        if self.hc_ffn_fn.dtype != x.dtype:
            self.hc_ffn_fn.data = self.hc_ffn_fn.data.to(x.dtype)

        # --- attention sub-layer -------------------------------------------
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        # --- FFN sub-layer --------------------------------------------------
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self.hc_post(x, residual, post, comb)


class IluDeepseekV4Fp8Model(nn.Module):
    """Embedding + 43 decoder layers + final norm + the global MHC head.

    Community counterpart: ``DeepseekV4Model``
    (``vllm/models/deepseek_v4/nvidia/model.py:960-1055``), single-stream
    and without the MegaMoE / Eagle plumbing.
    """

    # Subclass extension point, paired with ``IluDeepseekV4Fp8DecoderLayer.attention_cls``.
    decoder_layer_cls: type[nn.Module] = IluDeepseekV4Fp8DecoderLayer

    stacked_params_mapping: list[tuple[str, str, int]] = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
        ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config

        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Shared across every indexer layer (community: nvidia/model.py:991).
        # Community allocates on CPU; the W4A8 fork passes an explicit device
        # (model.py:658-663). Following the fork.
        self.device = current_platform.device_type
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Global MHC head: [hc_mult, hc_mult * hidden_size] /
        # [hc_mult] / [1], all fp32.  Checkpoint keys are ``hc_head_fn`` /
        # ``hc_head_base`` / ``hc_head_scale``, lifted to ``model.hc_head_*`` by
        # the dot-less ``hc_head`` prefix rule in _make_fp8_weights_mapper().
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        # Pre-hc_head residual stream buffer for the MTP draft. Only allocated
        # on the last PP rank -- that's where MTP target hidden states are
        # produced (W4A8 ``model.py:712-723``).
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

        # Filled by ``load_weights``, read by the Load_Verifier on
        # ``IluDeepseekV4Fp8ForCausalLM``: the checkpoint-side names
        # (post-mapper, pre-fusion) that the main loop handled.  Accumulated
        # across every ``load_weights`` call of one load and cleared by
        # ``reset_consumed_ckpt_names()``.
        self.consumed_ckpt_names: set[str] = set()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states of shape
        # (num_tokens, hc_mult, hidden_size): V4 expands the token embedding to
        # hc_mult streams before the first decoder layer and keeps that shape
        # until the hc_head collapse on the last rank.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Collapse the ``hc_mult`` residual streams down to ``hidden_size``.

        Community: ``vllm/models/deepseek_v4/nvidia/model.py:1137-1144``
        (``hc_head_fused_kernel_tilelang`` over ``hc_head_fn`` / ``_scale`` /
        ``_base``); here the ixformer kernel, as in W4A8 ``model.py:765-772``.
        ``hc_head_fn`` stays fp32 -- unlike the per-layer ``hc_*_fn`` it is
        validated as float32 by the kernel, so it must not be cast.
        """
        return ixfops.ds4_hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            rms_norm_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Embed -> 43 decoder layers -> hc_head -> norm.

        Mirrors ``DeepseekV4Model.forward``
        (``vllm/models/deepseek_v4/nvidia/model.py:1080-1145``) / W4A8
        ``model.py:738-772``, minus the aux streams.
        The token embedding is expanded to ``hc_mult`` streams before the loop
        and stays that shape all the way to ``hc_head``: each layer does its own
        closing ``hc_post``, so unlike NVIDIA there is no per-layer state to
        thread and no deferred post to reconstruct here.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            # (T, H) -> (T, hc_mult, H): every stream starts identical, which
            # makes the plain hc_pre equivalent to community's broadcast pre.
            # ``repeat`` also gives the contiguous tensor the kernels require.
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions, input_ids)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (only on last rank).
        num_tokens = hidden_states.shape[0]
        assert self._mtp_hidden_buffer is not None
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Routed-expert name fragments, built once per ``load_weights`` call.

        Community counterpart: ``DeepseekV4Model.get_expert_mapping``
        (``nvidia/model.py:1277-1291``) minus its MegaMoE branch --
        ``make_deepseek_v4_expert_params_mapping`` is only for
        ``use_mega_moe``, which this model never enables.

        The returned 4-tuples are ``(param_name, weight_name, expert_id,
        shard_id)`` where both names are **fragments**, e.g.
        ``("experts.routed_experts.w13_", "experts.7.w1.", 7, "w1")``. The
        loader turns them into real names with
        ``name.replace(weight_name, param_name)``, which is what folds
        ``...ffn.experts.7.w1.weight`` into
        ``...ffn.experts.routed_experts.w13_weight`` and
        ``...ffn.experts.7.w1.weight_scale`` into
        ``...ffn.experts.routed_experts.w13_weight_scale`` from the same entry.
        """
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def _pad_shared_expert_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate axis.

        Community counterpart: ``DeepseekV4Model._pad_shared_expert_weight``
        (``nvidia/model.py:1259-1281``).

        The intermediate axis is rounded up to a whole number of TP shards so
        the ordinary column/row-parallel loaders below split it evenly and the
        trailing ranks land on the zero pad. The axis is counted in *elements*
        for weights (step = ``block_size[0]``) and in *blocks* for the
        ``weight_scale_inv`` companions (step = 1); gate (``w1``) / up (``w3``)
        are ``[I, H]`` so they pad dim 0, down (``w2`` -> ``down_proj``) is
        ``[H, I]`` so it pads dim 1.

        Must agree with the matching round-up in ``IluDeepseekV4Fp8MLP.__init__``
        -- the pad is only correct if the layer was built at the padded size.

        On this checkpoint (``I = 2048``, ``block = [128, 128]``, ``tp = 8``)
        ``mult = 8 * 128 = 1024`` and ``cdiv(2048, 1024) * 1024 - 2048 == 0``, so
        this is a **no-op that early-returns the input object itself**.  It stays
        because a different checkpoint or TP size makes it fire.
        """
        block_size = getattr(self.quant_config, "weight_block_size", None)
        assert block_size is not None
        step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def reset_consumed_ckpt_names(self) -> None:
        """Drop the Load_Verifier bookkeeping before a fresh load starts.

        The clear must happen once per *load*, not once per ``load_weights``
        call, because ``AutoWeightsLoader`` calls ``load_weights`` once per
        contiguous run of ``model.*`` names (``itertools.groupby`` only merges
        adjacent keys), so the method may be entered more than once.
        """
        self.consumed_ckpt_names = set()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Main loading loop, transcribed from community ``nvidia/model.py:1150-1257``.

        Names arriving here have already been rewritten by
        ``hf_to_vllm_mapper`` and ``mtp.*`` has already been dropped by
        ``AutoWeightsLoader(skip_substrs=["mtp."])``.

        Four branches, in the order the community loop tries them:

        1. **stacked fusion** -- pairs of checkpoint tensors that share one parameter, per
           ``self.stacked_params_mapping``; by default ``attn.wq_a`` / ``attn.wkv`` ->
           ``attn.fused_wqa_wkv``, ``compressor.wkv`` / ``compressor.wgate`` ->
           ``compressor.fused_wkv_wgate``, and the shared expert's ``w1`` /
           ``w3`` -> ``gate_up_proj``.
        2. **routed experts** -- ``expert_mapping`` is tried entry by entry with
           ``return_success=True``, because a weight for a non-local expert
           must not consume the slot of a replica that *is* local.  e8m0 scales
           are re-interpreted with ``view(torch.uint8)`` to preserve the raw
           exponent bytes.
        3. **attn_sink** -- the checkpoint holds all ``num_attention_heads``
           sinks; this rank takes its own head slice.
        4. **plain** -- the parameter's own ``weight_loader`` (TP-aware) or
           ``default_weight_loader``.

        Every checkpoint name this loop *handles* is recorded in
        ``self.consumed_ckpt_names`` for the Load_Verifier.
        """
        stacked_params_mapping = self.stacked_params_mapping
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Accumulate, never reset here -- this method may be entered more than
        # once per load (see ``reset_consumed_ckpt_names``).
        consumed_ckpt_names = self.consumed_ckpt_names

        # attn_sink head slice for this rank (community nvidia/model.py:1163-1169).
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Built once: the mapping is O(num_experts * 3) tuples and independent of
        # the weight being loaded (community nvidia/model.py:1171-1172).
        expert_mapping = self.get_expert_mapping()

        # Same gate as community (nvidia/model.py:1174-1181): only block-FP8,
        # non-sequence-parallel shared experts need the intermediate pad.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.parallel_config.use_sequence_parallel_moe
        )

        for name, loaded_weight in weights:
            # ``name`` is rewritten in place by the stacked / expert branches, so
            # keep the incoming checkpoint name for the Load_Verifier.
            ckpt_name = name
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(name, loaded_weight)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Matched a stacked rule either way: on another PP rank the
                # tensor is deliberately dropped, not unmapped.
                consumed_ckpt_names.add(ckpt_name)
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in checkpoints but
                    # the MoE param is uint8. copy_() would do a numeric
                    # conversion (e.g. 2^-7 -> 0), destroying the raw exponent
                    # bytes -- so re-interpret instead.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    name_mapped: str | None = None
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        # A name match means the expert rules cover this tensor;
                        # whether *this* rank owns the expert is a separate
                        # question, answered by ``return_success`` below.
                        consumed_ckpt_names.add(ckpt_name)
                        candidate = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(candidate, self):
                            continue
                        name_mapped = candidate
                        param = params_dict[name_mapped]
                        # Ask the weight loader whether it actually consumed the
                        # tensor: a non-local expert returns False and we must
                        # keep trying, otherwise a replica that *is* local would
                        # be skipped.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    # Community adds ``name_mapped`` unconditionally here
                    # (nvidia/model.py:1232). Two deliberate hardenings, both
                    # no-ops on the happy path this checkpoint takes:
                    #   * the ``None`` guard -- community raises NameError if no
                    #     mapping matched at all; leaving the tensor unrecorded
                    #     hands it to the unconsumed report instead;
                    #   * assigning ``name_mapped`` only *after* the PP check, so
                    #     a parameter this rank does not own is never reported as
                    #     loaded (community would report it).
                    # Loading behaviour itself -- which slice of which parameter
                    # gets written -- is byte-identical to community.
                    if name_mapped is not None:
                        loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    consumed_ckpt_names.add(ckpt_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    # Front-fill only; the remaining padded_heads - n slots keep
                    # their -inf initialisation.
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        # Owned by another PP rank: handled, not unmapped.
                        consumed_ckpt_names.add(ckpt_name)
                        continue
                    if name not in params_dict:
                        # No parameter of that name: leave it for the
                        # unconsumed-tensor report instead of raising KeyError
                        # deep inside the loop -- a missing mapper rule must
                        # surface as a named list, not a stack trace.
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    consumed_ckpt_names.add(ckpt_name)
                    continue

        return loaded_params


class IluDeepseekV4Fp8ForCausalLM(nn.Module, SupportsPP):
    """Top-level causal-LM for the raw-FP8 DeepSeek-V4-Flash checkpoint.

    Registered under the community arch name ``DeepseekV4ForCausalLM``
    so the checkpoint is loaded unmodified.
    """

    model_cls = IluDeepseekV4Fp8Model
    hf_to_vllm_mapper = _make_fp8_weights_mapper()

    _LOAD_SKIP_SUBSTRS: tuple[str, ...] = _LOAD_SKIP_SUBSTRS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # No expert_dtype dispatch: this class is bound to the raw FP8 +
        # MXFP4-experts checkpoint, i.e. community's ``expert_dtype == "fp4"``.
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Community: vllm/models/deepseek_v4/nvidia/model.py:1443-1448
        # (self.logits_processor(self.lm_head, hidden_states)).
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # Community: vllm/models/deepseek_v4/nvidia/model.py:1450-1462
        # (delegates straight to self.model).
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def _tap_weight_names(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        seen: dict[str, str],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Pass ``weights`` through untouched while recording name + dtype.

        ``AutoWeightsLoader`` drains the weight generator, so the only way to
        learn which checkpoint tensors actually arrived is to tap the stream on
        the way in.  Only two short strings per tensor are kept -- never the
        tensor -- so the tap costs no extra host or device memory.

        Names are recorded *mapped* (via ``apply_list``, same
        ``_map_name_with_shard`` path as the stream ``apply``) so they line up
        with what ``IluDeepseekV4Fp8Model.load_weights`` records as consumed.
        Names the loader will skip (``mtp.*`` + rotary leftovers) are not
        recorded.
        """
        mapper = self.hf_to_vllm_mapper
        for name, tensor in weights:
            mapped = mapper.apply_list([name])
            if mapped and not _is_load_skipped(mapped[0], self._LOAD_SKIP_SUBSTRS):
                seen[mapped[0]] = fp8_load_stats.dtype_name(tensor.dtype)
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, then run the Load_Verifier.

        ``skip_substrs=["mtp."]`` matches community ``nvidia/model.py:1468-1469``;
        the loading rules themselves live in
        ``IluDeepseekV4Fp8Model.load_weights``.

        Three closing steps, in this order -- statistics first so the numbers are
        on record even when a check below fails, and the raise last so the
        WARNING it would otherwise hide is already out:

        1. **statistics**: the seven checkpoint-side categories plus the total,
           one ``key=value`` line each.  Classification is the scanner logic in
           ``fp8_load_stats``.
        2. **unconsumed tensor report**: checkpoint tensors that arrived but
           matched no rule, at WARNING level with a count and up to 20 names.
           Catches a missing mapper rule silently dropping tensors.
        3. **strict reverse check**: parameters that no weight ever filled,
           excluding ``mtp.*`` and this rank's PP-missing parameters, raising
           ``ValueError`` with a count and up to 20 names.  vLLM's own
           missing-parameter check does not run for this model
           (``DefaultModelLoader`` only performs it when
           ``model_config.quantization is None``).

        Steps 1/2 count **checkpoint tensors**, step 3 counts **parameters**.
        Stacked fusion and expert merging are many-to-one, so the two totals
        cannot be directly compared.
        """
        seen: dict[str, str] = {}
        # Once per load, before the loader runs: ``IluDeepseekV4Fp8Model``
        # accumulates and ``AutoWeightsLoader`` may enter its ``load_weights``
        # several times (see ``reset_consumed_ckpt_names``).
        self.model.reset_consumed_ckpt_names()
        loader = AutoWeightsLoader(self, skip_substrs=list(self._LOAD_SKIP_SUBSTRS))
        loaded = loader.load_weights(
            self._tap_weight_names(weights, seen), mapper=self.hf_to_vllm_mapper
        )

        # ``IluDeepseekV4Fp8Model.load_weights`` sees names with the ``model.``
        # prefix already stripped (AutoWeightsLoader hands each child module the
        # weights below its own attribute name), so put it back.
        consumed = {f"{_MODEL_PREFIX}{name}" for name in self.model.consumed_ckpt_names}
        # Tensors that never reach that loop: parameters AutoWeightsLoader fills
        # itself, i.e. everything *outside* the ``model.`` subtree
        # (``lm_head.weight`` here), whose parameter name *is* the mapped
        # checkpoint name. Names inside the subtree are deliberately **not**
        # unioned in: ``loaded`` is parameter-side, and for every 1:1 tensor the
        # parameter name equals the mapped checkpoint name, so unioning all of it
        # made the report blind to a whole missing bookkeeping set -- that alias
        # is why the 8-card WARNING looked like "only the stacked and expert
        # branches lost their entries" when in fact none of them had survived.
        consumed |= {name for name in loaded if not name.startswith(_MODEL_PREFIX)}

        # --- 1. statistics -------------------------------------------------
        stats = fp8_load_stats.LoadStats()
        for name, dtype in seen.items():
            if name in consumed:
                stats.record(name, dtype)
        for line in stats.log_lines():
            logger.info("%s", line)
        unknown_note = stats.describe_unknown()
        if unknown_note is not None:
            logger.warning("Load statistics: %s", unknown_note)

        # --- 2. unconsumed checkpoint tensors ------------------------------
        unconsumed = sorted(name for name in seen if name not in consumed)
        if unconsumed:
            logger.warning(
                "%d checkpoint tensor(s) were not consumed by any loading rule "
                "(stacked / experts / attn_sink / default); a missing mapper rule "
                "drops weights silently. First %d: %s",
                len(unconsumed),
                min(len(unconsumed), _MAX_REPORTED_NAMES),
                unconsumed[:_MAX_REPORTED_NAMES],
            )

        # --- 3. strict reverse check ---------------------------------------
        expected = {
            name
            for name, _ in self.named_parameters()
            if not _is_load_skipped(name, self._LOAD_SKIP_SUBSTRS)
            and not is_pp_missing_parameter(name, self)
        }
        missing = sorted(expected - loaded)
        if missing:
            raise ValueError(
                f"{len(missing)} parameter(s) were never filled by "
                f"{type(self).__name__}.load_weights. First "
                f"{min(len(missing), _MAX_REPORTED_NAMES)}: "
                f"{missing[:_MAX_REPORTED_NAMES]}"
            )

        logger.debug(
            "[IluDeepseekV4Fp8ForCausalLM.load_weights] loaded %d parameter(s) "
            "from %d checkpoint tensor(s)",
            len(loaded),
            len(seen),
        )
        return loaded


def register_model() -> None:
    """Re-point the community DS-V4 arch at the Iluvatar raw-FP8 model.

    Invoked by ``vllm_iluvatar.ilu_register._register_models()``.  Keeping the
    community arch name means the checkpoint's ``config.json`` stays untouched
    and ``tokenizer_mode`` is still auto-selected as ``"deepseek_v4"`` by
    community ``ModelConfig``.  Does not collide with the sibling W4A8 fork,
    which registers ``IluDeepseekV4ForCausalLM`` / ``IluDeepSeekV4MTPModel``.

    Also overrides community ``DeepSeekV4MTPModel`` with the FP8 MTP draft so
    ``--speculative-config method=mtp`` resolves without touching
    ``_ILU_MTP_ARCH_MAP`` (that map only remaps the W4A8 ``Ilu*`` target archs).
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_iluvatar.models.deepseek_v4.fp8_model:"
        "IluDeepseekV4Fp8ForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_iluvatar.models.deepseek_v4.fp8_mtp:"
        "IluDeepseekV4Fp8MTP",
    )
