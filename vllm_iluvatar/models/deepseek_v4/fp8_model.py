# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4-Flash model skeleton for the *raw FP8* checkpoint.

Stage 1 scope (spec ``deepseek-v4-fp8-weight-loading``, tasks 2.2 / 3.4 / 4.4 /
4.5): build the three-layer module tree (``ForCausalLM`` / ``Model`` /
``DecoderLayer``) plus the MoE FFN (``IluDeepseekV4Fp8MoE``), the
``WeightsMapper`` that renames the checkpoint keys, the weight-loading rules and
the Load_Verifier, and the arch registration. Every forward entry stops at a
numbered TODO; no kernel is wired, the vendor kernel package is not imported, and
no torch custom op is registered.

Stage 2 wiring status
---------------------
* ``IluDeepseekV4Fp8MoE.forward`` is wired (Module 4): a torch-equivalent FFN
  (gate GEMM -> sqrtsoftplus top-k -> local routed experts with on-the-fly
  MXFP4 dequant + grouped MLP -> ``+ shared`` -> EP/TP all-reduce), bypassing
  the community ``FusedMoE`` runner whose kernels are absent on Iluvatar.
* the shared-expert activation is a plain torch SwiGLU (``_Fp8SwiGLU``), not a
  community ``CustomOp``: those bind ``torch.ops._C.*`` inside ``__init__`` with
  no fallback on a CUDA-like platform, so building one would crash at
  construction time on Iluvatar (design §4.4).

Weight loading is split over the two classes: ``IluDeepseekV4Fp8Model.load_weights``
holds the stacked / experts / ``attn_sink`` main loop (design §4.5.1-§4.5.4, task
4.4) and ``IluDeepseekV4Fp8ForCausalLM.load_weights`` wraps it with the
Load_Verifier -- statistics, unconsumed-tensor report, strict reverse check
(design §4.5.6, task 4.5). Tensor classification for the statistics lives in
``fp8_load_stats``, ported from the offline scanner that produced the acceptance
numbers.

Construction-order constraint (task 3.3 / 3.4)
----------------------------------------------
``IluDeepseekV4Fp8MoE`` builds a community ``FusedMoE``, whose
``Mxfp4MoEMethod.__init__`` calls ``select_deepseek_v4_mxfp4_moe_backend``
during construction. Off ROCm that community function never returns
``Mxfp4MoeBackend.NONE`` -- it raises ``NotImplementedError("No MXFP4 MoE
backend supports the deployment configuration.")``. So the P1 wrapper patch
(``ilu_patch/general_patches/patch_dsv4_mxfp4_backend.py``) must already be
applied when the first layer is built. In production that is guaranteed by
``apply_general_patches()`` running from the ``vllm.general_plugins`` entry
point, well before the model loader instantiates anything; unit tests have to
call ``register()`` themselves before constructing a MoE layer.

Design references (``.kiro/specs/deepseek-v4-fp8-weight-loading/design.md``):

* §4.1 -- self-written model instead of subclassing community
  ``DeepseekV4Model`` / ``DeepseekV4DecoderLayer``: their forwards are wired to
  the tilelang MHC kernels and the model allocates three aux streams at build
  time.
* §4.5.5 -- ``_make_fp8_weights_mapper()``, the plugin-side equivalent of
  community ``_make_deepseek_v4_weights_mapper("fp4")``
  (``vllm/models/deepseek_v4/nvidia/model.py:1311-1350``). Note the sibling
  W4A8 fork's mapper is the *opposite* of what the raw checkpoint needs (it
  neutralises the ``.scale`` regex, see ``model.py:875-909``) -- do not copy it.
* §4.6 -- arch routing: ``register_model()`` re-points the community arch name
  ``DeepseekV4ForCausalLM`` at this class, so the checkpoint's ``config.json``
  stays untouched and ``tokenizer_mode`` is still auto-selected by community
  logic. Discovered automatically by ``ilu_register._register_models()`` because
  this file is a top-level module of an already-registered package -- no
  ``_OOT_MODEL_PACKAGES`` edit needed.
* §4.4 -- MoE skeleton: community ``Mxfp4MoEMethod`` forced onto
  ``Mxfp4MoeBackend.NONE``, so ``create_weights`` builds the four routed
  ``uint8`` parameters at exactly the checkpoint shapes and
  ``process_weights_after_loading`` early-returns instead of swizzling them.
* §5.3 / §5.4 -- MoE and model-level parameter shapes.
* §6.2 -- component checklist.
* §7 -- TODO numbering: ``E1``..``E5`` (model orchestration) were wired in
  Module 1 and the group-C MoE FFN gap was wired in Module 4, so this file no
  longer carries a ``TODO(ilu_fp8_*)`` stub.

Risk notes
----------
* R8: ``ilu_register._iter_package_modules`` imports every top-level module of
  this package at start-up, so this module coexists with the W4A8 ``model.py`` /
  ``attention.py``. It therefore registers no torch custom op and no
  quantization config, and its arch name (``DeepseekV4ForCausalLM``) differs
  from the W4A8 pair (``IluDeepseekV4ForCausalLM`` / ``IluDeepSeekV4MTPModel``).
* R9: ``topk_indices_buffer`` is allocated with an explicit ``device=`` (the
  W4A8 fork's choice, ``model.py:658-663``) rather than community's implicit CPU
  allocation (``nvidia/model.py:991-995``). Stage 1 never reads the buffer;
  Stage 2 has to confirm the placement at first use.
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
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

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    _e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.models.deepseek_v4 import fp8_load_stats, fp8_mhc
from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention import (
    IluDeepseekV4Fp8Attention,
)

# The logger name deliberately sits **inside the ``vllm.`` namespace**, unlike the
# ``init_logger(__name__)`` the rest of the plugin uses.
#
# ``vllm.logger._configure_vllm_root_logger()`` attaches a handler to the
# ``"vllm"`` logger only (``DEFAULT_LOGGING_CONFIG["loggers"]``) and configures no
# root handler. A logger named ``vllm_iluvatar.…`` therefore finds no handler
# anywhere on its ancestor chain, so ``logging.Logger.callHandlers`` falls through
# to ``logging.lastResort`` -- whose level is WARNING. That is exactly what the
# 8-card run showed: the unconsumed-tensor WARNING came out (message-only, no
# ``INFO 00-00 …[file:line]`` prefix, because lastResort has no formatter) while
# every one of the eight ``[ilu_fp8_load_stats]`` INFO lines was dropped before it
# reached a stream. Requirement 3.6 / task 5.2 need those eight lines in the log,
# so this module borrows vLLM's configured logger tree. ``propagate=False`` on
# ``"vllm"`` keeps the record from being emitted twice.
logger = init_logger(f"vllm.{__name__}")

# ``mtp.*`` is skipped end to end: at load time via AutoWeightsLoader, and in
# both directions of the Load_Verifier -- the strict reverse check and the
# unconsumed-tensor report (design C5, community ``nvidia/model.py:1468-1469``,
# requirements 3.1 / 3.2 / 3.4).
_LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("mtp.",)

# Requirements 3.3 / 3.5 both cap the reported names at 20.
_MAX_REPORTED_NAMES = 20

# The attribute name ``IluDeepseekV4Fp8Model`` is registered under on
# ``IluDeepseekV4Fp8ForCausalLM``, i.e. the single prefix level
# ``AutoWeightsLoader._groupby_prefix`` strips off (``name.split(".", 1)[1]``)
# before handing a weight to that child. The Load_Verifier has to put it back to
# compare the child's bookkeeping against the checkpoint-side names.
_MODEL_PREFIX = "model."


def _is_load_skipped(name: str) -> bool:
    """Whether ``AutoWeightsLoader`` would skip this (mapped) name.

    Mirrors ``AutoWeightsLoader._can_skip`` for the substrings this model passes:
    our own ``mtp.`` plus the rotary leftovers the loader appends to every
    ``skip_substrs`` list. Used by both Load_Verifier directions so the whitelist
    is identical on the parameter and the checkpoint side.
    """
    return any(substr in name for substr in _LOAD_SKIP_SUBSTRS) or any(
        substr in name for substr in AutoWeightsLoader.ROTARY_EMBEDS_UNUSED_WEIGHTS
    )


def _make_fp8_weights_mapper() -> WeightsMapper:
    """Checkpoint-name -> parameter-name map for the raw FP8 checkpoint.

    Equivalent to community ``_make_deepseek_v4_weights_mapper("fp4")``
    (``vllm/models/deepseek_v4/nvidia/model.py:1311-1350``); design §4.5.5.

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
            # ``hc_head_base`` / ``hc_head_scale`` (design §5.4), so a dot would
            # produce ``model.hc_head._fn``. Community is dot-less too
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
            # fused into ``gate_up_proj`` by the loader's stacked mapping
            # (design §4.5.1), and routed ``w2`` is untouched because the
            # pattern is anchored on ``.shared_experts.``.
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


# --- Module 4 (FFN/MoE, C1) torch-equivalent helpers ------------------------
# FP4 E2M1 magnitude lookup (3-bit magnitude -> fp32), sign applied separately.
# Same table community ``nvfp4_emulation_utils.break_fp4_bytes`` uses.
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# MXFP4 microscaling block size (32 fp4 values share one e8m0 exponent scale).
_MXFP4_BLOCK = 32


def _dequant_mxfp4_to_bf16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = _MXFP4_BLOCK,
) -> torch.Tensor:
    """Pure-torch MXFP4 -> bf16 dequant for one routed expert's weight.

    ``weight`` ``[out, in // 2]`` uint8: two fp4 (e2m1) nibbles per byte, low
    nibble first (community ``break_fp4_bytes`` order). ``scale`` ``[out,
    in // block]`` e8m0 (exponent-only) block scale. Returns ``[out, in]``
    bf16. Reuses module 2's ``_e8m0_to_fp32`` for the exponent decode; there is
    no MXFP4 global scale (unlike NVFP4).
    """
    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    # element 2i = low nibble of byte i, element 2i+1 = high nibble
    codes = torch.stack((low, high), dim=-1).flatten(start_dim=-2)  # [out, in]
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    mag = (codes & 0x07).long()
    lut = _FP4_E2M1_LUT.to(weight.device)
    vals = lut[mag] * sign  # fp32 [out, in]

    in_dim = codes.shape[-1]
    scale_f32 = _e8m0_to_fp32(scale)  # [out, in // block] fp32
    vals = vals.reshape(*vals.shape[:-1], in_dim // block, block)
    vals = vals * scale_f32.unsqueeze(-1)
    return vals.reshape(*vals.shape[:-2], in_dim).to(torch.bfloat16)


def _swiglu_clamp(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Torch equivalent of ``SiluAndMulWithClamp.forward_native`` (alpha=1,
    beta=0): ``gate=clamp(x[:d], max=limit)``, ``up=clamp(x[d:], +-limit)``,
    ``silu(gate) * up``. Computed in fp32, cast back to the input dtype.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    if limit is not None:
        gate = gate.clamp(max=float(limit))
        up = up.clamp(min=-float(limit), max=float(limit))
    out = gate * torch.sigmoid(gate) * up
    return out.to(x.dtype)


class _Fp8SwiGLU(nn.Module):
    """SwiGLU-with-clamp activation for the shared-expert MLP (Module 4).

    Torch equivalent of community ``SiluAndMulWithClamp`` (alpha=1, beta=0), i.e.
    ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)`` -- see
    ``_swiglu_clamp``. It is a plain ``nn.Module``, **not** a ``CustomOp``: the
    community activations resolve ``torch.ops._C.silu_and_mul_with_clamp`` inside
    ``__init__`` on any CUDA-like platform with no fallback, and Iluvatar CoreX
    does not export that symbol, so constructing one crashes before a single
    weight exists (design §4.4.1 / risk R10). The plugin's OOT
    ``IluSiluAndMulWithClamp`` would instead bind an ``ixformer`` kernel; doing
    the SwiGLU in torch keeps this module tree kernel-free and vendor-free.

    Holds no parameter and no buffer, so it stays invisible to
    ``named_parameters()`` / the loader; occupies the same ``act_fn`` slot the
    community MLP uses so the module tree diff stays 1:1.
    """

    def __init__(self, swiglu_limit: float | None = None) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def extra_repr(self) -> str:
        return f"swiglu_limit={self.swiglu_limit!r}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamp(x, self.swiglu_limit)


class IluDeepseekV4Fp8MLP(nn.Module):
    """Shared-expert MLP: block-FP8 ``gate_up_proj`` + ``down_proj``.

    Community counterpart: ``DeepseekV4MLP``
    (``vllm/models/deepseek_v4/nvidia/model.py:82-139``). Re-implemented rather
    than imported per design §4.1, but kept byte-compatible: the parameter names
    (``gate_up_proj`` / ``down_proj``) and the intermediate-size round-up are
    what the checkpoint mapping and design §5.3 are written against.

    Intermediate round-up (design §4.5.4, requirement 7.4): block-FP8 shards in
    whole ``block_size[0]`` rows, so the per-rank block count is rounded up and
    multiplied back out, which keeps the linear's even TP split block-aligned.
    On this checkpoint (``I = 2048``, ``block = [128, 128]``, ``tp = 8``) the
    round-up is a no-op -- ``cdiv(2048 // 128, 8) * 128 * 8 == 2048`` -- but it
    has to stay, because the loader's zero-fill of the trailing ranks
    (``_pad_shared_expert_weight``) is only correct if the two agree.

    ``forward`` (Module 4) runs ``gate_up_proj`` -> SwiGLU(clamp) ->
    ``down_proj``. The two projections are block-FP8 and dispatch to module 2's
    ``IluFp8BlockScaledMMKernel``; the SwiGLU is done in torch by ``act_fn``
    (``_Fp8SwiGLU``), which is a plain ``nn.Module`` rather than a community
    activation ``CustomOp`` (those bind ``torch.ops._C`` in ``__init__`` with no
    Iluvatar fallback, see ``_Fp8SwiGLU``).
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
        # Community builds ``SiluAndMulWithClamp(swiglu_limit)`` /
        # ``SiluAndMul()`` here (``nvidia/model.py:130-133``); both bind
        # ``torch.ops._C.*`` inside ``__init__`` on any CUDA-like platform and
        # Iluvatar has no such symbol, so we use a plain torch SwiGLU module
        # instead. Full reasoning on ``_Fp8SwiGLU``.
        self.act_fn = _Fp8SwiGLU(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Module 4: gate_up_proj -> SwiGLU(clamp) -> down_proj. The two
        # projections are block-FP8 and run through module 2's
        # ``IluFp8BlockScaledMMKernel``; ``act_fn`` does the SwiGLU in torch.
        # ``down_proj`` has ``reduce_results=False`` -- the MoE forward owns the
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
    (``kernel_config.moe_backend == "deep_gemm_mega_moe"`` needs EP plus a
    DeepGEMM kernel; Stage 1 has neither).

    Owned parameters (design §5.3; the ``experts`` ones are created by community
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
    ``topk_method == "noaux_tc"`` and get the bias. That is what makes
    ``scan_report.md`` show 3 ``tid*eid`` and 40 ``ffn.gate.bias`` tensors
    (requirement 7.6).

    The routed shapes above are the design §5.3 numbers, i.e. 32 local experts
    holding the *full* intermediate size. That layout is what community
    ``FusedMoE`` produces when the experts are expert-parallel over the 8 ranks
    (``num_local_experts = 256 / 8``, ``intermediate_size_per_partition =
    2048``). Nothing here forces that split: ``FusedMoE`` derives it from
    ``FusedMoEParallelConfig.make``, so a TP-only launch would instead give
    ``[256, 512, 2048]`` / ``[256, 4096, 128]`` (same element count, different
    axis). The Stage 1 8-card run (task 5.3) is what pins which of the two the
    checkpoint slicing has to match; the shared-expert MLP is TP-sharded either
    way.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.n_shared_experts = config.n_shared_experts or 0
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
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
        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        assert self.n_physical_experts % self.tp_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"tp_size={self.tp_size}. Adjust num_redundant_experts."
        )
        self.n_local_experts = self.n_physical_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

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
        # Timing: ``Mxfp4MoEMethod.__init__`` runs inside this call and asks
        # ``select_deepseek_v4_mxfp4_moe_backend`` for a backend -- patch P1 has
        # to be applied already, see the module docstring.
        self.experts = FusedMoE(
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
        # Module 4 (C1): torch-equivalent FFN aligned to NV
        # ``DeepseekV4MoE.forward`` (``nvidia/model.py:702-730``). We bypass the
        # community ``FusedMoE.forward`` / ``MoERunner`` (their gate GEMM, top-k
        # and EP dispatch all bind kernels Iluvatar lacks) and implement the
        # path in torch, reading the parameters directly:
        #   gate GEMM -> sqrtsoftplus top-k -> local routed experts (MXFP4
        #   dequant + grouped MLP, weighted by topk) -> + shared -> EP/TP
        #   all-reduce.
        # NV semantics: ``final = routed + shared`` (routed_scaling_factor is
        # multiplied into topk_weights, not the output); shared is dense per
        # token with ``reduce_results=False`` so the single all-reduce below
        # reduces both the expert-parallel routed partial and the TP-sharded
        # shared partial at once.
        org_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)

        topk_ids, topk_weights = self._route(x, input_ids)
        routed = self._routed_experts(x, topk_ids, topk_weights)

        if self.shared_experts is not None:
            out = routed + self.shared_experts(x)
        else:
            out = routed

        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.view(org_shape)

    def _route(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """gate GEMM + top-k, torch equivalent of ``fused_topk_bias`` /
        ``_topk_softplus_sqrt_torch`` (``router/fused_topk_bias_router.py``).

        Returns ``topk_ids`` ``[T, k]`` (long, global expert ids) and
        ``topk_weights`` ``[T, k]`` (input dtype, already scaled). Bias is used
        for expert *selection* only; weights come from the unbiased scores.
        """
        # gate GEMM in fp32 (weight is bf16, unquantised); done here rather than
        # via ``self.gate(...)`` because ``GateLinear.forward`` dispatches to
        # cuteDSL/dsv3 GEMMs absent on Iluvatar.
        logits = torch.matmul(x.to(torch.float32), self.gate.weight.t().to(torch.float32))
        scores = torch.sqrt(F.softplus(logits))

        if self.gate.tid2eid is not None:
            # hash layer: expert ids come from the lookup table
            assert input_ids is not None, "hash MoE layer needs input_ids"
            ids = input_ids.view(-1).long()
            topk_ids = self.gate.tid2eid[ids].long()
            topk_weights = scores.gather(1, topk_ids)
        else:
            bias = self.gate.e_score_correction_bias
            scores_for_choice = scores + bias.to(torch.float32) if bias is not None else scores
            _, topk_ids = torch.topk(
                scores_for_choice, k=self.n_activated_experts, dim=-1
            )
            topk_ids = topk_ids.long()
            topk_weights = scores.gather(1, topk_ids)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-20)
        topk_weights = (topk_weights * self.routed_scaling_factor).to(x.dtype)
        return topk_ids, topk_weights

    def _routed_experts(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Local routed-expert contribution (torch equivalent).

        For each expert owned by this rank ``[experts_start_idx,
        experts_end_idx)``, dequant its MXFP4 ``w13``/``w2`` to bf16 on the fly,
        run ``w13 -> SwiGLU(clamp) -> w2`` on the tokens routed to it, weight by
        ``topk_weights`` and scatter-add into a ``[T, H]`` partial. Non-local
        experts contribute zero; the caller's all-reduce sums the EP partials.
        Dequant is per expert (no full-layer cache) to bound peak memory.
        """
        re = self.experts.routed_experts
        routed = torch.zeros_like(x)
        for local_eid in range(self.n_local_experts):
            global_eid = self.experts_start_idx + local_eid
            sel = topk_ids == global_eid
            if not bool(sel.any()):
                continue
            tok_idx, slot_idx = sel.nonzero(as_tuple=True)
            xin = x[tok_idx]

            w13 = _dequant_mxfp4_to_bf16(
                re.w13_weight[local_eid], re.w13_weight_scale[local_eid]
            )
            gate_up = torch.matmul(xin, w13.t())
            act = _swiglu_clamp(gate_up, self.swiglu_limit)

            w2 = _dequant_mxfp4_to_bf16(
                re.w2_weight[local_eid], re.w2_weight_scale[local_eid]
            )
            y = torch.matmul(act.to(w2.dtype), w2.t())

            wgt = topk_weights[tok_idx, slot_idx].unsqueeze(-1).to(y.dtype)
            routed.index_add_(0, tok_idx, (y * wgt).to(routed.dtype))
        return routed


class IluDeepseekV4Fp8DecoderLayer(nn.Module):
    """One DS-V4-Flash decoder layer: MLA attention + MoE FFN + MHC parameters.

    Community counterpart: ``DeepseekV4DecoderLayer``
    (``vllm/models/deepseek_v4/nvidia/model.py:794-867``). The MHC parameter
    shapes follow design §5.4 and are derived, not hard-coded, so the mini
    config used by the unit tests stays self-consistent:
    ``mix_hc = (2 + hc_mult) * hc_mult`` (24 at ``hc_mult == 4``) and
    ``hc_dim = hc_mult * hidden_size`` (16384 at the real ``hidden_size``).
    All six are fp32 plain tensors loaded by ``default_weight_loader``.
    """

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

        self.attn = IluDeepseekV4Fp8Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        self.ffn = IluDeepseekV4Fp8MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        # MHC (multi-head channel) scalars kept under community names so Stage 2
        # can call the kernels without renaming anything.
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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MHC pre/post around attention and MoE, aligned to community NVIDIA.

        Structure mirrors ``DeepseekV4DecoderLayer.forward``
        (``vllm/models/deepseek_v4/nvidia/model.py:868-957``): the previous
        sub-layer's post is deferred and fused with the current sub-layer's pre
        (``mhc_fused_post_pre``); ``attn_norm`` / ``ffn_norm`` are folded into
        ``mhc_pre`` (so they are *not* called separately here). The MHC ops are
        the torch equivalents in ``fp8_mhc`` (module 1); the kernel swap to
        ``ixfops.ds4_mhc_*`` is a later optimisation behind the same interface.

        The first layer arrives with ``residual is None`` and ``x`` already
        expanded to ``(T, hc_mult, H)`` by ``Model.forward`` (the ``repeat``
        there makes the community ``x.dim() == 3`` standalone-``mhc_pre`` branch
        apply, so no broadcast special-case / ``fn_broadcast`` is needed).

        Returns the carried state ``(x, residual, post_mix, res_mix)``; the
        final ``mhc_post`` that reconstructs the last deferred post runs once in
        ``Model.forward`` after the loop.
        """
        # --- attention sub-layer -------------------------------------------
        if residual is None:
            residual = x
            post_mix, res_mix, x = fp8_mhc.mhc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            residual, post_mix, res_mix, x = fp8_mhc.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )

        x = self.attn(positions, x, None)

        # --- FFN sub-layer: attn's post fused with the ffn pre --------------
        residual, post_mix, res_mix, x = fp8_mhc.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
        )

        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


class IluDeepseekV4Fp8Model(nn.Module):
    """Embedding + 43 decoder layers + final norm + the global MHC head.

    Community counterpart: ``DeepseekV4Model``
    (``vllm/models/deepseek_v4/nvidia/model.py:960-1055``), minus the three aux
    streams (design §6.1 keeps Stage 1 single-stream) and minus the MegaMoE /
    Eagle plumbing.
    """

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
        # Risk R9: community allocates this on CPU, the W4A8 fork passes an
        # explicit device (model.py:658-663). Following the fork -- Stage 1 never
        # reads the buffer, so the choice only has to be revisited when Stage 2
        # first consumes it.
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
            lambda prefix: IluDeepseekV4Fp8DecoderLayer(
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

        # Global MHC head, design §5.4: [hc_mult, hc_mult * hidden_size] /
        # [hc_mult] / [1], all fp32. Checkpoint keys are ``hc_head_fn`` /
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

        # Filled by ``load_weights``, read by the Load_Verifier on
        # ``IluDeepseekV4Fp8ForCausalLM`` (task 4.5): the checkpoint-side names
        # (post-mapper, pre-fusion) that the main loop handled. Accumulated
        # across every ``load_weights`` call of one load and cleared by
        # ``reset_consumed_ckpt_names()`` -- see that method for why the clear
        # cannot live inside ``load_weights``.
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
        ``_base``); implemented via the torch equivalent in ``fp8_mhc``.
        """
        return fp8_mhc.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Embed -> 43 decoder layers -> final mhc_post -> hc_head -> norm.

        Mirrors ``DeepseekV4Model.forward``
        (``vllm/models/deepseek_v4/nvidia/model.py:1080-1145``), minus the aux
        streams and the MTP buffer (design §6.1). The token embedding is
        expanded to ``hc_mult`` streams before the loop (W4A8
        ``model.py:750``); the ``(residual, post_mix, res_mix)`` triple is
        threaded through the layers so the fused post/pre boundaries line up,
        and the last deferred post is reconstructed once after the loop.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            # (T, H) -> (T, hc_mult, H): every stream starts identical, which
            # makes the plain mhc_pre equivalent to community's broadcast pre.
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual: torch.Tensor | None = None
        post_mix: torch.Tensor | None = None
        res_mix: torch.Tensor | None = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Reconstruct the last sub-layer's deferred post (community
        # nvidia/model.py:1126-1128), then collapse the streams and norm.
        hidden_states = fp8_mhc.mhc_post(hidden_states, residual, post_mix, res_mix)
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Routed-expert name fragments, built once per ``load_weights`` call.

        Community counterpart: ``DeepseekV4Model.get_expert_mapping``
        (``nvidia/model.py:1277-1291``) minus its MegaMoE branch --
        ``make_deepseek_v4_expert_params_mapping`` is only for
        ``use_mega_moe``, which Stage 1 never enables (design §4.4).

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
        (``nvidia/model.py:1259-1281``); design §4.5.4, requirement 7.4.

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
        this is a **no-op that early-returns the input object itself**. It stays
        because a different checkpoint or TP size makes it fire (T7 asserts both
        the no-op and the object identity).
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

        **This is the fix for the 8-card statistics bug** (see the note in
        ``load_weights``): the clear must happen once per *load*, not once per
        ``load_weights`` call, because ``AutoWeightsLoader`` calls this model's
        ``load_weights`` **once per contiguous run of ``model.*`` names in the
        weight stream** -- ``_groupby_prefix`` uses ``itertools.groupby``, which
        only groups *adjacent* keys. The checkpoint interleaves ``head.weight``
        (mapped to ``lm_head.weight``) among the ``model.*`` tensors, so the run
        is split and the method is entered more than once.

        ``IluDeepseekV4Fp8ForCausalLM.load_weights`` calls this exactly once,
        before handing the stream to the loader. A caller that drives
        ``IluDeepseekV4Fp8Model.load_weights`` directly (unit tests) has to call
        it too if it wants the previous run's names gone.
        """
        self.consumed_ckpt_names = set()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Main loading loop, transcribed from community ``nvidia/model.py:1150-1257``.

        Names arriving here have already been rewritten by
        ``hf_to_vllm_mapper`` (design §4.5.5) and ``mtp.*`` has already been
        dropped by ``AutoWeightsLoader(skip_substrs=["mtp."])``.

        Four branches, in the order the community loop tries them:

        1. **stacked fusion** (design §4.5.1) -- three pairs of checkpoint
           tensors that share one parameter: ``attn.wq_a`` / ``attn.wkv`` ->
           ``attn.fused_wqa_wkv``, ``compressor.wkv`` / ``compressor.wgate`` ->
           ``compressor.fused_wkv_wgate``, and the shared expert's ``w1`` /
           ``w3`` -> ``gate_up_proj``. ``.experts.`` is ``continue``-skipped
           inside this loop so routed experts fall through to branch 2
           (``.shared_experts.`` does *not* contain ``.experts.`` -- the
           preceding character is an underscore -- so shared ``w1`` / ``w3``
           still fuse here).
        2. **routed experts** (design §4.5.2) -- ``expert_mapping`` is tried
           entry by entry with ``return_success=True``, because a weight for a
           non-local expert must not consume the slot of a replica that *is*
           local. e8m0 scales are re-interpreted with ``view(torch.uint8)``:
           the parameter is ``uint8`` and ``copy_()`` would run a numeric
           conversion (``2^-7`` -> ``0``) that destroys the raw exponent bytes
           (requirement 1.2 / 1.3, T3).
        3. **attn_sink** (design §4.5.3) -- the checkpoint holds all
           ``num_attention_heads`` sinks; this rank takes its own head slice and
           writes it into the *front* of a ``[padded_heads]`` parameter whose
           tail stays at ``-inf`` (no sink effect).
        4. **plain** -- the parameter's own ``weight_loader`` (TP-aware) or
           ``default_weight_loader``.

        Bookkeeping for the Load_Verifier (task 4.5, design §4.5.6): every
        checkpoint name this loop *handles* is recorded in
        ``self.consumed_ckpt_names`` before ``name`` is rewritten, so
        ``IluDeepseekV4Fp8ForCausalLM.load_weights`` can diff it against the
        names that actually arrived. "Handled" is wider than "written into a
        parameter on this rank" on purpose -- a routed-expert tensor owned by
        another EP rank, or a parameter owned by another PP rank, was still
        consumed by a rule and must not show up as a mapping gap. Requirement 3.7
        depends on that reading: all 8 ranks iterate the same 67612 trunk tensors
        and must print the same statistics.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Accumulate, never reset here. This method is entered once per
        # *contiguous run* of ``model.*`` names in the weight stream, not once per
        # load: ``AutoWeightsLoader._groupby_prefix`` groups with
        # ``itertools.groupby``, which only merges *adjacent* keys, and the
        # checkpoint's ``head.weight`` (-> ``lm_head.weight``) sits in the middle
        # of the ``model.*`` tensors, splitting the run. An earlier revision
        # rebound ``self.consumed_ckpt_names`` to a fresh set here, so the last
        # (often single-tensor) run wiped everything the previous runs had
        # recorded -- the 8-card "66516 tensors were not consumed" WARNING while
        # the weights had in fact all loaded. The clear now lives in
        # ``reset_consumed_ckpt_names()``, called once per load by
        # ``IluDeepseekV4Fp8ForCausalLM.load_weights``.
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
                    # bytes -- so re-interpret instead (design §4.5.2).
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
                    #     hands it to task 4.5's unconsumed report instead;
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
                    # their -inf initialisation (design §4.5.3).
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
                        # deep inside the loop (design §4.5.6 -- a missing mapper
                        # rule must surface as a named list, not a stack trace).
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
    (design §4.6), so the checkpoint is loaded unmodified.
    """

    model_cls = IluDeepseekV4Fp8Model
    hf_to_vllm_mapper = _make_fp8_weights_mapper()

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

    def _tap_weight_names(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        seen: dict[str, str],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Pass ``weights`` through untouched while recording name + dtype.

        ``AutoWeightsLoader`` drains the weight generator, so the only way to
        learn which checkpoint tensors actually arrived is to tap the stream on
        the way in (design §4.5.6). Only two short strings per tensor are kept --
        never the tensor -- so the tap costs no extra host or device memory.

        Names are recorded *mapped*, so they line up with what
        ``IluDeepseekV4Fp8Model.load_weights`` records as consumed; the mapper is
        applied a second time here purely for bookkeeping and has no side effect
        (``apply_list`` returns ``[]`` for a name the mapper drops).

        ``apply_list`` is *not* a different renaming path from the one
        ``AutoWeightsLoader`` takes: community ``WeightsMapper.apply`` (the stream
        version) and ``apply_list`` both go through ``_map_name_with_shard``, so
        the four rule families (regex / substr / prefix / suffix) are applied in
        the same order with the same non-short-circuiting semantics. The only
        difference is that ``apply`` additionally stamps ``data.shard_id`` for
        ``orig_to_new_stacked`` rules, and this mapper declares none. So the tap's
        key and the loader's key are the same string by construction -- checked
        against the community source, because a mismatch here would put ``seen``
        and ``consumed`` in different namespaces and make the whole report
        meaningless. Names the
        loader will skip (``mtp.*`` plus the ``AutoWeightsLoader`` rotary
        leftovers) are not recorded at all -- they are the whitelist of
        requirement 3.4.

        One known gap, empty on this checkpoint: ``AutoWeightsLoader`` also ORs
        in ``quant_config.get_cache_scale_mapper()`` (KV-cache scale renames such
        as ``.k_scale`` -> ``.attn.k_scale``). None of those patterns can match
        here -- our own ``.scale`` regex has already rewritten every scale key to
        ``weight_scale`` / ``weight_scale_inv`` -- so the tapped name equals the
        name the loop sees. If a future checkpoint ships KV-cache scales, the
        affected tensors would surface in the unconsumed report and the mapper
        would have to be applied here too.
        """
        mapper = self.hf_to_vllm_mapper
        for name, tensor in weights:
            mapped = mapper.apply_list([name])
            if mapped and not _is_load_skipped(mapped[0]):
                seen[mapped[0]] = fp8_load_stats.dtype_name(tensor.dtype)
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, then run the Load_Verifier (design §4.5.6).

        ``skip_substrs=["mtp."]`` matches community ``nvidia/model.py:1468-1469``
        (design C5, requirement 3.1); the loading rules themselves live in
        ``IluDeepseekV4Fp8Model.load_weights``.

        Three closing steps, in this order -- statistics first so the numbers are
        on record even when a check below fails, and the raise last so the
        WARNING it would otherwise hide is already out:

        1. **statistics** (requirement 3.6/3.7): the seven checkpoint-side
           categories plus the total, one ``key=value`` line each. Classification
           is the ported scanner logic in ``fp8_load_stats`` -- same buckets as
           ``weight_inspect/scan_report.md``, which is where the acceptance
           numbers come from.
        2. **unconsumed tensor report** (requirement 3.4/3.5): the checkpoint
           tensors that arrived but matched no rule, at WARNING level with a
           count and up to 20 names. Catches the failure mode the strict check
           cannot see -- a missing mapper rule silently dropping tensors.
        3. **strict reverse check** (requirement 3.2/3.3): parameters that no
           weight ever filled, excluding ``mtp.*`` and this rank's PP-missing
           parameters, raising ``ValueError`` with a count and up to 20 names.
           vLLM's own missing-parameter check does not run for this model --
           ``DefaultModelLoader`` only performs it when
           ``model_config.quantization is None``, and this checkpoint declares
           ``deepseek_v4_fp8``.

        The two directions are independent (design §4.5.6) and the counts they
        report are different口径: step 1/2 count **checkpoint tensors**, step 3
        counts **parameters**. Stacked fusion and expert merging are many-to-one,
        so the two totals cannot be compared (requirement 3.8/3.9).
        """
        seen: dict[str, str] = {}
        # Once per load, before the loader runs: ``IluDeepseekV4Fp8Model``
        # accumulates and ``AutoWeightsLoader`` may enter its ``load_weights``
        # several times (see ``reset_consumed_ckpt_names``).
        self.model.reset_consumed_ckpt_names()
        loader = AutoWeightsLoader(self, skip_substrs=list(_LOAD_SKIP_SUBSTRS))
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
            if not _is_load_skipped(name) and not is_pp_missing_parameter(name, self)
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

    Invoked by ``vllm_iluvatar.ilu_register._register_models()``, which imports
    every top-level module of ``vllm_iluvatar.model_executor.models.deepseek_v4``
    and calls its ``register_model()`` -- this file needs no orchestration change
    (design §4.6, requirement 8.4).

    Keeping the community arch name means the checkpoint's ``config.json`` stays
    untouched (requirement 8.2) and ``tokenizer_mode`` is still auto-selected as
    ``"deepseek_v4"`` by community ``ModelConfig`` (requirement 8.7). It does not
    collide with the sibling W4A8 fork, which registers
    ``IluDeepseekV4ForCausalLM`` / ``IluDeepSeekV4MTPModel``.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model:"
        "IluDeepseekV4Fp8ForCausalLM",
    )
