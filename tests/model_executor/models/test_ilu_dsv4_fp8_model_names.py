# SPDX-License-Identifier: Apache-2.0
"""T2 (whole model): parameter-name / shape reconciliation for the DS-V4-Flash
raw-FP8 model skeleton (``IluDeepseekV4Fp8ForCausalLM``), in two scopes --
``partial`` (non-FFN, task 2.3) and ``full`` (everything, task 3.5).

Spec: ``.kiro/specs/deepseek-v4-fp8-weight-loading`` tasks 2.3 and 3.5
  * design §4.5.1 (the three stacked fusions), §4.5.5 (``WeightsMapper``),
    §5.2 (attention shape table), §5.3 (MoE / MLP shape table),
    §5.4 (model-level parameters), §6.2 (component design)
  * requirements 2.4 (routed experts go through ``expert_mapping``),
    2.5 (bidirectional name diff is empty, ``mtp.*`` excluded),
    2.6 (shapes match the reconciliation table)

Direction of the comparison
---------------------------
The expected set is built **from the checkpoint side**: the 45 main-trunk
patterns of ``weight_inspect/scan_report.md`` are expanded into concrete tensor
names for the mini config, pushed through ``_make_fp8_weights_mapper()`` and then
through the stacked-fusion rules of design §4.5.1. Only then is the result
compared against ``named_parameters()``. ``named_parameters()`` is never copied
into the expectation, and no shape is read back out of the code under test --
every shape below is transcribed from design §5.2 / §5.4 and folded down to the
mini config with the same formulas the TP8 table uses. (The §5.2 half is
imported from ``test_ilu_dsv4_fp8_attention``, which is where that table was
hand-transcribed for task 1.2; §5.4 is transcribed again here.)

Scope ``partial`` (task 2.3)
---------------------------
The comparison is restricted to the **non-FFN** subtree. Exclusion rule, applied
identically to both sides:

    a name is in the ``partial`` scope  <=>  ``".ffn." not in mapped_name``

Why the whole ``.ffn.`` subtree and not just ``.experts.`` / ``w13_weight`` /
``w2_weight`` (the wording tasks.md 2.3 originally used): when this scope was
drawn, task 2.2 had landed ``IluDeepseekV4Fp8MoE`` as a zero-parameter
placeholder, so *every* FFN tensor was unlandable -- including the ones without
``.experts.`` in the name:

  * ``layers.*.ffn.gate.weight``
  * ``layers.*.ffn.gate.bias``    -> ``ffn.gate.e_score_correction_bias``
  * ``layers.*.ffn.gate.tid*eid``
  * ``layers.*.ffn.shared_experts.w{1,3}.{weight,scale}``
    -> fused into ``ffn.shared_experts.gate_up_proj.*``
  * ``layers.*.ffn.shared_experts.w2.{weight,scale}``
    -> ``ffn.shared_experts.down_proj.*``

Note ``shared_experts`` carries an *underscore* (``shared_experts.``), so it is
not caught by a ``.experts.`` filter either. Handing the entire FFN subtree to
task 3.5 keeps the rule one line long and the two scopes disjoint by
construction. ``attn_norm`` / ``ffn_norm`` / ``hc_ffn_*`` are unaffected: none of
them contains ``.ffn.``.

Task 3.4 has since built all of those parameters, so the exclusion is no longer
about missing parameters -- it is about keeping the two scopes disjoint so a
failure lands in one place. What 3.4 did change here is
``test_moe_owns_the_expected_parameter_families`` (it replaced an assertion that
the FFN subtree was empty) and ``build_mini_fp8_model``, which now has to apply
the P1 MXFP4-backend patch before construction.

Scope ``full`` (task 3.5)
-------------------------
``full`` keeps nothing out (``mtp.*`` is already gone before any predicate
runs), so it is the whole-model diff requirement 2.4 / 2.5 asks for. Everything
except the scope predicate is shared with ``partial``; what 3.5 filled in was:

  1. ``SCOPE_PREDICATES["full"]``;
  2. ``_expert_param_names()`` -- the routed-expert fold, driven by the real
     ``fused_moe_make_expert_params_mapping`` (see ``build_expert_mapping()`` for
     its signature and the fragment-replacement contract);
  3. ``EXPECTED_FFN_PARAM_TABLE`` + ``EXPECTED_FFN_{HASH,BIAS}_GATE`` -- the
     design §5.3 shapes, folded to the mini config.

Both scopes are parametrised on every requirement test, so a failure names the
scope it came from and the two can be selected independently
(``-k "full"`` / ``-k "partial"``).

The routed fold is many-to-one: at TP1 all 8 experts are local, so per layer the
48 checkpoint tensors (8 experts x {w1,w2,w3} x {weight,scale}) collapse into 4
parameters. ``build_expected_params()`` returns ``param_name -> [checkpoint
names]`` precisely so that fan-in stays visible in the failure message.

Runtime notes
-------------
* CPU only, ``meta`` device, TP1. No kernel runs, no checkpoint is read, the
  vendor kernel package is not imported.
* The TP1 process group fixture is imported from
  ``test_ilu_dsv4_fp8_attention`` (task 1.2) rather than duplicated -- it carries
  the fix from task 1.4 (``initialize_model_parallel`` must run inside
  ``set_current_vllm_config``).
* Building the *whole* model is what first drove community code into the derived
  corners of ``ModelConfig`` (``LogitsProcessor`` -> ``head_dtype`` ->
  ``runner_type``), so ``make_mini_vllm_config`` populates every attribute a real
  initialisation produces, not just the ones a single layer reads.
  ``test_mini_model_config_is_as_complete_as_the_engines`` guards that.
* Building the ``cr=4`` / ``cr=128`` layers allocates a few KB on the
  accelerator regardless of the ``meta`` context (community
  ``DeepseekCompressor`` passes an explicit ``device=``); same for
  ``topk_indices_buffer``. No assertion depends on placement.

Validates: Requirements 2.4, 2.5, 2.6
"""

from __future__ import annotations

import contextlib
import re

import pytest
import torch

# Reused from task 1.2. ``_tp1_distributed_env`` is a session-scoped autouse
# fixture; importing it into this module's namespace is what makes it apply to
# the tests here.
from test_ilu_dsv4_fp8_attention import (  # noqa: F401
    BF16,
    E8M0,
    EXPECTED_PARAMS as ATTENTION_EXPECTED_PARAMS,
    FP8,
    FP32,
    LAYER_CR4,
    LAYER_CR128,
    LAYER_DENSE,
    MINI_COMPRESS_RATIOS,
    MINI_CONFIG_FIELDS,
    MINI_HIDDEN_SIZE,
    MINI_NUM_LAYERS,
    MINI_TP_SIZE,
    MINI_VOCAB_SIZE,
    _tp1_distributed_env,
    make_mini_vllm_config,
)

MINI_HC_MULT = MINI_CONFIG_FIELDS["hc_mult"]
MINI_NUM_HASH_LAYERS = MINI_CONFIG_FIELDS["num_hash_layers"]
MINI_NUM_ROUTED_EXPERTS = MINI_CONFIG_FIELDS["n_routed_experts"]

# ``mtp.*`` is skipped end to end (AutoWeightsLoader ``skip_substrs``, design C5
# / requirement 2.5), so it never takes part in the diff.
LOAD_SKIP_SUBSTRS: tuple[str, ...] = ("mtp.",)

# --------------------------------------------------------------------------- #
# checkpoint side: scan_report.md patterns -> concrete mini-config names
# --------------------------------------------------------------------------- #
ALWAYS = "always"
COMPRESSOR = "compressor"  # compress_ratio > 1  (scan: layers 2..42)
INDEXER = "indexer"  # compress_ratio == 4  (scan: 21 layers)
HASH_GATE = "hash_gate"  # layer_id <  num_hash_layers (scan: tid2eid on 0,1,2)
BIAS_GATE = "bias_gate"  # layer_id >= num_hash_layers (scan: bias on 3..42)

# One entry per main-trunk per-layer pattern of scan_report.md.
_LAYER_TENSOR_TEMPLATES: tuple[tuple[str, str], ...] = (
    # other_fp8_weight / other_fp8_scale
    ("attn.wq_a.weight", ALWAYS),
    ("attn.wq_a.scale", ALWAYS),
    ("attn.wkv.weight", ALWAYS),
    ("attn.wkv.scale", ALWAYS),
    ("attn.wq_b.weight", ALWAYS),
    ("attn.wq_b.scale", ALWAYS),
    ("attn.wo_a.weight", ALWAYS),
    ("attn.wo_a.scale", ALWAYS),
    ("attn.wo_b.weight", ALWAYS),
    ("attn.wo_b.scale", ALWAYS),
    ("attn.indexer.wq_b.weight", INDEXER),
    ("attn.indexer.wq_b.scale", INDEXER),
    # plain_tensor -- attention
    ("attn.attn_sink", ALWAYS),
    ("attn.q_norm.weight", ALWAYS),
    ("attn.kv_norm.weight", ALWAYS),
    ("attn.compressor.ape", COMPRESSOR),
    ("attn.compressor.norm.weight", COMPRESSOR),
    ("attn.compressor.wkv.weight", COMPRESSOR),
    ("attn.compressor.wgate.weight", COMPRESSOR),
    ("attn.indexer.weights_proj.weight", INDEXER),
    ("attn.indexer.compressor.ape", INDEXER),
    ("attn.indexer.compressor.norm.weight", INDEXER),
    ("attn.indexer.compressor.wkv.weight", INDEXER),
    ("attn.indexer.compressor.wgate.weight", INDEXER),
    # plain_tensor -- layer level
    ("attn_norm.weight", ALWAYS),
    ("ffn_norm.weight", ALWAYS),
    ("hc_attn_fn", ALWAYS),
    ("hc_attn_base", ALWAYS),
    ("hc_attn_scale", ALWAYS),
    ("hc_ffn_fn", ALWAYS),
    ("hc_ffn_base", ALWAYS),
    ("hc_ffn_scale", ALWAYS),
    # plain_tensor -- FFN gate (excluded from the ``partial`` scope, kept here so
    # the pattern coverage self-check stays honest and 3.5 needs no new table)
    ("ffn.gate.weight", ALWAYS),
    ("ffn.gate.tid2eid", HASH_GATE),
    ("ffn.gate.bias", BIAS_GATE),
    # shared_expert_weight / shared_expert_scale
    ("ffn.shared_experts.w1.weight", ALWAYS),
    ("ffn.shared_experts.w1.scale", ALWAYS),
    ("ffn.shared_experts.w2.weight", ALWAYS),
    ("ffn.shared_experts.w2.scale", ALWAYS),
    ("ffn.shared_experts.w3.weight", ALWAYS),
    ("ffn.shared_experts.w3.scale", ALWAYS),
)

_GLOBAL_TENSORS: tuple[str, ...] = (
    "embed.weight",
    "head.weight",
    "norm.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)

# Representative ``mtp.*`` names. Full mtp pattern coverage belongs to task 2.1;
# here they only have to prove the skip rule actually removes something.
_MTP_TENSORS: tuple[str, ...] = (
    "mtp.0.attn.wq_a.weight",
    "mtp.0.attn.wq_a.scale",
    "mtp.0.attn_norm.weight",
    "mtp.0.ffn.gate.weight",
    "mtp.0.ffn.experts.0.w1.weight",
    "mtp.0.hc_head_fn",
    "mtp.0.norm.weight",
)

# scan_report.md: 82 distinct patterns, 37 of them ``mtp.*`` -> 45 main trunk.
# Transcribed from the seven category tables of scan_report.md.
MAIN_TRUNK_PATTERNS: frozenset[str] = frozenset(
    {
        # routed_expert_weight / routed_expert_scale (1 + 1)
        "layers.*.ffn.experts.*.w*.weight",
        "layers.*.ffn.experts.*.w*.scale",
        # shared_expert_weight / shared_expert_scale (1 + 1)
        "layers.*.ffn.shared_experts.w*.weight",
        "layers.*.ffn.shared_experts.w*.scale",
        # other_fp8_weight (6)
        "layers.*.attn.wkv.weight",
        "layers.*.attn.wo_a.weight",
        "layers.*.attn.wo_b.weight",
        "layers.*.attn.wq_a.weight",
        "layers.*.attn.wq_b.weight",
        "layers.*.attn.indexer.wq_b.weight",
        # other_fp8_scale (6)
        "layers.*.attn.wkv.scale",
        "layers.*.attn.wo_a.scale",
        "layers.*.attn.wo_b.scale",
        "layers.*.attn.wq_a.scale",
        "layers.*.attn.wq_b.scale",
        "layers.*.attn.indexer.wq_b.scale",
        # plain_tensor (29)
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "layers.*.attn.attn_sink",
        "layers.*.attn.kv_norm.weight",
        "layers.*.attn.q_norm.weight",
        "layers.*.attn_norm.weight",
        "layers.*.ffn_norm.weight",
        "layers.*.ffn.gate.weight",
        "layers.*.ffn.gate.bias",
        "layers.*.ffn.gate.tid*eid",
        "layers.*.hc_attn_fn",
        "layers.*.hc_attn_base",
        "layers.*.hc_attn_scale",
        "layers.*.hc_ffn_fn",
        "layers.*.hc_ffn_base",
        "layers.*.hc_ffn_scale",
        "layers.*.attn.compressor.ape",
        "layers.*.attn.compressor.norm.weight",
        "layers.*.attn.compressor.wgate.weight",
        "layers.*.attn.compressor.wkv.weight",
        "layers.*.attn.indexer.compressor.ape",
        "layers.*.attn.indexer.compressor.norm.weight",
        "layers.*.attn.indexer.compressor.wgate.weight",
        "layers.*.attn.indexer.compressor.wkv.weight",
        "layers.*.attn.indexer.weights_proj.weight",
    }
)


def _normalise_pattern(name: str) -> str:
    """``scan_v4_weights.py::normalise_pattern``: digit runs become ``*``."""
    return re.sub(r"\d+", "*", name)


def iter_checkpoint_names(*, include_mtp: bool = True):
    """Expand the scan_report main-trunk patterns for the mini config."""
    yield from _GLOBAL_TENSORS
    for layer_id in range(MINI_NUM_LAYERS):
        compress_ratio = max(1, MINI_COMPRESS_RATIOS[layer_id])
        for template, scope in _LAYER_TENSOR_TEMPLATES:
            if scope == COMPRESSOR and compress_ratio == 1:
                continue
            if scope == INDEXER and compress_ratio != 4:
                continue
            if scope == HASH_GATE and layer_id >= MINI_NUM_HASH_LAYERS:
                continue
            if scope == BIAS_GATE and layer_id < MINI_NUM_HASH_LAYERS:
                continue
            yield f"layers.{layer_id}.{template}"
        for expert_id in range(MINI_NUM_ROUTED_EXPERTS):
            for w_id in ("w1", "w2", "w3"):
                for suffix in ("weight", "scale"):
                    yield (
                        f"layers.{layer_id}.ffn.experts.{expert_id}."
                        f"{w_id}.{suffix}"
                    )
    if include_mtp:
        yield from _MTP_TENSORS


# --------------------------------------------------------------------------- #
# checkpoint name -> parameter name (mapper + stacked fusion, design §4.5.1)
# --------------------------------------------------------------------------- #
# Community ``DeepseekV4Model.load_weights`` stacked_params_mapping
# (``vllm/models/deepseek_v4/nvidia/model.py:1151-1158``); the shard id is
# irrelevant for a name-level comparison, so only (param, ckpt fragment) is kept.
# Order matters: the first fragment found in the name wins, exactly as the
# community loop does (it breaks out after the first hit).
STACKED_FUSION: tuple[tuple[str, str], ...] = (
    ("gate_up_proj", "w1"),
    ("gate_up_proj", "w3"),
    ("attn.fused_wqa_wkv", "attn.wq_a"),
    ("attn.fused_wqa_wkv", "attn.wkv"),
    ("compressor.fused_wkv_wgate", "compressor.wkv"),
    ("compressor.fused_wkv_wgate", "compressor.wgate"),
)


def apply_stacked_fusion(mapped_name: str) -> str:
    """Fold a mapped checkpoint name into its stacked parameter name."""
    if ".experts." in mapped_name:
        # Routed experts bypass the stacked mapping entirely (community
        # ``continue``, nvidia/model.py:1187-1188); without this guard the
        # ``w1`` / ``w3`` fragments would also rewrite routed expert names.
        return mapped_name
    for param_fragment, ckpt_fragment in STACKED_FUSION:
        if ckpt_fragment in mapped_name:
            return mapped_name.replace(ckpt_fragment, param_fragment)
    return mapped_name


def _is_skipped(checkpoint_name: str) -> bool:
    """``AutoWeightsLoader(skip_substrs=["mtp."])`` semantics."""
    return any(substr in checkpoint_name for substr in LOAD_SKIP_SUBSTRS)


def build_expert_mapping() -> list[tuple[str, str, int, str]]:
    """The community routed-expert mapping, exactly as the loader will build it.

    ``fused_moe_make_expert_params_mapping``
    (``vllm/model_executor/layers/fused_moe/layer.py``, delegating to
    ``RoutedExperts.build_expert_params_mapping`` in ``routed_experts.py``) has
    this signature::

        fused_moe_make_expert_params_mapping(
            model, ckpt_gate_proj_name, ckpt_down_proj_name,
            ckpt_up_proj_name, num_experts,
            num_redundant_experts=0, routed_experts_prefix="routed_experts",
        ) -> list[tuple[param_name, weight_name, expert_id, shard_id]]

    and returns, for every physical expert and every ``shard_id`` in
    ``("w1", "w2", "w3")``::

        param_name  = "experts.routed_experts.w13_"   (w1 / w3)
                      "experts.routed_experts.w2_"    (w2)
        weight_name = f"experts.{logical_expert_id}.{w1|w2|w3}."

    Both are **fragments**, not full names: the loader does
    ``name.replace(weight_name, param_name)``
    (community ``nvidia/model.py:1211-1218``), which turns
    ``model.layers.0.ffn.experts.0.w1.weight`` into
    ``model.layers.0.ffn.experts.routed_experts.w13_weight`` and
    ``...w1.weight_scale`` into ``...routed_experts.w13_weight_scale``. The
    trailing dot in ``weight_name`` is what makes ``weight`` / ``weight_scale``
    fall out of the same rule, and what keeps ``experts.1.`` from matching
    ``experts.11.``.

    ``model`` is only scanned for a ``.base_layer.`` prefix (LoRA), so a bare
    ``nn.Module`` is passed deliberately: the expectation must not read anything
    out of the model under test.

    ``num_redundant_experts`` stays 0 (no EPLB in Stage 1), so
    ``physical_to_logical_map`` is the identity and ``expert_id`` equals the
    checkpoint's expert index. Neither ``expert_id`` nor ``shard_id`` affects a
    name-level comparison -- they only steer *which slice* of the parameter the
    weight loader writes into, which is task 4.4's business.
    """
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )

    return fused_moe_make_expert_params_mapping(
        torch.nn.Module(),
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=MINI_NUM_ROUTED_EXPERTS,
    )


def _expert_param_names(mapped_expert_names) -> dict[str, list[str]]:
    """Routed-expert names -> ``w13/w2`` parameter names (many-to-one).

    Filled in by task 3.5. Input is ``[(checkpoint_name, mapped_name)]`` for the
    tensors the loader hands to its ``.experts.`` branch; output is
    ``param_name -> [checkpoint names]``. At TP1 every expert is local, so all
    ``MINI_NUM_ROUTED_EXPERTS`` experts fold into the same four parameters per
    layer -- hence the list-valued provenance.

    The first matching mapping wins, mirroring the loader's ``break`` on
    ``success`` (``nvidia/model.py:1226-1231``). Because ``weight_name`` pins
    both the expert id and the projection, at most one mapping can match anyway.
    """
    expert_mapping = build_expert_mapping()

    provenance: dict[str, list[str]] = {}
    for checkpoint_name, mapped_name in mapped_expert_names:
        for param_fragment, weight_name, _expert_id, _shard_id in expert_mapping:
            if weight_name not in mapped_name:
                continue
            param_name = mapped_name.replace(weight_name, param_fragment)
            provenance.setdefault(param_name, []).append(checkpoint_name)
            break
        else:
            raise AssertionError(
                f"no expert mapping matches {mapped_name!r} (from "
                f"{checkpoint_name!r}); the loader would leave it unconsumed"
            )
    return provenance


def build_expected_params(in_scope) -> tuple[dict[str, list[str]], list[str]]:
    """Build the expected parameter names for one scope, from the checkpoint side.

    Returns ``(param_name -> [checkpoint names feeding it], skipped_names)``.
    ``skipped_names`` holds the checkpoint tensors dropped by the ``mtp.`` rule.
    """
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        _make_fp8_weights_mapper,
    )

    mapper = _make_fp8_weights_mapper()

    skipped: list[str] = []
    expert_names: list[tuple[str, str]] = []
    provenance: dict[str, list[str]] = {}

    for checkpoint_name in iter_checkpoint_names():
        if _is_skipped(checkpoint_name):
            skipped.append(checkpoint_name)
            continue
        mapped = mapper.apply_list([checkpoint_name])
        assert mapped, f"mapper dropped {checkpoint_name!r}"
        mapped_name = mapped[0]
        if ".experts." in mapped_name:
            # Routed experts never go through the stacked mapping (community
            # ``continue``s on them, nvidia/model.py:1187-1188); they are folded
            # by the expert mapping instead. ``.shared_experts.`` does not match
            # this test (the character before ``experts`` is ``_``), same as
            # community.
            expert_names.append((checkpoint_name, mapped_name))
            continue
        param_name = apply_stacked_fusion(mapped_name)
        if not in_scope(param_name):
            continue
        provenance.setdefault(param_name, []).append(checkpoint_name)

    for param_name, sources in _expert_param_names(expert_names).items():
        if in_scope(param_name):
            provenance.setdefault(param_name, []).extend(sources)

    return provenance, skipped


# --------------------------------------------------------------------------- #
# scopes
# --------------------------------------------------------------------------- #
def _in_partial_scope(param_name: str) -> bool:
    """The whole FFN subtree belongs to task 3.5 -- see the module docstring."""
    return ".ffn." not in param_name


def _in_full_scope(_param_name: str) -> bool:
    """Everything, FFN included (task 3.5). ``mtp.*`` is dropped earlier, by
    ``_is_skipped`` on the checkpoint side and by ``skip_substrs`` in the
    loader, so it never reaches a scope predicate."""
    return True


SCOPE_PREDICATES = {
    "partial": _in_partial_scope,
    "full": _in_full_scope,
}


# --------------------------------------------------------------------------- #
# expected shapes (design §5.2 via task 1.2, §5.4 transcribed here)
# --------------------------------------------------------------------------- #
# Mini config folding of design §5.4:
#   mix_hc  = (2 + hc_mult) * hc_mult      = 24
#   hc_dim  = hc_mult * hidden_size        = 4 * 256 = 1024
#   hc_head_fn = [hc_mult, hc_dim]         = [4, 1024]
_MIX_HC = (2 + MINI_HC_MULT) * MINI_HC_MULT
_HC_DIM = MINI_HC_MULT * MINI_HIDDEN_SIZE

_EXPECTED_LAYER_LEVEL: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "attn_norm.weight": ((MINI_HIDDEN_SIZE,), BF16),
    "ffn_norm.weight": ((MINI_HIDDEN_SIZE,), BF16),
    "hc_attn_fn": ((_MIX_HC, _HC_DIM), FP32),
    "hc_ffn_fn": ((_MIX_HC, _HC_DIM), FP32),
    "hc_attn_base": ((_MIX_HC,), FP32),
    "hc_ffn_base": ((_MIX_HC,), FP32),
    # community keeps a literal 3 (alpha / beta / gamma), independent of hc_mult
    "hc_attn_scale": ((3,), FP32),
    "hc_ffn_scale": ((3,), FP32),
}

_EXPECTED_MODEL_LEVEL: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "model.embed_tokens.weight": ((MINI_VOCAB_SIZE, MINI_HIDDEN_SIZE), BF16),
    "model.norm.weight": ((MINI_HIDDEN_SIZE,), BF16),
    "model.hc_head_fn": ((MINI_HC_MULT, _HC_DIM), FP32),
    "model.hc_head_base": ((MINI_HC_MULT,), FP32),
    "model.hc_head_scale": ((1,), FP32),
    "lm_head.weight": ((MINI_VOCAB_SIZE, MINI_HIDDEN_SIZE), BF16),
}

# --------------------------------------------------------------------------- #
# design §5.3 (task 3.5): FFN parameters, folded to the mini config
# --------------------------------------------------------------------------- #
# Same folding as ``test_ilu_dsv4_fp8_moe.py`` (task 3.4), reproduced here
# instead of imported so a silent change on either side shows up as a literal
# mismatch rather than as two tables agreeing on the wrong number.
#
#   E_local = n_routed_experts // tp_size   = 8 // 1 = 8
#   H       = hidden_size                   = 256
#   I_local = moe_intermediate_size // tp   = 256
#   shared I = moe_intermediate_size * n_shared_experts // tp = 256
#   MXFP4 packs 2 values per byte and carries one scale per 32 values;
#   block-FP8 scale dims are ceil(size / 128).
#
# =========================  ====================  ==================
# parameter                  design §5.3 (TP8)     mini config (TP1)
# =========================  ====================  ==================
# w13_weight                 [32, 4096, 2048]      [8, 512, 128]
# w13_weight_scale           [32, 4096,  128]      [8, 512,   8]
# w2_weight                  [32, 4096, 1024]      [8, 256, 128]
# w2_weight_scale            [32, 4096,   64]      [8, 256,   8]
# gate_up_proj.weight        [512, 4096]           [512, 256]
# gate_up_proj.scale_inv     [4, 32]               [4, 2]
# down_proj.weight           [4096, 256]           [256, 256]
# down_proj.scale_inv        [32, 2]               [2, 2]
# gate.weight                [256, 4096]           [8, 256]
# gate.e_score_..._bias      [256]                 [8]
# gate.tid2eid               [129280, 6]           [1024, 2]
# =========================  ====================  ==================
INT32 = torch.int32
UINT8 = torch.uint8
MXFP4_BLOCK = 32

_TOP_K = MINI_CONFIG_FIELDS["num_experts_per_tok"]
_MOE_INTERMEDIATE_SIZE = MINI_CONFIG_FIELDS["moe_intermediate_size"]
_N_SHARED_EXPERTS = MINI_CONFIG_FIELDS["n_shared_experts"]
_FP8_BLOCK = MINI_CONFIG_FIELDS["quantization_config"]["weight_block_size"]

_E_LOCAL = MINI_NUM_ROUTED_EXPERTS // MINI_TP_SIZE
_I_LOCAL = _MOE_INTERMEDIATE_SIZE // MINI_TP_SIZE
_SHARED_I = (_MOE_INTERMEDIATE_SIZE * _N_SHARED_EXPERTS) // MINI_TP_SIZE


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# Layer-independent FFN parameters (expanded under every ``model.layers.{i}``).
EXPECTED_FFN_PARAM_TABLE: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    # gate: ``GateLinear`` gets no ``params_dtype``, so the weight follows the
    # default dtype (bf16); only its *output* is forced to fp32.
    "ffn.gate.weight": ((MINI_NUM_ROUTED_EXPERTS, MINI_HIDDEN_SIZE), BF16),
    # shared expert, block-FP8
    "ffn.shared_experts.gate_up_proj.weight": (
        (2 * _SHARED_I, MINI_HIDDEN_SIZE),
        FP8,
    ),
    "ffn.shared_experts.gate_up_proj.weight_scale_inv": (
        (
            _ceil_div(2 * _SHARED_I, _FP8_BLOCK[0]),
            _ceil_div(MINI_HIDDEN_SIZE, _FP8_BLOCK[1]),
        ),
        E8M0,
    ),
    "ffn.shared_experts.down_proj.weight": ((MINI_HIDDEN_SIZE, _SHARED_I), FP8),
    "ffn.shared_experts.down_proj.weight_scale_inv": (
        (
            _ceil_div(MINI_HIDDEN_SIZE, _FP8_BLOCK[0]),
            _ceil_div(_SHARED_I, _FP8_BLOCK[1]),
        ),
        E8M0,
    ),
    # routed experts (community ``Mxfp4MoEMethod.create_weights``); one level
    # deeper than the checkpoint name suggests because ``FusedMoE`` returns a
    # ``MoERunner`` whose ``routed_experts`` child owns them
    "ffn.experts.routed_experts.w13_weight": (
        (_E_LOCAL, 2 * _I_LOCAL, MINI_HIDDEN_SIZE // 2),
        UINT8,
    ),
    "ffn.experts.routed_experts.w13_weight_scale": (
        (_E_LOCAL, 2 * _I_LOCAL, MINI_HIDDEN_SIZE // MXFP4_BLOCK),
        UINT8,
    ),
    "ffn.experts.routed_experts.w2_weight": (
        (_E_LOCAL, MINI_HIDDEN_SIZE, _I_LOCAL // 2),
        UINT8,
    ),
    "ffn.experts.routed_experts.w2_weight_scale": (
        (_E_LOCAL, MINI_HIDDEN_SIZE, _I_LOCAL // MXFP4_BLOCK),
        UINT8,
    ),
}

# The two mutually exclusive gate flavours (design §5.3, requirement 7.6):
# ``layer_id < num_hash_layers`` -> ``tid2eid``, otherwise the correction bias.
EXPECTED_FFN_HASH_GATE: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "ffn.gate.tid2eid": ((MINI_VOCAB_SIZE, _TOP_K), INT32),
}
EXPECTED_FFN_BIAS_GATE: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "ffn.gate.e_score_correction_bias": ((MINI_NUM_ROUTED_EXPERTS,), FP32),
}

# Guard the folding itself, same as task 3.4 does: if the mini config is shrunk
# further, this fails here (pointing at the table) instead of inside a test.
assert (_E_LOCAL, MINI_HIDDEN_SIZE, _I_LOCAL, _SHARED_I) == (8, 256, 256, 256)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.experts.routed_experts.w13_weight"
][0] == (8, 512, 128)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.experts.routed_experts.w13_weight_scale"
][0] == (8, 512, 8)
assert EXPECTED_FFN_PARAM_TABLE["ffn.experts.routed_experts.w2_weight"][0] == (
    8,
    256,
    128,
)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.experts.routed_experts.w2_weight_scale"
][0] == (8, 256, 8)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.shared_experts.gate_up_proj.weight"
][0] == (512, 256)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.shared_experts.gate_up_proj.weight_scale_inv"
][0] == (4, 2)
assert EXPECTED_FFN_PARAM_TABLE["ffn.shared_experts.down_proj.weight"][0] == (
    256,
    256,
)
assert EXPECTED_FFN_PARAM_TABLE[
    "ffn.shared_experts.down_proj.weight_scale_inv"
][0] == (2, 2)

# design §5.2, keyed by compress_ratio (tables transcribed in task 1.2).
_ATTENTION_TABLE_BY_COMPRESS_RATIO = {
    1: ATTENTION_EXPECTED_PARAMS[LAYER_DENSE],
    4: ATTENTION_EXPECTED_PARAMS[LAYER_CR4],
    128: ATTENTION_EXPECTED_PARAMS[LAYER_CR128],
}


def build_expected_shape_table() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Full expected ``name -> (shape, dtype)`` table for the mini model."""
    table: dict[str, tuple[tuple[int, ...], torch.dtype]] = dict(
        _EXPECTED_MODEL_LEVEL
    )
    for layer_id in range(MINI_NUM_LAYERS):
        compress_ratio = max(1, MINI_COMPRESS_RATIOS[layer_id])
        prefix = f"model.layers.{layer_id}"
        for name, spec in _ATTENTION_TABLE_BY_COMPRESS_RATIO[
            compress_ratio
        ].items():
            table[f"{prefix}.attn.{name}"] = spec
        for name, spec in _EXPECTED_LAYER_LEVEL.items():
            table[f"{prefix}.{name}"] = spec
        for name, spec in EXPECTED_FFN_PARAM_TABLE.items():
            table[f"{prefix}.{name}"] = spec
        # exactly one gate flavour per layer (requirement 7.6)
        gate_table = (
            EXPECTED_FFN_HASH_GATE
            if layer_id < MINI_NUM_HASH_LAYERS
            else EXPECTED_FFN_BIAS_GATE
        )
        for name, spec in gate_table.items():
            table[f"{prefix}.{name}"] = spec
    return table


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def build_mini_fp8_model():
    """Build ``IluDeepseekV4Fp8ForCausalLM`` on ``meta`` under the mini config."""
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        IluDeepseekV4Fp8ForCausalLM,
    )

    vllm_config = make_mini_vllm_config()
    # No patch registration needed since module 4.1: ``make_mini_vllm_config``
    # hands out ``IluDeepseekV4FP8Config``, so the routed experts land on
    # ``IluMxfp4MoEMethod`` and nothing asks the community MXFP4 backend oracle.
    with set_default_torch_dtype(BF16), set_current_vllm_config(vllm_config):
        with torch.device("meta"):
            model = IluDeepseekV4Fp8ForCausalLM(vllm_config=vllm_config)
        yield model, vllm_config


def _actual_param_table(model) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        name: (tuple(param.shape), param.dtype)
        for name, param in model.named_parameters()
    }


@pytest.fixture(scope="module")
def mini_model_params() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``named_parameters()`` of the mini model, built once for this module."""
    with build_mini_fp8_model() as (model, _):
        return _actual_param_table(model)


# --------------------------------------------------------------------------- #
# self checks on the expectation itself (no dependency on the code under test)
# --------------------------------------------------------------------------- #
def test_generated_checkpoint_names_cover_every_main_trunk_pattern():
    patterns = {
        _normalise_pattern(name)
        for name in iter_checkpoint_names(include_mtp=False)
    }
    missing = sorted(MAIN_TRUNK_PATTERNS - patterns)
    unexpected = sorted(patterns - MAIN_TRUNK_PATTERNS)
    assert not missing, f"scan_report patterns never generated: {missing}"
    assert not unexpected, f"generated names outside scan_report: {unexpected}"
    # 82 distinct patterns overall, 37 of them mtp (see task 2.1)
    assert len(MAIN_TRUNK_PATTERNS) == 82 - 37


def test_all_three_layer_flavours_are_covered_by_the_mini_config():
    ratios = [
        max(1, MINI_COMPRESS_RATIOS[layer_id])
        for layer_id in range(MINI_NUM_LAYERS)
    ]
    assert set(ratios) == {1, 4, 128}, (
        "the mini config must cover dense / cr4 / cr128 layers, got "
        f"{ratios}"
    )


def test_mini_model_config_is_as_complete_as_the_engines():
    """The fixture's ``ModelConfig`` must define every attribute a real one does.

    This module is the only place the *whole* model is built, so it is the only
    place community code reaches the derived corners of ``ModelConfig`` --
    ``LogitsProcessor.__init__`` reads
    ``get_current_vllm_config().model_config.head_dtype``, and ``head_dtype``
    reads ``runner_type``, which ``__post_init__`` derives rather than declares.
    Filling in one missing attribute per crash would be endless (the next
    property depends on the next derived field), so ``make_mini_model_config``
    replays *all* of them and this test checks the list against community source:
    every ``self.<attr> = ...`` that a real initialisation performs must be
    present on the fixture's object too.

    A vLLM bump that derives a new attribute therefore fails here, naming the
    attribute, instead of raising ``AttributeError`` inside a layer constructor.
    """
    from test_ilu_dsv4_fp8_attention import post_init_assigned_attrs

    model_config = make_mini_vllm_config().model_config

    expected = post_init_assigned_attrs()
    assert "runner_type" in expected, (
        "the source scan found no 'runner_type' assignment -- it is the attribute "
        "that broke this fixture, so the scan itself must be wrong"
    )
    missing = sorted(name for name in expected if not hasattr(model_config, name))
    assert not missing, (
        "make_mini_model_config does not replay every ModelConfig.__post_init__ "
        f"assignment; missing attributes: {missing}"
    )


def test_mtp_tensors_are_dropped_before_the_diff():
    _, skipped = build_expected_params(_in_partial_scope)
    assert skipped, "the sample set must contain mtp.* tensors"
    assert all(name.startswith("mtp.") for name in skipped)


# --------------------------------------------------------------------------- #
# requirement 2.5 -- bidirectional name diff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", sorted(SCOPE_PREDICATES))
def test_parameter_names_match_checkpoint_side_expectation(scope, mini_model_params):
    in_scope = SCOPE_PREDICATES[scope]
    provenance, _ = build_expected_params(in_scope)

    expected = set(provenance)
    actual = {name for name in mini_model_params if in_scope(name)}

    # checkpoint side: tensors whose target parameter does not exist
    unconsumed = sorted(
        f"{checkpoint_name} -> {param_name}"
        for param_name, sources in provenance.items()
        if param_name not in actual
        for checkpoint_name in sources
    )
    # parameter side: parameters no checkpoint tensor feeds
    unfilled = sorted(actual - expected)

    assert not unconsumed and not unfilled, (
        f"[{scope}] name reconciliation failed\n"
        f"  未消费张量清单 ({len(unconsumed)}):\n"
        + "".join(f"    {entry}\n" for entry in unconsumed)
        + f"  未填充参数清单 ({len(unfilled)}):\n"
        + "".join(f"    {entry}\n" for entry in unfilled)
    )


# --------------------------------------------------------------------------- #
# requirement 2.6 -- shapes / dtypes match the reconciliation table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", sorted(SCOPE_PREDICATES))
def test_parameter_shapes_match_design_tables(scope, mini_model_params):
    in_scope = SCOPE_PREDICATES[scope]
    expected_table = {
        name: spec
        for name, spec in build_expected_shape_table().items()
        if in_scope(name)
    }
    actual = {
        name: spec for name, spec in mini_model_params.items() if in_scope(name)
    }

    missing = sorted(set(expected_table) - set(actual))
    unexpected = sorted(set(actual) - set(expected_table))
    assert not missing, f"[{scope}] shape table entries with no parameter: {missing}"
    assert not unexpected, f"[{scope}] parameters absent from the tables: {unexpected}"

    mismatches = {
        name: {"expected": expected_table[name], "actual": actual[name]}
        for name in sorted(expected_table)
        if actual[name] != expected_table[name]
    }
    assert not mismatches, f"[{scope}] shape/dtype mismatches: {mismatches}"


@pytest.mark.parametrize("scope", sorted(SCOPE_PREDICATES))
def test_shape_table_and_checkpoint_expectation_agree(scope):
    """Two independent hand-built constructions must describe the same set:
    the checkpoint-side expansion (scan_report patterns + mapper + fusion) and
    the design §5.2 / §5.4 shape tables."""
    in_scope = SCOPE_PREDICATES[scope]
    provenance, _ = build_expected_params(in_scope)
    table = {name for name in build_expected_shape_table() if in_scope(name)}

    assert sorted(provenance) == sorted(table), (
        f"[{scope}] checkpoint-side names and shape-table names disagree\n"
        f"  only in checkpoint expansion: {sorted(set(provenance) - table)}\n"
        f"  only in shape table: {sorted(table - set(provenance))}"
    )


# --------------------------------------------------------------------------- #
# scope bookkeeping: what the ``partial`` scope defers to tasks 3.4 / 3.5
# --------------------------------------------------------------------------- #
def test_partial_scope_defers_the_whole_ffn_subtree():
    """The FFN checkpoint tensors excluded here are exactly the ones whose
    parameters task 3.4 creates; task 3.5 then covers them in the ``full``
    scope. Keeping the boundary asserted means the two scopes can never start
    overlapping or leaving a gap."""
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        _make_fp8_weights_mapper,
    )

    mapper = _make_fp8_weights_mapper()
    deferred = set()
    for checkpoint_name in iter_checkpoint_names(include_mtp=False):
        param_name = apply_stacked_fusion(
            mapper.apply_list([checkpoint_name])[0]
        )
        if not _in_partial_scope(param_name):
            deferred.add(_normalise_pattern(param_name))

    # the five families that a ``.experts.`` / ``w13_weight`` / ``w2_weight``
    # filter would have missed (``shared_experts`` has an underscore, not a dot)
    assert {
        "model.layers.*.ffn.gate.weight",
        "model.layers.*.ffn.gate.e_score_correction_bias",
        "model.layers.*.ffn.gate.tid*eid",
        "model.layers.*.ffn.shared_experts.gate_up_proj.weight",
        "model.layers.*.ffn.shared_experts.gate_up_proj.weight_scale_inv",
        "model.layers.*.ffn.shared_experts.down_proj.weight",
        "model.layers.*.ffn.shared_experts.down_proj.weight_scale_inv",
    } <= deferred
    # and nothing outside the FFN subtree is deferred
    assert all(".ffn." in pattern for pattern in deferred)


def test_routed_expert_fold_is_many_to_one():
    """Task 3.5: the expert mapping must collapse every local expert into the
    same four parameters per layer, and nothing must be dropped on the way.

    Per layer the checkpoint holds ``n_routed_experts x {w1,w2,w3} x
    {weight,scale}`` tensors; at TP1 all experts are local, so the expected set
    contains four routed parameter names per layer with these fan-ins:

      * ``w13_weight`` / ``w13_weight_scale``  <- ``w1`` and ``w3`` (2 x E)
      * ``w2_weight``  / ``w2_weight_scale``   <- ``w2`` (E)
    """
    provenance, _ = build_expected_params(_in_full_scope)
    routed = {
        name: sources
        for name, sources in provenance.items()
        if ".experts.routed_experts." in name
    }

    assert len(routed) == 4 * MINI_NUM_LAYERS, (
        "expected exactly 4 routed parameters per layer, got "
        f"{sorted(routed)}"
    )
    expected_fan_in = {
        "w13_weight": 2 * MINI_NUM_ROUTED_EXPERTS,
        "w13_weight_scale": 2 * MINI_NUM_ROUTED_EXPERTS,
        "w2_weight": MINI_NUM_ROUTED_EXPERTS,
        "w2_weight_scale": MINI_NUM_ROUTED_EXPERTS,
    }
    suffixes = {name.rsplit(".", 1)[-1] for name in routed}
    assert suffixes == set(expected_fan_in), (
        f"unexpected routed parameter suffixes: {sorted(suffixes)}"
    )
    wrong = {
        name: {
            "actual": len(sources),
            "expected": expected_fan_in[name.rsplit(".", 1)[-1]],
        }
        for name, sources in routed.items()
        if len(sources) != expected_fan_in[name.rsplit(".", 1)[-1]]
    }
    assert not wrong, f"routed expert fan-in disagrees with the mapping: {wrong}"

    # every routed checkpoint tensor of the mini config was consumed exactly once
    consumed = [src for sources in routed.values() for src in sources]
    assert len(consumed) == len(set(consumed))
    assert len(consumed) == (
        MINI_NUM_LAYERS * MINI_NUM_ROUTED_EXPERTS * 3 * 2
    ), f"routed checkpoint tensors consumed: {len(consumed)}"


def test_moe_owns_the_expected_parameter_families(mini_model_params):
    """Updated by task 3.4: the FFN subtree is no longer empty.

    Task 2.2 left ``IluDeepseekV4Fp8MoE`` as a zero-parameter placeholder and
    this test asserted exactly that. Task 3.4 built the gate, the shared-expert
    MLP and the four routed MXFP4 parameters, so the old premise is gone. What
    survives is the *reason* the ``partial`` scope exists: the FFN parameter
    names are checked here only as families (they must be exactly the ones
    design §5.3 predicts), while the name-by-name and shape-by-shape
    reconciliation against the checkpoint side is task 3.5's ``full`` scope.

    Note the routed parameters sit one level deeper than the checkpoint mapping
    suggests -- ``FusedMoE`` is a factory returning a ``MoERunner`` whose
    ``routed_experts`` child owns them, hence ``ffn.experts.routed_experts.w*``.
    """
    ffn_patterns = {
        _normalise_pattern(name)
        for name in mini_model_params
        if ".ffn." in name
    }

    expected = {
        # gate: bias on the 40 non-hash layers, tid2eid on layers 0/1/2
        "model.layers.*.ffn.gate.weight",
        "model.layers.*.ffn.gate.e_score_correction_bias",
        "model.layers.*.ffn.gate.tid*eid",
        # shared expert (design §5.3, block-FP8)
        "model.layers.*.ffn.shared_experts.gate_up_proj.weight",
        "model.layers.*.ffn.shared_experts.gate_up_proj.weight_scale_inv",
        "model.layers.*.ffn.shared_experts.down_proj.weight",
        "model.layers.*.ffn.shared_experts.down_proj.weight_scale_inv",
        # routed experts (community Mxfp4MoEMethod.create_weights)
        "model.layers.*.ffn.experts.routed_experts.w*_weight",
        "model.layers.*.ffn.experts.routed_experts.w*_weight_scale",
    }

    assert ffn_patterns == expected, (
        "FFN parameter families disagree with design §5.3\n"
        f"  missing: {sorted(expected - ffn_patterns)}\n"
        f"  unexpected: {sorted(ffn_patterns - expected)}"
    )


def test_hash_and_bias_gates_are_mutually_exclusive_per_layer(mini_model_params):
    """Requirement 7.6 at whole-model level: no layer carries both.

    ``num_hash_layers == 3`` in the mini config, so layers 0/1/2 own ``tid2eid``
    and the rest own ``e_score_correction_bias`` -- which is why
    ``scan_report.md`` counts 3 ``tid*eid`` against 40 ``ffn.gate.bias``.
    """
    hash_layers = {
        int(name.split(".")[2])
        for name in mini_model_params
        if name.endswith(".ffn.gate.tid2eid")
    }
    bias_layers = {
        int(name.split(".")[2])
        for name in mini_model_params
        if name.endswith(".ffn.gate.e_score_correction_bias")
    }

    assert hash_layers == set(range(MINI_NUM_HASH_LAYERS))
    assert bias_layers == set(range(MINI_NUM_HASH_LAYERS, MINI_NUM_LAYERS))
    assert not (hash_layers & bias_layers)
