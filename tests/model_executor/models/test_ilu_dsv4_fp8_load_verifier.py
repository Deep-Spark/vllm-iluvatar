# SPDX-License-Identifier: Apache-2.0
"""CPU regression test for the Load_Verifier bookkeeping of
``IluDeepseekV4Fp8ForCausalLM.load_weights`` (design §4.5.6, requirements
3.4-3.7). Part of task 4.6.

What this pins, and why it did not exist before
-----------------------------------------------
The 8-card run of task 5.3 printed, on all 8 ranks::

    66516 checkpoint tensor(s) were not consumed by any loading rule ...

while the weights had in fact loaded correctly (no strict-reverse-check raise,
19.78 GiB resident, 148.66 GiB / 8). 66516 is *exactly* the number of trunk
tensors that the stacked branch (468) plus the routed-expert branch (66048)
handle, so the report -- not the loading -- was wrong. Two independent defects,
both invisible to every test that existed at the time:

1. ``AutoWeightsLoader._load_module`` calls the child's ``load_weights`` **once
   per contiguous run of ``model.*`` names**: ``_groupby_prefix`` groups with
   ``itertools.groupby``, which only merges *adjacent* keys. The checkpoint's
   ``head.weight`` maps to ``lm_head.weight``, i.e. a different top-level prefix,
   and it sits in the middle of the ``model.*`` tensors -- so the run is split and
   ``IluDeepseekV4Fp8Model.load_weights`` is entered twice. It used to rebind
   ``self.consumed_ckpt_names`` to a fresh set on entry, so the *last* run (a
   single tensor in the real checkpoint's file layout) discarded everything the
   first run had recorded.

2. The verifier then did ``consumed |= loaded``, unioning the whole
   **parameter-side** result into a **checkpoint-side** set. For every 1:1 tensor
   the mapped checkpoint name *equals* the parameter name, so those names were
   silently restored and only the many-to-one families (stacked fusion, routed
   experts) still showed up as unconsumed. That alias is what made the failure
   look branch-specific and hid defect 1 for an entire task.

Both defects are key-space defects: ``seen`` is keyed by mapped checkpoint names,
``consumed`` must be keyed the same way, and every earlier test only ever
exercised one side of that equality (T1 the mapper alone, T2 the parameter names
alone, T6 the statistics classifier alone). This module closes the gap: it drives
the *real* ``ForCausalLM.load_weights`` -- real mapper, real
``AutoWeightsLoader``, real four-branch loop, real verifier -- over a synthetic
stream of the mini config's checkpoint tensors, and asserts

  * the unconsumed report is empty,
  * ``total_trunk_tensor`` equals the number of non-``mtp.`` tensors fed in,
  * the two key sets are the same set (modulo ``lm_head.weight``),

for three stream orderings, including the one that reproduces the 8-card file
layout (``head.weight`` second-to-last, so the trailing ``model.*`` run holds one
tensor). Against the pre-fix source, ``head_before_tail`` reports 66516-style
losses and ``head_early`` loses the first run as well; the fixed source keeps the
books straight for any ordering.

Why the weight loaders are neutralised
--------------------------------------
``_neutralised_weight_loaders`` swaps every parameter's ``weight_loader`` for a
recording no-op. The subject here is the *name* plumbing: which checkpoint name
each rule claims, and in which namespace that claim is recorded. Byte-level
loading behaviour is covered elsewhere and deliberately not re-tested here --
T3 (e8m0 ``view`` vs ``copy_``), T7 (shared-expert pad), T4/T5 (attention and MoE
shapes). Keeping the copies out means this test needs no accelerator, no kernel
and no dtype-conversion support, so it runs anywhere the model can be built.

Two paths stay real because they do not go through ``weight_loader``:
``attn_sink`` (the loop copies the head slice itself) and
``_pad_shared_expert_weight``; the synthetic tensors are shaped and typed so both
behave exactly as they do on the real checkpoint.

The synthetic stream is shaped from design §5.2 / §5.3 / §5.4 folded to the mini
config -- the same tables ``test_ilu_dsv4_fp8_attention`` /
``test_ilu_dsv4_fp8_model_names`` transcribe -- and typed from
``weight_inspect/scan_report.md``, so the statistics classification is exercised
on realistic ``(pattern, dtype)`` pairs rather than on placeholders.

Validates: Requirements 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import contextlib
import logging

import pytest
import torch

from test_ilu_dsv4_fp8_attention import (
    BF16,
    E8M0,
    FP8,
    FP32,
    MINI_COMPRESS_RATIOS,
    MINI_CONFIG_FIELDS,
    MINI_HIDDEN_SIZE,
    MINI_NUM_HEADS,
    MINI_VOCAB_SIZE,
    _tp1_distributed_env,  # noqa: F401  session autouse fixture (TP1 group)
    make_mini_vllm_config,
)
from test_ilu_dsv4_fp8_model_names import (
    MINI_NUM_ROUTED_EXPERTS,
    iter_checkpoint_names,
    p1_mxfp4_backend_patch,
)

INT8 = torch.int8
INT64 = torch.int64

# --------------------------------------------------------------------------- #
# checkpoint-side shapes / dtypes for the mini config
# --------------------------------------------------------------------------- #
_H = MINI_HIDDEN_SIZE  # 256
_HEADS = MINI_NUM_HEADS  # 8
_HEAD_DIM = MINI_CONFIG_FIELDS["head_dim"]  # 512
_Q_LORA = MINI_CONFIG_FIELDS["q_lora_rank"]  # 1024
_O_LORA = MINI_CONFIG_FIELDS["o_lora_rank"]  # 1024
_O_GROUPS = MINI_CONFIG_FIELDS["o_groups"]  # 8
_INDEX_N_HEADS = MINI_CONFIG_FIELDS["index_n_heads"]  # 64
_INDEX_HEAD_DIM = MINI_CONFIG_FIELDS["index_head_dim"]  # 128
_HC_MULT = MINI_CONFIG_FIELDS["hc_mult"]  # 4
_MOE_I = MINI_CONFIG_FIELDS["moe_intermediate_size"]  # 256
_SHARED_I = _MOE_I * MINI_CONFIG_FIELDS["n_shared_experts"]  # 256
_TOP_K = MINI_CONFIG_FIELDS["num_experts_per_tok"]  # 2
_FP8_BLOCK = MINI_CONFIG_FIELDS["quantization_config"]["weight_block_size"][0]  # 128
_MXFP4_GROUP = 32

_HC_DIM = _HC_MULT * _H  # 1024
_MIX_HC = (2 + _HC_MULT) * _HC_MULT  # 24


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _block_scale_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Block-FP8 companion scale of a ``[out, in]`` weight (block 128x128)."""
    return (_ceil_div(shape[0], _FP8_BLOCK), _ceil_div(shape[1], _FP8_BLOCK))


def _fp8_pair(name: str, shape: tuple[int, int]) -> dict:
    """``{name.weight: fp8, name.scale: e8m0}`` -- the checkpoint's spelling."""
    return {
        f"{name}.weight": (shape, FP8),
        f"{name}.scale": (_block_scale_shape(shape), E8M0),
    }


# Layer-flavour independent tensors, keyed by the name below ``layers.{i}.``.
_LAYER_SPECS_ALWAYS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    # --- attention, block-FP8 (design §5.2) --------------------------------
    **_fp8_pair("attn.wq_a", (_Q_LORA, _H)),
    **_fp8_pair("attn.wkv", (_HEAD_DIM, _H)),
    **_fp8_pair("attn.wq_b", (_HEADS * _HEAD_DIM, _Q_LORA)),
    **_fp8_pair("attn.wo_a", (_O_GROUPS * _O_LORA, _HEADS * _HEAD_DIM // _O_GROUPS)),
    **_fp8_pair("attn.wo_b", (_H, _O_GROUPS * _O_LORA)),
    # --- attention, plain --------------------------------------------------
    # all heads live in the checkpoint; the loop narrows this rank's slice and
    # front-fills a [padded_heads] parameter (design §4.5.3)
    "attn.attn_sink": ((_HEADS,), FP32),
    "attn.q_norm.weight": ((_Q_LORA,), BF16),
    "attn.kv_norm.weight": ((_HEAD_DIM,), BF16),
    # --- layer level, plain -------------------------------------------------
    "attn_norm.weight": ((_H,), BF16),
    "ffn_norm.weight": ((_H,), BF16),
    "hc_attn_fn": ((_MIX_HC, _HC_DIM), FP32),
    "hc_attn_base": ((_MIX_HC,), FP32),
    "hc_attn_scale": ((3,), FP32),
    "hc_ffn_fn": ((_MIX_HC, _HC_DIM), FP32),
    "hc_ffn_base": ((_MIX_HC,), FP32),
    "hc_ffn_scale": ((3,), FP32),
    # --- FFN gate + shared expert ------------------------------------------
    "ffn.gate.weight": ((MINI_NUM_ROUTED_EXPERTS, _H), BF16),
    "ffn.gate.tid2eid": ((MINI_VOCAB_SIZE, _TOP_K), INT64),
    "ffn.gate.bias": ((MINI_NUM_ROUTED_EXPERTS,), FP32),
    **_fp8_pair("ffn.shared_experts.w1", (_SHARED_I, _H)),
    **_fp8_pair("ffn.shared_experts.w3", (_SHARED_I, _H)),
    **_fp8_pair("ffn.shared_experts.w2", (_H, _SHARED_I)),
}


def _compressor_specs(
    prefix: str, compress_ratio: int, head_dim: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``DeepseekCompressor`` tensors: ``coff = 1 + (compress_ratio == 4)``.

    ``wkv`` / ``wgate`` are the two shards of ``compressor.fused_wkv_wgate``, so
    each is half of the parameter's rows -- the numbers agree with the
    ``[2 * coff * head_dim, H]`` fused shapes of design §5.2 (transcribed in
    ``test_ilu_dsv4_fp8_attention``).
    """
    coff = 1 + (compress_ratio == 4)
    return {
        f"{prefix}.ape": ((compress_ratio, coff * head_dim), FP32),
        f"{prefix}.norm.weight": ((head_dim,), BF16),
        f"{prefix}.wkv.weight": ((coff * head_dim, _H), BF16),
        f"{prefix}.wgate.weight": ((coff * head_dim, _H), BF16),
    }


_INDEXER_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    **_fp8_pair("attn.indexer.wq_b", (_INDEX_N_HEADS * _INDEX_HEAD_DIM, _Q_LORA)),
    "attn.indexer.weights_proj.weight": ((_INDEX_N_HEADS, _H), BF16),
    # the indexer's own compressor always runs at compress_ratio 4
    **_compressor_specs("attn.indexer.compressor", 4, _INDEX_HEAD_DIM),
}

# MXFP4 routed experts (scan_report.md: weight I8 packed 2/byte, scale E8M0 with
# one exponent per 32 values). w1/w3 are ``[I, H/2]``, w2 is ``[H, I/2]``.
_EXPERT_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "w1.weight": ((_MOE_I, _H // 2), INT8),
    "w1.scale": ((_MOE_I, _H // _MXFP4_GROUP), E8M0),
    "w3.weight": ((_MOE_I, _H // 2), INT8),
    "w3.scale": ((_MOE_I, _H // _MXFP4_GROUP), E8M0),
    "w2.weight": ((_H, _MOE_I // 2), INT8),
    "w2.scale": ((_H, _MOE_I // _MXFP4_GROUP), E8M0),
}

_GLOBAL_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "embed.weight": ((MINI_VOCAB_SIZE, _H), BF16),
    "head.weight": ((MINI_VOCAB_SIZE, _H), BF16),
    "norm.weight": ((_H,), BF16),
    "hc_head_fn": ((_HC_MULT, _HC_DIM), FP32),
    "hc_head_base": ((_HC_MULT,), FP32),
    "hc_head_scale": ((1,), FP32),
}

# ``mtp.*`` never reaches a loading rule (AutoWeightsLoader ``skip_substrs``), so
# its shapes are irrelevant -- only its presence matters (requirement 3.4).
_MTP_SPEC: tuple[tuple[int, ...], torch.dtype] = ((1,), FP32)


def checkpoint_spec(name: str) -> tuple[tuple[int, ...], torch.dtype]:
    """``(shape, dtype)`` of one mini-config checkpoint tensor."""
    if name.startswith("mtp."):
        return _MTP_SPEC
    if name in _GLOBAL_SPECS:
        return _GLOBAL_SPECS[name]

    assert name.startswith("layers."), name
    _, layer_id, rest = name.split(".", 2)
    compress_ratio = max(1, MINI_COMPRESS_RATIOS[int(layer_id)])

    if rest.startswith("ffn.experts."):
        # ffn.experts.{expert_id}.{w1|w2|w3}.{weight|scale}
        return _EXPERT_SPECS[rest.split(".", 3)[3]]
    if rest in _LAYER_SPECS_ALWAYS:
        return _LAYER_SPECS_ALWAYS[rest]
    if rest in _INDEXER_SPECS:
        return _INDEXER_SPECS[rest]
    compressor = _compressor_specs("attn.compressor", compress_ratio, _HEAD_DIM)
    if rest in compressor:
        return compressor[rest]
    raise AssertionError(f"no checkpoint spec for {name!r}")


def make_checkpoint_tensor(name: str) -> torch.Tensor:
    """A zero-filled CPU tensor with the checkpoint's shape and dtype.

    ``torch.zeros`` has no CPU kernel for the FP8 dtypes, so those are allocated
    as ``uint8`` and reinterpreted -- the same construction
    ``test_ilu_dsv4_fp8_e8m0.py`` uses. Values are never asserted on; only the
    dtype (statistics classification, the e8m0 ``view`` in the expert branch) and
    the shape (``attn_sink`` narrowing, the shared-expert pad) matter.
    """
    shape, dtype = checkpoint_spec(name)
    if dtype in (FP8, E8M0):
        return torch.zeros(shape, dtype=torch.uint8).view(dtype)
    return torch.zeros(shape, dtype=dtype)


# --------------------------------------------------------------------------- #
# stream orderings
# --------------------------------------------------------------------------- #
# ``AutoWeightsLoader`` groups by *adjacent* top-level prefix, so where
# ``head.weight`` (-> ``lm_head.weight``) sits decides how many times
# ``IluDeepseekV4Fp8Model.load_weights`` is entered. Each ordering below is a
# layout a real shard set can produce (shard files are globbed, and keys inside a
# safetensors file come out sorted, so ``head.weight`` lands wherever its shard
# does). The expected run count is asserted, so a community change to the
# grouping shows up here instead of silently weakening the test.
ORDERINGS: dict[str, int] = {
    # one contiguous ``model.*`` run -- the only layout the pre-fix code survived
    "single_run": 1,
    # ``head.weight`` early, as ``iter_checkpoint_names`` emits it
    "head_early": 2,
    # ``head.weight`` second to last: the trailing run holds a single tensor.
    # This is the 8-card layout that produced "66516 ... not consumed".
    "head_before_tail": 2,
}

_TAIL = "norm.weight"
_HEAD = "head.weight"


def ordered_checkpoint_names(order: str) -> list[str]:
    names = list(iter_checkpoint_names())
    if order == "head_early":
        return names
    without_head = [name for name in names if name != _HEAD]
    if order == "single_run":
        return [*without_head, _HEAD]
    if order == "head_before_tail":
        body = [name for name in without_head if name != _TAIL]
        return [*body, _HEAD, _TAIL]
    raise AssertionError(f"unknown ordering {order!r}")


def synthetic_weight_stream(order: str):
    """Lazily materialise the stream, one tensor at a time (as the loader does)."""
    for name in ordered_checkpoint_names(order):
        yield name, make_checkpoint_tensor(name)


def mapped_trunk_names() -> set[str]:
    """The mapped, non-``mtp.`` checkpoint names -- the key space of ``seen``."""
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        _is_load_skipped,
        _make_fp8_weights_mapper,
    )

    mapper = _make_fp8_weights_mapper()
    mapped = mapper.apply_list(list(iter_checkpoint_names()))
    return {name for name in mapped if not _is_load_skipped(name)}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
_MISSING = object()


@contextlib.contextmanager
def neutralised_weight_loaders(model):
    """Replace every ``param.weight_loader`` with a permissive recording no-op.

    Returns the list of parameter names the loop asked to load. ``return_success``
    (the routed-expert protocol) is answered ``True``: at TP1 every expert is
    local, so the real loader would accept on the first matching mapping too.
    """
    called: list[str] = []
    originals: list[tuple[torch.nn.Parameter, object]] = []

    def make_loader(param_name: str):
        def loader(param, loaded_weight, *args, **kwargs):
            called.append(param_name)
            if kwargs.get("return_success"):
                return True
            return None

        return loader

    for name, param in model.named_parameters():
        originals.append((param, getattr(param, "weight_loader", _MISSING)))
        param.weight_loader = make_loader(name)
    try:
        yield called
    finally:
        for param, original in originals:
            if original is _MISSING:
                with contextlib.suppress(AttributeError):
                    del param.weight_loader
            else:
                param.weight_loader = original


class RecordingLogger:
    """Stand-in for the module logger, capturing formatted messages per level.

    ``caplog`` cannot be used: the module logger now lives under ``vllm.…`` and
    the ``"vllm"`` logger has ``propagate=False``, so its records never reach the
    root handler pytest installs. Swapping the module-level object is both
    simpler and independent of the logging configuration -- which
    ``test_module_logger_reaches_a_handler_at_info_level`` checks separately.
    """

    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []
        self.debug_lines: list[str] = []

    @staticmethod
    def _format(msg: str, args: tuple) -> str:
        return msg % args if args else msg

    def info(self, msg, *args) -> None:
        self.info_lines.append(self._format(msg, args))

    def warning(self, msg, *args) -> None:
        self.warning_lines.append(self._format(msg, args))

    def debug(self, msg, *args) -> None:
        self.debug_lines.append(self._format(msg, args))


@pytest.fixture(scope="module")
def mini_model():
    """``IluDeepseekV4Fp8ForCausalLM`` with real CPU storage (not ``meta``).

    ``meta`` is unusable here: the ``attn_sink`` branch copies into the parameter
    directly, which a meta tensor cannot accept. The mini config is a few MB, so
    real storage is cheap. Built once -- ``load_weights`` writes nothing (the
    loaders are neutralised) and the bookkeeping is cleared per load, so the
    tests stay independent.
    """
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        IluDeepseekV4Fp8ForCausalLM,
    )

    vllm_config = make_mini_vllm_config()
    with p1_mxfp4_backend_patch():
        with set_default_torch_dtype(BF16), set_current_vllm_config(vllm_config):
            model = IluDeepseekV4Fp8ForCausalLM(vllm_config=vllm_config)
    return model


def run_load(model, order, monkeypatch):
    """Drive the real ``ForCausalLM.load_weights`` and report what it logged.

    Returns ``(loaded, recorder, model_load_weights_calls)``.
    """
    from vllm_iluvatar.model_executor.models.deepseek_v4 import fp8_model

    # Drop any patch a previous call in the same test installed, so the wrapper
    # below always wraps the genuine method instead of stacking on itself.
    monkeypatch.undo()

    recorder = RecordingLogger()
    monkeypatch.setattr(fp8_model, "logger", recorder)

    # Count how often AutoWeightsLoader enters the child's load_weights: that
    # count *is* the condition the bug needed, so the test has to prove it hit it.
    calls: list[int] = []
    original = fp8_model.IluDeepseekV4Fp8Model.load_weights

    def counting_load_weights(self, weights):
        calls.append(1)
        return original(self, weights)

    monkeypatch.setattr(
        fp8_model.IluDeepseekV4Fp8Model, "load_weights", counting_load_weights
    )

    with neutralised_weight_loaders(model):
        loaded = model.load_weights(synthetic_weight_stream(order))
    return loaded, recorder, len(calls)


# --------------------------------------------------------------------------- #
# self checks on the synthetic stream
# --------------------------------------------------------------------------- #
def test_every_generated_name_has_a_checkpoint_spec():
    for name in iter_checkpoint_names():
        shape, dtype = checkpoint_spec(name)
        assert shape and dtype is not None, name


def test_stream_reproduces_the_grouping_the_bug_needed(mini_model, monkeypatch):
    """Each ordering must produce the number of child ``load_weights`` calls it
    claims -- otherwise the regression below would pass for the wrong reason."""
    for order, expected_calls in ORDERINGS.items():
        _, _, calls = run_load(mini_model, order, monkeypatch)
        assert calls == expected_calls, (
            f"[{order}] AutoWeightsLoader entered IluDeepseekV4Fp8Model."
            f"load_weights {calls} time(s), expected {expected_calls}; the "
            "grouping this test relies on has changed"
        )


# --------------------------------------------------------------------------- #
# requirement 3.4 / 3.5 -- the unconsumed report must be empty
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", sorted(ORDERINGS))
def test_no_checkpoint_tensor_is_reported_unconsumed(mini_model, order, monkeypatch):
    _, recorder, _ = run_load(mini_model, order, monkeypatch)
    unconsumed = [line for line in recorder.warning_lines if "not consumed" in line]
    assert not unconsumed, f"[{order}] {unconsumed}"
    # nothing else may warn either (an ``unknown`` classification would)
    assert not recorder.warning_lines, (
        f"[{order}] unexpected warnings: {recorder.warning_lines}"
    )


@pytest.mark.parametrize("order", sorted(ORDERINGS))
def test_consumed_and_seen_share_one_key_space(mini_model, order, monkeypatch):
    """The core invariant this module exists for.

    ``seen`` (checkpoint side, mapped) and ``consumed`` (what the loop recorded,
    re-prefixed with ``model.``) must be the *same* set of strings. The only
    legitimate difference is ``lm_head.weight``: it lives outside the ``model.``
    subtree, so ``AutoWeightsLoader`` loads it itself and the loop never sees it.
    """
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        _MODEL_PREFIX,
    )

    run_load(mini_model, order, monkeypatch)

    seen = mapped_trunk_names()
    consumed = {
        f"{_MODEL_PREFIX}{name}" for name in mini_model.model.consumed_ckpt_names
    }

    assert not consumed - seen, (
        f"[{order}] the loop recorded names that are not mapped checkpoint "
        f"names: {sorted(consumed - seen)}"
    )
    assert seen - consumed == {"lm_head.weight"}, (
        f"[{order}] checkpoint names the loop did not record: "
        f"{sorted(seen - consumed - {'lm_head.weight'})}"
    )


# --------------------------------------------------------------------------- #
# requirement 3.6 / 3.7 -- the eight statistics lines
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", sorted(ORDERINGS))
def test_statistics_cover_every_trunk_tensor(mini_model, order, monkeypatch):
    from vllm_iluvatar.model_executor.models.deepseek_v4 import fp8_load_stats

    _, recorder, _ = run_load(mini_model, order, monkeypatch)

    stats_lines = [
        line
        for line in recorder.info_lines
        if line.startswith(f"[{fp8_load_stats.STATS_LOG_TAG}]")
    ]
    assert len(stats_lines) == len(fp8_load_stats.STATS_KEYS), (
        f"[{order}] expected one line per published key, got {stats_lines}"
    )

    reported = {}
    for line in stats_lines:
        key, _, value = line.split("] ", 1)[1].partition("=")
        reported[key] = int(value)
    assert list(reported) == list(fp8_load_stats.STATS_KEYS), (
        f"[{order}] statistics keys drifted: {list(reported)}"
    )

    # Every trunk tensor fed in must have been counted exactly once (the mapper is
    # 1:1 on names, so the mapped set has one entry per checkpoint tensor), and the
    # seven categories must account for all of them (``unknown`` == 0).
    expected_total = len(mapped_trunk_names())
    assert reported[fp8_load_stats.TOTAL_KEY] == expected_total, (
        f"[{order}] total_trunk_tensor={reported[fp8_load_stats.TOTAL_KEY]}, "
        f"expected {expected_total}"
    )
    category_sum = sum(reported[key] for key in fp8_load_stats.CATEGORY_KEYS)
    assert category_sum == reported[fp8_load_stats.TOTAL_KEY], (
        f"[{order}] unclassified tensors: "
        f"{reported[fp8_load_stats.TOTAL_KEY] - category_sum}"
    )


def test_mtp_tensors_are_excluded_from_the_statistics(mini_model, monkeypatch):
    """Requirement 3.4's whitelist: the stream carries ``mtp.*`` tensors and the
    statistics must not count them (the trunk total is mtp-free by definition)."""
    from vllm_iluvatar.model_executor.models.deepseek_v4 import fp8_load_stats

    names = list(iter_checkpoint_names())
    assert any(name.startswith("mtp.") for name in names)

    _, recorder, _ = run_load(mini_model, "head_early", monkeypatch)
    total_line = next(
        line for line in recorder.info_lines if fp8_load_stats.TOTAL_KEY in line
    )
    total = int(total_line.rsplit("=", 1)[1])
    trunk = [name for name in names if not name.startswith("mtp.")]
    assert total == len(trunk), f"total={total}, trunk tensors fed in={len(trunk)}"


# --------------------------------------------------------------------------- #
# the logging defect: eight INFO lines that never reached a handler
# --------------------------------------------------------------------------- #
def test_module_logger_reaches_a_handler_at_info_level():
    """``[ilu_fp8_load_stats]`` lines must actually make it into the log.

    ``vllm.logger`` configures a handler for the ``"vllm"`` logger only and none
    for the root logger, so a logger named ``vllm_iluvatar.…`` resolves to no
    handler at all and ``logging.Logger.callHandlers`` falls back to
    ``logging.lastResort`` -- level WARNING. That is why the 8-card log showed the
    unconsumed WARNING but not one of the eight INFO lines. This asserts the two
    conditions those lines need: a reachable handler, and INFO enabled.
    """
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
        logger as module_logger,
    )

    assert isinstance(module_logger, logging.Logger)
    assert module_logger.name.startswith("vllm."), (
        f"{module_logger.name!r} is outside the configured 'vllm' logger tree, "
        "so its INFO records are dropped by logging.lastResort"
    )

    node = module_logger
    while node is not None:
        if node.handlers:
            break
        node = node.parent if node.propagate else None
    assert node is not None and node.handlers, (
        f"no handler is reachable from {module_logger.name!r}"
    )
    assert module_logger.isEnabledFor(logging.INFO)
