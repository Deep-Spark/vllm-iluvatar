# SPDX-License-Identifier: Apache-2.0
"""CPU-only test (T7) for "the shared-expert intermediate padding is a no-op here".

design §4.5.4 describes two cooperating pieces of the community block-FP8
shared-expert handling:

1. ``DeepseekV4MLP.__init__`` (community ``models/deepseek_v4/nvidia/model.py``
   :104-108) lifts ``intermediate_size`` to a whole number of 128-blocks per TP
   rank::

       n_local = cdiv(intermediate_size // block_size[0], tp_size)
       intermediate_size = n_local * block_size[0] * tp_size

2. ``DeepseekV4Model._pad_shared_expert_weight`` (same file, :1259-1279)
   zero-pads the checkpoint tensor on its intermediate axis so the ordinary TP
   loaders split it evenly, with ``step = 1`` for ``weight_scale_inv`` (the axis
   is counted in blocks) and ``step = block_size[0]`` for weights, on
   ``dim = 1`` for ``.down_proj.`` and ``dim = 0`` otherwise.

For DeepSeek-V4-Flash (``I = 2048``, ``block = [128, 128]``, ``tp = 8``) both
pieces are identities: ``cdiv(16, 8) * 128 * 8 == 2048`` and
``cdiv(2048, 8 * 128) * 1024 - 2048 == 0``, so ``_pad_shared_expert_weight``
takes its ``pad == 0`` early return and hands the loader **the very same tensor
object** — no copy, no re-layout, the raw FP8 / e8m0 bytes stay untouched.
Requirements 7.4 asks exactly for that: pad ``0``, shape unchanged, object
identity.

What is under test
-----------------
The **community** method itself (``DeepseekV4Model._pad_shared_expert_weight``),
called as an unbound function with a minimal stand-in ``self``. That method only
reads ``self.quant_config``, so a stub carrying a real ``DeepseekV4FP8Config``
is enough — no model construction, no distributed init, no GPU. This is
deliberate: Task 4.4 will land an equivalent implementation in the plugin, and a
test written against a local copy of the logic would only ever validate the copy.
The TP world size is redirected to the value under test because ``tp = 8``
cannot be produced by a single-process CPU test any other way; everything else
is real community code.

Part 1 (``DeepseekV4MLP.__init__``) cannot be exercised without building the
linears (which needs a distributed group and a quant method), so it is covered
by evaluating the same formula with the real ``cdiv`` plus a drift guard that
reads the community source text.

Scope notes:

* Pure CPU, no ``ixformer`` import, no plugin code — this file passes against
  community code alone, before Task 4.4 exists.
* Every byte comparison goes through a ``uint8`` view: the FP8 dtypes carry NaN
  bit patterns and have no meaningful CPU comparison kernels, and byte identity
  is the property that matters here.
* Alongside the no-op cases there are counter-examples (``tp = 32``, non-aligned
  intermediate) proving the mechanism is live rather than dead code — it must be
  kept because a different checkpoint or TP size will engage it.

Validates: Requirements 7.4
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------
# Constants from DeepSeek-V4-Flash-config.json / scan_report.md.
# --------------------------------------------------------------------------

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = [128, 128]
TP_SIZE = 8

# Checkpoint shapes (scan_report.md: `layers.*.ffn.shared_experts.w*.weight`
# is [2048, 4096] for w1/w3 and [4096, 2048] for w2; the `.scale` companions
# are [16, 32] and [32, 16]).
GATE_UP_WEIGHT_SHAPE = (INTERMEDIATE_SIZE, HIDDEN_SIZE)  # [2048, 4096]
GATE_UP_SCALE_SHAPE = (16, 32)
DOWN_WEIGHT_SHAPE = (HIDDEN_SIZE, INTERMEDIATE_SIZE)  # [4096, 2048]
DOWN_SCALE_SHAPE = (32, 16)

# Names as seen by `_pad_shared_expert_weight`: the WeightsMapper has already
# run (`.scale` -> `.weight_scale_inv`, `.shared_experts.w2` ->
# `.shared_experts.down_proj`) but the stacked-param renaming has not, so w1/w3
# still carry their checkpoint names.
LAYER_PREFIX = "model.layers.5.ffn.shared_experts"
GATE_WEIGHT = f"{LAYER_PREFIX}.w1.weight"
GATE_SCALE = f"{LAYER_PREFIX}.w1.weight_scale_inv"
UP_WEIGHT = f"{LAYER_PREFIX}.w3.weight"
UP_SCALE = f"{LAYER_PREFIX}.w3.weight_scale_inv"
DOWN_WEIGHT = f"{LAYER_PREFIX}.down_proj.weight"
DOWN_SCALE = f"{LAYER_PREFIX}.down_proj.weight_scale_inv"

WEIGHT_DTYPE = torch.float8_e4m3fn
SCALE_DTYPE = torch.float8_e8m0fnu


# --------------------------------------------------------------------------
# Community handles + a minimal stand-in `self`.
# --------------------------------------------------------------------------


def _community_model_module():
    import vllm.models.deepseek_v4.nvidia.model as community_model

    return community_model


def _pad_fn():
    """The community method, unbound (called with the stub `self` below)."""
    return _community_model_module().DeepseekV4Model._pad_shared_expert_weight


def _cdiv():
    from vllm.utils.math_utils import cdiv

    return cdiv


class _StubModel:
    """Carries the single attribute ``_pad_shared_expert_weight`` reads."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


def _make_quant_config(weight_block_size=BLOCK_SIZE):
    """A real ``DeepseekV4FP8Config``, as built by the mini config in the other
    Stage 1 tests."""
    from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

    return DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=weight_block_size,
    )


@pytest.fixture
def pad(monkeypatch):
    """``pad(name, tensor, tp_size=8, block_size=[128, 128])``.

    Redirects the community module's TP world size for the duration of the test
    (a single CPU process cannot join an 8-rank group) and leaves the rest of the
    method's inputs real.
    """

    def _call(name, tensor, tp_size=TP_SIZE, block_size=BLOCK_SIZE):
        monkeypatch.setattr(
            _community_model_module(),
            "get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        stub = _StubModel(_make_quant_config(block_size))
        return _pad_fn()(stub, name, tensor)

    return _call


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _byte_patterned(shape, dtype):
    """A tensor whose bytes are a deterministic non-zero pattern.

    Built through a ``uint8`` view so no FP8 arithmetic is involved; values
    start at 1 so "not the pad region" is distinguishable from the zero pad.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    raw = (torch.arange(numel, dtype=torch.int64) % 255 + 1).to(torch.uint8)
    return raw.reshape(shape).view(dtype)


def _bytes(tensor):
    """Byte view of a whole (contiguous) tensor; slice the result, not the input."""
    return tensor.view(torch.uint8)


def _expected_pad(size, step, tp_size):
    """The community pad arithmetic (:1273-1274), evaluated with the real cdiv."""
    mult = tp_size * step
    return _cdiv()(size, mult) * mult - size


def _expected_mlp_intermediate(intermediate_size, block, tp_size):
    """The community ``DeepseekV4MLP.__init__`` lift (:106-108), real cdiv."""
    n_local = _cdiv()(intermediate_size // block, tp_size)
    return n_local * block * tp_size


# --------------------------------------------------------------------------
# Requirements 7.4: pad == 0 and the early return hands back the same object.
# --------------------------------------------------------------------------


NO_OP_CASES = [
    # (name, shape, dtype, dim, step)
    (GATE_WEIGHT, GATE_UP_WEIGHT_SHAPE, WEIGHT_DTYPE, 0, BLOCK_SIZE[0]),
    (UP_WEIGHT, GATE_UP_WEIGHT_SHAPE, WEIGHT_DTYPE, 0, BLOCK_SIZE[0]),
    (GATE_SCALE, GATE_UP_SCALE_SHAPE, SCALE_DTYPE, 0, 1),
    (UP_SCALE, GATE_UP_SCALE_SHAPE, SCALE_DTYPE, 0, 1),
    (DOWN_WEIGHT, DOWN_WEIGHT_SHAPE, WEIGHT_DTYPE, 1, BLOCK_SIZE[0]),
    (DOWN_SCALE, DOWN_SCALE_SHAPE, SCALE_DTYPE, 1, 1),
]

NO_OP_IDS = ["w1.weight", "w3.weight", "w1.scale", "w3.scale", "w2.weight", "w2.scale"]


@pytest.mark.parametrize("name,shape,dtype,dim,step", NO_OP_CASES, ids=NO_OP_IDS)
def test_pad_amount_is_zero_for_this_checkpoint(name, shape, dtype, dim, step):
    """The arithmetic itself: every shared-expert tensor needs 0 padding."""
    assert _expected_pad(shape[dim], step, TP_SIZE) == 0


@pytest.mark.parametrize("name,shape,dtype,dim,step", NO_OP_CASES, ids=NO_OP_IDS)
def test_early_return_yields_the_same_tensor_object(pad, name, shape, dtype, dim, step):
    """Zero-copy proof: identity of the object, not just of the shape."""
    original = _byte_patterned(shape, dtype)

    result = pad(name, original)

    # The hard evidence: no copy was made.
    assert result is original
    assert result.data_ptr() == original.data_ptr()
    # And the loader-visible properties are untouched.
    assert tuple(result.shape) == tuple(shape)
    assert result.dtype == dtype


def test_no_op_holds_for_every_realistic_tp_size():
    """I = 2048 with 128-blocks is TP-aligned for all single-node TP sizes."""
    for tp_size in (1, 2, 4, 8, 16):
        assert _expected_pad(INTERMEDIATE_SIZE, BLOCK_SIZE[0], tp_size) == 0
        assert (
            _expected_mlp_intermediate(INTERMEDIATE_SIZE, BLOCK_SIZE[0], tp_size)
            == INTERMEDIATE_SIZE
        )


# --------------------------------------------------------------------------
# design §4.5.4 part 1: the MLP-side intermediate lift is an identity too.
# --------------------------------------------------------------------------


def test_mlp_intermediate_lift_is_identity_for_this_config():
    assert (
        _expected_mlp_intermediate(INTERMEDIATE_SIZE, BLOCK_SIZE[0], TP_SIZE)
        == INTERMEDIATE_SIZE
    )


def test_mlp_intermediate_lift_formula_matches_community_source():
    """Guard against community drift: part 1 is replicated arithmetically above,
    so pin the two source lines it replicates."""
    import inspect

    source = inspect.getsource(_community_model_module().DeepseekV4MLP.__init__)

    assert "n_local = cdiv(intermediate_size // block_size[0], tp_size)" in source
    assert "intermediate_size = n_local * block_size[0] * tp_size" in source


# --------------------------------------------------------------------------
# Counter-examples: the padding path is live, not dead code.
# --------------------------------------------------------------------------


def test_padding_engages_at_tp32():
    """tp = 32 (multi-node) is the realistic setting that breaks the identity."""
    unaligned_i = INTERMEDIATE_SIZE  # 2048, but 32 * 128 = 4096 per whole shard
    assert _expected_pad(unaligned_i, BLOCK_SIZE[0], 32) == 2048
    assert _expected_mlp_intermediate(unaligned_i, BLOCK_SIZE[0], 32) == 4096


def test_weight_is_zero_padded_on_dim0_when_not_tp_aligned(pad):
    intermediate = 1152  # 9 blocks: not a multiple of tp(8) * 128
    original = _byte_patterned((intermediate, 256), WEIGHT_DTYPE)

    result = pad(GATE_WEIGHT, original)

    assert result is not original
    assert tuple(result.shape) == (2048, 256)  # cdiv(1152, 1024) * 1024
    assert result.dtype == WEIGHT_DTYPE
    # Original bytes preserved, pad region zeroed.
    assert torch.equal(_bytes(result)[:intermediate], _bytes(original))
    assert not _bytes(result)[intermediate:].any()


def test_scale_is_zero_padded_with_step_one(pad):
    n_blocks = 9  # scales count the axis in blocks, so mult is tp * 1 = 8
    original = _byte_patterned((n_blocks, 32), SCALE_DTYPE)

    result = pad(GATE_SCALE, original)

    assert result is not original
    assert tuple(result.shape) == (16, 32)  # cdiv(9, 8) * 8
    assert result.dtype == SCALE_DTYPE
    assert torch.equal(_bytes(result)[:n_blocks], _bytes(original))
    assert not _bytes(result)[n_blocks:].any()


def test_down_proj_is_padded_on_dim1(pad):
    intermediate = 1152
    original = _byte_patterned((64, intermediate), WEIGHT_DTYPE)

    result = pad(DOWN_WEIGHT, original)

    assert result is not original
    assert tuple(result.shape) == (64, 2048)
    assert torch.equal(_bytes(result)[:, :intermediate], _bytes(original))
    assert not _bytes(result)[:, intermediate:].any()


# --------------------------------------------------------------------------
# The two discriminators that make the no-op cases above meaningful: the same
# shape behaves differently depending on the suffix (step) and on the name (dim).
# --------------------------------------------------------------------------


def test_scale_suffix_selects_step_one_not_block_size(pad):
    """[16, 32] survives as a scale (mult = 8) but pads as a weight (mult = 1024)."""
    scale_like_shape = GATE_UP_SCALE_SHAPE

    as_scale = pad(GATE_SCALE, _byte_patterned(scale_like_shape, SCALE_DTYPE))
    assert tuple(as_scale.shape) == scale_like_shape

    as_weight = pad(GATE_WEIGHT, _byte_patterned(scale_like_shape, WEIGHT_DTYPE))
    assert tuple(as_weight.shape) == (1024, 32)


def test_down_proj_substring_selects_dim1(pad):
    """One shape, two names: [2048, 1152] is aligned on dim 0 but not on dim 1."""
    shape = (INTERMEDIATE_SIZE, 1152)

    gate_up = _byte_patterned(shape, WEIGHT_DTYPE)
    assert pad(GATE_WEIGHT, gate_up) is gate_up  # dim 0 == 2048 -> no pad

    down = _byte_patterned(shape, WEIGHT_DTYPE)
    padded = pad(DOWN_WEIGHT, down)
    assert padded is not down
    assert tuple(padded.shape) == (INTERMEDIATE_SIZE, 2048)  # dim 1 padded


# --------------------------------------------------------------------------
# Precondition: the method asserts a block size is configured. The caller gate
# (community :1177-1180) only invokes it when `weight_block_size is not None`.
# --------------------------------------------------------------------------


def test_requires_a_configured_weight_block_size(pad):
    original = _byte_patterned(GATE_UP_WEIGHT_SHAPE, WEIGHT_DTYPE)

    with pytest.raises(AssertionError):
        pad(GATE_WEIGHT, original, block_size=None)
