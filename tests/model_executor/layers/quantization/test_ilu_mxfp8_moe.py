# SPDX-License-Identifier: Apache-2.0
"""MXFP8 routed experts loaded straight into the IxInfer W4A8 contract.

MXFP8 is block-FP8 on a ``[1, 32]`` grid, so the loader reuses the block-FP8
staging and the ``fp8_block_to_w4a8`` converter; what is specific to MXFP8 is
the UE8M0 scale decode and the ivcore11-only dispatch. Both are covered here,
the conversion end to end against the converter's own torch reference.
"""

import pytest
import torch
from safetensors.torch import save_file

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)
from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
    BlockFp8LoadSpec,
)
from vllm_iluvatar.model_executor.layers.quantization.common.mxfp8_loader import (
    MXFP8_BLOCK_SHAPE,
    make_mxfp8_loader,
)
from vllm_iluvatar.model_executor.model_loader.ilu_default_loader import (
    PairedLoadSession,
)
from vllm_iluvatar.model_executor.model_loader.ilu_paired_weights import (
    DeferredCheckpointTensor,
    checkpoint_references,
)

NN_SPEC = MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18)
BLOCK = MXFP8_BLOCK_SHAPE[1]


def _decode(e8m0):
    return (e8m0.to(torch.int32) << 23).view(torch.float32)


def _route(layer, refs, session, total):
    """Deliver each checkpoint tensor through the community expert callback."""
    for ref in refs:
        name = ref.source.name
        group = name.split(".")[0]
        is_scale = name.endswith("scale")
        param = getattr(layer, group + ("_weight_scale" if is_scale else "_weight"))
        value = DeferredCheckpointTensor(ref, session.readers)
        for expert in range(total):
            if group == "w13":
                half = value.shape[1] // 2
                for shard, offset in (("w1", 0), ("w3", half)):
                    param.weight_loader(
                        param,
                        value[expert].narrow(0, offset, half),
                        name,
                        shard,
                        expert,
                    )
            else:
                param.weight_loader(param, value[expert], name, "w2", expert)


def _checkpoint(tmp_path, total, h, m, tp):
    """A whole-layer MXFP8 checkpoint: e4m3 values plus a UE8M0 [1, 32] grid."""
    torch.manual_seed(321)
    source = {}
    for group, n, k in (("w13", 2 * m * tp, h), ("w2", h, m * tp)):
        source[group] = torch.randint(-32, 33, (total, n, k)).to(torch.float8_e4m3fn)
        source[group + "_scale"] = torch.randint(
            118, 136, (total, n, k // BLOCK), dtype=torch.uint8
        )
    weights = {g + ".weight": v for g, v in source.items() if not g.endswith("_scale")}
    scales = {
        g.removesuffix("_scale") + ".scale": v
        for g, v in source.items()
        if g.endswith("_scale")
    }
    files = [str(tmp_path / "scale.safetensors"), str(tmp_path / "weight.safetensors")]
    save_file(scales, files[0])
    save_file(weights, files[1])
    return source, checkpoint_references(files)


def _layer(local, ep):
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda e: (
        e - (ep - 1) * local if (ep - 1) * local <= e < ep * local else -1
    )
    return layer


def _local_source(source, group, local, ep, tp):
    """The slice of the checkpoint this rank owns, values and decoded scales."""
    experts = slice((ep - 1) * local, ep * local)
    src = source[group][experts]
    scale = _decode(source[group + "_scale"][experts])
    if group == "w13":
        src = torch.cat([x.chunk(tp, 1)[-1] for x in src.chunk(2, 1)], 1)
        scale = torch.cat([x.chunk(tp, 1)[-1] for x in scale.chunk(2, 1)], 1)
    else:
        src = src.chunk(tp, 2)[-1]
        scale = scale.chunk(tp, 2)[-1]
    return src, scale


@pytest.mark.parametrize("tp,ep,m", [(2, 1, 256), (1, 2, 256), (4, 1, 64)])
def test_e8m0_decode_and_sharding_reach_the_converter(tmp_path, monkeypatch, tp, ep, m):
    """Each shard arrives dequantized to exactly its checkpoint value."""
    local, h = 2, 256
    total = local * ep
    layer = _layer(local, ep)
    loader = make_mxfp8_loader(
        NN_SPEC, BlockFp8LoadSpec(local, h, m, MXFP8_BLOCK_SHAPE, tp, tp - 1)
    )
    loader.create_weights(layer, {})
    source, refs = _checkpoint(tmp_path, total, h, m, tp)
    seen = []
    original_write = loader._write

    def check(group, expert, shard, weight, scale):
        global_e = (ep - 1) * local + expert
        raw = source[group][global_e].float()
        expanded = _decode(source[group + "_scale"][global_e]).repeat_interleave(
            BLOCK, 1
        )[: raw.shape[0], : raw.shape[1]]
        expected = raw * expanded
        if group == "w13":
            expected = expected.chunk(2, 0)[int(shard == "w3")].chunk(tp, 0)[-1]
        else:
            expected = expected.chunk(tp, 1)[-1]
        bn, bk = loader._conversion_block
        actual = (
            weight.float()
            * scale.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[
                : weight.shape[0], : weight.shape[1]
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seen.append((expert, shard))
        original_write(group, expert, shard, weight, scale)

    monkeypatch.setattr(loader, "_write", check)
    monkeypatch.setattr(
        loader, "convert_shard", lambda w, s, out, os: (out.zero_(), os.fill_(1))
    )
    session = PairedLoadSession({loader: "experts"})
    session.bind()
    try:
        _route(layer, refs, session, total)
        assert session.readers.reads == 0
        session.finish()
    finally:
        session.close()
    loader.finalize()
    assert set(seen) == set(loader.expected_pairs())
    # MXFP8 names its grid ``weight_scale``, so ``finalize`` renames nothing.
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert layer.w13_weight_scale.dtype == torch.float32


@pytest.mark.parametrize("tp,ep", [(2, 1), (1, 2)])
def test_native_load_matches_the_converter_reference(tmp_path, tp, ep):
    """The loaded layer is byte-identical to a per-expert reference conversion."""
    from ixformer.inference.backend_registry import registry

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    if cuda_arch() != "ivcore11":
        pytest.skip("MXFP8 experts convert to IxInfer W4A8 on ivcore11")
    local, h, m = 2, 256, 256
    total = local * ep
    layer = _layer(local, ep)
    loader = make_mxfp8_loader(
        NN_SPEC, BlockFp8LoadSpec(local, h, m, MXFP8_BLOCK_SHAPE, tp, tp - 1)
    )
    with torch.device("cuda"):
        loader.create_weights(layer, {})
    source, refs = _checkpoint(tmp_path, total, h, m, tp)
    session = PairedLoadSession({loader: "experts"})
    session.bind()
    try:
        _route(layer, refs, session, total)
        with torch.device("cuda"):
            session.finish()
    finally:
        session.close()
    weights = loader.finalize()

    ref = registry.get_reference_impl("fp8_block_to_w4a8")
    for group in ("w13", "w2"):
        src, scales = _local_source(source, group, local, ep, tp)
        actual = getattr(weights, group + "_weight")
        actual_scale = getattr(weights, group + "_weight_scale")
        expected, es = torch.empty_like(actual), torch.empty_like(actual_scale)
        for e in range(local):
            ref(
                src[e],
                scales[e],
                expected[e],
                es[e],
                block_shape=MXFP8_BLOCK_SHAPE,
                weight_format="ixinfer_nn",
            )
        assert torch.equal(actual, expected)
        torch.testing.assert_close(actual_scale, es, rtol=1e-6, atol=0)


def test_loader_rejects_foreign_contracts_and_grids():
    spec = BlockFp8LoadSpec(2, 256, 256, MXFP8_BLOCK_SHAPE)
    unknown = MoEWeightSpec("ixinfer_w4a8_nn_v1", "NN", torch.int8)
    with pytest.raises(ValueError, match="No MXFP8 loader"):
        make_mxfp8_loader(unknown, spec)
    # The spec must match the contract exactly, code multiplier included: the
    # IxInfer kernel applies 18 itself, so a spec without it would double it.
    # ``ixmm_w4a8_nn`` is a real contract whose kernel does not, which is why
    # the multiplier cannot be waved through on the format id alone.
    without_multiplier = MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8)
    with pytest.raises(ValueError, match="No MXFP8 loader"):
        make_mxfp8_loader(without_multiplier, spec)
    with pytest.raises(ValueError, match="block shape"):
        make_mxfp8_loader(NN_SPEC, BlockFp8LoadSpec(2, 256, 256, (128, 128)))


def test_weight_spec_carries_the_kernel_applied_multiplier():
    """``fp8_block_to_w4a8`` leaves 18 out of ixinfer scales; the spec adds it."""
    assert NN_SPEC.code_multiplier == 18
    loader = make_mxfp8_loader(
        NN_SPEC, BlockFp8LoadSpec(2, 256, 256, MXFP8_BLOCK_SHAPE)
    )
    assert loader.weight_spec.code_multiplier == 18
    assert loader.source_scale_dtypes == (torch.uint8,)


def test_every_block_fp8_target_is_reachable_from_mxfp8():
    """The two sources differ only in the scale encoding, so a target either
    loader can produce must not depend on which one the checkpoint picked."""
    from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
        make_block_fp8_loader,
    )
    from vllm_iluvatar.model_executor.layers.quantization.common.mxfp8_loader import (
        MxFp8ToFP4FP8Loader,
        MxFp8ToW4A8Loader,
    )

    spec = BlockFp8LoadSpec(2, 256, 256, MXFP8_BLOCK_SHAPE)
    targets = {
        "ixinfer_w4a8_nn_v2": (NN_SPEC, MxFp8ToW4A8Loader),
        "ixmm_w4a8_nn": (
            MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8),
            MxFp8ToW4A8Loader,
        ),
        "ixmm_fp4fp8": (
            MoEWeightSpec(
                "ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256, alignment=32
            ),
            MxFp8ToFP4FP8Loader,
        ),
    }
    for weight_spec, expected in targets.values():
        loader = make_mxfp8_loader(weight_spec, spec)
        assert isinstance(loader, expected)
        # Same target, same conversion: only the source decode may differ.
        assert isinstance(loader, type(make_block_fp8_loader(weight_spec, spec)))
        # Every target reads the same source, so the decode must not have forked.
        assert loader.scale_parameter_name == "weight_scale"
        assert loader.source_scale_dtypes == (torch.uint8,)


def test_both_mxfp8_configs_share_one_expert_method():
    """Neither tagging convention may grow its own expert class again."""
    from vllm_iluvatar.model_executor.layers.quantization import (
        _registration,  # noqa: F401
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.moe import (
        IluMxFp8MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.configs.modelopt.mxfp8 import (
        IluModelOptMxFp8Config,
    )
    from vllm_iluvatar.model_executor.layers.quantization.configs.mxfp8 import (
        IluMxFp8Config,
    )

    assert IluMxFp8Config.FusedMoEMethodCls is IluMxFp8MoEMethod
    assert IluModelOptMxFp8Config.FusedMoEMethodCls is IluMxFp8MoEMethod
