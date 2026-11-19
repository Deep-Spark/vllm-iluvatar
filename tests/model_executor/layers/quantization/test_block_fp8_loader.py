# SPDX-License-Identifier: Apache-2.0
"""Check TP gate/up halves and EP expert ranges in the paired FP8 upload slot."""

import pytest
import torch
from safetensors.torch import save_file

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)
from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
    BlockFp8LoadSpec,
    make_block_fp8_loader,
)
from vllm_iluvatar.model_executor.model_loader.ilu_default_loader import (
    PairedLoadSession,
)
from vllm_iluvatar.model_executor.model_loader.ilu_paired_weights import (
    DeferredCheckpointTensor,
    checkpoint_references,
)


def _route(layer, refs, session, total, *, whole):
    for ref in refs:
        name = ref.source.name
        group = name.split(".")[0]
        is_scale = name.endswith("scale")
        param = getattr(layer, group + ("_weight_scale_inv" if is_scale else "_weight"))
        value = DeferredCheckpointTensor(ref, session.readers)
        if whole:
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
        else:
            _, expert, shard, _ = name.split(".")
            param.weight_loader(param, value, name, shard, int(expert))


def _checkpoint(tmp_path, total, h, m, tp, scale_dtype, whole):
    torch.manual_seed(321)
    source = {}
    for group, n, k in (("w13", 2 * m * tp, h), ("w2", h, m * tp)):
        source[group] = torch.randint(-32, 33, (total, n, k)).to(torch.float8_e4m3fn)
        source[group + "_scale"] = (
            torch.rand((total, (n + 127) // 128, (k + 127) // 128)) + 0.1
        ).to(scale_dtype)
    weights, scales = {}, {}
    for name, value in source.items():
        is_scale = name.endswith("scale")
        group = name.removesuffix("_scale")
        collection = scales if is_scale else weights
        if whole:
            collection[group + "." + ("scale" if is_scale else "weight")] = value
        else:
            for expert in range(total):
                parts = ("w1", "w3") if group == "w13" else ("w2",)
                for shard, part in zip(parts, value[expert].chunk(len(parts), dim=0)):
                    collection[
                        f"{group}.{expert}.{shard}."
                        + ("scale" if is_scale else "weight")
                    ] = part.contiguous()
    files = [str(tmp_path / "scale.safetensors"), str(tmp_path / "weight.safetensors")]
    save_file(scales, files[0])
    save_file(weights, files[1])
    return source, checkpoint_references(files)


@pytest.mark.parametrize(
    "whole,tp,ep,m,dtype",
    [
        (True, 2, 1, 256, torch.bfloat16),
        (False, 2, 1, 256, torch.float32),
        (True, 1, 2, 256, torch.bfloat16),
        (False, 4, 1, 32, torch.float32),
        (True, 2, 1, 192, torch.float32),
    ],
)
def test_source_views_tp_refinement_and_ep(
    tmp_path, monkeypatch, whole, tp, ep, m, dtype
):
    local, h = 2, 256
    total = local * ep
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda e: (
        e - (ep - 1) * local if (ep - 1) * local <= e < ep * local else -1
    )
    spec = MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18)
    loader = make_block_fp8_loader(
        spec, BlockFp8LoadSpec(local, h, m, (128, 128), tp, tp - 1)
    )
    loader.create_weights(layer, {})
    source, refs = _checkpoint(tmp_path, total, h, m, tp, dtype, whole)
    seen = []
    original_write = loader._write

    def check(group, expert, shard, weight, scale):
        global_e = (ep - 1) * local + expert
        raw = source[group][global_e].float()
        expanded = (
            source[group + "_scale"][global_e]
            .float()
            .repeat_interleave(128, 0)
            .repeat_interleave(128, 1)[: raw.shape[0], : raw.shape[1]]
        )
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
        _route(layer, refs, session, total, whole=whole)
        assert session.readers.reads == 0
        session.finish()
    finally:
        session.close()
    loader.finalize()
    assert set(seen) == set(loader.expected_pairs())
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert layer.w13_weight_scale.dtype == torch.float32
    with pytest.raises(RuntimeError, match="finalized"):
        loader.finalize()


@pytest.mark.parametrize("mode", ["auto", "w4a8"])
@pytest.mark.parametrize("tp,ep,whole", [(2, 1, True), (2, 1, False), (1, 2, True)])
def test_native_load_compute_contract(tmp_path, monkeypatch, mode, tp, ep, whole):
    from ixformer.inference.backend_registry import registry

    from vllm_iluvatar.config.ilu_compute import select_four_bit_expert_compute
    from vllm_iluvatar.ilu_util.arch import cuda_arch
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        get_moe_backend,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        preprocess_fp8_block_weight_for_ixmm,
    )

    arch = cuda_arch()
    if arch not in ("ivcore11", "ivcore30"):
        pytest.skip("requires a supported Iluvatar GPU")
    mode = (
        ("ixinfer_w4a8_tn_v2" if arch == "ivcore11" else "ixmm_w4a8_nn")
        if mode == "w4a8"
        else "auto"
    )
    selection = select_four_bit_expert_compute(arch, mode, source="block-FP8")
    spec = get_moe_backend(selection).weight_specs[selection.layout]
    local, h, m = 2, 256, 256
    total = local * ep
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda e: (
        e - (ep - 1) * local if (ep - 1) * local <= e < ep * local else -1
    )
    loader = make_block_fp8_loader(
        spec, BlockFp8LoadSpec(local, h, m, (128, 128), tp, tp - 1)
    )
    with torch.device("cuda"):
        loader.create_weights(layer, {})
    source, refs = _checkpoint(tmp_path, total, h, m, tp, torch.bfloat16, whole)
    session = PairedLoadSession({loader: "experts"})
    session.bind()
    try:
        _route(layer, refs, session, total, whole=whole)
        with torch.device("cuda"):
            session.finish()
    finally:
        session.close()
    weights = loader.finalize()
    for group in ("w13", "w2"):
        src = source[group][(ep - 1) * local : ep * local]
        scales = source[group + "_scale"][(ep - 1) * local : ep * local].float()
        if group == "w13":
            src = torch.cat([x.chunk(tp, 1)[-1] for x in src.chunk(2, 1)], 1)
            scales = torch.cat([x.chunk(tp, 1)[-1] for x in scales.chunk(2, 1)], 1)
        else:
            src = src.chunk(tp, 2)[-1]
            scales = scales.chunk(tp, 2)[-1]
        actual = getattr(weights, group + "_weight")
        actual_scale = getattr(weights, group + "_weight_scale")
        if spec.format_id == "ixmm_fp4fp8":
            expected, es = preprocess_fp8_block_weight_for_ixmm(
                src.cuda(), scales.cuda(), (128, 128)
            )
        else:
            ref = registry.get_reference_impl("fp8_block_to_w4a8")
            expected, es = torch.empty_like(actual), torch.empty_like(actual_scale)
            fmt = (
                "ixmm_nn"
                if spec.format_id == "ixmm_w4a8_nn"
                else "ixinfer_" + spec.layout.lower()
            )
            for e in range(local):
                ref(
                    src[e],
                    scales[e],
                    expected[e],
                    es[e],
                    block_shape=(128, 128),
                    weight_format=fmt,
                )
        assert torch.equal(actual, expected)
        torch.testing.assert_close(actual_scale, es, rtol=1e-6, atol=0)
