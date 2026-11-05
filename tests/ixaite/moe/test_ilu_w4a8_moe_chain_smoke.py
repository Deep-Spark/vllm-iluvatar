# SPDX-License-Identifier: Apache-2.0
"""W4A8 MoE chain smoke under VLLM_ILU_KERNEL_PROVIDER.

Prefill + decode through ``IluCompressedTensorsW4A8Int8MoEMethod`` at the
Qwen3.6 Q-M2 envelope and the DeepSeek-V4 TP=8, EP-off envelope. Compares
ixaite vs ixformer: expert ids must match exactly and both complete MoE paths
must produce finite outputs. Also asserts the MoE ops and grouped
GEMM/GEMV route to ixaite when that provider is forced.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_THIS = Path(__file__).resolve()
_PLUGIN_ROOT = _THIS.parents[3]
_EXPECTED_IXAITE_OPS = (
    "moe_fused_topk_bias",
    "moe_compute_token_index",
    "moe_compute_token_index_for_decode",
    "moe_expand_input_dynamic_scaled_int8",
    "moe_output_reduce_sum",
    "moe_w4a8_group_gemm",
    "moe_w4a8_group_gemv",
)

_PREFILL_TOKENS = 128
_DECODE_TOKENS = 8


@dataclass(frozen=True)
class _Shape:
    experts: int
    topk: int
    hidden: int
    intermediate: int


_SHAPES = {
    "qwen36_qm2": _Shape(experts=256, topk=8, hidden=2048, intermediate=512),
    # DeepSeek-V4 TP=8 with expert parallelism disabled keeps all 256 experts
    # resident and partitions the global expert intermediate size 2048 by TP.
    "deepseek_v4_tp8": _Shape(
        experts=256,
        topk=6,
        hidden=4096,
        intermediate=256,
    ),
}


def _build_layer(device: torch.device, shape: _Shape):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
        W4A8GroupGemmInfo,
    )

    generator = torch.Generator().manual_seed(2954)
    gemm1 = W4A8GroupGemmInfo("TN", k=shape.hidden, n=2 * shape.intermediate)
    gemm2 = W4A8GroupGemmInfo("TN", k=shape.intermediate, n=shape.hidden)

    # Every byte encodes two valid signed INT4 values, so generating the
    # version-2 TN storage directly avoids materializing multi-GB logical
    # weights for the DeepSeek-V4 shape.
    w13_weight = torch.randint(
        -128,
        128,
        (shape.experts, *gemm1.get_weight_shape()),
        dtype=torch.int8,
        generator=generator,
    )
    w2_weight = torch.randint(
        -128,
        128,
        (shape.experts, *gemm2.get_weight_shape()),
        dtype=torch.int8,
        generator=generator,
    )
    layer = SimpleNamespace(
        expert_map=None,
        global_num_experts=shape.experts,
        local_num_experts=shape.experts,
        top_k=shape.topk,
        activation=SimpleNamespace(value="silu"),
        swiglu_limit=None,
        _ilu_output_scale=1.0,
        w13_weight=w13_weight.to(device),
        w2_weight=w2_weight.to(device),
        w13_weight_scale=(
            torch.rand(
                shape.experts,
                2 * shape.intermediate,
                dtype=torch.float32,
                generator=generator,
            ).to(device)
            * 0.02
        ),
        w2_weight_scale=(
            torch.rand(
                shape.experts,
                shape.hidden,
                dtype=torch.float32,
                generator=generator,
            ).to(device)
            * 0.02
        ),
        w13_i8_weight_scale=None,
        w2_i8_weight_scale=None,
        w13_i8_weight_zero=None,
        w2_i8_weight_zero=None,
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
    )
    return layer, gemm1, gemm2


# Separate processes bind their own callables before constructing test objects.
def _selected_module():
    if os.environ.get("VLLM_ILU_KERNEL_PROVIDER") == "ixaite":
        from vllm_iluvatar.backends.ixaite.moe import w4a8

        w4a8.configure_moe_providers()
        return w4a8
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_compressed_tensors_moe_w4a8_int8,
    )

    return ilu_compressed_tensors_moe_w4a8_int8


def _make_method(gemm1, gemm2):
    w4a8_moe = _selected_module()
    IluCompressedTensorsW4A8Int8MoEMethod = (
        w4a8_moe.IluCompressedTensorsW4A8Int8MoEMethod
    )

    method = IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        IluCompressedTensorsW4A8Int8MoEMethod
    )
    method.group_size = -1
    method.version = 2
    method.gemm1_info = gemm1
    method.gemm2_info = gemm2
    # __new__ bypasses __init__; mirror the workspace slots it would set.
    method._decode_ws = {}
    method._prefill_index_ws = None
    method._hist_d2h_stream = None
    method._hist_d2h_event = None
    return method


# Inspect bound entry points for diagnostics; the guard verifies actual execution.
def _route_ops():
    if os.environ.get("VLLM_ILU_KERNEL_PROVIDER") == "ixaite":
        from vllm_iluvatar.backends.ixaite import ops
        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

        module = _selected_module()
        return {
            "moe_fused_topk_bias": load_ixaite_python_op(
                "moe_fused_topk_bias", "moe_fused_topk_bias"
            )[0],
            **{name: getattr(module, "_" + name) for name in _EXPECTED_IXAITE_OPS[1:]},
        }
    import ixformer.inference.functions as ops

    return {name: getattr(ops, name) for name in _EXPECTED_IXAITE_OPS}


def _tokens_per_expert(
    sorted_ids: torch.Tensor, sizes: torch.Tensor
) -> list[list[int]]:
    grouped: list[list[int]] = []
    cursor = 0
    for count in sizes.tolist():
        grouped.append(sorted(sorted_ids[cursor : cursor + int(count)].tolist()))
        cursor += int(count)
    return grouped


def _impl_origin(fn) -> str:
    module = getattr(fn, "__module__", "") or ""
    if module.startswith("ixaite."):
        return "ixaite"
    if "ixformer" in module:
        return "ixformer"
    return module or type(fn).__name__


def _worker(dump_path: str) -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")
    provider = os.environ.get("VLLM_ILU_KERNEL_PROVIDER", "baseline")
    case = os.environ["_W4A8_MOE_CASE"]
    shape = _SHAPES[case]
    device = torch.device("cuda")
    layer, gemm1, gemm2 = _build_layer(device, shape)
    method = _make_method(gemm1, gemm2)

    torch.manual_seed(7)
    gating = torch.randn(
        _PREFILL_TOKENS,
        shape.experts,
        device=device,
        dtype=torch.bfloat16,
    )
    bias = torch.randn(shape.experts, device=device, dtype=torch.bfloat16)
    topk_w = torch.empty(
        _PREFILL_TOKENS, shape.topk, device=device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        _PREFILL_TOKENS, shape.topk, device=device, dtype=torch.int32
    )
    unused = torch.empty_like(topk_ids)

    from vllm_iluvatar.backends.ops import implementation

    ilu_ops = implementation
    w4a8_moe = _selected_module()
    import ixformer.inference.functions as vendor_ops

    token_index = (
        w4a8_moe._moe_compute_token_index
        if provider == "ixaite"
        else vendor_ops.moe_compute_token_index
    )
    expand = (
        w4a8_moe._moe_expand_input_dynamic_scaled_int8
        if provider == "ixaite"
        else vendor_ops.moe_expand_input_dynamic_scaled_int8
    )
    reduce = (
        w4a8_moe._moe_output_reduce_sum
        if provider == "ixaite"
        else vendor_ops.moe_output_reduce_sum
    )

    ilu_ops.topk_softmax(
        topk_w,
        topk_ids,
        unused,
        gating,
        renormalize=True,
        e_score_correction_bias=bias,
    )

    x = torch.randn(
        _PREFILL_TOKENS,
        shape.hidden,
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-1, 1)
    shared = torch.randn(
        _PREFILL_TOKENS,
        shape.hidden,
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-1, 1)
    prefill_out = method._apply_general(
        layer=layer,
        x=x,
        topk_weights=topk_w,
        topk_ids=topk_ids,
        shared_experts_input=shared,
    )
    (
        src_to_dst,
        sorted_token_ids,
        expert_sizes_gpu,
        expert_sizes_cpu,
    ) = token_index(
        topk_ids=topk_ids,
        num_experts=shape.experts,
    )
    expand_tokens = _PREFILL_TOKENS * shape.topk
    expand_out, expand_scale = expand(
        hidden_states=x,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=shape.topk,
        src_to_dst=src_to_dst,
        topk_ids=None,
        smooth_scales=None,
        output_format=0,
        check_valid=False,
    )
    torch.manual_seed(11)
    fake_expert_out = torch.randn(
        _PREFILL_TOKENS,
        shape.topk,
        shape.hidden,
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-1, 1)
    reduce_out = reduce(
        input=fake_expert_out,
        topk_weight=topk_w,
        extra_residual=shared,
        scaling_factor=1.0,
    )

    decode_x = x[:_DECODE_TOKENS].contiguous()
    decode_w = topk_w[:_DECODE_TOKENS].contiguous()
    decode_ids = topk_ids[:_DECODE_TOKENS].contiguous()
    decode_shared = shared[:_DECODE_TOKENS].contiguous()
    decode_out = method._apply_decode(
        layer=layer,
        x=decode_x,
        topk_weights=decode_w,
        topk_ids=decode_ids,
        shared_experts_input=decode_shared,
    )

    origins = {name: _impl_origin(fn) for name, fn in _route_ops().items()}
    payload = {
        "provider": provider,
        "case": case,
        "origins": origins,
        "topk_ids": topk_ids.cpu(),
        "topk_w": topk_w.cpu(),
        "prefill_out": prefill_out.cpu(),
        "decode_out": decode_out.cpu(),
        "src_to_dst": src_to_dst.cpu(),
        "sorted_token_ids": sorted_token_ids.cpu(),
        "expert_sizes": expert_sizes_gpu.cpu(),
        "expand_out": expand_out.cpu(),
        "expand_scale": expand_scale.cpu(),
        "reduce_out": reduce_out.cpu(),
    }
    Path(dump_path).write_bytes(pickle.dumps(payload))
    print("ORIGINS", origins, flush=True)


def _run_provider(
    provider: str, case: str, dump_path: Path
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["VLLM_ILU_KERNEL_PROVIDER"] = provider
    env["_SWCOMP2954_SMOKE_WORKER"] = "1"
    env["_W4A8_MOE_CASE"] = case
    env["FLYDSL_COMPILE_BACKEND"] = "iluvatar"
    env["FLYDSL_RUNTIME_KIND"] = "iluvatar"
    env["ARCH"] = env.get("ARCH", "ivcore11")
    plugin_src = str(_PLUGIN_ROOT)
    ixaite_src = str(_PLUGIN_ROOT.parent / "ixaite" / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_PLUGIN_ROOT / "tests/ixaite/backend_isolation/worker_bootstrap"),
            plugin_src,
            ixaite_src,
            env.get("PYTHONPATH", ""),
        ]
    )
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(_THIS), "--dump", str(dump_path)],
        cwd=str(_PLUGIN_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Iluvatar required")
@pytest.mark.parametrize("case", sorted(_SHAPES))
def test_w4a8_moe_ixaite_chain_matches_baseline(tmp_path, case):
    base_dump = tmp_path / f"{case}_baseline.pkl"
    ix_dump = tmp_path / f"{case}_ixaite.pkl"
    base = _run_provider("baseline", case, base_dump)
    ix = _run_provider("ixaite", case, ix_dump)
    if base.returncode != 0:
        raise AssertionError(
            f"baseline worker failed\nstdout:\n{base.stdout}\nstderr:\n{base.stderr}"
        )
    if ix.returncode != 0:
        raise AssertionError(
            f"ixaite worker failed\nstdout:\n{ix.stdout}\nstderr:\n{ix.stderr}"
        )

    base_data = pickle.loads(base_dump.read_bytes())
    ix_data = pickle.loads(ix_dump.read_bytes())
    for op in _EXPECTED_IXAITE_OPS:
        assert ix_data["origins"][op] == "ixaite", (
            f"{op} origin={ix_data['origins'][op]}"
        )
        assert base_data["origins"][op] == "ixformer", (
            f"baseline {op} origin={base_data['origins'][op]}"
        )

    assert torch.equal(base_data["topk_ids"], ix_data["topk_ids"]), (
        "expert ids must match baseline exactly"
    )
    assert torch.equal(base_data["expert_sizes"], ix_data["expert_sizes"]), (
        "expert histogram must match baseline exactly"
    )
    assert _tokens_per_expert(
        base_data["sorted_token_ids"], base_data["expert_sizes"]
    ) == _tokens_per_expert(ix_data["sorted_token_ids"], ix_data["expert_sizes"]), (
        "per-expert token sets must match (order within an expert may differ)"
    )

    torch.testing.assert_close(
        ix_data["topk_w"],
        base_data["topk_w"],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        ix_data["reduce_out"],
        base_data["reduce_out"],
        rtol=1e-2,
        atol=1e-1,
    )
    assert torch.isfinite(ix_data["prefill_out"].float()).all(), (
        "ixaite prefill produced non-finite values"
    )
    assert torch.isfinite(base_data["prefill_out"].float()).all()
    assert torch.isfinite(ix_data["decode_out"].float()).all(), (
        "ixaite decode produced non-finite values "
        "(check grouped-MR block-table cache keys include bm/grid_n)"
    )
    assert torch.isfinite(base_data["decode_out"].float()).all()


if __name__ == "__main__":
    if os.environ.get("_SWCOMP2954_SMOKE_WORKER") == "1":
        dump = sys.argv[sys.argv.index("--dump") + 1]
        _worker(dump)
    else:
        pytest.main([str(_THIS), "-s", "--tb=short"])
