# SPDX-License-Identifier: Apache-2.0
"""Check baseline Hash/sqrtsoftplus routing and the strict ixaite capability gap."""

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

_PREFILL_TOKENS = 128
_DECODE_TOKENS = 8
_VOCAB = 64
_E = 256
_TOPK = 6
_HIDDEN = 4096
_INTERMEDIATE = 256
_RENORM = True
_ROUTED_SCALE = 1.0
# GEMM2 uses N=4096 -> grid_n=32. Each active expert costs grid_n merge
# blocks even with M=1, so 256 sparse experts overflow _MAX_MERGE_BLOCKS=4096.
_NORMAL_POOL = 64


@dataclass(frozen=True)
class _Case:
    route: str
    dist: str


def _build_layer(device: torch.device):
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
        W4A8GroupGemmInfo,
    )

    generator = torch.Generator().manual_seed(2970)
    gemm1 = W4A8GroupGemmInfo("TN", k=_HIDDEN, n=2 * _INTERMEDIATE)
    gemm2 = W4A8GroupGemmInfo("TN", k=_INTERMEDIATE, n=_HIDDEN)
    w13_weight = torch.randint(
        -128,
        128,
        (_E, *gemm1.get_weight_shape()),
        dtype=torch.int8,
        generator=generator,
    )
    w2_weight = torch.randint(
        -128,
        128,
        (_E, *gemm2.get_weight_shape()),
        dtype=torch.int8,
        generator=generator,
    )
    layer = SimpleNamespace(
        expert_map=None,
        global_num_experts=_E,
        local_num_experts=_E,
        top_k=_TOPK,
        activation=SimpleNamespace(value="silu"),
        swiglu_limit=None,
        _ilu_output_scale=1.0,
        w13_weight=w13_weight.to(device),
        w2_weight=w2_weight.to(device),
        w13_weight_scale=(
            torch.rand(
                _E, 2 * _INTERMEDIATE, dtype=torch.float32, generator=generator
            ).to(device)
            * 0.02
        ),
        w2_weight_scale=(
            torch.rand(_E, _HIDDEN, dtype=torch.float32, generator=generator).to(device)
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


def _make_method(gemm1, gemm2):
    from tests.ixaite.moe.test_ilu_w4a8_moe_chain_smoke import _selected_module

    IluCompressedTensorsW4A8Int8MoEMethod = (
        _selected_module().IluCompressedTensorsW4A8Int8MoEMethod
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


def _unique_row(generator: torch.Generator, n: int, k: int) -> torch.Tensor:
    perm = torch.randperm(n, generator=generator)
    return perm[:k].to(torch.int32)


def _tokens_per_expert(
    sorted_ids: torch.Tensor, sizes: torch.Tensor
) -> list[list[int]]:
    grouped: list[list[int]] = []
    cursor = 0
    for count in sizes.tolist():
        grouped.append(sorted(sorted_ids[cursor : cursor + int(count)].tolist()))
        cursor += int(count)
    return grouped


def _hash_table_and_ids(
    dist: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    table = torch.empty(_VOCAB, _TOPK, dtype=torch.int32)
    if dist == "empty":
        pool = torch.tensor([0, 10, 20, 30, 40, 50], dtype=torch.int32)
        for row in range(_VOCAB):
            table[row] = pool
        input_ids = torch.randint(0, _VOCAB, (_PREFILL_TOKENS,), generator=generator)
    elif dist == "hot":
        hot = torch.arange(_TOPK, dtype=torch.int32)
        for row in range(_VOCAB):
            if row < 8:
                table[row] = hot
            else:
                table[row] = _unique_row(generator, _NORMAL_POOL, _TOPK)
        # Most tokens hash through the hot rows.
        hot_ids = torch.randint(0, 8, (_PREFILL_TOKENS - 16,), generator=generator)
        tail = torch.randint(8, _VOCAB, (16,), generator=generator)
        input_ids = torch.cat([hot_ids, tail], dim=0)
    else:
        for row in range(_VOCAB):
            table[row] = _unique_row(generator, _NORMAL_POOL, _TOPK)
        input_ids = torch.randint(0, _VOCAB, (_PREFILL_TOKENS,), generator=generator)
    return table, input_ids.to(torch.int32)


def _routed_gating_and_bias(
    dist: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    gating = torch.randn(_PREFILL_TOKENS, _E, dtype=torch.float32, generator=generator)
    bias = torch.randn(_E, dtype=torch.float32, generator=generator)
    if dist == "normal":
        bias[_NORMAL_POOL:] = -1.0e4
    if dist == "empty":
        bias[:] = -1.0e4
        bias[:_TOPK] = 10.0
        gating = gating * 0.01
    elif dist == "hot":
        bias[:] = -1.0e3
        bias[:3] = 50.0
        bias[3:_TOPK] = 5.0
        gating[:, :3] = gating[:, :3] + 20
    return gating, bias


def _worker(dump_path: str) -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")
    provider = os.environ.get("VLLM_ILU_KERNEL_PROVIDER", "baseline")
    route = os.environ["_SWCOMP2970_ROUTE"]
    dist = os.environ["_SWCOMP2970_DIST"]
    device = torch.device("cuda")
    layer, gemm1, gemm2 = _build_layer(device)
    method = _make_method(gemm1, gemm2)

    generator = torch.Generator().manual_seed(2970)
    table = None
    input_ids = None
    python_ids = None
    if route == "hash":
        table, input_ids = _hash_table_and_ids(dist, generator)
        gating = torch.randn(
            _PREFILL_TOKENS, _E, dtype=torch.float32, generator=generator
        )
        table = table.to(device)
        input_ids = input_ids.to(device)
        python_ids = table[input_ids.long()].cpu()
        gating = gating.to(device)
        bias = None
    else:
        gating, bias = _routed_gating_and_bias(dist, generator)
        gating = gating.to(device)
        bias = bias.to(device)

    topk_w = torch.empty(_PREFILL_TOKENS, _TOPK, device=device, dtype=torch.float32)
    topk_ids = torch.empty(_PREFILL_TOKENS, _TOPK, device=device, dtype=torch.int32)
    unused = torch.empty_like(topk_ids)

    import ixformer.inference.functions as vendor_ops

    from vllm_iluvatar.backends.ops import implementation as ilu_ops

    ilu_ops.topk_hash_softplus_sqrt(
        topk_w,
        topk_ids,
        unused,
        gating,
        renormalize=_RENORM,
        routed_scaling_factor=_ROUTED_SCALE,
        e_score_correction_bias=bias,
        input_tokens=input_ids,
        hash_indices_table=table,
    )

    x = torch.randn(
        _PREFILL_TOKENS, _HIDDEN, device=device, dtype=torch.bfloat16
    ).clamp(-1, 1)
    prefill_out = method._apply_general(
        layer=layer,
        x=x,
        topk_weights=topk_w,
        topk_ids=topk_ids,
        shared_experts_input=None,
    )
    (
        src_to_dst,
        sorted_token_ids,
        expert_sizes_gpu,
        expert_sizes_cpu,
    ) = vendor_ops.moe_compute_token_index(
        topk_ids=topk_ids,
        num_experts=_E,
    )

    fake_gen = torch.Generator().manual_seed(11)
    fake_expert_out = (
        torch.randn(
            _PREFILL_TOKENS,
            _TOPK,
            _HIDDEN,
            dtype=torch.bfloat16,
            generator=fake_gen,
        )
        .clamp(-1, 1)
        .to(device)
    )
    reduce_out = vendor_ops.moe_output_reduce_sum(
        input=fake_expert_out,
        topk_weight=topk_w,
        extra_residual=None,
        scaling_factor=1.0,
    )

    decode_out = method._apply_decode(
        layer=layer,
        x=x[:_DECODE_TOKENS].contiguous(),
        topk_weights=topk_w[:_DECODE_TOKENS].contiguous(),
        topk_ids=topk_ids[:_DECODE_TOKENS].contiguous(),
        shared_experts_input=None,
    )

    # The fp32 score formula is independent of the native router implementation.
    scores = torch.nn.functional.softplus(gating.float()).sqrt()
    oracle_weights = scores.gather(1, topk_ids.long())
    if _RENORM:
        oracle_weights = oracle_weights / oracle_weights.sum(dim=-1, keepdim=True)
    oracle_weights = oracle_weights * _ROUTED_SCALE
    hist = expert_sizes_gpu.cpu()
    payload = {
        "provider": provider,
        "route": route,
        "dist": dist,
        "topk_ids": topk_ids.cpu(),
        "topk_w": topk_w.cpu(),
        "oracle_weights": oracle_weights.cpu(),
        "python_ids": python_ids,
        "expert_sizes": hist,
        "active_experts": int((hist > 0).sum().item()),
        "max_expert_tokens": int(hist.max().item()),
        "prefill_out": prefill_out.cpu(),
        "decode_out": decode_out.cpu(),
        "reduce_out": reduce_out.cpu(),
        "sorted_token_ids": sorted_token_ids.cpu(),
    }
    Path(dump_path).write_bytes(pickle.dumps(payload))
    print(
        f"2970 route={route} dist={dist} active={payload['active_experts']} "
        f"max_tokens={payload['max_expert_tokens']}",
        flush=True,
    )


def _run_provider(
    provider: str, case: _Case, dump_path: Path
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["VLLM_ILU_KERNEL_PROVIDER"] = provider
    env["_SWCOMP2970_WORKER"] = "1"
    env["_SWCOMP2970_ROUTE"] = case.route
    env["_SWCOMP2970_DIST"] = case.dist
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


_CASES = [
    _Case("hash", "normal"),
    _Case("hash", "empty"),
    _Case("hash", "hot"),
    _Case("routed", "normal"),
    _Case("routed", "empty"),
    _Case("routed", "hot"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Iluvatar required")
@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c.route}_{c.dist}")
def test_dsv4_hash_and_routed_w4a8_chain(tmp_path, case: _Case):
    base_dump = tmp_path / f"{case.route}_{case.dist}_baseline.pkl"
    ix_dump = tmp_path / f"{case.route}_{case.dist}_ixaite.pkl"
    base = _run_provider("baseline", case, base_dump)
    ix = _run_provider("ixaite", case, ix_dump)
    if base.returncode != 0:
        raise AssertionError(
            f"baseline worker failed\nstdout:\n{base.stdout}\nstderr:\n{base.stderr}"
        )
    # The current ixaite catalog has no softplus/sqrt or hash router.
    assert ix.returncode != 0
    assert "does not support operator topk_hash_softplus_sqrt" in ix.stderr
    assert not ix_dump.exists()

    base_data = pickle.loads(base_dump.read_bytes())
    if case.dist == "empty":
        assert base_data["active_experts"] <= _TOPK + 2
    if case.dist == "hot":
        assert base_data["max_expert_tokens"] >= _PREFILL_TOKENS // 2
    if case.route == "hash":
        assert torch.equal(base_data["topk_ids"], base_data["python_ids"])
    torch.testing.assert_close(
        base_data["topk_w"], base_data["oracle_weights"], rtol=1e-2, atol=1e-2
    )
    assert torch.equal(
        base_data["expert_sizes"],
        torch.bincount(base_data["topk_ids"].flatten().long(), minlength=_E),
    )
    assert torch.isfinite(base_data["prefill_out"].float()).all()
    assert torch.isfinite(base_data["decode_out"].float()).all()


if __name__ == "__main__":
    if os.environ.get("_SWCOMP2970_WORKER") == "1":
        dump = sys.argv[sys.argv.index("--dump") + 1]
        _worker(dump)
    else:
        pytest.main([str(_THIS), "-s", "--tb=short"])
