# SPDX-License-Identifier: Apache-2.0
"""W4A8 (compressed-tensors) MiniMax-M3 graph contracts."""

from __future__ import annotations

import ast
from pathlib import Path

_MODEL_PY = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar"
    / "models"
    / "minimax_m3"
    / "w4a8"
    / "model.py"
)


def test_routed_experts_receive_the_config_swiglu_parameters():
    tree = ast.parse(_MODEL_PY.read_text(), filename=str(_MODEL_PY))
    moe_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "IluMiniMaxM3MoE"
    )
    factory = next(
        node
        for node in ast.walk(moe_cls)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedMoEFactory"
    )
    passed = {
        kw.arg: ast.unparse(kw.value) for kw in factory.keywords if kw.arg is not None
    }
    for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta"):
        assert passed.get(name) == f"config.{name}", (
            f"IluMiniMaxM3MoE must forward {name} to the routed experts"
        )
    assert passed.get("activation") == "'swigluoai_uninterleave'"


def test_dense_mlp_uses_community_act_then_linear():
    tree = ast.parse(_MODEL_PY.read_text(), filename=str(_MODEL_PY))
    mlp_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "IluMiniMaxM3MLP"
    )
    init_fn = next(
        node
        for node in mlp_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    forward_fn = next(
        node
        for node in mlp_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assigned = {
        ast.unparse(stmt.targets[0]): ast.unparse(stmt.value)
        for stmt in init_fn.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
    }
    assert assigned.get("self.act_fn", "").startswith("SiluAndMulWithClamp(")
    source = ast.unparse(forward_fn)
    assert "self.act_fn(gate_up)" in source
    assert "self.down_proj(x)" in source
    assert "enable_fuse_quant" not in ast.unparse(mlp_cls)
    assert "swigluoai_uninterleave_dynamic_scaled_fp8" not in ast.unparse(mlp_cls)
