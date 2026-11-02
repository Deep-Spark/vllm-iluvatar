# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "vllm_iluvatar/models/glm_moe_dsa/model.py",
        "vllm_iluvatar/models/glm_moe_dsa/attention.py",
    ],
)
def test_glm_norm_quant_uses_exported_ixformer_api(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    ixf_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ixf_ops"
    }

    assert "residual_rms_norm_with_quant" not in ixf_calls
    assert "residual_rmsnorm_with_quant" in ixf_calls
