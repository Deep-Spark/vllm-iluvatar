# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for import compatibility shims."""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import patch_import_compat


def test_missing_torch_c_compile_schema_ops_are_dispatcher_overloads():
    import torch
    from torch._ops import OpOverload

    patch_import_compat._define_missing_torch_c_compile_schema_ops()

    assert isinstance(torch.ops._C.static_scaled_fp8_quant.default, OpOverload)
    assert isinstance(torch.ops._C.dynamic_scaled_fp8_quant.default, OpOverload)
    assert isinstance(
        torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
        OpOverload,
    )
    assert isinstance(torch.ops._C.silu_and_mul.default, OpOverload)
    assert isinstance(torch.ops._C.silu_and_mul_quant.default, OpOverload)
    assert isinstance(torch.ops._C.fused_add_rms_norm.default, OpOverload)
    assert isinstance(torch.ops._C.fused_qk_norm_rope.default, OpOverload)
    assert isinstance(torch.ops._C.scaled_fp4_quant.default, OpOverload)
    assert isinstance(torch.ops._C.scaled_fp4_quant.out, OpOverload)
