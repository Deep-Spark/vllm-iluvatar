# SPDX-License-Identifier: Apache-2.0
"""Independent activation quantization exports and numeric ABI coverage."""

from unittest.mock import Mock

import pytest
import torch

from vllm_iluvatar import _custom_ops as ilu_ops


# Tuple-returning activation quantization binds its Python ABI, never a vendor op.
def test_own_scaled_int8_quant_loads_python_entry(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as own

    implementation = Mock()
    loader = Mock(return_value=(implementation, "available"))
    monkeypatch.setattr(own, "load_ixaite_python_op", loader)
    monkeypatch.setattr(
        own, "load_ixaite_torch_op", lambda *_: pytest.fail("wrong quant ABI")
    )
    assert own._load_ixaite_routed_op("scaled_int8_quant")[0] is implementation
    loader.assert_called_once_with("scaled_int8_quant", "scaled_int8_quant")


# A missing export must fail before any baseline call or mutation of its input.
def test_missing_own_quantization_does_not_fall_back(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as own

    monkeypatch.setattr(own, "load_ixaite_python_op", lambda *_: (None, "missing"))
    assert own._load_ixaite_routed_op("scaled_int8_quant")[0] is None
    reject = own._unsupported("scaled_int8_quant", ilu_ops.scaled_int8_quant)
    x = torch.ones(2, 8)
    with pytest.raises(RuntimeError, match="does not support operator"):
        reject(x)
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
# The two activation widths a Qwen3.5-9B W8A8 forward produces: 4096 for every
# projection off the residual stream, 12288 for down_proj.
@pytest.mark.parametrize("hidden_size", [4096, 12288])
def test_ixaite_dynamic_matches_ixformer_on_device(hidden_size: int) -> None:
    import ixaite

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    arch = cuda_arch()
    if arch is None:
        pytest.skip("Iluvatar GPU architecture cannot be detected")
    ixaite.initialize(arch)

    torch.manual_seed(0)
    # Row counts either side of the config heuristic's row bands.
    for rows in (1, 64, 192, 1024):
        x = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16) * 4
        expected_q, expected_s, _ = ilu_ops.scaled_int8_quant(x)
        actual_q, actual_s, azp = ixaite.scaled_int8_quant(x)

        assert azp is None
        torch.testing.assert_close(actual_s, expected_s, rtol=1e-5, atol=1e-6)
        # Both round to nearest even, but ties can land either side.
        torch.testing.assert_close(actual_q, expected_q, atol=1, rtol=0.0)
