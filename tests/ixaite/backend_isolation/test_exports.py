# SPDX-License-Identifier: Apache-2.0
"""Copied custom-op references must already point to the selected backend."""

import os
import subprocess
import sys

import pytest


# Run the actual early hook before a consumer copies a function from the module.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_early_hook_publishes_selected_exports(mode):
    env = os.environ.copy()
    env["VLLM_ILU_KERNEL_PROVIDER"] = mode
    env.pop("VLLM_ILU_IXAITE_OP_ALLOWLIST", None)
    script = """
import sys
from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
with forbid_ixformer_calls():
    import vllm_iluvatar
    vllm_iluvatar.register_platform()
    from vllm import _custom_ops as published
    from vllm._custom_ops import silu_and_mul as copied
    from vllm_iluvatar.backends.ops import implementation
    assert copied is implementation.silu_and_mul
    if sys.argv[1] == 'ixaite':
        assert published.linear is implementation.linear
    else:
        assert not hasattr(implementation, 'linear')
    assert published.rms_norm is implementation.rms_norm
    assert published.scaled_int8_quant is implementation.scaled_int8_quant
    if sys.argv[1] == 'baseline':
        assert not any(n == 'ixaite' or n.startswith('ixaite.')
                       or n.startswith('vllm_iluvatar.backends.ixaite')
                       for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
