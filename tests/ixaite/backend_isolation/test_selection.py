# SPDX-License-Identifier: Apache-2.0
"""Launch selection is checked in fresh interpreters, as in spawned workers."""

import os
import subprocess
import sys

import pytest


# Each subprocess gets an independent module cache and process-wide mode.
@pytest.mark.parametrize("mode", [None, "baseline", "ixaite"])
def test_process_selection(mode):
    env = os.environ.copy()
    for name in (
        "VLLM_ILU_KERNEL_PROVIDER",
        "VLLM_ILU_IXAITE_OP_ALLOWLIST",
    ):
        env.pop(name, None)
    if mode is not None:
        env["VLLM_ILU_KERNEL_PROVIDER"] = mode
    expected = mode or "baseline"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from vllm_iluvatar.backends import "
                "provider_mode,worker_class_path,communicator_class_path; "
                f"assert provider_mode() == {expected!r}; "
                f"assert ('backends.ixaite' in worker_class_path()) == {expected == 'ixaite'}; "
                f"assert ('backends.ixaite' in communicator_class_path()) == {expected == 'ixaite'}; "
                "assert provider_mode() == 'ixaite' or not any(n == 'ixaite' "
                "or n.startswith('ixaite.') or n.startswith('vllm_iluvatar.backends.ixaite') "
                "for n in sys.modules)"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


# Reject invalid modes and mixed backends before importing an implementation.
@pytest.mark.parametrize(
    "name,value",
    [
        ("VLLM_ILU_KERNEL_PROVIDER", "auto"),
        ("VLLM_ILU_KERNEL_PROVIDER", "both"),
        ("VLLM_ILU_IXAITE_OP_ALLOWLIST", "linear"),
    ],
)
def test_invalid_launch_controls(monkeypatch, name, value):
    from vllm_iluvatar.backends import provider_mode

    monkeypatch.delenv("VLLM_ILU_KERNEL_PROVIDER", raising=False)
    monkeypatch.delenv("VLLM_ILU_IXAITE_OP_ALLOWLIST", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        provider_mode()
