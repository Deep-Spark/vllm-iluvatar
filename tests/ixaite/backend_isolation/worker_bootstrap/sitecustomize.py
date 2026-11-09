# SPDX-License-Identifier: Apache-2.0
"""Test process bootstrap, inherited through PYTHONPATH by spawned workers."""

import atexit
import os
import sys
import traceback
from pathlib import Path

if os.environ.get("VLLM_ILU_KERNEL_PROVIDER") == "ixaite":
    # A separate test entry keeps monitoring out of the production import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ixformer_guard import forbid_ixformer_calls

    # Worker teardown must report attempts that a warmup error handler swallowed.
    def _finish_guard():
        try:
            _guard.__exit__(None, None, None)
        except BaseException:
            traceback.print_exc()
            os._exit(1)

    try:
        _guard = forbid_ixformer_calls()
        _guard.__enter__()
        atexit.register(_finish_guard)
    except Exception:
        # Python ignores sitecustomize exceptions; an unaudited run must fail.
        traceback.print_exc()
        os._exit(1)
