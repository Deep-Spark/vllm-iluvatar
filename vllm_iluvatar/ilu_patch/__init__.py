# SPDX-License-Identifier: Apache-2.0
"""Iluvatar patch layer.

Two kinds of patches live here:

  1. Early import-compat shims + import-hook registration
     (``patch_import_compat.py`` / ``patch_custom_ops.py``).  These are applied
     at startup via ``vllm_iluvatar._load_import_compat_patch()`` (main process
     + spawned workers), BEFORE deep ``import vllm`` — they are NOT imported by
     this package ``__init__`` to keep startup cheap.

  2. Late-phase patches that need a fully initialised vLLM (e.g. parallel_state
     ready).  Import them HERE so ``register_general`` (the
     ``vllm.general_plugins`` entry point) applies them.  Currently none.

To add a late-phase patch: create ``patch_<target>.py`` whose import side
effect performs the patch, then add ``from . import patch_<target>`` below.
"""
