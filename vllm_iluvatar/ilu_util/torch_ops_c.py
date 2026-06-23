# SPDX-License-Identifier: Apache-2.0
"""
torch_ops_c — load the plugin's native ``torch.ops._C`` operators.

The Iluvatar plugin ships a small native extension (``vllm_iluvatar._C``) that
registers the vLLM-community-signature operators used by unmodified community
code:

  * ``rms_norm``                     (vllm/kernels/vllm_c.py)
  * ``weak_ref_tensor``              (vllm/utils/torch_utils.py)
  * ``get_cuda_view_from_cpu_tensor``(vllm/utils/torch_utils.py)

``import vllm_iluvatar._C`` triggers the static ``TORCH_LIBRARY`` registrations,
so this module just imports it early — before community code first calls these
ops.

No schema validation is done here on purpose: any such check would compare our
registration against our own copy of the signature (tautological) and could not
detect an upstream signature change anyway. If a signature ever diverges, the
torch dispatcher raises at the first call site (and the e2e run fails), which is
the real, non-silent backstop.

No vllm imports; safe to call at interpreter-startup time.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

NATIVE_MODULE = "vllm_iluvatar._C"


def load_native_ops(*, strict: bool = False) -> bool:
    """Import the native module so ``torch.ops._C.*`` is registered.

    Returns True on success. In non-strict mode (startup) an import failure is
    logged and False returned so a missing build never breaks interpreter / vLLM
    startup; in strict mode the underlying ImportError propagates.
    """
    try:
        importlib.import_module(NATIVE_MODULE)
    except Exception as exc:  # pragma: no cover - depends on build/toolchain
        if strict:
            raise
        logger.debug(
            "vllm_iluvatar: native ops module %r is not importable: %s",
            NATIVE_MODULE,
            exc,
        )
        return False
    return True
