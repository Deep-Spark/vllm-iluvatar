# SPDX-License-Identifier: Apache-2.0
"""Patch layer: redirect vLLM's bundled pynvml to Iluvatar ixML.

Corresponds to: vllm/third_party/pynvml.py

Registered via the post-import hook keyed on ``vllm.third_party.pynvml``. The
hook fires after the community module has defined its globals and before callers
use it. We only replace that module's ``CDLL`` global, not ``ctypes.CDLL``
process-wide.

Target   : vllm.third_party.pynvml.CDLL
Baseline : main_6e448d0
Why      : Iluvatar environments provide the NVML-compatible ixML library as
           ``libixml.so`` rather than NVIDIA's ``libnvidia-ml.so.1``.
How      : wrap the module-local ``CDLL`` so only that library name is
           redirected. Other dynamic library loads are untouched.
Scope    : vLLM bundled pynvml library loading on Iluvatar.
Remove   : when ixML is exposed through the standard NVML library soname.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch.import_hook import register

_PATCH_ATTR = "_ilu_ixml_redirect"
_ORIGINAL_ATTR = "_ilu_original_cdll"


def _patch_pynvml(module) -> None:
    """Patch ``vllm.third_party.pynvml.CDLL`` to load ixML on Iluvatar."""
    original_cdll = module.CDLL
    if getattr(original_cdll, _PATCH_ATTR, False):
        return

    class IluCDLL(original_cdll):
        def __init__(self, name, *args, **kwargs):
            if name == "libnvidia-ml.so.1":
                name = "libixml.so"
            super().__init__(name, *args, **kwargs)

    setattr(IluCDLL, _PATCH_ATTR, True)
    setattr(IluCDLL, _ORIGINAL_ATTR, original_cdll)
    module.CDLL = IluCDLL


register("vllm.third_party.pynvml", _patch_pynvml)
