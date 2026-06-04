"""Interpreter-startup bootstrap for vllm-iluvatar.

This module is imported very early -- before ``import vllm`` -- via a ``.pth``
file installed into site-packages by ``scripts/install_bootstrap.py``. Its only
job is to install the mirror-tree override finder onto ``sys.meta_path`` so that
the FIRST import of any ``vllm.*`` module already resolves to the iluvatar
overlay.

Why this matters: ``vllm.distributed.parallel_state`` (and the device
communicators) are imported during normal vLLM startup, well before the
``vllm.general_plugins`` entry point fires. Installing the finder here -- at
interpreter startup -- means the community copy of those modules is NEVER
imported; only the overlay is. That keeps a SINGLE module object per path, so
the global TP/PP groups, the ``vllm::all_reduce`` custom-op registration and the
ixformer communication path all stay consistent, preserving the iluvatar comm
optimization without double-registration or split state.

Hard rules:

* MUST NOT import vllm (this runs in every Python process on this env).
* MUST NOT raise (a failure here must never break interpreter startup).
* Cheap: only inserts one ``MetaPathFinder`` that returns ``None`` for any
  non-``vllm.*`` import.
"""

from __future__ import annotations


def _install() -> None:
    try:
        from vllm_iluvatar.patch.patch_sys_modules import install_module_redirects

        install_module_redirects()
    except Exception:
        # Never break interpreter startup; the entry-point path will retry
        # (without the early-import benefit) when vLLM loads plugins.
        pass


_install()
