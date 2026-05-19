"""Install a meta-path finder that redirects 6 stock vLLM module paths
to the support_dsv4 versions stored under :mod:`vllm_iluvatar._replacements`.

We use a :class:`importlib.abc.MetaPathFinder` rather than mutating
``sys.modules`` directly so that the redirect works correctly even when
one of the redirected modules is imported lazily later in the process
(e.g. ``ilu_compressed_tensors_moe_w4a8_int8`` is only imported from
``IluDeepseekV4W4A8Config.get_quant_method`` at model-build time).

If one of the redirected modules has already been imported by the time
:func:`install_module_redirects` runs, we additionally drop it from
``sys.modules`` so the next ``import`` statement goes through our
finder. This relies on no live references being held to the old module
attributes -- which is true for the affected modules because they are
only imported from a handful of well-defined call sites that all happen
AFTER plugin loading.

The 6 redirected paths:

* ``vllm.v1.attention.backends.mla.sparse_swa``
* ``vllm.v1.attention.backends.mla.ilu_flashmla_sparse``
* ``vllm.v1.attention.backends.mla.ilu_indexer``
* ``vllm.model_executor.layers.fused_moe.iluvatar_runner.moe_runner``
* ``vllm.model_executor.layers.fused_moe.iluvatar_runner.moe_runner_interface``
* ``vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.iluvatar.ilu_compressed_tensors_moe_w4a8_int8``
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

# Map: vllm module path -> filename inside ``vllm_iluvatar/_replacements/``.
_REDIRECTS: dict[str, str] = {
    "vllm.v1.attention.backends.mla.sparse_swa": "sparse_swa.py",
    "vllm.v1.attention.backends.mla.ilu_flashmla_sparse": "ilu_flashmla_sparse.py",
    "vllm.v1.attention.backends.mla.ilu_indexer": "ilu_indexer.py",
    "vllm.model_executor.layers.fused_moe.iluvatar_runner.moe_runner": "moe_runner.py",
    "vllm.model_executor.layers.fused_moe.iluvatar_runner.moe_runner_interface": (
        "moe_runner_interface.py"
    ),
    (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.iluvatar.ilu_compressed_tensors_moe_w4a8_int8"
    ): "ilu_compressed_tensors_moe_w4a8_int8.py",
}


def _replacements_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "_replacements"


class _IluRedirectFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that intercepts the 6 paths in ``_REDIRECTS``.

    For each intercepted ``fullname`` we build a ``ModuleSpec`` whose
    ``origin`` points at our replacement file, so the standard
    :class:`SourceFileLoader` does the actual loading. The replacement
    file keeps its original ``vllm....`` qualified name (its ``__name__``
    attribute), which means any ``from vllm.... import foo`` inside it
    will resolve through the normal import machinery and we don't have
    to rewrite intra-module imports.
    """

    _REPL_DIR = _replacements_dir()

    def find_spec(self, fullname, path, target=None):  # noqa: D401
        if fullname not in _REDIRECTS:
            return None
        filename = _REDIRECTS[fullname]
        filepath = self._REPL_DIR / filename
        if not filepath.is_file():
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            os.fspath(filepath),
        )


_INSTALLED = False


def install_module_redirects() -> None:
    """Idempotently install the redirect finder.

    Also drops any already-imported copies of the redirected modules so
    the next ``import`` statement goes through our finder.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    finder = _IluRedirectFinder()
    # Put the finder at the FRONT so it wins over the default file finders.
    sys.meta_path.insert(0, finder)

    # If any of the redirected modules are already imported, evict them so
    # the next import goes through our finder. Stale references already
    # bound by upstream callers would still see the old class objects --
    # we rely on the fact that no upstream code imports these modules
    # before the plugin loads (FusedMoE / DeepseekV4 attention paths are
    # imported lazily from model build, well after plugin init).
    for modname in _REDIRECTS:
        if modname in sys.modules:
            logger.warning(
                "vllm-iluvatar: evicting already-imported %s; live "
                "references to its classes may need to be re-imported",
                modname,
            )
            del sys.modules[modname]

    logger.info(
        "vllm-iluvatar: installed sys.modules redirects for %d modules",
        len(_REDIRECTS),
    )
