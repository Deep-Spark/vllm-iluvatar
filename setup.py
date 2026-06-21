# SPDX-License-Identifier: Apache-2.0
"""Build entry point for the vllm_iluvatar plugin (pure-Python, no native ext).

Project metadata lives in pyproject.toml (PEP 621); the version is loaded from
vllm_iluvatar/version.py. The only build-time customisation here is installing
an env-gated startup-patch ``.pth`` so spawned vLLM worker subprocesses apply
the early corex/torch compat shims before unpickling engine modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:  # pragma: no cover - older setuptools
    editable_wheel = None

ROOT_DIR = Path(__file__).resolve().parent
STARTUP_PATCH_ENV = "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH"
PTH_NAME = "vllm_iluvatar_startup_patch.pth"
# Inert unless the env flag is "1" (set by register_platform); only then does
# Python startup call into the plugin to apply the early compat shims.
PTH_CONTENT = (
    f"import os; os.environ.get('{STARTUP_PATCH_ENV}') == '1' "
    "and __import__('vllm_iluvatar')._load_import_compat_patch()\n"
)


def _load_version() -> str:
    version_file = ROOT_DIR / "vllm_iluvatar" / "version.py"
    namespace: dict[str, str] = {}
    exec(version_file.read_text(encoding="utf-8"), namespace)  # noqa: S102
    version = namespace.get("__version__")
    if not version:
        raise RuntimeError(f"__version__ not found in {version_file}")
    return str(version)


class BuildPyWithStartupPatch(build_py):
    """Emit the env-gated startup hook .pth alongside the built package."""

    def _pth_path(self) -> str:
        return os.path.join(self.build_lib, PTH_NAME)

    def run(self):
        super().run()
        if self.dry_run:
            return
        self.mkpath(self.build_lib)
        with open(self._pth_path(), "w", encoding="utf-8") as f:
            f.write(PTH_CONTENT)

    def get_outputs(self, include_bytecode=True):
        outputs = list(super().get_outputs(include_bytecode))
        pth_path = self._pth_path()
        if pth_path not in outputs:
            outputs.append(pth_path)
        return outputs


CMDCLASS: dict[str, type] = {"build_py": BuildPyWithStartupPatch}

if editable_wheel is not None:

    class _EditableStrategyWithStartupPatch:
        def __init__(self, strategy):
            self._strategy = strategy

        def __enter__(self):
            self._strategy.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._strategy.__exit__(exc_type, exc_value, traceback)

        def __call__(self, wheel, files, mapping):
            self._strategy(wheel, files, mapping)
            # Editable installs bypass build_py outputs, so inject the same
            # env-gated startup hook directly into the editable wheel.
            wheel.writestr(PTH_NAME, PTH_CONTENT.encode("utf-8"))

    class EditableWheelWithStartupPatch(editable_wheel):
        def _select_strategy(self, name, tag, build_lib):
            strategy = super()._select_strategy(name, tag, build_lib)
            return _EditableStrategyWithStartupPatch(strategy)

    CMDCLASS["editable_wheel"] = EditableWheelWithStartupPatch


setup(
    version=_load_version(),
    cmdclass=CMDCLASS,
)
