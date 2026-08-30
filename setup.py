# SPDX-License-Identifier: Apache-2.0
"""Build entry point for the vllm_iluvatar plugin.

Project metadata lives in pyproject.toml (PEP 621); the version is loaded from
version.json.

Two build-time customisations:
  1. An env-gated startup-patch ``.pth`` so spawned vLLM worker subprocesses
     apply the early corex/torch compat shims before unpickling engine modules.
  2. A CMake-built native extension ``vllm_iluvatar._C`` that registers the
     vLLM-community-signature ``torch.ops._C`` operators (rms_norm backed by
     ixformer; weak_ref_tensor / get_cuda_view_from_cpu_tensor self-contained).
     The extension is always built; it requires CMake + a CUDA/ixformer
     toolchain (the Iluvatar target machine).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:  # pragma: no cover - older setuptools
    editable_wheel = None

logging.basicConfig(level=logging.INFO)

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
    """Read the version from ``version.json``, the release spec's data source.

    Baking it into the wheel metadata here is what lets ``version.py`` recover
    it at runtime, when version.json itself is no longer around.
    """
    version_file = ROOT_DIR / "version.json"
    version = json.loads(version_file.read_text(encoding="utf-8")).get("version")
    if not version:
        raise RuntimeError(f"'version' not found in {version_file}")
    return str(version)


def get_requirements() -> list[str]:
    """Resolve runtime dependencies from ``requirements.txt``.

    Mirrors vLLM's setup.py: supports ``-r other.txt`` includes and skips blank
    lines, comments (``#``) and pip options (``--``). torch / ixformer are
    intentionally NOT listed (provided by the corex environment).
    """

    def _read_requirements(filename: str) -> list[str]:
        resolved: list[str] = []
        with open(ROOT_DIR / filename, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        for line in lines:
            if line.startswith("-r "):
                resolved += _read_requirements(line.split()[1])
                continue
            # pip treats " #" as the start of an inline comment
            spec = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if spec and not spec.startswith(("--", "#")):
                resolved.append(spec)
        return resolved

    return _read_requirements("requirements.txt")


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.cmake_lists_dir = str((ROOT_DIR / cmake_lists_dir).resolve())


class CMakeBuildExt(build_ext):
    """Configure + build the CMake-based ``vllm_iluvatar._C`` extension.

    Mirrors the reference vllm_iluvatar repo's build_ext (Ninja detection,
    persistent per-config build dir, ivcore architecture resolution) but trimmed
    to this plugin's single small target.
    """

    def _cuda_architectures(self) -> str:
        # Same order as ixformer / ixmm: CUDA_ARCH, then COREX_ARCH from the SDK.
        override = (
            os.environ.get("CUDA_ARCH")
            or os.environ.get("COREX_ARCH")
            or os.environ.get("CMAKE_CUDA_ARCHITECTURES")
            or os.environ.get("TORCH_CUDA_ARCH_LIST")
        )
        if override:
            return override
        try:
            import torch

            return torch.corex_arch
        except Exception:  # pragma: no cover - non-corex host
            return "ivcore11"

    def _num_jobs(self) -> int:
        max_jobs = os.environ.get("VLLM_ILUVATAR_MAX_JOBS") or os.environ.get(
            "MAX_JOBS"
        )
        if max_jobs:
            return int(max_jobs)
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    def _find_ninja(self) -> str | None:
        ninja_path = shutil.which("ninja")
        if ninja_path and Path(ninja_path).is_file():
            return str(Path(ninja_path).resolve())
        try:
            import ninja
        except ImportError:
            return None
        candidate = Path(ninja.__file__).resolve().parent / "data" / "bin" / "ninja"
        return str(candidate) if candidate.is_file() else None

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        if shutil.which("cmake") is None:
            raise RuntimeError("CMake is required to build vllm_iluvatar extensions")

        cuda_architectures = self._cuda_architectures()
        cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")

        generator_args: list[str] = []
        ninja_path = self._find_ninja()
        if ninja_path is not None:
            generator_args = ["-G", "Ninja"]
        else:
            logging.warning("Ninja not found; CMake will use its default generator.")

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        override_build = os.environ.get("VLLM_ILUVATAR_CMAKE_BUILD_DIR")
        if override_build:
            build_temp = Path(override_build).expanduser().resolve()
        else:
            py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
            safe = "".join(
                c if c.isalnum() or c in "._-" else "_"
                for c in f"{cfg}-{py_tag}-{cuda_architectures}"
            )
            build_temp = (ROOT_DIR / "build" / safe).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir.mkdir(parents=True, exist_ok=True)

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXTENSION_SUFFIX={ext_suffix}",
        ]
        if ninja_path is not None:
            cmake_args.append(f"-DCMAKE_MAKE_PROGRAM={ninja_path}")
        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            cmake_args.extend(extra_cmake_args.split())

        cmake_env = os.environ.copy()
        cmake_env.setdefault("TORCH_CUDA_ARCH_LIST", cuda_architectures)
        num_jobs = self._num_jobs()
        cmake_env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(num_jobs))

        configured_build_dirs = getattr(self, "_configured_build_dirs", set())
        if build_temp not in configured_build_dirs:
            configure_cmd = ["cmake", ext.cmake_lists_dir, *generator_args, *cmake_args]
            logging.info("Running CMake configure: %s", " ".join(configure_cmd))
            subprocess.run(configure_cmd, cwd=build_temp, env=cmake_env, check=True)
            configured_build_dirs.add(build_temp)
            self._configured_build_dirs = configured_build_dirs

        target = ext.name.rsplit(".", 1)[-1]
        build_cmd = [
            "cmake",
            "--build",
            ".",
            "--target",
            target,
            "--parallel",
            str(num_jobs),
        ]
        logging.info("Running CMake build: %s", " ".join(build_cmd))
        subprocess.run(build_cmd, cwd=build_temp, env=cmake_env, check=True)


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


CMDCLASS: dict[str, type] = {
    "build_py": BuildPyWithStartupPatch,
    "build_ext": CMakeBuildExt,
}

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


EXT_MODULES = [
    CMakeExtension(name="vllm_iluvatar._C"),
    CMakeExtension(name="vllm_iluvatar._onchip_check"),
]


setup(
    version=_load_version(),
    install_requires=get_requirements(),
    ext_modules=EXT_MODULES,
    cmdclass=CMDCLASS,
)
