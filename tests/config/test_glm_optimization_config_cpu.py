# SPDX-License-Identifier: Apache-2.0
"""CPU tests of actual config registration and shared architecture ownership."""

import builtins
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MOE = "VLLM_ILU_GLM_MOE_PREPARE_FUSION"
DSA = "VLLM_ILU_GLM_DSA_PREPARE_FUSION"
SAMPLING = "VLLM_ILU_GLM_SAMPLING_MODE"


def source_module(path):
    module = ModuleType(path)
    exec(compile((ROOT / path).read_text(), path, "exec"), module.__dict__)
    return module


class ConfigurationCPU(unittest.TestCase):
    def test_config_import_is_lazy(self):
        original = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] in ("vllm", "ixformer", "torch"):
                raise AssertionError(f"unexpected eager import: {name}")
            return original(name, *args, **kwargs)

        with (
            patch("builtins.__import__", blocked),
            patch.dict(os.environ, {MOE: "1", DSA: "1", SAMPLING: "invalid"}),
        ):
            source_module("vllm_iluvatar/envs.py")
            source_module("vllm_iluvatar/ilu_util/arch.py")

    def test_registered_boolean_contract(self):
        module = source_module("vllm_iluvatar/envs.py")
        vllm = ModuleType("vllm")
        envs = ModuleType("vllm.envs")
        envs.environment_variables = {}
        vllm.envs = envs
        with patch.dict(sys.modules, {"vllm": vllm, "vllm.envs": envs}):
            module.apply()
        for name in (MOE, DSA):
            reader = envs.environment_variables[name]
            self.assertIs(reader, module._ILU_CANONICAL_ONLY_ENV_VARS[name])
            for value, expected in (
                (None, False),
                ("0", False),
                ("1", True),
                (" TRUE ", True),
                ("yes", True),
                ("ON", True),
                ("false", False),
                ("2", False),
                ("", False),
            ):
                with (
                    self.subTest(name=name, value=value),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    if value is not None:
                        os.environ[name] = value
                    self.assertIs(reader(), expected)
                    # Existing canonical-only registration has no legacy alias.
                    os.environ[name.replace("VLLM_ILU_", "VLLM_")] = "1"
                    self.assertIs(reader(), expected)

    def test_sampling_modes_and_errors(self):
        module = source_module("vllm_iluvatar/envs.py")
        reader = module._ILU_CANONICAL_ONLY_ENV_VARS[SAMPLING]
        for value, expected in (
            (None, "off"),
            ("OFF", "off"),
            (" sorted ", "sorted"),
            ("RADIX", "radix"),
        ):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ[SAMPLING] = value
                self.assertEqual(reader(), expected)
        for value in ("", "true", "1", "other"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {SAMPLING: value}),
                self.assertRaisesRegex(ValueError, "must be one of"),
            ):
                reader()

    def test_arch_unknown_and_loaded_binary_cache(self):
        for value in (None, "unknown", "ivcore11", "ivcore40", "ivcore30"):
            module = source_module("vllm_iluvatar/ilu_util/arch.py")
            ixformer = ModuleType("ixformer")
            cuda = ModuleType("ixformer.cuda")
            cuda.cuda_arch = value
            ixformer.cuda = cuda
            with patch.dict(sys.modules, {"ixformer": ixformer, "ixformer.cuda": cuda}):
                self.assertEqual(module.is_ivcore30(), value == "ivcore30")
                self.assertEqual(module.cuda_arch(), value)
                cuda.cuda_arch = "ivcore30" if value != "ivcore30" else None
                self.assertEqual(module.cuda_arch(), value)

    def test_missing_binary_falls_back(self):
        original = builtins.__import__
        module = source_module("vllm_iluvatar/ilu_util/arch.py")

        def unavailable(name, *args, **kwargs):
            if name == "ixformer":
                raise ImportError("missing")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", unavailable):
            self.assertIsNone(module.cuda_arch())
            self.assertFalse(module.is_ivcore30())

    def test_binary_initialization_error_propagates(self):
        original = builtins.__import__
        for error in (RuntimeError("initialization failed"), OSError("load failed")):
            for query in ("cuda_arch", "is_ivcore30", "is_ivcore11"):
                with self.subTest(error=type(error).__name__, query=query):
                    module = source_module("vllm_iluvatar/ilu_util/arch.py")

                    def unavailable(name, *args, _error=error, **kwargs):
                        if name == "ixformer":
                            raise _error
                        return original(name, *args, **kwargs)

                    with (
                        patch("builtins.__import__", unavailable),
                        self.assertRaises(type(error)) as caught,
                    ):
                        getattr(module, query)()
                    self.assertIs(caught.exception, error)

    def test_arch_read_error_propagates(self):
        for error in (AttributeError("missing cuda_arch"), RuntimeError("read failed")):
            for query in ("cuda_arch", "is_ivcore30", "is_ivcore11"):
                with self.subTest(error=type(error).__name__, query=query):
                    module = source_module("vllm_iluvatar/ilu_util/arch.py")

                    class UnreadableCuda:
                        @property
                        def cuda_arch(self, _error=error):
                            raise _error

                    ixformer = ModuleType("ixformer")
                    ixformer.cuda = UnreadableCuda()
                    with (
                        patch.dict(sys.modules, {"ixformer": ixformer}),
                        self.assertRaises(type(error)) as caught,
                    ):
                        getattr(module, query)()
                    self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
