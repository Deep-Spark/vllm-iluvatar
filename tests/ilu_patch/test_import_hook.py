# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Post-import callbacks must remain scoped when a finder shares its loader."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
_TARGET = "_iluvatar_hook_target"
_OTHER = "_iluvatar_hook_other"


class _SharedLoader(importlib.abc.Loader):
    def __init__(self):
        self.bodies = {}
        self.created = []

    def create_module(self, spec):
        self.created.append(spec.name)
        module = ModuleType(spec.name)
        module.created_by_loader = True
        return module

    def exec_module(self, module):
        body = self.bodies.get(module.__name__)
        if body is not None:
            body(module)
        module.ready = True

    def get_source(self, fullname):
        return "source for " + fullname


class _SharedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, loader):
        self.loader = loader

    def find_spec(self, fullname, path=None, target=None):
        if fullname in (_TARGET, _OTHER):
            return importlib.util.spec_from_loader(fullname, self.loader)
        return None


class TestPostImportHooks(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "_isolated_iluvatar_import_hook",
            _ROOT / "vllm_iluvatar/ilu_patch/import_hook.py",
        )
        self.hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hooks)
        self.loader = _SharedLoader()
        self.finder = _SharedFinder(self.loader)
        paths = patch.object(sys, "meta_path", [self.finder, *sys.meta_path])
        paths.start()
        self.addCleanup(paths.stop)
        for name in (_TARGET, _OTHER):
            previous = sys.modules.pop(name, None)
            self.addCleanup(self._restore_module, name, previous)

    @staticmethod
    def _restore_module(name, previous):
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous

    def test_shared_loader_does_not_fire_target_hook_for_nested_import(self):
        observed = []
        original_exec = self.loader.exec_module
        self.loader.bodies[_TARGET] = lambda module: importlib.import_module(_OTHER)
        self.hooks.register(
            _TARGET, lambda module: observed.append((module.__name__, module.ready))
        )

        target = importlib.import_module(_TARGET)
        importlib.import_module(_TARGET)
        importlib.import_module(_OTHER)

        self.assertEqual(observed, [(_TARGET, True)])
        self.assertIs(importlib.import_module(_TARGET), target)
        self.assertEqual(self.loader.exec_module, original_exec)
        self.assertTrue(target.created_by_loader)
        self.assertEqual(self.loader.created, [_TARGET, _OTHER])
        self.assertEqual(
            target.__spec__.loader.get_source(_TARGET), "source for " + _TARGET
        )

    def test_two_targets_using_one_loader_fire_after_their_own_bodies(self):
        observed = []
        self.loader.bodies[_TARGET] = lambda module: importlib.import_module(_OTHER)
        for name in (_TARGET, _OTHER):
            self.hooks.register(
                name, lambda module: observed.append((module.__name__, module.ready))
            )

        importlib.import_module(_TARGET)

        self.assertEqual(observed, [(_OTHER, True), (_TARGET, True)])

    def test_failed_module_body_does_not_fire_callback(self):
        observed = []

        def fail(module):
            raise RuntimeError("module body failed")

        self.loader.bodies[_TARGET] = fail
        self.hooks.register(_TARGET, observed.append)
        with self.assertRaisesRegex(RuntimeError, "module body failed"):
            importlib.import_module(_TARGET)
        self.assertEqual(observed, [])
        self.assertNotIn(_TARGET, self.hooks._fired)

    def test_already_loaded_module_fires_once(self):
        module = ModuleType(_TARGET)
        module.ready = True
        sys.modules[_TARGET] = module
        observed = []
        self.hooks.register(_TARGET, observed.append)
        self.hooks._fire(_TARGET, module)
        self.assertEqual(observed, [module])


if __name__ == "__main__":
    unittest.main()
