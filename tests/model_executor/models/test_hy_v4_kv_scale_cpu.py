# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for HY4 calibration; native KV insertion is mocked.

Execute the actual methods without importing the GPU backend. These checks
cover calibration values and insert wiring, not FP8 kernels or Graph replay.
"""

import __future__

import ast
import logging
import math
import sys
import unittest
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_iluvatar/models/hy_v4/flashmla_sparse.py"
)


def load_method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV4FlashMLASparseImpl"
    )
    fn = next(node for node in cls.body if getattr(node, "name", None) == name)
    namespace = {
        "torch": torch,
        "math": math,
        "logger": logging.getLogger(__name__),
        "_HYV4_DSA_FP8_E4M3_MAX": 448.0,
    }
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class HYV4KVScaleCPU(unittest.TestCase):
    def setUp(self):
        self.layer = SimpleNamespace(
            _fp8_kv_scale_frozen=False, prefix="test.layer", use_fp8_attn=True
        )
        self.layer._maybe_init_fp8_kv_scale = MethodType(
            load_method("_maybe_init_fp8_kv_scale"), self.layer
        )
        self.scale = torch.tensor(1.0)
        self.capture = patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        )
        self.capture.start()
        self.addCleanup(self.capture.stop)

    def calibrate(self, values, slots):
        kv = torch.tensor(values).reshape(-1, 1).expand(-1, 512)
        self.layer._maybe_init_fp8_kv_scale(kv, self.scale, torch.tensor(slots))

    def test_all_invalid_warmup_does_not_freeze(self):
        self.calibrate([40.0, float("nan"), float("inf")], [-1, -1, -1])
        self.assertFalse(self.layer._fp8_kv_scale_frozen)
        self.assertEqual(self.scale.item(), 1.0)

    def test_padding_does_not_change_valid_amax(self):
        self.calibrate(
            [4.0, -2.0, 40.0, float("nan"), float("inf")], [0, 7, -1, -1, -1]
        )
        self.assertTrue(self.layer._fp8_kv_scale_frozen)
        torch.testing.assert_close(self.scale, torch.tensor(4.0 / 448.0))

    def test_invalid_warmup_then_valid_insert_freezes_once(self):
        self.calibrate([40.0], [-1])
        self.calibrate([-4.0], [3])
        self.calibrate([400.0], [4])
        self.assertTrue(self.layer._fp8_kv_scale_frozen)
        torch.testing.assert_close(self.scale, torch.tensor(4.0 / 448.0))

    def test_zero_and_nonfinite_valid_rows_do_not_freeze(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.calibrate([value], [0])
                self.assertFalse(self.layer._fp8_kv_scale_frozen)
                self.assertEqual(self.scale.item(), 1.0)

    def test_empty_and_capture_do_not_freeze(self):
        self.calibrate([], [])
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.calibrate([4.0], [0])
        self.assertFalse(self.layer._fp8_kv_scale_frozen)
        self.assertEqual(self.scale.item(), 1.0)

    def test_explicit_frozen_scale_is_preserved(self):
        self.layer._fp8_kv_scale_frozen = True
        self.scale.fill_(0.25)
        self.calibrate([4.0], [0])
        self.assertEqual(self.scale.item(), 0.25)

    def test_update_passes_slots_and_calibrates_before_insert(self):
        kv = torch.tensor([4.0, 40.0]).reshape(-1, 1).expand(-1, 512)
        slots = torch.tensor([[5], [-1]])
        calls = []

        def insert(kv_arg, rope, cache, slots_arg, scale_arg, block_size):
            self.assertIs(kv_arg, kv)
            self.assertIs(scale_arg, self.scale)
            self.assertTrue(self.layer._fp8_kv_scale_frozen)
            torch.testing.assert_close(scale_arg, torch.tensor(4.0 / 448.0))
            torch.testing.assert_close(slots_arg, slots.flatten())
            calls.append(block_size)

        modules = {
            name: ModuleType(name)
            for name in (
                "ixformer",
                "ixformer.inference",
                "ixformer.inference.functions",
            )
        }
        modules["ixformer.inference.functions"].glm_dsa_fp8_kv_insert = insert
        update = load_method("do_kv_cache_update")
        with patch.dict(sys.modules, modules):
            update(
                self.layer,
                kv,
                torch.zeros(2, 64),
                torch.empty(1, 64, 640, dtype=torch.uint8),
                slots,
                "fp8",
                self.scale,
            )
        self.assertEqual(calls, [64])


if __name__ == "__main__":
    unittest.main()
