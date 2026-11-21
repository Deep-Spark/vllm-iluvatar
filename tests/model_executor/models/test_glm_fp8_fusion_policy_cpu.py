# SPDX-License-Identifier: Apache-2.0
"""Exercise source calibration logic without importing torch or GPU packages.

Run directly with Python to avoid the repository's GPU-probing conftest.
These tests cover fusion state transitions, not tensor or kernel numerics.
"""

import __future__

import ast
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get("GLM_DSA_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
MODEL_DIR = ROOT / "vllm_iluvatar/models/glm_moe_dsa"


def source_class(filename, name):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def method(cls, name):
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def assigns_attribute(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == name
        for target in node.targets
    )


def execute(nodes, namespace):
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            "<actual GLM attention source>",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


class CalibrationValue:
    """Scalar stand-in for the calibration method's amax and scale storage."""

    def __init__(self, value):
        self.value = value

    def numel(self):
        return 1

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return self

    def amax(self):
        return self.value

    def fill_(self, value):
        self.value = value


class FusionPolicyCPU(unittest.TestCase):
    def build(
        self, *, fp8, frozen=False, requested=True, available=True, capture=False
    ):
        base = source_class("attention.py", "IluGlmMoeDsaSparseMLA")
        sub = source_class("fp8_attention.py", "IluGlmMoeDsaFp8SparseMLA")
        obj = SimpleNamespace(
            use_fp8_attn=True,
            _decode_prepare_enabled=True,
            _fp8_kv_scale_calibration_enabled=True,
            _fp8_kv_scale_frozen=frozen,
            _k_scale=CalibrationValue(0.25),
            dcp_world_size=1,
            prefix="test",
        )
        for cls in [base, sub] if fp8 else [base]:
            for node in cls.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "_allow_rms_rope_fp8_kv_fusion"
                ):
                    setattr(obj, node.target.id, ast.literal_eval(node.value))
        namespace = {
            "self": obj,
            "fusion_requested": requested,
            "fusion_op_available": available,
            "fusion_heads_supported": True,
            "use_pcp": False,
            "math": math,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture)
            ),
            "logger": SimpleNamespace(info=lambda *args: None),
            "_GLM_DSA_FP8_E4M3_MAX": 448.0,
        }
        names = ("_rms_rope_fp8_kv_fusion_available", "use_rms_rope_fp8_kv_fusion")
        execute(
            [
                n
                for n in method(base, "__init__").body
                if any(assigns_attribute(n, name) for name in names)
            ],
            namespace,
        )
        if fp8:
            execute(
                [
                    n
                    for n in method(sub, "__init__").body
                    if assigns_attribute(n, "use_rms_rope_fp8_kv_fusion")
                ],
                namespace,
            )
        execute([method(base, "_maybe_init_fp8_kv_scale")], namespace)
        return obj, namespace["_maybe_init_fp8_kv_scale"]

    def test_fp8_online_calibration_preserves_fusion_ban(self):
        obj, calibrate = self.build(fp8=True)
        self.assertFalse(obj.use_rms_rope_fp8_kv_fusion)
        calibrate(obj, CalibrationValue(448.0))
        self.assertTrue(obj._fp8_kv_scale_frozen)
        self.assertEqual(obj._k_scale.value, 1.0)
        self.assertFalse(obj.use_rms_rope_fp8_kv_fusion)
        self.assertTrue(obj._decode_prepare_enabled)
        calibrate(obj, CalibrationValue(896.0))
        self.assertEqual(obj._k_scale.value, 1.0)

    def test_fp8_offline_scale_keeps_fusion_disabled(self):
        obj, calibrate = self.build(fp8=True, frozen=True)
        calibrate(obj, CalibrationValue(448.0))
        self.assertFalse(obj.use_rms_rope_fp8_kv_fusion)
        self.assertFalse(obj._rms_rope_fp8_kv_fusion_available)

    def test_w4a8_can_enable_fusion_after_calibration(self):
        obj, calibrate = self.build(fp8=False)
        calibrate(obj, CalibrationValue(448.0))
        self.assertTrue(obj.use_rms_rope_fp8_kv_fusion)
        offline, _ = self.build(fp8=False, frozen=True)
        self.assertTrue(offline.use_rms_rope_fp8_kv_fusion)

    def test_missing_op_or_disabled_switch_prevents_fusion(self):
        for requested, available in [(False, True), (True, False), (False, False)]:
            with self.subTest(requested=requested, available=available):
                obj, calibrate = self.build(
                    fp8=False, requested=requested, available=available
                )
                calibrate(obj, CalibrationValue(448.0))
                self.assertFalse(obj.use_rms_rope_fp8_kv_fusion)

    def test_capture_and_invalid_amax_do_not_freeze_scale(self):
        obj, calibrate = self.build(fp8=True, capture=True)
        calibrate(obj, CalibrationValue(448.0))
        self.assertFalse(obj._fp8_kv_scale_frozen)
        for value in [0.0, float("nan"), float("inf")]:
            with self.subTest(amax=value):
                obj, calibrate = self.build(fp8=True)
                calibrate(obj, CalibrationValue(value))
                self.assertFalse(obj._fp8_kv_scale_frozen)
                self.assertFalse(obj.use_rms_rope_fp8_kv_fusion)


if __name__ == "__main__":
    unittest.main(verbosity=2)
