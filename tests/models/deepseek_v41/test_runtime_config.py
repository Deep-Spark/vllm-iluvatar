# SPDX-License-Identifier: Apache-2.0
"""Keep input-storage policy independent of launcher flags and inherited state."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_iluvatar/models/deepseek_v41/runtime_config.py"
)
spec = importlib.util.spec_from_file_location("v41_runtime_config_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InputStoragePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS=False)
        core = ModuleType("ixformer.core")
        core.config = self.config
        ixformer = ModuleType("ixformer")
        ixformer.core = core
        modules = patch.dict(sys.modules, {"ixformer": ixformer, "ixformer.core": core})
        modules.start()
        self.addCleanup(modules.stop)

    def configure(self, role, seqs):
        transfer = None if role is None else SimpleNamespace(kv_role=role)
        return module.configure_w8a8_input_rows(
            SimpleNamespace(
                kv_transfer_config=transfer,
                scheduler_config=SimpleNamespace(max_num_seqs=seqs),
            )
        )

    def test_decode_seq4_enables_without_launcher_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self.configure("kv_consumer", 4))
        self.assertTrue(self.config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS)

    def test_other_roles_and_sequence_limits_clear_inherited_enable(self):
        with patch.dict(os.environ, {"IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS": "1"}):
            for role in (None, "kv_producer", "kv_consumer", "kv_both"):
                for seqs in (1, 2, 3, 4, 5, 8, 32):
                    if role == "kv_consumer" and seqs == 4:
                        continue
                    with self.subTest(role=role, max_num_seqs=seqs):
                        self.config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS = True
                        self.assertFalse(self.configure(role, seqs))
                        self.assertFalse(self.config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS)

    def test_decode_seq4_ignores_inherited_disable(self):
        with patch.dict(os.environ, {"IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS": "0"}):
            self.assertTrue(self.configure("kv_consumer", 4))
        self.assertTrue(self.config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS)

    def test_reconfiguration_does_not_keep_previous_model_policy(self):
        for role, seqs, expected in (
            ("kv_consumer", 4, True),
            ("kv_producer", 4, False),
            ("kv_consumer", 3, False),
            (None, 4, False),
            ("kv_consumer", 4, True),
        ):
            self.assertEqual(self.configure(role, seqs), expected)
            self.assertEqual(self.config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
