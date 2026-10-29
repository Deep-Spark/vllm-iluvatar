# SPDX-License-Identifier: Apache-2.0
"""Retained-query host layout must preserve original causal/output row IDs."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_iluvatar/models/deepseek_v41/prefill_query.py"
spec = importlib.util.spec_from_file_location("producer_query_test_module", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Selection = module.ProducerQuerySelection


class QueryLayoutTests(unittest.TestCase):
    def test_even_and_odd_tail_keep_exact128_or129(self):
        for n in (128, 129, 4096, 4097, 1040384, 1040385):
            start = max(0, n - n % 2 - 128)
            plan = Selection.from_rows(tuple(range(start, n)), n)
            self.assertEqual(plan.spans, ((start, n),))
            self.assertEqual(plan.count, n - start)

    def test_nonfinal_chunk_has_no_retained_queries(self):
        plan = Selection.from_rows((), 4096)
        self.assertEqual(plan.count, 0)
        self.assertEqual(list(plan.intersections(0, 4096)), [])

    def test_ragged_requests_keep_packed_offsets_across_score_tiles(self):
        plan = Selection.from_rows(tuple(range(6, 10)) + tuple(range(21, 26)), 30)
        self.assertEqual(list(plan.intersections(8, 24)), [(8, 10, 2), (21, 24, 4)])
        self.assertEqual(list(plan.intersections(0, 8)), [(6, 8, 0)])
        self.assertEqual(list(plan.intersections(24, 30)), [(24, 26, 7)])

    def test_invalid_host_rows_are_rejected_before_device_reads(self):
        for rows in ((1, 1), (2, 1), (-1,), (3,), (True,)):
            with self.assertRaises(ValueError):
                Selection.from_rows(rows, 3)

    def test_configuration_defaults_off_and_requires_real_boolean(self):
        self.assertFalse(module.producer_query_compaction_enabled({}))
        self.assertTrue(
            module.producer_query_compaction_enabled(
                {"ilu_prefill": {"producer_query_compaction": True}}
            )
        )
        for value in ("1", 1, None):
            with self.assertRaises(ValueError):
                module.producer_query_compaction_enabled(
                    {"ilu_prefill": {"producer_query_compaction": value}}
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
