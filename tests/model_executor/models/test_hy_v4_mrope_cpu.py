# SPDX-License-Identifier: Apache-2.0
"""CPU checks for shape-based M-RoPE routing and indexer tables.

Execute the model helpers on CPU, resolving their vLLM imports from the
installed package. These tests do not establish device RoPE accuracy or
CUDA Graph replay correctness.
"""

import __future__

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_iluvatar/models/hy_v4"


def load_functions(path, names):
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
    ]
    assert len(body) == len(names)
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


class HYV4MRopeCPU(unittest.TestCase):
    def setUp(self):
        self.apply = load_functions(MODEL / "attention.py", {"apply_qk_rope"})[
            "apply_qk_rope"
        ]
        self.indexer = load_functions(
            MODEL / "fp8_indexer.py", {"_mixed_mrope_cos_sin", "_indexer_rope_table"}
        )

    def test_three_rows_always_use_mrope_without_reading_device_values(self):
        q, k = torch.zeros(2, 4, 8), torch.zeros(2, 1, 8)
        for positions in (
            torch.tensor([[1, 2], [1, 2], [1, 2]]),
            torch.tensor([[1, 2], [3, 4], [5, 6]]),
        ):
            with self.subTest(positions=positions.tolist()):
                rotary = Mock(side_effect=AssertionError("1-D branch selected"))
                mrope = Mock(side_effect=lambda p, a, b: (a + 1, b + 2))
                with patch.object(
                    torch, "equal", side_effect=AssertionError("host read")
                ):
                    out_q, out_k = self.apply(rotary, positions, q, k, mrope)
                self.assertIs(mrope.call_args.args[0], positions)
                self.assertEqual(mrope.call_args.args[1].shape, (2, 32))
                torch.testing.assert_close(out_q, q + 1)
                torch.testing.assert_close(out_k, k + 2)

    def test_one_dimensional_text_keeps_ordinary_rope(self):
        positions = torch.tensor([1, 2])
        q, k = torch.zeros(2, 4, 8), torch.zeros(2, 1, 8)
        rotary = Mock(return_value=(q, k))
        mrope = Mock(side_effect=AssertionError("M-RoPE branch selected"))
        self.apply(rotary, positions, q, k, mrope)
        self.assertIs(rotary.call_args.args[0], positions)

    def test_inplace_mrope_outputs_do_not_alias_mla_slices(self):
        for tokens in (1, 7):
            with self.subTest(tokens=tokens):
                q = torch.arange(tokens * 4 * 12, dtype=torch.float32).view(
                    tokens, 4, 12
                )
                kv = torch.arange(tokens * 20, dtype=torch.float32).view(tokens, 20)
                q_pe = q[..., 4:]
                k_pe = kv[..., 12:].unsqueeze(1)
                q_before, k_before = q_pe.clone(), k_pe.clone()

                def rotate_inplace(positions, query, key):
                    query.add_(1)
                    key.add_(2)
                    return query.view_as(query), key.view_as(key)

                out_q, out_k = self.apply(
                    Mock(), torch.zeros(3, tokens), q_pe, k_pe, rotate_inplace
                )
                torch.testing.assert_close(q_pe, q_before)
                torch.testing.assert_close(k_pe, k_before)
                # Mirrors _prepare_mla_qkv's writeback, including T=1 decode.
                q_pe.copy_(out_q.reshape(q_pe.shape))
                k_pe.copy_(out_k.reshape(k_pe.shape))
                torch.testing.assert_close(q_pe, q_before + 1)
                torch.testing.assert_close(k_pe, k_before + 2)

    def test_missing_module_or_invalid_rows_fail_explicitly(self):
        q = torch.zeros(2, 1, 8)
        for positions, module in (
            (torch.zeros(3, 2), None),
            (torch.zeros(2, 2), Mock()),
        ):
            with self.assertRaisesRegex(ValueError, "three rows"):
                self.apply(Mock(), positions, q, q, module)

    def test_indexer_tables_match_independent_channel_selection(self):
        # Distinct values identify both the selected position and channel.
        cache = torch.arange(10 * 12, dtype=torch.float32).view(10, 12)
        uniform = torch.tensor([[1, 2], [1, 2], [1, 2]])
        mixed = torch.tensor([[1, 2], [3, 4], [5, 6]])
        for interleaved in (False, True):
            for positions in (uniform, mixed):
                with self.subTest(
                    interleaved=interleaved, positions=positions.tolist()
                ):
                    emb = SimpleNamespace(
                        mrope_section=[2, 2, 2],
                        mrope_interleaved=interleaved,
                        _match_cos_sin_cache_dtype=lambda _: cache,
                    )
                    indexer = SimpleNamespace(mrope_emb=emb)
                    rotary = Mock()
                    with patch.object(
                        torch, "equal", side_effect=AssertionError("host read")
                    ):
                        pos, actual = self.indexer["_indexer_rope_table"](
                            indexer, positions, rotary, cache
                        )
                    rows = [0, 1, 2, 0, 1, 2] if interleaved else [0, 0, 1, 1, 2, 2]
                    expected = torch.stack(
                        [
                            cache[positions[row], channel + half * 6]
                            for half in (0, 1)
                            for channel, row in enumerate(rows)
                        ],
                        dim=-1,
                    )
                    torch.testing.assert_close(pos, torch.arange(2))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    rotary._match_cos_sin_cache_dtype.assert_not_called()

    def test_indexer_missing_mrope_does_not_silently_use_first_row(self):
        with self.assertRaisesRegex(ValueError, "three rows"):
            self.indexer["_indexer_rope_table"](
                SimpleNamespace(), torch.zeros(3, 2), Mock(), torch.zeros(1)
            )


if __name__ == "__main__":
    unittest.main()
