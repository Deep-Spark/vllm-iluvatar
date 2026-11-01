"""CPU proofs for payload ownership, exact bit patterns and traffic selection."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

_PATH = Path(__file__).resolve().parents[2] / (
    "vllm_iluvatar/distributed/device_communicators/compact_ragged.py"
)
_SPEC = importlib.util.spec_from_file_location("compact_ragged", _PATH)
compact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compact)


class CompactRaggedTest(unittest.TestCase):
    def test_disjoint_word_sum_preserves_payload(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError):
            for dtype, width in (
                (torch.bfloat16, 8), (torch.float16, 8), (torch.float32, 4),
                (torch.int64, 2), (torch.int8, 16),
            ):
                for sizes in ((0, 3, 1), (2, 0, 5, 1), (0, 0, 0)):
                    with self.subTest(dtype=dtype, sizes=sizes):
                        inputs, words = [], []
                        for rank, count in enumerate(sizes):
                            bits = torch.arange(count * 4, dtype=torch.int32).reshape(count, 4)
                            bits += rank * 8347981
                            if count:
                                bits[0] = torch.tensor(
                                    [-2147483648, 2143294004, 2139095040, -8388608],
                                    dtype=torch.int32,
                                )
                            inputs.append(bits.view(dtype).reshape(count, width))
                        for rank, tensor in enumerate(inputs):
                            compact.compact_all_gather(tensor, sizes, rank, words.append)
                        reduced = torch.zeros_like(words[0])
                        for tensor in words:
                            reduced.add_(tensor)
                        expected = torch.cat(inputs)
                        self.assertTrue(torch.equal(
                            reduced.view(torch.uint8).reshape(-1),
                            expected.view(torch.uint8).reshape(-1),
                        ))

    def test_traffic_policy(self):
        self.assertTrue(compact.compact_gather_beneficial([129] + [12] * 15, 10240))
        self.assertFalse(compact.compact_gather_beneficial([12] * 16, 10240))
        self.assertFalse(compact.compact_gather_beneficial([129] + [12] * 15, 24))
        self.assertFalse(compact.compact_gather_beneficial([129] + [12] * 15, 3))
        self.assertFalse(compact.compact_gather_beneficial([0] * 16, 10240))
        self.assertFalse(compact.compact_gather_beneficial([-1, 7], 10240))

    def test_invalid_local_layout(self):
        with self.assertRaises(ValueError):
            compact.compact_all_gather(torch.empty(3, 8), [1, 2], 0, lambda _: None)
        with self.assertRaises(ValueError):
            compact.compact_all_gather(torch.empty(1, 3, dtype=torch.uint8), [1, 2], 0, lambda _: None)


if __name__ == "__main__":
    unittest.main()
