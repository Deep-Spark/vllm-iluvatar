# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Regression tests for KV transfer with more than 101 model layers."""

import unittest

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.layer_names import (
    find_number_on_layer_name,
)


class TestLayerNameParser(unittest.TestCase):
    def test_all_layers_of_a_deep_model_are_mapped(self):
        names = [f"model.layers.{index}.self_attn.attn" for index in range(130)]
        mapping = {name: int(find_number_on_layer_name(name)[0]) for name in names}

        self.assertEqual(list(mapping.values()), list(range(130)))
        self.assertEqual(find_number_on_layer_name("model.layers.1000.attn"), ["1000"])

    def test_preserves_layer_name_format(self):
        self.assertEqual(
            find_number_on_layer_name("model.2.layers.101.attn"), ["2", "101"]
        )
        for name in ("model.layers.-1.attn", "model.layers.01.attn", "model.layers.1"):
            with self.subTest(name=name):
                self.assertEqual(find_number_on_layer_name(name), [])


if __name__ == "__main__":
    unittest.main()
