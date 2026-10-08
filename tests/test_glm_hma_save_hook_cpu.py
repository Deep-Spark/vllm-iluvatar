# SPDX-License-Identifier: Apache-2.0
"""CPU interface checks for the real GLM HMA save-hook statements.

Execute the source forward tail without importing Torch/vLLM or constructing a
model. This checks metadata gating and callback ordering, not a full forward.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_hook(namespace):
    path = ROOT / "vllm_iluvatar/models/glm_moe_dsa/attention.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "IluGlmMoeDsaMLAAttention"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    tail = forward.body[-3:]
    assert isinstance(tail[0], ast.Assign)
    assert ast.unparse(tail[0].targets[0]) == "attn_metadata"
    assert isinstance(tail[1], ast.If) and isinstance(tail[2], ast.Return)
    function = ast.FunctionDef(
        name="run_hook",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="attn_out")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=tail,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_hook"]


class HmaSaveHookCPU(unittest.TestCase):
    def run_case(self, metadata, *, group=True, connector_metadata=True, error=None):
        events = []
        cached = {"updated": True}
        inner = SimpleNamespace(prefix="model.layers.3.self_attn.attn", kv_cache=cached)
        output = object()
        activation = object()

        class Connector:
            def has_connector_metadata(self):
                events.append("connector_metadata")
                return connector_metadata

            def save_kv_layer(connector, layer_name, cache, actual_metadata):
                self.assertEqual(layer_name, "model.layers.3.self_attn")
                self.assertIs(cache, cached)
                self.assertTrue(cache["updated"])
                self.assertIs(actual_metadata, metadata)
                events.append("save")
                if error is not None:
                    raise error

        connector = Connector()

        def has_group():
            events.append("has_group")
            return group

        def get_group():
            self.assertTrue(group)
            events.append("get_group")
            return connector

        def project(value):
            self.assertIs(value, activation)
            events.append("project")
            return output, None

        layer = SimpleNamespace(
            prefix="model.layers.3.self_attn", mla_attn=inner, o_proj=project
        )
        hook = load_hook(
            {
                "get_forward_context": lambda: SimpleNamespace(attn_metadata=metadata),
                "has_kv_transfer_group": has_group,
                "get_kv_transfer_group": get_group,
            }
        )
        if error is None:
            self.assertIs(hook(layer, activation), output)
        else:
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                hook(layer, activation)
        return events

    def test_saves_outer_prefix_cache_and_complete_metadata_before_projection(self):
        metadata = {
            "model.layers.3.self_attn.attn": object(),
            "model.layers.3.self_attn.indexer.k_cache": object(),
        }
        self.assertEqual(
            self.run_case(metadata),
            ["has_group", "get_group", "connector_metadata", "save", "project"],
        )

    def test_missing_metadata_or_connector_skips_save(self):
        key = "model.layers.3.self_attn.attn"
        cases = [
            (SimpleNamespace(), True, True, ["project"]),
            ({}, True, True, ["project"]),
            ({key: None}, True, True, ["project"]),
            ({key: object()}, False, True, ["has_group", "project"]),
            (
                {key: object()},
                True,
                False,
                ["has_group", "get_group", "connector_metadata", "project"],
            ),
        ]
        for metadata, group, connector_metadata, expected in cases:
            with self.subTest(
                group=group,
                connector_metadata=connector_metadata,
                metadata_kind=type(metadata).__name__,
            ):
                self.assertEqual(
                    self.run_case(
                        metadata, group=group, connector_metadata=connector_metadata
                    ),
                    expected,
                )

    def test_save_failure_propagates_before_projection(self):
        events = self.run_case(
            {"model.layers.3.self_attn.attn": object()},
            error=RuntimeError("save failed"),
        )
        self.assertEqual(
            events, ["has_group", "get_group", "connector_metadata", "save"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
