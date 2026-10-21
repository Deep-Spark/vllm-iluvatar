# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU regression checks for paired recipes and legacy command-line paths."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
import quant  # noqa: E402
import recipes  # noqa: E402


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RecipeCliTests(unittest.TestCase):
    recipe = "qwen/qwen3_dense_w8a8"

    def call_cli(self, arguments):
        with (
            patch.object(sys, "argv", [str(TOOL / "quant.py"), *map(str, arguments)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            quant.main()

    def make_checkpoint(self, root):
        source = root / "input"
        source.mkdir()
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                24, dtype=torch.bfloat16
            ).reshape(6, 4)
            - 12,
            "lm_head.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        }
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, str(source / shard))
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: shard for key in tensors}})
        )
        (source / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        return source, tensors

    def test_all_vendor_profiles_are_paired_and_loadable(self):
        configs = {
            p.relative_to(TOOL / "configs") for p in (TOOL / "configs").glob("*/*.json")
        }
        rules = {
            p.relative_to(TOOL / "rules") for p in (TOOL / "rules").glob("*/*.json")
        }
        self.assertEqual(configs, rules)
        self.assertTrue(configs)
        self.assertFalse(list((TOOL / "configs").glob("*.json")))
        self.assertFalse(list((TOOL / "rules").glob("*.json")))
        self.assertEqual(len(recipes.available_recipes()), len(configs))
        for name in recipes.available_recipes():
            with self.subTest(recipe=name):
                config, rule = recipes.recipe_paths(name)
                quant.RuleSet.load(rule, json.loads(config.read_text()))

    def test_recipe_explicit_legacy_and_custom_conversion_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, original = self.make_checkpoint(root)
            config, rule = recipes.recipe_paths(self.recipe)
            custom = root / "custom"
            custom.mkdir()
            # A custom flat path with a built-in basename must remain custom.
            shutil.copy(config, custom / config.name)
            custom_rule = json.loads(rule.read_text())
            custom_rule["rules"] = []
            custom_rule["default_action"] = "keep"
            (custom / rule.name).write_text(json.dumps(custom_rule))
            variants = [
                ["--recipe", self.recipe],
                ["--configs", config, "--rules", rule],
                [
                    "--configs",
                    TOOL / "configs" / config.name,
                    "--rules",
                    TOOL / "rules" / rule.name,
                ],
            ]
            results = []
            with working_directory(root):
                for index, options in enumerate(variants):
                    output = root / f"output-{index}"
                    self.call_cli(
                        [
                            "--input",
                            source,
                            "--output",
                            output,
                            "--device",
                            "cpu",
                            *options,
                        ]
                    )
                    results.append(
                        (
                            load_file(str(output / "model-00001-of-00001.safetensors")),
                            json.loads((output / "config.json").read_text()),
                        )
                    )
                self.call_cli(
                    [
                        "--input",
                        source,
                        "--output",
                        root / "custom-output",
                        "--device",
                        "cpu",
                        "--configs",
                        custom / config.name,
                        "--rules",
                        custom / rule.name,
                    ]
                )
            for weights, config_doc in results[1:]:
                self.assertEqual(weights.keys(), results[0][0].keys())
                self.assertEqual(config_doc, results[0][1])
                for key in weights:
                    torch.testing.assert_close(
                        weights[key], results[0][0][key], rtol=0, atol=0
                    )
            key = "model.layers.0.self_attn.q_proj.weight"
            self.assertEqual(results[0][0][key].dtype, torch.int8)
            torch.testing.assert_close(
                results[0][0]["lm_head.weight"], original["lm_head.weight"]
            )
            custom_result = load_file(
                str(root / "custom-output/model-00001-of-00001.safetensors")
            )
            torch.testing.assert_close(
                custom_result[key], original[key], rtol=0, atol=0
            )

    def test_match_only_relative_legacy_paths_and_recipe_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.make_checkpoint(root)
            options = [
                ["--recipe", self.recipe],
                [
                    "--configs",
                    "configs/qwen3_dense_w8a8.json",
                    "--rules",
                    "rules/qwen3_dense_w8a8.json",
                ],
            ]
            reports = []
            with working_directory(TOOL):
                for index, option in enumerate(options):
                    report = root / f"match-{index}.json"
                    self.call_cli(
                        [
                            "--input",
                            source,
                            "--match-only",
                            "--match-report",
                            report,
                            *option,
                        ]
                    )
                    reports.append(json.loads(report.read_text())["stats"])
            self.assertEqual(reports[0], reports[1])

    def test_invalid_options_fail_before_reading_checkpoint(self):
        cases = [
            [],
            ["--configs", "custom.json"],
            ["--rules", "custom.json"],
            ["--recipe", self.recipe, "--configs", "custom.json"],
            ["--recipe", self.recipe, "--rules", "custom.json"],
            ["--recipe", "unknown/missing"],
            ["--recipe", "../qwen3_dense_w8a8"],
            ["--recipe", "/qwen/qwen3_dense_w8a8"],
            ["--recipe", self.recipe, "--match-only"],
            ["--recipe", self.recipe, "--match-report", "report.json"],
            ["--recipe", self.recipe],
        ]
        for options in cases:
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    self.call_cli(["--input", "/does-not-exist", *options])
                self.assertEqual(error.exception.code, 2)

    def test_legacy_lookup_does_not_guess_custom_or_ambiguous_paths(self):
        value = "not-the-tool/configs/qwen3_dense_w8a8.json"
        self.assertEqual(recipes.legacy_profile_path(value, "configs"), Path(value))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for vendor in ("one", "two"):
                p = root / "rules" / vendor / "same.json"
                p.parent.mkdir(parents=True)
                p.write_text("{}")
            with (
                patch.object(recipes, "TOOL_DIR", root),
                self.assertRaisesRegex(ValueError, "Ambiguous"),
            ):
                recipes.legacy_profile_path(str(root / "rules/same.json"), "rules")


if __name__ == "__main__":
    unittest.main()
