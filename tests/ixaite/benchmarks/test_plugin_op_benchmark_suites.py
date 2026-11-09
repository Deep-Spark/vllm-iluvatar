# SPDX-License-Identifier: Apache-2.0
"""Fail if a plugin-routed ixaite op has no ``python -m tests.ixaite.benchmarks`` suite."""

from __future__ import annotations

import tests.ixaite.benchmarks.suites  # noqa: F401 -- register suites
from tests.ixaite.benchmarks.core.model_catalog import listed_e2e_ops
from tests.ixaite.benchmarks.core.plugin_op_suites import (
    PLUGIN_OP_SUITES,
    discover_plugin_op_names,
)
from tests.ixaite.benchmarks.core.registry import list_suites


def test_discovered_plugin_ops_are_catalogued() -> None:
    discovered = discover_plugin_op_names()
    missing = sorted(discovered - PLUGIN_OP_SUITES.keys())
    extra = sorted(PLUGIN_OP_SUITES.keys() - discovered)
    assert not missing, (
        "new plugin ixaite bind(s) have no PLUGIN_OP_SUITES entry; "
        "add a benchmarks suite and map the op name: "
        f"{missing}"
    )
    assert not extra, (
        "PLUGIN_OP_SUITES has stale op names (not discovered in "
        f"vllm_iluvatar): {extra}"
    )


def test_catalogued_ops_have_registered_suites() -> None:
    registered = set(list_suites())
    missing: dict[str, tuple[str, ...]] = {}
    for op_name, suites in PLUGIN_OP_SUITES.items():
        absent = tuple(suite for suite in suites if suite not in registered)
        if absent:
            missing[op_name] = absent
    assert not missing, (
        "plugin op(s) map to unregistered benchmark suites; "
        f"register them in tests/ixaite/benchmarks/suites/: {missing}"
    )


def test_discovered_plugin_ops_are_covered_by_e2e() -> None:
    discovered = discover_plugin_op_names()
    covered = listed_e2e_ops()
    missing = sorted(discovered - covered)
    extra = sorted(covered - discovered)
    assert not missing, (
        "new plugin ixaite bind(s) have no e2e.ops coverage; add the op "
        "to tests/ixaite/benchmarks/models/<model>.yaml e2e.ops on a serve path that "
        f"hits it: {missing}"
    )
    assert not extra, (
        f"e2e.ops has stale op names (not discovered in vllm_iluvatar): {extra}"
    )


# Discover supported binding interfaces without treating dependency probes as exports.
def test_discovery_reads_direct_provider_binding_and_conv_table(tmp_path):
    (tmp_path / "bindings.py").write_text(
        "def configure_embedding_gather_provider():\n"
        "    load_ixaite_python_op('embedding_gather', 'embedding_gather')\n"
        "@_routed_op\n"
        "def linear(): pass\n"
        "_TRITON_OPS = {'causal_conv1d_fn': None}\n"
        "def validate_checkpoint():\n"
        "    load_ixaite_python_op('unavailable_repacker', 'repack')\n"
    )
    assert discover_plugin_op_names(tmp_path) == {
        "embedding_gather",
        "linear",
        "causal_conv1d_fn",
    }
