# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar sparse attention indexer CustomOp wrapper."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

import vllm_iluvatar.ilu_register as ilu_register


# Test the selected class through the official API with an isolated registry entry.
def _load_sparse_attn_indexer(monkeypatch):
    module = importlib.import_module(
        "vllm_iluvatar.backends.ixaite.layers.sparse_attn_indexer"
    )
    monkeypatch.delitem(op_registry_oot, "SparseAttnIndexer", raising=False)
    SparseAttnIndexer.register_oot(module.IluSparseAttnIndexer)
    return module


def test_sparse_attn_indexer_package_is_auto_discovered():
    assert "vllm_iluvatar.model_executor.layers" in ilu_register._OOT_OP_PACKAGES


def test_ilu_sparse_attn_indexer_registers_oot(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)

    assert issubclass(sparse_attn_indexer.IluSparseAttnIndexer, SparseAttnIndexer)
    assert (
        op_registry_oot["SparseAttnIndexer"] is sparse_attn_indexer.IluSparseAttnIndexer
    )


def test_plugin_indexer_op_name_does_not_reuse_fork_schema(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)

    assert sparse_attn_indexer._PLUGIN_INDEXER_OP != "ilu_sparse_attn_indexer"
    assert hasattr(torch.ops.vllm, sparse_attn_indexer._PLUGIN_INDEXER_OP)


# Select ixaite before platform discovery, matching a real service process.
def _run_isolated_indexer_registration(script: str) -> subprocess.CompletedProcess[str]:
    # Custom-op schemas are process-global; a fresh interpreter is the only way
    # to assert fork-then-plugin registration order against a clean registry.
    env = os.environ.copy()
    env["VLLM_ILU_KERNEL_PROVIDER"] = "ixaite"
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


# Occupy the fork schema name with a dummy op. CI vLLM is not the Iluvatar fork
# and has no ``vllm.model_executor.layers.ilu_sparse_attn_indexer``; the bug is
# the colliding schema name, not that module's implementation.
_OCCUPY_FORK_INDEXER_SCHEMA = """
import importlib
import importlib.util

import torch
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_FORK_INDEXER = "vllm.model_executor.layers.ilu_sparse_attn_indexer"


def occupy_fork_indexer_schema():
    if importlib.util.find_spec(_FORK_INDEXER) is not None:
        importlib.import_module(_FORK_INDEXER)
        return

    def _dummy(
        hidden_states: torch.Tensor,
        k_cache_prefix: str,
        kv_cache: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        total_seq_lens: int,
        topk_indices_buffer: torch.Tensor | None,
    ) -> torch.Tensor:
        return topk_indices_buffer

    direct_register_custom_op(
        op_name="ilu_sparse_attn_indexer",
        op_func=_dummy,
        mutates_args=["topk_indices_buffer"],
        fake_impl=_dummy,
        dispatch_key=current_platform.dispatch_key,
    )


occupy_fork_indexer_schema()
assert hasattr(torch.ops.vllm, "ilu_sparse_attn_indexer")
"""


def test_plugin_indexer_registers_after_fork_schema():
    result = _run_isolated_indexer_registration(
        _OCCUPY_FORK_INDEXER_SCHEMA
        + textwrap.dedent(
            """
            plugin = importlib.import_module(
                "vllm_iluvatar.backends.ixaite.layers.sparse_attn_indexer"
            )
            assert plugin._PLUGIN_INDEXER_OP != "ilu_sparse_attn_indexer"
            assert hasattr(torch.ops.vllm, "ilu_sparse_attn_indexer")
            assert hasattr(torch.ops.vllm, plugin._PLUGIN_INDEXER_OP)
            """
        )
    )
    assert result.returncode == 0, result.stderr + result.stdout


# Exercise the full entry point so baseline helper imports stay inside the filter.
def test_oot_scan_imports_indexer_and_dsa_models_after_fork_schema():
    result = _run_isolated_indexer_registration(
        _OCCUPY_FORK_INDEXER_SCHEMA
        + textwrap.dedent(
            """
            import io
            import logging

            from vllm.model_executor.custom_op import op_registry_oot

            log = io.StringIO()
            handler = logging.StreamHandler(log)
            logger = logging.getLogger("vllm_iluvatar.ilu_register")
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)

            from vllm_iluvatar import register_general

            register_general()

            plugin = importlib.import_module(
                "vllm_iluvatar.backends.ixaite.layers.sparse_attn_indexer"
            )
            assert hasattr(torch.ops.vllm, "ilu_sparse_attn_indexer")
            assert hasattr(torch.ops.vllm, plugin._PLUGIN_INDEXER_OP)
            assert (
                op_registry_oot["SparseAttnIndexer"].__module__
                == "vllm_iluvatar.backends.ixaite.layers.sparse_attn_indexer"
            )

            text = log.getvalue()
            needles = ("sparse_attn_indexer", "deepseek_v32", "glm_moe_dsa")
            for line in text.splitlines():
                if "failed to import OOT" not in line:
                    continue
                for needle in needles:
                    assert needle not in line, line
            """
        )
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_ilu_sparse_attn_indexer_fake_returns_mutated_buffer(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)
    buffer = torch.empty((2, 4), dtype=torch.int32)

    result = sparse_attn_indexer.ilu_sparse_attn_indexer_fake(
        hidden_states=torch.empty((2, 8)),
        k_cache_prefix="layer.0.indexer",
        kv_cache=torch.empty((1, 1, 8)),
        q=torch.empty((2, 1, 8)),
        k=torch.empty((2, 8)),
        weights=torch.empty((2,)),
        topk_tokens=4,
        head_dim=8,
        max_model_len=16,
        total_seq_lens=16,
        topk_indices_buffer=buffer,
    )

    assert result is buffer


def test_ilu_sparse_attn_indexer_fake_requires_output_buffer(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)

    with pytest.raises(AssertionError):
        sparse_attn_indexer.ilu_sparse_attn_indexer_fake(
            hidden_states=torch.empty((2, 8)),
            k_cache_prefix="layer.0.indexer",
            kv_cache=torch.empty((1, 1, 8)),
            q=torch.empty((2, 1, 8)),
            k=torch.empty((2, 8)),
            weights=torch.empty((2,)),
            topk_tokens=4,
            head_dim=8,
            max_model_len=16,
            total_seq_lens=16,
            topk_indices_buffer=None,
        )


@pytest.mark.parametrize(
    "constructor_skip,call_skip", [(False, False), (True, False), (False, True)]
)
def test_ilu_sparse_attn_indexer_forward_cuda_calls_registered_op(
    monkeypatch,
    default_vllm_config,
    constructor_skip,
    call_skip,
):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)
    calls = []
    output = torch.empty((2, 4), dtype=torch.int32)

    def fake_op(*args):
        calls.append(args)
        return output

    k_cache = SimpleNamespace(
        prefix="model.layers.0.self_attn.indexer",
        kv_cache=object(),
    )
    topk_indices_buffer = torch.empty((2, 4), dtype=torch.int32)
    layer = sparse_attn_indexer.IluSparseAttnIndexer(
        k_cache=k_cache,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=8,
        max_model_len=16,
        max_total_seq_len=32,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=constructor_skip,
    )

    hidden_states = torch.empty((2, 8))
    q = torch.empty((2, 1, 8))
    k = torch.empty((2, 8))
    weights = torch.empty((2,))

    with patch.object(
        torch.ops.vllm, sparse_attn_indexer._PLUGIN_INDEXER_OP, fake_op, create=True
    ):
        result = layer.forward_cuda(
            hidden_states, q, k, weights, skip_k_cache_insert=call_skip
        )

    assert result is output
    assert len(calls) == 1
    args = calls[0]
    assert args[0] is hidden_states
    assert args[1] == k_cache.prefix
    assert args[2] is k_cache.kv_cache
    assert args[3] is q
    assert args[4] is k
    assert args[5] is weights
    assert args[6:10] == (4, 8, 16, 32)
    assert args[10] is topk_indices_buffer
    assert args[11] is (constructor_skip or call_skip)
