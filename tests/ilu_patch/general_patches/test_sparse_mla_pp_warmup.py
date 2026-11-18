# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as warmup

from vllm_iluvatar.ilu_patch.general_patches import patch_sparse_mla_pp_warmup as patch


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ILU_DEEPSEEK_SPARSE_SWA", True),
        ("ILU_DEEPSEEK_SPARSE_SWA_FP8", True),
        ("FLASHMLA_SPARSE_DSV4_FP8", True),
        ("DEEPSEEK_SPARSE_SWA", True),
        ("UNRELATED_BACKEND", False),
    ],
)
def test_local_stage_selects_sparse_warmup(monkeypatch, name, expected):
    monkeypatch.setattr(
        warmup, "_DEEPSEEK_V4_SPARSE_MLA_BACKENDS", frozenset({"DEEPSEEK_SPARSE_SWA"})
    )
    patch._apply()
    group = SimpleNamespace(backend=SimpleNamespace(get_name=lambda: name))
    runner = SimpleNamespace(attn_groups=[[group]])
    assert warmup._has_deepseek_v4_sparse_mla_backend(runner) is expected
