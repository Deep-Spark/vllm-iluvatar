# SPDX-License-Identifier: Apache-2.0
"""Producer and legacy attention must not require the optional INT8 reader."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("grouped", [False, True])
def test_legacy_forward_without_optional_reader(grouped):
    # A fresh interpreter also checks model registration: previously the
    # module-level type import prevented producers from loading their model.
    code = r'''
import importlib.abc
import sys

class RejectOptionalReader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ixformer.inference.functions.deepseek.quantized_sparse_mla_int8":
            raise ModuleNotFoundError("optional decode reader unavailable", name=fullname)
        return None

sys.meta_path.insert(0, RejectOptionalReader())
from vllm.plugins import load_general_plugins
load_general_plugins()
from types import SimpleNamespace
import torch
from vllm_iluvatar.models.deepseek_v41 import attention as module

layer = module.IluDeepseekV41Attention.__new__(module.IluDeepseekV41Attention)
torch.nn.Module.__init__(layer)
layer.n_local_heads, layer.head_dim = 2, 512
layer._grouped_int8_o_proj = sys.argv[1] == "True"
layer._run_parallel_input_projections = lambda x: (x, None, None)
layer._split_qkv_and_norm = lambda x: (x, x)
layer._prepare_and_attn = lambda *args: None
layer._uses_fused_inverse_rope = lambda: True
layer._o_proj = lambda x: x.flatten(1)
layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1, 64))
layer.wo_a = SimpleNamespace(quant_method=SimpleNamespace(
    apply_with_rope=lambda owner, x, *args: x))
layer.wo_b = lambda x: x
result = layer.forward(torch.zeros(1, dtype=torch.long), torch.zeros(1, 1024))
assert result.shape == (1, 1024)
assert "ixformer.inference.functions.deepseek.quantized_sparse_mla_int8" not in sys.modules
assert not torch.cuda.is_initialized()
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-B", "-c", code, str(grouped)],
                   env=env, check=True, capture_output=True, text=True, timeout=90)
