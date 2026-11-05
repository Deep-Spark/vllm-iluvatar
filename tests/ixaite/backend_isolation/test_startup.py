# SPDX-License-Identifier: Apache-2.0
"""Verify real startup objects in fresh processes, including absent optional libraries."""

import json
import os
import subprocess
import sys

import pytest


# Default and explicit baseline must construct the same objects without own imports.
@pytest.mark.parametrize("mode", [None, "baseline", "ixaite"])
def test_real_registration_and_constructed_classes(mode):
    from vllm_iluvatar.backends.ixaite.oot_registration import _SELECTED_OP_MODULES

    env = os.environ.copy()
    env.pop("VLLM_ILU_KERNEL_PROVIDER", None)
    if mode is not None:
        env["VLLM_ILU_KERNEL_PROVIDER"] = mode
    script = r"""
import importlib
import importlib.abc
import json
import sys

mode = sys.argv[1]
selected_op_modules = json.loads(sys.argv[2])
blocked = []

# A failed optional import attempt still violates baseline independence.
class NoOptionalBackend(importlib.abc.MetaPathFinder):
    # Record attempts before raising so caught imports still fail baseline independence.
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"ixaite", "flydsl"} or fullname.startswith(
            "vllm_iluvatar.backends.ixaite"
        ):
            blocked.append(fullname)
            raise ImportError("optional backend disabled by regression test")

if mode == "baseline":
    sys.meta_path.insert(0, NoOptionalBackend())

import vllm_iluvatar
platform_path = vllm_iluvatar.register_platform()
from vllm.model_executor.custom_op import CustomOp, PluggableLayer, op_registry_oot
saved_apis = {
    owner: owner.__dict__["register_oot"]
    for owner in (CustomOp, PluggableLayer)
}
vllm_iluvatar.register_general()
if mode == "ixaite":
    vllm_iluvatar.register_general()
# Repeating the public OOT entry must preserve the already selected objects.
from vllm_iluvatar.ilu_register import apply_oot_registrations
registered_ops = dict(op_registry_oot)
apply_oot_registrations()
assert op_registry_oot == registered_ops
for owner, descriptor in saved_apis.items():
    assert owner.__dict__["register_oot"] is descriptor
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm import ModelRegistry
from vllm_iluvatar.backends import (
    worker_class_path, communicator_class_path, resolve_device_communicator_cls,
)

# General patches must bind the same selected implementations as OOT discovery.
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.utils import default_unquantized_gemm
expected_runner = (
    "vllm_iluvatar.backends.ixaite.moe.runner" if mode == "ixaite"
    else "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner"
)
expected_utils = (
    "vllm_iluvatar.backends.ixaite.layers.utils" if mode == "ixaite"
    else "vllm_iluvatar.model_executor.layers.utils"
)
assert moe_runner.MoERunner.__module__ == expected_runner
assert default_unquantized_gemm.__module__ == expected_utils
assert UnquantizedLinearMethod.process_weights_after_loading.__module__ == expected_utils
if mode == "ixaite":
    for name in ("patch_fused_moe_layer", "patch_grouped_topk_router",
                 "patch_layer_utils", "patch_sparse_mla_warmup",
                 "patch_stateless_coordinator"):
        assert "vllm_iluvatar.ilu_patch.general_patches." + name not in sys.modules

# The stateless constructor and platform must use the same actual class.
from vllm.distributed import stateless_coordinator
assert stateless_coordinator.CudaCommunicator is resolve_device_communicator_cls()

prefix = ("vllm_iluvatar.backends.ixaite" if mode == "ixaite"
          else "vllm_iluvatar")
expected_norm = (prefix + ".layers.layernorm" if mode == "ixaite"
                 else prefix + ".model_executor.layers.layernorm")
with set_current_vllm_config(VllmConfig()):
    for norm_cls in (RMSNorm, GemmaRMSNorm):
        norm = norm_cls(64)
        assert type(norm).__module__ == expected_norm, type(norm)

# All selected layers must use the chosen backend, including conditional GDN.
from vllm import envs
for original, (own, pairs) in selected_op_modules.items():
    expected = prefix + "." + own if mode == "ixaite" else original
    for base_name, impl_name in pairs:
        if mode == "baseline" and (
            base_name == "RMSNormGated" or (
                base_name == "QwenGatedDeltaNetAttention"
                and not envs.VLLM_ILU_GDN_ATTN_BACKEND
            )
        ):
            assert base_name not in op_registry_oot, base_name
            continue
        implementation = op_registry_oot[base_name]
        assert implementation.__module__ == expected, (base_name, implementation)
        assert implementation.__name__ == impl_name
        assert implementation.name == base_name

# Actual helper imports remain cached; the private indexer avoids the legacy schema.
if mode == "ixaite":
    registry = dict(op_registry_oot)
    helpers = {
        "vllm_iluvatar.model_executor.layers.activation",
        "vllm_iluvatar.model_executor.layers.layernorm",
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn",
    }
    assert "vllm_iluvatar.model_executor.layers.sparse_attn_indexer" not in sys.modules
    for module_name in helpers:
        assert module_name in sys.modules, module_name
        importlib.import_module(module_name)
    assert op_registry_oot == registry
quant = get_quantization_config("compressed-tensors")
expected_quant = (prefix + ".quantization.config" if mode == "ixaite"
                  else prefix + ".model_executor.layers.quantization.compressed_tensors.compressed_tensors")
assert quant.__module__ == expected_quant, quant
for class_path in (platform_path, worker_class_path(), communicator_class_path()):
    module, name = class_path.rsplit(".", 1)
    implementation = getattr(importlib.import_module(module), name)
    assert implementation.__module__ == module, class_path
    assert class_path.startswith(prefix + "."), class_path
from tests.ixaite.backend_isolation.prefix_worker import PrefixCacheWorker
from tests.ixaite.backend_isolation.chunked_worker import ChunkedPrefillWorker
for worker in (PrefixCacheWorker, ChunkedPrefillWorker):
    assert worker.__mro__[1].__module__ == worker_class_path().rsplit(".", 1)[0]
model = ModelRegistry.models["Qwen3_5ForConditionalGeneration"]
expected_model = (prefix + ".models.qwen3_5" if mode == "ixaite"
                  else prefix + ".model_executor.models.qwen3_5")
assert model.module_name == expected_model, model
assert not blocked, blocked
if mode == "baseline":
    assert not any(name == "ixaite" or name.startswith(("ixaite.", "flydsl.",
                       "vllm_iluvatar.backends.ixaite")) for name in sys.modules)
print("real startup objects verified:", mode)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode or "baseline",
         json.dumps(_SELECTED_OP_MODULES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
