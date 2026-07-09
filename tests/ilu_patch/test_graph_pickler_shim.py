# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for import compatibility shims."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from vllm_iluvatar.ilu_patch import patch_import_compat


def test_graph_pickler_options_is_added_to_existing_module():
    module = ModuleType("torch.fx._graph_pickler")
    module.GraphPickler = object
    module.__all__ = ["GraphPickler"]

    patch_import_compat._ensure_graph_pickler_options(module)

    assert module.__all__ == ["GraphPickler", "Options"]
    assert module.Options(ops_filter="ops").ops_filter == "ops"


def test_torch_config_key_is_added_from_existing_entry_type():
    @dataclass
    class FakeEntry:
        config: object

    module = ModuleType("torch._inductor.config")
    module._config = {"existing": FakeEntry(object())}

    added = patch_import_compat._ensure_torch_config_key(
        module,
        "alignment_asserts",
        False,
        bool,
    )

    assert added is True
    assert isinstance(module._config["alignment_asserts"], FakeEntry)
    assert module._config["alignment_asserts"].config.default is False
    assert module._config["alignment_asserts"].config.value_type is bool


def test_torch_config_key_helper_supports_functorch_keys():
    @dataclass
    class FakeEntry:
        config: object

    module = ModuleType("torch._functorch.config")
    module._config = {"existing": FakeEntry(object())}

    for key in ("bundled_autograd_cache", "autograd_cache_normalize_inputs"):
        added = patch_import_compat._ensure_torch_config_key(
            module,
            key,
            False,
            bool,
        )

        assert added is True
        assert isinstance(module._config[key], FakeEntry)
        assert module._config[key].config.default is False
        assert module._config[key].config.value_type is bool


def test_torch_config_key_is_not_overwritten():
    sentinel = object()
    module = ModuleType("torch._inductor.config")
    module._config = {"alignment_asserts": sentinel}

    added = patch_import_compat._ensure_torch_config_key(
        module,
        "alignment_asserts",
        False,
        bool,
    )

    assert added is False
    assert module._config["alignment_asserts"] is sentinel


def test_missing_torch_c_compile_schema_ops_are_dispatcher_overloads():
    import torch
    from torch._ops import OpOverload

    patch_import_compat._define_missing_torch_c_compile_schema_ops()

    assert isinstance(torch.ops._C.static_scaled_fp8_quant.default, OpOverload)
    assert isinstance(torch.ops._C.dynamic_scaled_fp8_quant.default, OpOverload)
    assert isinstance(
        torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
        OpOverload,
    )
    assert isinstance(torch.ops._C.silu_and_mul.default, OpOverload)
    assert isinstance(torch.ops._C.silu_and_mul_quant.default, OpOverload)
    assert isinstance(torch.ops._C.fused_add_rms_norm.default, OpOverload)
    assert isinstance(torch.ops._C.fused_qk_norm_rope.default, OpOverload)
    assert isinstance(torch.ops._C.scaled_fp4_quant.default, OpOverload)
    assert isinstance(torch.ops._C.scaled_fp4_quant.out, OpOverload)
