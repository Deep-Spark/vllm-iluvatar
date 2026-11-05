# SPDX-License-Identifier: Apache-2.0
"""Exercise the official registration boundary without constructing GPU layers."""

from types import SimpleNamespace

import pytest


# Baseline still marks completion before discovery, including a partial failure.
@pytest.mark.parametrize("failure", [False, True])
def test_public_baseline_registration_keeps_original_lifecycle(monkeypatch, failure):
    import vllm_iluvatar.ilu_register as registration

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "baseline")
    monkeypatch.setattr(registration, "_applied", False)
    calls = []

    # Discovery must see the original early completion flag even when it fails.
    def register_ops():
        assert registration._applied
        calls.append("ops")
        if failure:
            raise RuntimeError("discovery failed")

    monkeypatch.setattr(registration, "_register_ops", register_ops)
    monkeypatch.setattr(registration, "_register_models", lambda: calls.append("models"))
    monkeypatch.setattr(registration, "_validate_oot_overrides", lambda: calls.append("validate"))
    if failure:
        with pytest.raises(RuntimeError, match="discovery failed"):
            registration.apply_oot_registrations()
    else:
        registration.apply_oot_registrations()
    registration.apply_oot_registrations()
    assert calls == (["ops"] if failure else ["ops", "models", "validate"])
    assert registration._applied


# Direct public calls filter baseline decorators; nested startup contexts also restore.
# A failed private registration remains retryable without changing the baseline flag.
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_public_ixaite_registration_scope_and_retry(monkeypatch, nested, failure):
    import sys
    from contextlib import nullcontext
    from types import ModuleType

    import vllm.model_executor.custom_op as custom_op

    import vllm_iluvatar.ilu_register as baseline
    from vllm_iluvatar.backends.ixaite import oot_registration as registration

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(baseline, "_applied", True)
    monkeypatch.setattr(registration, "_applied", False)
    monkeypatch.setattr(
        registration, "_SELECTED_OP_MODULES",
        {"test.original": ("unused", (("ExampleBase", "IluExample"),))},
    )
    registry = {}
    monkeypatch.setattr(custom_op, "op_registry_oot", registry)
    owners = (custom_op.CustomOp, custom_op.PluggableLayer)
    saved = {owner: owner.__dict__["register_oot"] for owner in owners}
    base = type("ExampleBase", (custom_op.CustomOp,), {})
    original = type("IluExample", (base,), {"__module__": "test.original"})
    calls = []

    # Exercise an actual import-time decorator through the public entry's guard.
    def register_ops():
        calls.append("ops")
        assert base.register_oot(original) is original
        assert registry == {}

    # Fail after registration stages so cleanup and a complete retry are both checked.
    def validate():
        calls.append("validate")
        if failure and calls.count("validate") == 1:
            raise RuntimeError("validation failed")

    monkeypatch.setattr(registration, "_register_ops", register_ops)
    monkeypatch.setattr(registration, "_register_selected_ops", lambda: calls.append("selected"))
    monkeypatch.setattr(registration, "_register_models", lambda: calls.append("models"))
    monkeypatch.setattr(registration, "_validate_oot_overrides", validate)
    quant = ModuleType("vllm_iluvatar.backends.ixaite.quantization.config")
    quant.register_ixaite_quantization_config = lambda: calls.append("quant")
    monkeypatch.setitem(sys.modules, quant.__name__, quant)

    with registration.suppress_baseline_oot() if nested else nullcontext():
        outer = {owner: owner.__dict__["register_oot"] for owner in owners}
        if failure:
            with pytest.raises(RuntimeError, match="validation failed"):
                baseline.apply_oot_registrations()
            assert not registration._applied
            for owner, descriptor in outer.items():
                assert owner.__dict__["register_oot"] is descriptor
        baseline.apply_oot_registrations()
        baseline.apply_oot_registrations()
        for owner, descriptor in outer.items():
            assert owner.__dict__["register_oot"] is descriptor

    assert calls == ["ops", "selected", "models", "quant", "validate"] * (2 if failure else 1)
    assert registration._applied
    assert baseline._applied
    for owner, descriptor in saved.items():
        assert owner.__dict__["register_oot"] is descriptor


# A retry reuses the same class; a conflicting class must never overwrite it.
def test_single_registration_and_conflict(monkeypatch):
    import vllm.model_executor.custom_op as custom_op

    from vllm_iluvatar.backends.ixaite import oot_registration as registration

    registry, imports = {}, []
    monkeypatch.setattr(custom_op, "op_registry_oot", registry)
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")

    # Simulate the official registry's conflict check for the selected class.
    class Base:
        # Reject a second implementation instead of silently overwriting it.
        @classmethod
        def register_oot(cls, implementation):
            assert "Base" not in registry
            registry["Base"] = implementation

    # A retry must reuse this exact implementation object.
    class Implementation(Base):
        pass

    module = SimpleNamespace(Base=Base, Implementation=Implementation)
    monkeypatch.setattr(
        registration,
        "_SELECTED_OP_MODULES",
        {"test.original": ("test.selected", (("Base", "Implementation"),))},
    )

    # Track selected imports without requiring a GPU layer implementation.
    def import_module(name):
        imports.append(name)
        return module

    monkeypatch.setattr(registration.importlib, "import_module", import_module)
    registration._register_selected_ops()
    registration._register_selected_ops()
    assert registry == {"Base": Implementation}
    assert imports.count("vllm_iluvatar.backends.ixaite.test.selected") == 2
    registry["Base"] = object()
    with pytest.raises(RuntimeError, match="conflicting OOT"):
        registration._register_selected_ops()
    assert registry["Base"] is not Implementation


# Baseline keeps discovery's soft failure; ixaite requires a complete registration.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_selected_registration_import_failure(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    if mode == "ixaite":
        from vllm_iluvatar.backends.ixaite import oot_registration as registration

        monkeypatch.setattr(
            registration, "_SELECTED_OP_MODULES", {"missing": ("missing", ())}
        )
    else:
        import vllm_iluvatar.ilu_register as registration

    # Fail the selected import before it can install any classes.
    def import_module(name):
        raise ImportError("selected module unavailable")

    monkeypatch.setattr(registration.importlib, "import_module", import_module)
    if mode == "ixaite":
        with pytest.raises(ImportError, match="selected module"):
            registration._register_selected_ops()
    else:
        assert list(registration._iter_package_modules("missing")) == []


# Baseline discovery imports original decorators; ixaite skips their direct discovery.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_original_module_discovery_is_backend_specific(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    module_name = "test.original"
    if mode == "ixaite":
        from vllm_iluvatar.backends.ixaite import oot_registration as registration

        monkeypatch.setattr(
            registration, "_SELECTED_OP_MODULES", {module_name: ("selected", ())}
        )
    else:
        import vllm_iluvatar.ilu_register as registration
    monkeypatch.setattr(
        registration.pkgutil, "iter_modules",
        lambda *args: [SimpleNamespace(ispkg=False, name=module_name)],
    )
    imports = []
    module = SimpleNamespace()

    # Return a discoverable package and record whether its original layer is loaded.
    def import_module(name):
        imports.append(name)
        return SimpleNamespace(__path__=[], __name__="test") if name == "test" else module

    monkeypatch.setattr(registration.importlib, "import_module", import_module)
    discovered = list(registration._iter_package_modules("test"))
    assert discovered == ([module] if mode == "baseline" else [])
    assert (module_name in imports) == (mode == "baseline")


# Both official APIs must preserve decorator forms, metadata, and third-party conflicts.
@pytest.mark.parametrize("owner_name", ["CustomOp", "PluggableLayer"])
@pytest.mark.parametrize("form", ["bare", "parentheses", "named", "keyword"])
def test_scoped_oot_filter_preserves_registration(monkeypatch, owner_name, form):
    import inspect

    import vllm.model_executor.custom_op as custom_op

    from vllm_iluvatar.backends.ixaite import oot_registration as registration

    registry = {}
    monkeypatch.setattr(custom_op, "op_registry_oot", registry)
    monkeypatch.setattr(
        registration, "_SELECTED_OP_MODULES",
        {"test.original": ("selected", (("ExampleBase", "IluExample"),))},
    )
    owners = (custom_op.CustomOp, custom_op.PluggableLayer)
    saved = {owner: owner.__dict__["register_oot"] for owner in owners}
    owner = getattr(custom_op, owner_name)
    base = type("ExampleBase", (owner,), {})
    baseline = type("IluExample", (base,), {"__module__": "test.original"})
    own = type("IluExample", (base,), {"__module__": "test.selected"})
    third_party = type("IluExample", (base,), {"__module__": "test.third_party"})
    keyword = tuple(inspect.signature(saved[owner].__func__).parameters)[1]
    reg_name = "ExplicitName" if form == "named" else "ExampleBase"

    # Exercise the installed API's positional, decorator, and named-argument forms.
    def register(implementation):
        if form == "parentheses":
            return base.register_oot()(implementation)
        if form == "named":
            return base.register_oot(name=reg_name)(implementation)
        if form == "keyword":
            return base.register_oot(**{keyword: implementation})
        return base.register_oot(implementation)

    with registration.suppress_baseline_oot():
        assert register(baseline) is baseline
        assert baseline.name == reg_name
        assert registry == {}
        assert register(own) is own
        assert registry == {reg_name: own}
        with pytest.raises(AssertionError, match="Duplicate"):
            register(third_party)
        with pytest.raises(TypeError):
            base.register_oot(object())
        with pytest.raises(TypeError):
            base.register_oot(unknown_argument=True)
        base.register_oot(name="ThirdParty")(third_party)
        assert registry["ThirdParty"] is third_party

    for owner, descriptor in saved.items():
        assert owner.__dict__["register_oot"] is descriptor


# Restore both APIs when initialization fails before registration completes.
def test_scoped_oot_filter_restores_apis_on_failure():
    from vllm.model_executor.custom_op import CustomOp, PluggableLayer

    from vllm_iluvatar.backends.ixaite import oot_registration as registration

    saved = {
        owner: owner.__dict__["register_oot"]
        for owner in (CustomOp, PluggableLayer)
    }
    with (
        pytest.raises(RuntimeError, match="initialization failed"),
        registration.suppress_baseline_oot(),
    ):
        raise RuntimeError("initialization failed")
    for owner, descriptor in saved.items():
        assert owner.__dict__["register_oot"] is descriptor


# Required imports propagate their error instead of leaving an incomplete registry.
def test_required_registration_import_fails(monkeypatch):
    from vllm_iluvatar.backends.ixaite import oot_registration as registration

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")

    def import_module(name):
        raise ImportError("required module unavailable")

    monkeypatch.setattr(registration.importlib, "import_module", import_module)
    with pytest.raises(ImportError, match="required module"):
        list(registration._iter_package_modules("test.required"))


# Reinitialization must leave optional NIXL loaders lazy, even without NIXL installed.
def test_kv_registration_does_not_load_connectors_twice(monkeypatch):
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    from vllm_iluvatar.backends.ixaite import kv_transfer

    registry = {}
    monkeypatch.setattr(KVConnectorFactory, "_registry", registry)
    monkeypatch.setattr(kv_transfer, "_registered_loaders", {})
    kv_transfer.apply_kv_transfer_registrations()
    loaders = dict(registry)
    for name in registry:
        loader = Mock(side_effect=AssertionError("connector imported during startup"))
        registry[name] = loader
        kv_transfer._registered_loaders[name] = loader
    kv_transfer.apply_kv_transfer_registrations()
    assert registry.keys() == loaders.keys()
    for loader in registry.values():
        loader.assert_not_called()


# Baseline deliberately retains mainline's repeated-registration loader call.
def test_baseline_kv_registration_retains_loader_call(monkeypatch):
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    from vllm_iluvatar.distributed import kv_transfer

    name = kv_transfer._ILU_KV_CONNECTORS[0][0]
    loader = Mock(side_effect=RuntimeError("mainline loader invoked"))
    monkeypatch.setattr(KVConnectorFactory, "_registry", {name: loader})
    with pytest.raises(RuntimeError, match="mainline loader invoked"):
        kv_transfer.apply_kv_transfer_registrations()
    loader.assert_called_once_with()


# Duplicate target hints remain first-group in baseline and last-group in ixaite.
def test_quant_group_precedence_is_backend_specific():
    from copy import deepcopy

    from vllm_iluvatar.backends.ixaite.quantization.config import (
        IxaiteCompressedTensorsConfig,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        IluQuantConfig,
    )

    group = {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "strategy": "channel"},
        "input_activations": {
            "num_bits": 8, "type": "int", "strategy": "token", "dynamic": True,
        },
    }
    config = {
        "quant_method": "compressed-tensors", "format": "int-quantized",
        "config_groups": {
            "first": {**deepcopy(group), "iluvatar": {"gemm_format": "TN"}},
            "last": {**deepcopy(group), "iluvatar": {"gemm_format": "NN"}},
        },
    }
    baseline = IluQuantConfig.from_config(deepcopy(config))
    ixaite = IxaiteCompressedTensorsConfig.from_config(deepcopy(config))._ilu_config
    assert baseline.target_scheme_map["Linear"]["ilu_params"] == {"gemm_format": "TN"}
    assert ixaite.target_scheme_map["Linear"]["ilu_params"] == {"gemm_format": "NN"}
