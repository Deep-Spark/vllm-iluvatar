# Source: ilu_register/__init__.py at 6f0942c; move backend adaptation out of frozen baseline code.
# SPDX-License-Identifier: Apache-2.0
"""Ixaite OOT registration and scoped filtering of baseline helper decorators.

Layer, model and quantization imports stay inside registration or discovery so
loading this module cannot register baseline helpers before filtering starts.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from contextlib import contextmanager
from functools import wraps

from vllm.logger import init_logger

from vllm_iluvatar.ilu_register import _OOT_MODEL_PACKAGES, _OOT_OP_PACKAGES

logger = init_logger(__name__)

_SELECTED_MODEL_MODULES = {
    "vllm_iluvatar.model_executor.models.qwen2_5_vl": "vllm_iluvatar.backends.ixaite.models.qwen2_5_vl",
    "vllm_iluvatar.model_executor.models.qwen3_5": "vllm_iluvatar.backends.ixaite.models.qwen3_5",
    "vllm_iluvatar.model_executor.models.pixtral": "vllm_iluvatar.backends.ixaite.models.pixtral",
    "vllm_iluvatar.models.deepseek_v4.model": "vllm_iluvatar.backends.ixaite.models.deepseek_v4.model",
    "vllm_iluvatar.models.deepseek_v4.attention": "vllm_iluvatar.backends.ixaite.models.deepseek_v4.attention",
    "vllm_iluvatar.models.deepseek_v4.mtp": "vllm_iluvatar.backends.ixaite.models.deepseek_v4.mtp",
    "vllm_iluvatar.models.deepseek_v4.dspark": "vllm_iluvatar.backends.ixaite.models.deepseek_v4.dspark",
    "vllm_iluvatar.model_executor.models.step3p5": "vllm_iluvatar.backends.ixaite.models.step3p5",
    "vllm_iluvatar.model_executor.models.step3p5_mtp": "vllm_iluvatar.backends.ixaite.models.step3p5_mtp",
    "vllm_iluvatar.models.minimax_m3.model": "vllm_iluvatar.backends.ixaite.models.minimax_m3.model",
}

__all__ = ["apply_oot_registrations", "suppress_baseline_oot"]

_applied = False


# Protect direct public-entry calls as well as calls inside the startup context.
def apply_oot_registrations() -> None:
    """Register the selected ixaite implementations once after successful setup."""
    global _applied
    if _applied:
        return
    with suppress_baseline_oot():
        _register_ops()
        _register_selected_ops()
        _register_models()
        from vllm_iluvatar.backends.ixaite.quantization.config import (
            register_ixaite_quantization_config,
        )

        register_ixaite_quantization_config()
        _validate_oot_overrides()
    # Partial registration remains retryable; existing class conflicts are checked.
    _applied = True


# Skip replaced modules before import, including their decorators and Torch schemas.
def _iter_package_modules(pkg_name: str):
    """Yield selected top-level modules; required import failures propagate."""
    pkg = importlib.import_module(pkg_name)
    if pkg_name in {
        "vllm_iluvatar.models.deepseek_v32",
        "vllm_iluvatar.models.glm_moe_dsa",
        "vllm_iluvatar.models.hy_v4",
        "vllm_iluvatar.models.qwen4_exp",
    }:
        yield importlib.import_module(
            "vllm_iluvatar.backends.ixaite.models.unsupported"
        )
        return
    for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if info.ispkg or info.name in _SELECTED_OP_MODULES:
            continue
        if info.name in {
            "vllm_iluvatar.models.deepseek_v4.fp8_attention",
            "vllm_iluvatar.models.deepseek_v4.fp8_model",
            "vllm_iluvatar.models.deepseek_v4.fp8_mtp",
            "vllm_iluvatar.models.deepseek_v4.sparse_swa",
        }:
            continue
        module_name = _SELECTED_MODEL_MODULES.get(info.name, info.name)
        try:
            yield importlib.import_module(module_name)
        except Exception as exc:
            # Optional models (e.g. Pixtral) may need a newer/older transformers
            # API than the host ships; skip them so MiniMax/Flash registration
            # still completes.
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Skipping OOT module %s (%s: %s)",
                module_name,
                type(exc).__name__,
                exc,
            )


_SELECTED_OP_MODULES = {
    "vllm_iluvatar.model_executor.layers.sparse_attn_indexer": (
        "layers.sparse_attn_indexer",
        (("SparseAttnIndexer", "IluSparseAttnIndexer"),),
    ),
    "vllm_iluvatar.model_executor.layers.fused_moe.ilu_routed_experts": (
        "moe.routed_experts",
        (("RoutedExperts", "IluRoutedExperts"),),
    ),
    "vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_gate_linear": (
        "moe.gate_linear",
        (("GateLinear", "IluGateLinear"),),
    ),
    "vllm_iluvatar.model_executor.layers.fused_moe.ilu_unquantized_fused_moe_method": (
        "moe.unquantized",
        (("UnquantizedFusedMoEMethod", "IluUnquantizedFusedMoEMethod"),),
    ),
    "vllm_iluvatar.model_executor.layers.activation": (
        "layers.activation",
        (
            ("SiluAndMul", "IluSiluAndMul"),
            ("SiluAndMulWithClamp", "IluSiluAndMulWithClamp"),
            ("SwigluStepAndMul", "IluSwigluStepAndMul"),
        ),
    ),
    "vllm_iluvatar.model_executor.layers.layernorm": (
        "layers.layernorm",
        (
            ("RMSNorm", "IluRMSNorm"),
            ("GemmaRMSNorm", "IluGemmaRMSNorm"),
            ("RMSNormGated", "IluRMSNormGated"),
        ),
    ),
    "vllm_iluvatar.model_executor.layers.attention.mm_encoder_attention": (
        "attention.mm_encoder_attention",
        (("MMEncoderAttention", "IluMMEncoderAttention"),),
    ),
    "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn": (
        "layers.gdn",
        (
            ("ChunkGatedDeltaRule", "IluChunkGatedDeltaRule"),
            ("QwenGatedDeltaNetAttention", "IluQwenGatedDeltaNetAttention"),
        ),
    ),
}


# Keep both register_oot signatures, including their different decorator keywords.
def _filtered_register_oot(original, blocked):
    signature = inspect.signature(original)
    decorated_arg = tuple(signature.parameters)[1]

    # Bind with the original signature so invalid calls retain argument validation.
    @wraps(original)
    def register(cls, *args, **kwargs):
        bound = signature.bind(cls, *args, **kwargs)
        decorated = bound.arguments.get(decorated_arg)
        name = bound.arguments.get("name")
        original_decorator = original(cls, name=name)

        # Preserve metadata even when an imported baseline class is not registered.
        def decorate(implementation):
            if isinstance(implementation, type) and (
                implementation.__module__, implementation.__name__
            ) in blocked:
                implementation.name = name if name is not None else cls.__name__
                return implementation
            return original_decorator(implementation)

        if decorated is None:
            return decorate
        if isinstance(decorated, type):
            return decorate(decorated)
        return original(cls, *args, **kwargs)

    return classmethod(register)


# Ixaite imports baseline helpers; filter their decorators only during startup.
@contextmanager
def suppress_baseline_oot():
    from vllm.model_executor.custom_op import CustomOp, PluggableLayer

    blocked = {
        (module, implementation)
        for module, (_, pairs) in _SELECTED_OP_MODULES.items()
        for _, implementation in pairs
    }
    saved = {
        owner: owner.__dict__["register_oot"]
        for owner in (CustomOp, PluggableLayer)
    }
    try:
        for owner, descriptor in saved.items():
            owner.register_oot = _filtered_register_oot(descriptor.__func__, blocked)
        # Imported helpers are already cached by Python. Importing unused baseline
        # modules here would also install their unrelated OOT or Torch schemas.
        yield
    finally:
        for owner, descriptor in saved.items():
            owner.register_oot = descriptor


# Use the official OOT API once, including retries after a failed initialization.
def _register_selected_ops() -> None:
    from vllm.model_executor.custom_op import op_registry_oot

    for ixaite_module, pairs in _SELECTED_OP_MODULES.values():
        module = importlib.import_module(
            f"vllm_iluvatar.backends.ixaite.{ixaite_module}"
        )
        for base_name, impl_name in pairs:
            base = getattr(module, base_name)
            implementation = getattr(module, impl_name)
            previous = op_registry_oot.get(base_name)
            if previous is implementation:
                continue
            if previous is not None:
                raise RuntimeError(f"conflicting OOT registration for {base_name}")
            base.register_oot(implementation)
    # Embedding has no Iluvatar baseline override.
    importlib.import_module("vllm_iluvatar.backends.ixaite.layers.embedding")


# Keep discovery limited to supported modules before explicit private registration.
def _register_ops() -> None:
    """Op OOT registration: import triggers @register_oot / @CustomOp.register."""
    for pkg_name in _OOT_OP_PACKAGES:
        for _module in _iter_package_modules(pkg_name):
            pass  # import side effect (decorators) is the registration


# Install the DS4 dispatcher first and avoid repeating mapped model hooks.
def _register_models() -> None:
    """Model registration: import then call each module's register_model() hook."""
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch import register_model

    register_model()
    seen = set()
    for pkg_name in _OOT_MODEL_PACKAGES:
        for module in _iter_package_modules(pkg_name):
            if module.__name__ in seen:
                continue
            seen.add(module.__name__)
            hook = getattr(module, "register_model", None)
            if callable(hook):
                hook()


# Validate private implementations with the existing strict signature contract.
def _validate_oot_overrides() -> None:
    """Run the prefix-compatible signature check on our OOT subclasses.

    Reuses the shared ``VLLM_ILU_PATCH_CHECK_SIGNATURE`` / ``VLLM_ILU_PATCH_VERBOSE``
    switches. No fingerprint check — OOT subclasses are replacements, not copies.
    """
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_iluvatar.ilu_patch.general_patch import (
        ENV_CHECK_SIGNATURE,
        ENV_VERBOSE,
        _enabled,
    )
    from vllm_iluvatar.ilu_util.signature import validate_oot_override

    if not _enabled(ENV_CHECK_SIGNATURE):
        logger.warning(
            "ILU: OOT signature check disabled via %s; skipping", ENV_CHECK_SIGNATURE
        )
        return

    prefixes = tuple(_OOT_OP_PACKAGES + _OOT_MODEL_PACKAGES) + (
        "vllm_iluvatar.backends.ixaite",
    )
    log = logger.info if _enabled(ENV_VERBOSE) else logger.debug

    for ilu_cls in list(op_registry_oot.values()):
        if not getattr(ilu_cls, "__module__", "").startswith(prefixes):
            continue
        # The immediate non-object base is the community class being overridden.
        base = next((b for b in ilu_cls.__mro__[1:] if b is not object), None)
        if base is None:
            continue
        for attr, value in list(vars(ilu_cls).items()):
            if not inspect.isfunction(value):
                continue
            if getattr(base, attr, None) is None:
                continue
            validate_oot_override(base, ilu_cls, attr, strict=True)
            log(
                "ILU OOT signature ok: %s.%s vs %s",
                ilu_cls.__name__,
                attr,
                base.__name__,
            )
