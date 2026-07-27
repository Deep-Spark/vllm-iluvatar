# SPDX-License-Identifier: Apache-2.0
"""Wiring + wrapper for skipping modular-kernel init on the NONE MXFP4 backend
(patch P3).

Corresponds to: vllm/model_executor/layers/fused_moe/runner/moe_runner.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner.
           maybe_init_modular_kernel  (``moe_runner.py:852``)
Baseline : main_9c12a81ac
Why      : EP-only crash observed on an 8-card TP8+EP8 run, on every rank, right
           after the weights loaded successfully:

               gpu_model_runner.py:5476 load_model
                 -> parallel_state.py:2029 prepare_communication_buffer_for_model
                   -> parallel_state.py:1218 (EP group)
                     -> base_device_communicator.py:339
                        module.maybe_init_modular_kernel()
                       -> moe_runner.py:857
                          ``or self.routed_experts.quant_method.is_monolithic``
                         -> fused_moe_method_base.py:167
                            ``return self.experts_cls.is_monolithic()``
               AttributeError: 'NoneType' object has no attribute 'is_monolithic'

           Root cause is the intended Stage 1 configuration, not a bug in the
           loader: P1 pins the MXFP4 MoE backend to ``Mxfp4MoeBackend.NONE``, so
           ``Mxfp4MoEMethod.process_weights_after_loading``
           (``quantization/mxfp4.py:735``) early-returns, ``_setup_kernel`` never
           runs, and both ``moe_kernel`` and ``experts_cls`` stay ``None``.
           Community ``FusedMoEMethodBase.is_monolithic``
           (``fused_moe_method_base.py:163-170``) only guards with
           ``hasattr(self, "experts_cls")`` — ``Mxfp4MoEMethod.__init__``
           (``quantization/mxfp4.py:484``) *does* set the attribute, to ``None``
           — so the ``hasattr`` guard passes and the attribute access explodes.

           This path is reached only under expert parallelism:
           ``BaseDeviceCommunicator.prepare_communication_buffer_for_model``
           returns immediately unless ``is_ep_communicator``
           (``base_device_communicator.py:331-339``). Pure TP8 never asks the MoE
           method for a modular kernel, which is why the 8-card EP8 run is the
           first place this surfaced.
How      : delegation wrapper over the class method (no method-body copy, so no
           source-fingerprint guard is needed — patch登记表 §2, tier 2).
           ``make_maybe_init_modular_kernel`` returns ``None`` without touching
           the quant method when, and only when, the routed-experts quant method
           is an *unwired* MXFP4 method (backend ``NONE`` + no ``experts_cls`` +
           no ``moe_kernel``); every other configuration, and every non-Iluvatar
           platform, delegates to the community implementation untouched.

           Skipping is the semantically correct Stage 1 answer, not a workaround:
           a modular kernel *is* a compute kernel, and Stage 1 wires none
           (requirement 5.5, design §2 C2). Returning early also avoids the rest
           of the community body — ``_ensure_moe_quant_config_init()`` and
           ``maybe_make_prepare_finalize()`` — which would build a MoE quant
           config for a kernel that will never exist. (With ``dp_size == 1`` that
           call chain happens to end in ``prepare_finalize is None``
           (``all2all_utils.py:139-141``), i.e. no modular method is installed
           anyway, so the early return changes no observable state beyond
           ``self.moe_quant_config``.)

           The wrapper is installed with ``setattr`` on the class that *defines*
           the method, resolved through the MRO rather than read off the module
           global: ``patch_fused_moe_layer`` rebinds
           ``moe_runner.MoERunner`` to ``IluMoERunner``, so the module attribute
           is order-dependent while the MRO walk is not. ``IluMoERunner``
           inherits the method, so wrapping the defining class covers both.
           ``rebind_module_func`` is *not* used: this is a class attribute, and
           instances resolve it through the class at call time.
Scope    : Iluvatar + routed-MXFP4 MoE layers whose backend is ``NONE``, i.e.
           DeepSeek-V4 FP8 Stage 1 weight loading under expert parallelism. Must
           be in place before ``load_model`` calls
           ``prepare_communication_buffer_for_model``; ``apply_general_patches()``
           runs from ``register_general`` inside ``create_engine_config``, well
           before that.
Remove   : together with P1 — when an Iluvatar MXFP4 MoE kernel lands and a real
           backend (hence a real ``experts_cls``) can be selected (Stage 2+).

Note on placement: unlike P1, the few lines of override live in this wiring
module instead of an upstream-mirrored module. The wrapper copies no community
code and has no counterpart worth diffing against ``moe_runner.py``; if it ever
grows, its mirrored home is
``vllm_iluvatar/model_executor/layers/fused_moe/runner/ilu_moe_runner.py``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

_CLASS = "MoERunner"
_METHOD = "maybe_init_modular_kernel"

# ``IluPlatform.device_name`` (vllm_iluvatar/platforms/ilu_platform.py:115).
_ILU_DEVICE_NAME = "iluvatar"


def _is_iluvatar_platform() -> bool:
    """True when running on the Iluvatar platform.

    Same gate as ``model_executor/layers/fused_moe/oracle/ilu_mxfp4.py`` (P1),
    duplicated as a module-level zero-arg predicate so it resolves as a module
    global at call time (monkeypatchable per module in tests). Deliberately not
    ``isinstance(current_platform, IluPlatform)`` — importing
    ``platforms/ilu_platform.py`` stubs ``vllm._C`` and touches
    ``torch.backends`` as an import side effect — and not
    ``current_platform.is_cuda()``, which is always true because ``IluPlatform``
    subclasses ``CudaPlatform`` and would therefore hijack real CUDA runs.
    """
    from vllm.platforms import current_platform

    return getattr(current_platform, "device_name", None) == _ILU_DEVICE_NAME


def _is_unwired_mxfp4_method(quant_method: Any) -> bool:
    """True for an MXFP4 MoE method that owns no kernel and never will.

    All three conditions are required, so the predicate cannot short-circuit a
    healthy layer:

    * ``moe_kernel is None``     — nothing was built by
      ``Mxfp4MoEMethod._setup_kernel`` (``quantization/mxfp4.py:718-725``);
    * ``mxfp4_backend is Mxfp4MoeBackend.NONE`` — the P1 pin. Any real backend,
      and any non-MXFP4 quant method (no such attribute), delegates;
    * ``experts_cls is None``    — the exact attribute community
      ``is_monolithic`` dereferences. If a backend ever yields an experts class,
      community code works and we stay out of the way.
    """
    if quant_method is None:
        return False
    if getattr(quant_method, "moe_kernel", None) is not None:
        return False
    backend = getattr(quant_method, "mxfp4_backend", None)
    if backend is None:
        return False

    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    if backend is not Mxfp4MoeBackend.NONE:
        return False
    return getattr(quant_method, "experts_cls", "missing") is None


def make_maybe_init_modular_kernel(
    orig: Callable[[Any], None],
) -> Callable[[Any], None]:
    """Wrap community ``MoERunner.maybe_init_modular_kernel``.

    Mirrors P1's ``make_select_deepseek_v4_mxfp4_moe_backend``: the factory takes
    the community callable and returns a ``functools.wraps``-decorated wrapper,
    so name / signature / ``__wrapped__`` still point at the community method
    after the ``setattr``.
    """

    @functools.wraps(orig)
    def maybe_init_modular_kernel(self) -> None:
        # [ILU override] MoERunner.maybe_init_modular_kernel — delta: build no
        # modular kernel for an unwired MXFP4 method (backend NONE), instead of
        # dereferencing ``experts_cls is None`` inside community
        # ``FusedMoEMethodBase.is_monolithic``.
        routed_experts = getattr(self, "routed_experts", None)
        quant_method = getattr(routed_experts, "quant_method", None)
        if _is_iluvatar_platform() and _is_unwired_mxfp4_method(quant_method):
            return None
        return orig(self)

    return maybe_init_modular_kernel


def _defining_class(cls: type, method: str) -> type:
    """The most-derived class in ``cls.__mro__`` that defines ``method``."""
    for klass in cls.__mro__:
        if method in vars(klass):
            return klass
    raise AttributeError(f"{cls.__name__} has no own definition of {method}")


def register() -> None:
    """Register the Iluvatar modular-kernel-init skip for the NONE MXFP4 backend.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=f"{_CLASS}.{_METHOD} -> skip on Mxfp4MoeBackend.NONE",
    )


def _apply() -> None:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as _m

    # Resolve through the MRO, not the module global: patch_fused_moe_layer may
    # already have rebound ``moe_runner.MoERunner`` to ``IluMoERunner``, which
    # inherits (does not define) this method.
    target_cls = _defining_class(getattr(_m, _CLASS), _METHOD)
    orig = vars(target_cls)[_METHOD]

    setattr(target_cls, _METHOD, make_maybe_init_modular_kernel(orig))
