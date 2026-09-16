# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar early compat shims.

These are corex/torch-environment compat shims (NOT vLLM logic) that must run
before any deep ``import vllm`` touches them.  They are applied via two paths:

  1. Main process: ``vllm_iluvatar.register_platform()`` calls ``apply()``
     directly (at ``current_platform`` first access, before parallel_state is
     imported) and sets ``VLLM_ILUVATAR_ENABLE_STARTUP_PATCH=1`` so children
     inherit it.
  2. Spawned worker subprocess: the installed ``vllm_iluvatar.pth`` runs at
     interpreter startup; it is inert unless the env above is set, so it only
     fires for vLLM worker processes (never for unrelated ``python`` calls).

Shims (no-ops on stock CUDA torch):
  1. torch.accelerator — corex torch has a partial/missing module; map the
     subset vLLM uses onto torch.cuda equivalents.
  2. symmetric-memory guard — corex torch ships an INCOMPLETE
     ``torch.distributed._symmetric_memory`` (importable, but missing
     ``empty``).  Community ``device_communicators/symm_mem.py`` only guards
     ``ImportError`` (not the missing attribute), so it flips
     ``symm_mem_available=True`` and later crashes at ``torch_symm_mem.empty``
     during GroupCoordinator init.  We pre-inject a stub and force-disable
     symmetric-memory allreduce via ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``.
  3. torch.fx._graph_pickler — the CoreX Torch build used here ships
     GraphPickler but not Options. Community ``vllm.compilation.caching``
     imports both at module scope, so patch the missing Options symbol before
     vLLM imports it.
  4. torch._inductor.config / torch._functorch.config — some CoreX Torch builds
     lack config keys that upstream vLLM patches or serializes during compile
     setup/cache-key creation, so add compatible no-op defaults before vLLM
     builds the config.
  5. Triton Gluon import surface — community ``vllm.triton_utils`` imports
     ``triton.experimental.gluon`` and ``triton.language.core._aggregate`` at
     module scope. Older CoreX Triton builds do not ship those symbols, so add
     import-only stand-ins before ``vllm.triton_utils`` is imported.
  6. torch.ops._C compile-only quant schemas — compilation fusion modules bind
     several FP8 quant op overloads at import time. Register missing schemas as
     real dispatcher ops so FX matchers can import even when CoreX does not ship
     these CUDA kernels.
  7. MiniMax-M3 NVIDIA model entry points — importing
     ``vllm.models.minimax_m3.common`` first executes the parent package, whose
     CUDA branch imports the NVIDIA model and MTP implementations. Provide
     import-only stand-ins for those entry points so Iluvatar code can import
     the hardware-neutral ``common`` package directly.
Then installs module *stand-in* shims (``_install_import_shims`` ->
``import_shims.install_all``): whole community modules that would otherwise be
imported at *import time* and fail on corex are replaced in ``sys.modules`` with
ixformer-backed equivalents (e.g. ``vllm.vllm_flash_attn``, imported by the
community fa_utils ``is_cuda()`` branch since IluPlatform is cuda-like).  Each
stand-in mirrors the upstream module name under ``import_shims/``.

Finally registers post-import splat hooks (``_register_import_hooks`` ->
``patch_custom_ops``): ILU overrides for community modules whose functions
are called at *import time* (e.g. ``vllm._custom_ops`` used by ``w8a8_utils``
while it is imported in a spawned worker).  Registration is cheap — the target
modules are not imported here.

Env injection (``ilu_envs.apply()``) is intentionally NOT done here; it needs
``vllm.envs`` and is handled by ``register_platform()`` instead.
"""

import os
import sys
import types

_applied = False
_torch_c_schema_libraries: list[object] = []

_TORCH_C_COMPILE_SCHEMA_OPS = {
    "static_scaled_fp8_quant": (
        "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
        "int[]? group_shape=None) -> ()"
    ),
    "dynamic_scaled_fp8_quant": (
        "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) -> ()"
    ),
    "dynamic_per_token_scaled_fp8_quant": (
        "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
        "Tensor! scale, Tensor? scale_ub) -> ()"
    ),
    "rms_norm_static_fp8_quant": (
        "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
        "Tensor scale, float epsilon) -> ()"
    ),
    "fused_add_rms_norm": (
        "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
        "float epsilon) -> ()"
    ),
    "fused_add_rms_norm_static_fp8_quant": (
        "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
        "Tensor! residual, Tensor weight, Tensor scale, float epsilon) -> ()"
    ),
    "rms_norm_dynamic_per_token_quant": (
        "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
        "Tensor weight, Tensor! scale, float epsilon, Tensor? scale_ub, "
        "Tensor!? residual) -> ()"
    ),
    "rms_norm_per_block_quant": (
        "rms_norm_per_block_quant(Tensor! result, Tensor input, Tensor weight, "
        "Tensor! scale, float epsilon, Tensor? scale_ub, Tensor!? residual, "
        "int group_size, bool is_scale_transposed) -> ()"
    ),
    "fused_qk_norm_rope": (
        "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, "
        "int num_heads_v, int head_dim, float eps, Tensor q_weight, "
        "Tensor k_weight, Tensor cos_sin_cache, bool is_neox, "
        "Tensor position_ids, int forced_token_heads_per_warp=-1) -> ()"
    ),
    "scaled_fp4_quant": (
        "scaled_fp4_quant(Tensor input, Tensor input_scale, "
        "bool is_sf_swizzled_layout) -> (Tensor, Tensor)"
    ),
    "scaled_fp4_quant.out": (
        "scaled_fp4_quant.out(Tensor input, Tensor input_scale, "
        "bool is_sf_swizzled_layout, *, Tensor(a!) output, "
        "Tensor(b!) output_scale) -> ()"
    ),
    "silu_and_mul_quant": (
        "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()"
    ),
    "silu_and_mul": "silu_and_mul(Tensor! result, Tensor input) -> ()",
    "silu_and_mul_nvfp4_quant": (
        "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
        "Tensor input, Tensor input_global_scale) -> ()"
    ),
    "silu_and_mul_per_block_quant": (
        "silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, "
        "int group_size, Tensor? scale_ub=None, "
        "bool is_scale_transposed=False) -> ()"
    ),
}


def _has_torch_c_op(torch_ops_c: object, name: str) -> bool:
    base_name, sep, overload_name = name.partition(".")
    if not hasattr(torch_ops_c, base_name):
        return False
    if not sep:
        return True
    return hasattr(getattr(torch_ops_c, base_name), overload_name)


def _patch_torch() -> None:
    import torch

    if getattr(torch._C, "_accelerator_setAllocatorSettings", None) is None:
        torch._C._accelerator_setAllocatorSettings = lambda *args, **kwargs: None

    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = "float4_e2m1fn_x2"

    _patch_torch_fx_graph_pickler()
    _patch_torch_inductor_config()
    _patch_torch_functorch_config()

    if not hasattr(torch, "accelerator"):
        torch.accelerator = types.ModuleType("torch.accelerator")
    target_module = torch.accelerator
    patch_map = {
        "empty_cache": torch.cuda.empty_cache,
        "device_count": torch.cuda.device_count,
        "set_device_index": torch.cuda.set_device,
        "current_device_index": torch.cuda.current_device,
        "get_memory_info": torch.cuda.mem_get_info,
        "memory_stats": torch.cuda.memory_stats,
        "memory_reserved": torch.cuda.memory_reserved,
        "max_memory_allocated": torch.cuda.max_memory_allocated,
        "reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
        "synchronize": torch.cuda.synchronize,
        "device_index": torch.cuda.device,
    }
    for name, func in patch_map.items():
        if not hasattr(target_module, name):
            setattr(target_module, name, func)


def _ensure_graph_pickler_options(module: types.ModuleType) -> None:
    if hasattr(module, "Options"):
        return

    class Options:
        def __init__(self, ops_filter=None):
            self.ops_filter = ops_filter

    module.Options = Options
    existing_all = getattr(module, "__all__", None)
    if isinstance(existing_all, list) and "Options" not in existing_all:
        existing_all.append("Options")


def _patch_torch_fx_graph_pickler() -> None:
    try:
        import torch.fx._graph_pickler as graph_pickler
    except Exception:
        return
    _ensure_graph_pickler_options(graph_pickler)


def _ensure_torch_config_key(
    config_module: types.ModuleType,
    key: str,
    default: object,
    value_type: type,
) -> bool:
    config = getattr(config_module, "_config", None)
    if not isinstance(config, dict) or key in config:
        return False

    try:
        from torch.utils._config_module import Config
    except Exception:
        return False

    for entry in config.values():
        entry_type = type(entry)
        try:
            config[key] = entry_type(Config(default, value_type=value_type))
            return True
        except Exception:
            continue
    return False


def _patch_torch_inductor_config() -> None:
    """Add CoreX-missing Inductor config keys consumed by upstream vLLM.

    Community ``CompilationConfig`` may pass ``alignment_asserts`` to
    ``torch._inductor.config.patch``. Some CoreX Torch builds lack that key, so
    patching would raise before a late general patch can help. Keep this as an
    early, defensive Torch-runtime compatibility shim.
    """
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return

    _ensure_torch_config_key(
        inductor_config,
        "alignment_asserts",
        False,
        bool,
    )


def _patch_torch_functorch_config() -> None:
    """Add CoreX-missing Functorch config keys consumed by upstream vLLM.

    Community ``compiler_interface`` patches and serializes functorch config
    keys while computing compile config and cache keys. Some CoreX Torch builds
    lack those keys, so add defaults early instead of overriding vLLM's compiler
    helpers with a late general patch.
    """
    try:
        import torch._functorch.config as functorch_config
    except Exception:
        return

    for key in (
        "bundled_autograd_cache",
        "autograd_cache_normalize_inputs",
    ):
        _ensure_torch_config_key(functorch_config, key, False, bool)


def _define_missing_torch_c_compile_schema_ops() -> list[str]:
    """Register missing compile-time ``torch.ops._C`` schemas.

    These ops are used as FX pattern/replacement targets by upstream compilation
    passes. They must be real dispatcher overloads, not Python stand-ins, because
    matcher tables type them as ``OpOverload``. This intentionally defines only
    schemas for ops still missing after ``vllm_iluvatar._C`` has loaded; if a
    future native implementation registers an op, this shim skips it.
    """
    try:
        import torch
        from torch.library import Library
    except Exception:
        return []

    missing = [
        (name, schema)
        for name, schema in _TORCH_C_COMPILE_SCHEMA_OPS.items()
        if not _has_torch_c_op(torch.ops._C, name)
    ]
    if not missing:
        return []

    try:
        lib = Library("_C", "FRAGMENT")
    except Exception:
        return []

    registered: list[str] = []
    for name, schema in missing:
        try:
            lib.define(schema)
        except Exception:
            continue
        registered.append(name)

    if registered:
        _torch_c_schema_libraries.append(lib)
    return registered


def _stub_symmetric_memory() -> None:
    """Pre-inject a stub for ``torch.distributed._symmetric_memory``.

    On corex torch this module is either missing or incomplete (importable
    but lacks ``empty``).  Community ``parallel_state.py`` has a bare
    ``import torch.distributed._symmetric_memory`` at module scope that would
    fail with ``ImportError`` on corex.  By inserting a lightweight stub into
    ``sys.modules`` early, the import succeeds harmlessly and
    ``_disable_unsupported_symmetric_memory`` (below) ensures the actual usage
    is gated off via ``VLLM_ALLREDUCE_USE_SYMM_MEM=0``.
    """
    import sys

    key = "torch.distributed._symmetric_memory"
    module = sys.modules.setdefault(key, types.ModuleType(key))
    if not hasattr(module, "enable_symm_mem_for_group"):
        module.enable_symm_mem_for_group = lambda *args, **kwargs: None


def _disable_unsupported_symmetric_memory() -> None:
    try:
        import torch.distributed._symmetric_memory as torch_symm_mem
    except Exception:
        torch_symm_mem = None
    if torch_symm_mem is None or not hasattr(torch_symm_mem, "empty"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"


def _patch_triton_gdc_symbols() -> None:
    """Provide no-op Triton GDC symbols for CoreX AST dependency scanning.

    vLLM's LoRA Triton kernels reference ``tl.extra.cuda.gdc_*`` inside
    constexpr-gated branches. Even when PDL/GDC is disabled at runtime, Triton
    still resolves those helper symbols while building the JIT cache key. CoreX
    Triton does not expose them, so install no-op helpers early enough for the
    dependency scan to succeed.
    """
    try:
        import triton
        import triton.language as tl
    except Exception:
        return

    extra = getattr(tl, "extra", None)
    if extra is None:
        return

    cuda_extra = getattr(extra, "cuda", None)
    if cuda_extra is None:
        cuda_extra = types.SimpleNamespace()
        extra.cuda = cuda_extra

    if not hasattr(cuda_extra, "gdc_wait"):

        @triton.jit
        def _ilu_gdc_wait():
            pass

        cuda_extra.gdc_wait = _ilu_gdc_wait

    if not hasattr(cuda_extra, "gdc_launch_dependents"):

        @triton.jit
        def _ilu_gdc_launch_dependents():
            pass

        cuda_extra.gdc_launch_dependents = _ilu_gdc_launch_dependents


class _IluGluonPlaceholder:
    """Import-only placeholder for unavailable Triton Gluon symbols."""

    def __getattr__(self, _name: str):
        return self

    def __call__(self, *args, **kwargs):
        del kwargs
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return self


def _identity_aggregate(obj=None, *args, **kwargs):
    del args, kwargs
    if obj is None:
        return lambda wrapped: wrapped
    return obj


def _patch_triton_gluon_symbols() -> None:
    """Provide import-time Triton Gluon symbols missing in older CoreX Triton.

    These stand-ins are intentionally minimal. They let community modules that
    merely import ``vllm.triton_utils`` start up; they are not a real Gluon
    backend implementation.
    """
    try:
        import triton  # noqa: F401
        import triton.language.core as tl_core
    except Exception:
        return

    if not hasattr(tl_core, "_aggregate"):
        tl_core._aggregate = _identity_aggregate

    try:
        from triton.experimental import gluon as _gluon  # noqa: F401
        from triton.experimental.gluon import language as _gl  # noqa: F401
        return
    except Exception:
        pass

    experimental = sys.modules.setdefault(
        "triton.experimental", types.ModuleType("triton.experimental")
    )
    gluon_module = sys.modules.setdefault(
        "triton.experimental.gluon", types.ModuleType("triton.experimental.gluon")
    )
    language_module = sys.modules.setdefault(
        "triton.experimental.gluon.language",
        types.ModuleType("triton.experimental.gluon.language"),
    )
    for module in (experimental, gluon_module, language_module):
        module.__file__ = "<vllm_iluvatar.triton_gluon_shim>"

    placeholder = _IluGluonPlaceholder()
    for name in ("jit", "constexpr_function"):
        if not hasattr(gluon_module, name):
            setattr(gluon_module, name, _identity_aggregate)
    if not hasattr(gluon_module, "language"):
        gluon_module.language = language_module

    def _language_getattr(_name: str):
        if _name.startswith("__"):
            raise AttributeError(_name)
        return placeholder

    language_module.__getattr__ = _language_getattr
    if not hasattr(language_module, "constexpr"):
        language_module.constexpr = placeholder

    if not hasattr(experimental, "gluon"):
        experimental.gluon = gluon_module


class _IluMiniMaxM3NvidiaPlaceholder:
    """Import-only placeholder for unavailable MiniMax-M3 NVIDIA symbols."""

    def __getattr__(self, _name: str):
        return self

    def __call__(self, *args, **kwargs):
        del kwargs
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return self



def _patch_minimax_m3_nvidia_entrypoints() -> None:
    """Provide import-only MiniMax-M3 NV entry points when NV imports fail.

    Python executes ``vllm.models.minimax_m3.__init__`` before importing a
    child such as ``common.mm_preprocess``. On CUDA-like Iluvatar platforms
    that initializer selects ``nvidia.model`` and ``nvidia.mtp``. Their real
    implementations are not used by the OOT Iluvatar model and may fail while
    importing NVIDIA-only dependencies, so expose only the names required by
    the parent initializer.

    Follow the same pattern as :func:`_patch_triton_gluon_symbols`: use the
    upstream modules when they import successfully, otherwise install minimal
    stand-ins in ``sys.modules``.
    """

    try:
        from vllm.models.minimax_m3.nvidia import model as _nv_model  # noqa: F401
        from vllm.models.minimax_m3.nvidia import mtp as _nv_mtp  # noqa: F401

        return
    except Exception:
        # A failed parent-package import can leave partially initialized modules.
        # Remove only the NV leaves we replace; the package itself is retried
        # normally when ``vllm.models.minimax_m3.common`` is imported.
        sys.modules.pop("vllm.models.minimax_m3.nvidia.model", None)
        sys.modules.pop("vllm.models.minimax_m3.nvidia.mtp", None)
        sys.modules.pop("vllm.models.minimax_m3", None)

    model_module = sys.modules.setdefault(
        "vllm.models.minimax_m3.nvidia.model",
        types.ModuleType("vllm.models.minimax_m3.nvidia.model"),
    )
    mtp_module = sys.modules.setdefault(
        "vllm.models.minimax_m3.nvidia.mtp",
        types.ModuleType("vllm.models.minimax_m3.nvidia.mtp"),
    )
    for module in (model_module, mtp_module):
        module.__file__ = "<vllm_iluvatar.minimax_m3_nvidia_entrypoint_shim>"

    # Export the placeholder *class* (not an instance): community warmup does
    # ``isinstance(module, MiniMaxM3SparseAttention)``; an instance would raise.
    placeholder = _IluMiniMaxM3NvidiaPlaceholder
    for name in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "MiniMaxM3SparseAttention",
    ):
        # Always (re)bind: a prior empty/partial shim in sys.modules must not
        # leave required warmup symbols missing.
        setattr(model_module, name, placeholder)
    mtp_module.MiniMaxM3MTP = placeholder

    def _model_getattr(_name: str):
        if _name.startswith("__"):
            raise AttributeError(_name)
        return placeholder

    model_module.__getattr__ = _model_getattr
    mtp_module.__getattr__ = _model_getattr


def _install_import_shims() -> None:
    # Inject module *stand-in* shims into sys.modules BEFORE any deep
    # ``import vllm``.  Unlike the attribute-level shims above, these replace a
    # whole community module that would otherwise be imported at import-time and
    # fail / pull NV-specific code on corex (e.g. vllm.vllm_flash_attn, imported
    # by the community fa_utils ``is_cuda()`` branch — IluPlatform is cuda-like,
    # so that branch runs).  Each stand-in lives in import_shims/ mirroring the
    # upstream module name; install_all() auto-discovers and injects them.
    from vllm_iluvatar.ilu_patch.import_shims import install_all

    install_all()


def _register_import_hooks() -> None:
    # Register post-import splat hooks (e.g. vllm._custom_ops) so ILU overrides
    # apply the moment the target community module finishes importing — early
    # enough for import-time uses in spawned subprocesses, where general_plugins
    # would run too late.  This only REGISTERS the hooks; the target modules are
    # NOT imported here, so no heavy deps / platform resolution / ixformer are
    # forced.  Each patch_<target>.py calls import_hook.register(...) at import.
    import vllm_iluvatar.ilu_patch.patch_custom_ops  # noqa: F401
    import vllm_iluvatar.ilu_patch.patch_pynvml  # noqa: F401


def _load_native_torch_ops() -> None:
    # Import the native ``vllm_iluvatar._C`` extension so its TORCH_LIBRARY static
    # registrations populate ``torch.ops._C.{rms_norm, weak_ref_tensor,
    # get_cuda_view_from_cpu_tensor}`` BEFORE unmodified community code
    # (vllm/kernels/vllm_c.py, vllm/utils/torch_utils.py) first calls them.
    # Non-fatal in non-strict mode: a missing build degrades gracefully (the
    # community call would then raise with a clear torch error), and dev hosts
    # without the compiled extension still import the plugin.
    from vllm_iluvatar.ilu_util.torch_ops_c import load_native_ops

    load_native_ops(strict=False)


def apply() -> None:
    """Apply all early compat shims.  Idempotent and defensive."""
    global _applied
    if _applied:
        return
    _applied = True
    try:
        _patch_torch()
        _patch_triton_gluon_symbols()
        _patch_triton_gdc_symbols()
        _stub_symmetric_memory()
        _disable_unsupported_symmetric_memory()
        _patch_minimax_m3_nvidia_entrypoints()
        _install_import_shims()
        _register_import_hooks()
        _load_native_torch_ops()
        _define_missing_torch_c_compile_schema_ops()
    except Exception:
        # Compat shims must never break interpreter / vLLM startup.
        pass


# Apply on import. This module is loaded exactly once via
# ``vllm_iluvatar._load_import_compat_patch()`` (main process at
# register_platform time, and spawned workers via the installed .pth), so the
# early shims + import-hook registration run before any deep ``import vllm``.
apply()
