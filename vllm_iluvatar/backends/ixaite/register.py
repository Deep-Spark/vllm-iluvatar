# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Startup entry points moved from vllm_iluvatar/__init__.py at 3d7baeb.
"""Startup bindings owned by the ixaite integration."""

_general_patches_applied = False


# Install private paths before shared hooks can resolve a backend.
def register_platform() -> str:
    from ixformer.inference import functions

    # Shared attention metadata still imports this renamed vendor symbol.
    if not hasattr(functions, "flash_attn_varlen_fp8_func"):
        functions.flash_attn_varlen_fp8_func = functions.flash_attn_unpad_fp8

    from vllm_iluvatar.backends.ixaite.attention.register import (
        register_attention_backends,
    )

    register_attention_backends()
    from vllm_iluvatar.ilu_patch.patch_platform_import_hooks import (
        register_platform_import_hooks,
    )

    register_platform_import_hooks()
    return "vllm_iluvatar.backends.ixaite.platform.IxaitePlatform"


# Keep baseline helper decorators filtered throughout private initialization.
def register_general() -> None:
    from vllm_iluvatar.backends.ixaite.oot_registration import suppress_baseline_oot
    from vllm_iluvatar.ilu_register import apply_oot_registrations

    # Provider setup can import baseline helpers before explicit OOT discovery.
    with suppress_baseline_oot():
        from vllm_iluvatar.backends.ixaite.kv_transfer import (
            apply_kv_transfer_registrations,
        )

        complete_native_registration()
        apply_general_patches()
        configure_before_registration()
        apply_oot_registrations()
        configure_after_registration()
        apply_kv_transfer_registrations()


# Replace selected patches in place, before their baseline imports can bind kernels.
def apply_general_patches() -> None:
    import importlib
    import pkgutil

    from vllm_iluvatar.ilu_patch import general_patch as p

    global _general_patches_applied
    if _general_patches_applied:
        return

    replacements = {
        "patch_fused_moe_layer": (
            _apply_moe,
            "IluMoERunner (inherit MoERunner), IluSharedExperts",
        ),
        "patch_layer_utils": (
            _apply_linear,
            "layer utils default_unquantized_gemm override",
        ),
        "patch_sparse_mla_warmup": (
            _apply_warmup,
            "sparse_swa prefill metadata warmup kernel shim",
        ),
        "patch_stateless_coordinator": (_apply_stateless, "CudaCommunicator rebind"),
        "patch_kv_cache_utils": (
            _apply_kv_cache,
            "ixaite INT8 cache capacity and scales",
        ),
        "patch_ray_executor_v2": (
            _apply_ray_executor_v2,
            "RayExecutorV2 worker device-control env setup",
        ),
        "patch_grouped_topk_router": None,
    }
    package = importlib.import_module("vllm_iluvatar.ilu_patch.general_patches")
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.ispkg:
            continue
        name = info.name.rsplit(".", 1)[-1]
        if name in replacements:
            replacement = replacements[name]
            if replacement is not None:
                apply_fn, detail = replacement
                p.register_once(info.name, apply_fn, detail=detail)
            continue
        module = importlib.import_module(info.name)
        register = getattr(module, "register", None)
        if callable(register):
            register()

    # Import or registration errors propagate; a partial install is not complete.
    _general_patches_applied = True


def _apply_ray_executor_v2() -> None:
    import importlib

    try:
        importlib.import_module("vllm.v1.executor.ray_executor_v2")
    except (ImportError, NameError) as exc:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "Skipping RayExecutorV2 patch (ray executor unavailable: %s)", exc
        )
        return

    from vllm_iluvatar.ilu_patch.general_patches import patch_ray_executor_v2

    patch_ray_executor_v2._apply()


def _apply_kv_cache() -> None:
    """Keep shared planning hooks while selecting ixaite's cache policy."""
    import vllm.envs as envs
    from vllm.v1.core import kv_cache_utils as community

    from vllm_iluvatar.backends.ixaite.attention import kv_cache
    from vllm_iluvatar.ilu_patch import general_patch as p
    from vllm_iluvatar.ilu_patch.general_patches import patch_kv_cache_utils
    from vllm_iluvatar.v1.core import kv_cache_utils as common

    patch_kv_cache_utils._apply()
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE not in ("KINT8", "KVINT8"):
        return
    p.rebind_module_func(
        community,
        "get_kv_cache_config_from_groups",
        common.make_get_kv_cache_config_from_groups(
            kv_cache.get_kv_cache_config_from_groups
        ),
    )
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
        from vllm_iluvatar.v1.worker import gpu_model_runner

        # Only the selected runner alias changes; baseline's loader stays intact.
        gpu_model_runner.load_static_v_scales = kv_cache.load_ixaite_static_v_scales


# Bind the private runner while retaining the existing shared-expert implementation.
def _apply_moe() -> None:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as runner
    import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared

    from vllm_iluvatar.backends.ixaite.moe.runner import IluMoERunner
    from vllm_iluvatar.ilu_patch import general_patch as p
    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_shared_experts import (
        IluSharedExperts,
    )

    p.rebind_module_func(runner, "MoERunner", IluMoERunner)
    p.rebind_module_func(shared, "SharedExperts", IluSharedExperts)


# Keep signature checks and update copied GEMM references before weights are loaded.
def _apply_linear() -> None:
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.layers.utils as utils

    from vllm_iluvatar.backends.ixaite.layers import utils as impl
    from vllm_iluvatar.ilu_patch import general_patch as p

    p.check_signatures(utils, impl, ["default_unquantized_gemm"], strict=True)
    p.rebind_module_func(
        utils, "default_unquantized_gemm", impl.default_unquantized_gemm
    )
    linear.UnquantizedLinearMethod.process_weights_after_loading = (
        impl.process_unquantized_weights_after_loading
    )


# Preserve the warmup guards while installing only the ixaite kernel shim.
def _apply_warmup() -> None:
    try:
        from vllm.v1.attention.backends.mla import sparse_swa
    except ImportError:
        return

    name = "_compute_prefill_metadata_kernel"
    if hasattr(sparse_swa, name):
        return

    from vllm_iluvatar.backends.ixaite.attention.sparse_mla_warmup import (
        _IxaitePrefillMetadataWarmupKernel,
    )

    setattr(sparse_swa, name, _IxaitePrefillMetadataWarmupKernel())


# Match the platform's communicator without importing the frozen baseline patch.
def _apply_stateless() -> None:
    import vllm.distributed.stateless_coordinator as coordinator
    from vllm.logger import init_logger

    if not hasattr(coordinator, "CudaCommunicator"):
        init_logger(__name__).warning(
            "ILU stateless_coordinator patch: community module has no "
            "'CudaCommunicator' symbol; skipping rebind."
        )
        return

    from vllm_iluvatar.backends import resolve_device_communicator_cls

    coordinator.CudaCommunicator = resolve_device_communicator_cls()


# Platform discovery can run while torch imports; finish ixaite's compile setup later.
def complete_native_registration() -> None:
    import sys

    from vllm_iluvatar import _load_import_compat_patch
    from vllm_iluvatar.ilu_util.torch_ops_c import load_native_ops

    load_native_ops(strict=True)
    _load_import_compat_patch()
    sys.modules[
        "vllm_iluvatar._early_import_compat"
    ]._define_missing_torch_c_compile_schema_ops()


# Bind same-ABI callables before OOT models capture module-level references.
def configure_before_registration() -> None:
    from vllm_iluvatar.backends.ixaite.attention.flash_attn import (
        configure_flash_attn_provider,
    )
    from vllm_iluvatar.backends.ixaite.attention.kv_cache import (
        configure_int8_kv_cache_writer_providers,
    )
    from vllm_iluvatar.backends.ixaite.attention.vit import (
        configure_vit_qkv_layout_provider,
        configure_vit_qkv_rope_provider,
    )
    from vllm_iluvatar.backends.ixaite.layers.causal_conv1d import (
        configure_causal_conv1d_provider,
    )
    from vllm_iluvatar.backends.ixaite.models.qwen2_moe import (
        configure_shared_expert_gate_provider,
    )
    from vllm_iluvatar.backends.ixaite.models.qwen3_5 import (
        configure_attn_output_sigmoid_gate_provider,
        configure_fused_qk_provider,
        configure_video_evs_provider,
        configure_vision_gelu_provider,
    )

    configure_causal_conv1d_provider()
    configure_flash_attn_provider()
    configure_int8_kv_cache_writer_providers()
    configure_fused_qk_provider()
    configure_attn_output_sigmoid_gate_provider()
    configure_shared_expert_gate_provider()
    configure_vision_gelu_provider()
    configure_vit_qkv_layout_provider()
    configure_vit_qkv_rope_provider()
    configure_video_evs_provider()


# GDN classes register first; their stateful bindings must finish before loading.
def configure_after_registration() -> None:
    from vllm_iluvatar.backends.ixaite.layers.embedding import (
        configure_embedding_gather_provider,
    )
    from vllm_iluvatar.backends.ixaite.layers.gdn import (
        configure_gdn_chunk_provider,
        configure_gdn_packed_decode_provider,
        configure_gdn_post_conv_prep_provider,
        configure_gdn_spec_delta_update_provider,
        configure_gdn_strided_z_norm_provider,
    )

    configure_gdn_post_conv_prep_provider()
    configure_gdn_chunk_provider()
    configure_gdn_packed_decode_provider()
    configure_gdn_spec_delta_update_provider()
    configure_gdn_strided_z_norm_provider()
    configure_embedding_gather_provider()
    from vllm_iluvatar.backends.ixaite.patches.qwen2_moe_mlp import register

    register()
    from vllm_iluvatar.backends.ixaite.patches.mrope import register as register_mrope

    register_mrope()
    from vllm_iluvatar.backends.ixaite.patches.jit_monitor import (
        register as register_jit_monitor,
    )

    register_jit_monitor()
    from vllm_iluvatar.backends.ixaite.patches.sampler_greedy import (
        register as register_sampler_greedy,
    )

    register_sampler_greedy()
    from vllm_iluvatar.backends.ixaite.patches.block_table_slot_mapping import (
        register as register_slot_mapping,
    )

    register_slot_mapping()
