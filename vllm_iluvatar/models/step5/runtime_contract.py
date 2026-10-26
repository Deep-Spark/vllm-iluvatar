"""Step5 startup checks; importing this module never loads CUDA or ixformer.

These checks describe the Step5 service, not support for other Iluvatar models.
CPU/Meta reference tests remain independent of the native extension. Backend
overrides and the native binding must remain unchanged after model construction.
"""

from vllm_iluvatar import envs

_FACADE_MODULES = {
    "rmsnorm": "ixformer.inference.functions.models.step5.norm.float.rmsnorm",
    "residual_add": "ixformer.inference.functions.models.step5.norm.float.residual_add",
    "qk_rmsnorm": "ixformer.inference.functions.models.step5.norm.float.qk_rmsnorm",
    "mtp_norm_concat": "ixformer.inference.functions.models.step5.mtp.float.mtp_norm_concat",
    "gemv_bf16_bf16_fp32": "ixformer.inference.functions.ops.gemm.float.gemv_bf16_bf16_fp32",
    "logits_reorder": "ixformer.inference.functions.models.step5.logits.float.logits_reorder",
    "moe_swiglu": "ixformer.inference.functions.models.step5.moe.float.moe_swiglu",
    "csa_cache_update": "ixformer.inference.functions.models.step5.cache.fp8.csa_cache_update",
    "indexer_norm_rope": "ixformer.inference.functions.models.step5.indexer.fp8.indexer_norm_rope",
    "indexer_scores": "ixformer.inference.functions.models.step5.indexer.fp8.indexer_scores",
    "top512": "ixformer.inference.functions.models.step5.indexer.float.top512",
    "top512_workspace_size": "ixformer.inference.functions.models.step5.indexer.float.top512",
    "paged_decode": "ixformer.inference.functions.models.step5.attention.float.paged_decode",
    "sparse_gqa_fp8": "ixformer.inference.functions.models.step5.attention.fp8.sparse_gqa_fp8",
    "SparseGQAWorkspace": "ixformer.inference.functions.models.step5.attention.fp8.sparse_gqa_fp8",
    "per_channel_gemv_bf16_fp8_bf16": "ixformer.inference.functions.ops.gemm.fp8.per_channel_gemv_bf16_fp8_bf16",
    "vocab_parallel_embedding": "ixformer.inference.functions.ops.miscs.vocab_parallel_embedding",
    "moe_compute_token_index_ep_group_layout": "ixformer.inference.functions.ops.moe.float.token_index_ep_group_layout",
    "sparse_gqa_fp8_workspace_size": "ixformer.inference.functions.models.step5.attention.fp8.sparse_gqa_fp8",
}


def _is_cuda(vllm_config):
    device = vllm_config.device_config
    return device.device_type == "cuda"


def _native_requirements(vllm_config, *, draft):
    """Union of Step5 APIs used by the configured model across its PP ranks."""
    hf = vllm_config.model_config.hf_config
    parallel = vllm_config.parallel_config
    start = hf.num_hidden_layers if draft else 0
    count = hf.num_nextn_predict_layers if draft else hf.num_hidden_layers
    types = getattr(hf, "layer_types", None)
    layer_types = [
        types[i] if types else "full_attention" for i in range(start, start + count)
    ]
    other = getattr(hf, "attention_other_setting", None) or {}
    has_d192 = any(
        (other["head_dim"] if kind == other.get("attention_type") else hf.head_dim)
        == 192
        for kind in layer_types
    )
    ops = {"rmsnorm", "vocab_parallel_embedding"}
    defaults = set()
    if hf.fp32_residual_connection:
        ops.add("residual_add")
    if has_d192:
        ops.update(("qk_rmsnorm", "paged_decode"))
        defaults.add("paged_decode")
    if parallel.tensor_parallel_size > 1:
        ops.add("logits_reorder")
    if draft:
        ops.add("mtp_norm_concat")
    else:
        # Target Step5 requires MoE; checkpoint exception layers can remain
        # unquantized even when the other experts use the online FP4 method.
        ops.update(
            (
                "residual_add",
                "gemv_bf16_bf16_fp32",
                "moe_swiglu",
            )
        )
        defaults.add("moe_swiglu")
    if hf.fp32_residual_connection and envs.VLLM_ILU_STEP5_FP32_TP:
        raise ValueError(
            "FP32 TP communication is unsupported; disable VLLM_ILU_STEP5_FP32_TP (legacy VLLM_STEP5_FP32_TP)"
        )
    if (
        not draft
        and parallel.data_parallel_size > 1
        and parallel.enable_expert_parallel
        and parallel.all2all_backend
        in ("naive", "allgather_reducescatter", "ag-rs", "ag_rs")
        and envs.VLLM_ILU_STEP5_FP32_EP
    ):
        raise ValueError(
            "FP32 EP communication is unsupported; disable VLLM_ILU_STEP5_FP32_EP (legacy VLLM_STEP5_FP32_EP)"
        )
    sparse = (
        "full_attention" in layer_types
        and (getattr(hf, "sparse_config", None) or {}).get("enabled", False)
        and envs.VLLM_ILU_STEP5_SPARSE_ATTENTION
    )
    queries = {}
    if sparse:
        if not envs.VLLM_ILU_STEP5_SPARSE_FP8:
            raise ValueError("This Step5 migration requires sparse FP8 full attention")
        ops.update(
            (
                "indexer_norm_rope",
                "indexer_scores",
                "csa_cache_update",
                "top512",
                "sparse_gqa_fp8",
            )
        )
        queries = {"top512_workspace_size": (1, 1, 2)}
    return sorted(ops), defaults, queries


def _cuda_arch():
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    return cuda_arch()


def _native_runtime():
    from importlib import import_module

    import ixformer._C as extension

    class Facades:
        def __getattr__(self, name):
            module = _FACADE_MODULES.get(name)
            if module is None:
                raise AttributeError(name)
            return getattr(import_module(module), name)

    functions = Facades()
    from ixformer.core import config
    from ixformer.inference.backend_registry import registry

    binding = extension.infer
    if config.IXFORMER_USE_TORCH_OPS:
        import torch

        binding = torch.ops._ixformer_C
    return functions, extension.infer, binding, registry


def validate_step5_runtime(vllm_config, *, draft=False):
    """Fail before model allocation; validate the supported model configuration."""
    if not getattr(vllm_config.model_config.hf_config, "zero_centered", True):
        raise ValueError("Step5 requires zero_centered=True")
    if not _is_cuda(vllm_config):
        return None
    role = "MTP draft" if draft else "target"
    ops, defaults, queries = _native_requirements(vllm_config, draft=draft)
    try:
        arch = _cuda_arch()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Step5 {role} requires an ivcore30 native extension: {exc}"
        ) from exc
    if arch != "ivcore30":
        raise RuntimeError(
            f"Step5 {role} GPU execution requires ivcore30; detected {arch!r}. "
            "Disabling sparse attention does not remove this requirement."
        )
    try:
        functions, infer, binding, registry = _native_runtime()
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            f"Step5 {role} native extension could not be loaded: {exc}"
        ) from exc
    if envs.VLLM_ILU_STEP5_SPARSE_FP8:
        if vllm_config.cache_config.cache_dtype != "auto":
            raise ValueError("Step5 sparse-only FP8 requires --kv-cache-dtype auto")
        # The composition is one standard native binding, not an experimental
        # module with separately exported producer/probe stages.
        if not callable(getattr(infer, "sparse_gqa_fp8", None)):
            raise RuntimeError("missing native Step5 FP8 composition")
        if envs.VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND == "ixattn":
            # Facades register lazily when accessed through functions.
            _ = functions.sparse_gqa_fp8
            available = getattr(infer, "sparse_gqa_fp8_ixattn_available", None)
            if (
                not callable(available)
                or not available()
                or registry.resolve("sparse_gqa_fp8", "ixattn") != "ixattn"
            ):
                raise RuntimeError("missing Step5 ixattn split backend")
    if not draft:
        _ = functions.per_channel_gemv_bf16_fp8_bf16
        if (
            registry.resolve("per_channel_gemv_bf16_fp8_bf16", "ixformer_perf")
            != "ixformer_perf"
        ):
            raise RuntimeError("Step5 fused BF16/FP8 GEMV backend unavailable")
    for op in ops:
        try:
            for label, namespace in (
                ("facade", functions),
                ("native export", infer),
                ("active binding", binding),
            ):
                if not callable(getattr(namespace, op, None)):
                    raise RuntimeError(f"missing {label}")
            if "ixformer_native" not in registry.available_backends(op):
                raise RuntimeError("native backend unavailable")
            selected = registry.resolve(op, "ixformer_native")
            if selected != "ixformer_native":
                raise RuntimeError(f"forced native resolved to {selected!r}")
            if op in defaults:
                selected = registry.resolve(op)
                if selected != "ixformer_native":
                    raise RuntimeError(f"default backend resolved to {selected!r}")
        except (AttributeError, KeyError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Step5 {role} requires {op}: {exc}") from exc
    for name, args in queries.items():
        try:
            if not callable(getattr(infer, name, None)):
                raise RuntimeError("missing native workspace export")
            query = getattr(functions, name, None)
            if not callable(query):
                raise RuntimeError("missing workspace facade")
            sizes = query(*args)
            if type(sizes) is not int or sizes < 0:
                raise RuntimeError(f"invalid workspace sizes {sizes!r}")
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Step5 {role} requires {name}: {exc}") from exc
    return None
