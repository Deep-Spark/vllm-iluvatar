# SPDX-License-Identifier: Apache-2.0
"""Map plugin-routed ixaite ops to ``python -m tests.ixaite.benchmarks`` suite names.

``tests/ixaite/benchmarks/test_plugin_op_benchmark_suites.py`` fails if a new
startup/lifecycle bindings and exported operator tables is
missing from this table or if the mapped suite is not registered.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PLUGIN_ROOT = _REPO_ROOT / "vllm_iluvatar"

# Plugin op_name (cached loader first arg, @_routed_op def, or
# _TRITON_OPS key) -> registered suite.name values. Names may differ
# (e.g. gdn_chunk).
PLUGIN_OP_SUITES: dict[str, tuple[str, ...]] = {
    "silu_and_mul": ("silu_and_mul",),
    "linear": ("linear",),
    "embedding_gather": ("embedding_gather",),
    "w8a8": ("w8a8",),
    "scaled_int8_quant": ("scaled_int8_quant", "scaled_int8_quant_asym"),
    "activation_dynamic_scaled_int8": ("activation_dynamic_scaled_int8",),
    "moe_w4a8_group_gemm": ("moe_w4a8_group_gemm",),
    "moe_w4a8_group_gemv": ("moe_w4a8_group_gemv",),
    "moe_compute_token_index": ("moe_compute_token_index",),
    "moe_compute_token_index_ep": ("moe_compute_token_index",),
    "moe_compute_token_index_for_decode": ("moe_compute_token_index",),
    "moe_expand_input_dynamic_scaled_int8": ("moe_expand_input_dynamic_scaled_int8",),
    "moe_fused_topk_bias": ("moe_fused_topk_bias",),
    "moe_output_reduce_sum": ("moe_output_reduce_sum",),
    "rms_norm_gated": ("rms_norm_gated",),
    "rms_norm": ("rms_norm",),
    "rms_norm_qk": ("rms_norm_qk",),
    "ds4_fused_qnorm_rope_kv_insert": ("ds4_fused_qnorm_rope_kv_insert",),
    "ds4_inv_rope": ("ds4_inv_rope",),
    "ds4_compressor_save_partial_states": ("ds4_compressor_save_partial_states",),
    "ds4_compressor_pool_norm_rope_insert": ("ds4_compressor_pool_norm_rope_insert",),
    "gemm_bf16_bf16_fp32": ("gemm_bf16_bf16_fp32",),
    "marlin_w8a16": ("marlin_w8a16",),
    "ds4_mhc_pre": ("ds4_mhc_pre",),
    "ds4_mhc_pre_v2": ("ds4_mhc_pre",),
    "ds4_mhc_pre_v3": ("ds4_mhc_pre",),
    "ds4_mhc_post": ("ds4_mhc_post",),
    "ds4_hc_head": ("ds4_hc_head",),
    "ds4_compute_global_topk_indices_and_lens": ("ds4_compute_global_topk",),
    "ds4_combine_topk_swa_indices": ("ds4_combine_topk_swa",),
    "ds4_compute_swa_decode_indices_and_lens": ("ds4_compute_swa_decode_indices",),
    "ds4_compute_swa_prefill_gather_lens": ("ds4_compute_swa_prefill_gather_lens",),
    "ds4_gather_k_cache": ("ds4_gather_k_cache",),
    "ds4_merge_attn_outputs": ("ds4_merge_attn_outputs",),
    "ds4_fused_indexer_q_rope": ("ds4_fused_indexer_q_rope",),
    "dsa_indexer_mqa_logits_with_blocks_bf16": ("dsa_indexer_mqa_logits_bf16",),
    "dsa_update_topk_indices": ("dsa_update_topk_indices",),
    "ds4_flash_mla_with_kvcache": ("ds4_flash_mla_with_kvcache",),
    "ds4_flash_mla_sparse_fwd": ("ds4_flash_mla_sparse_fwd",),
    "ds4_swa_attn_fwd": ("ds4_swa_attn_fwd",),
    "ds4_paged_sparse_attn_fwd": ("ds4_paged_sparse_attn_fwd",),
    "ds4_dense_compressed_attn_fwd": ("ds4_dense_compressed_attn_fwd",),
    "fused_add_rms_norm": ("fused_add_rms_norm",),
    "gemma_rms_norm": ("gemma_rms_norm",),
    "layer_norm": ("layer_norm",),
    "reshape_and_cache_flash": ("reshape_and_cache_flash",),
    "reshape_and_cache_with_scale": ("reshape_and_cache_with_scale",),
    "reshape_and_cache_mix": ("reshape_and_cache_mix",),
    "causal_conv1d_fn": ("causal_conv1d_fn",),
    "causal_conv1d_update": ("causal_conv1d_update",),
    "gdn_post_conv_prep": ("fused_post_conv_prep",),
    "gdn_chunk": ("chunk_gated_delta_rule",),
    "gdn_packed_decode": ("gdn_packed_decode",),
    "gdn_spec_delta_update": ("gdn_spec_delta_update",),
    "fused_qk_rmsnorm_rope_gate": ("fused_qk_rmsnorm_rope_gate",),
    "fused_qk_rmsnorm_gate": ("fused_qk_rmsnorm_gate",),
    "attn_output_sigmoid_gate": ("attn_output_sigmoid_gate",),
    "gelu": ("gelu",),
    "video_evs": ("video_evs",),
    "vit_qkv_layout": ("vit_qkv_layout",),
    "vit_qkv_rope": ("vit_qkv_rope",),
    "flash_attn": ("attention",),
    "distributed": ("distributed",),
}


def _call_func_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _call_func_name(node)
    return None


# Scan the published ABI and startup bindings without importing GPU modules.
def discover_plugin_op_names(plugin_root: Path | None = None) -> set[str]:
    """Collect exported ops and fixed provider/lifecycle bindings for the catalog."""
    root = plugin_root or _PLUGIN_ROOT
    names: set[str] = set()
    loader_names = {
        "load_ixaite_python_op",
        "load_ixaite_torch_op",
        "_required",
        "_required_torch_op",
        "_required_python_op",
    }
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id in {"_ROUTED_OP_NAMES", "_TRITON_OPS"}
                for target in node.targets
            ):
                values = (
                    node.value.keys
                    if isinstance(node.value, ast.Dict)
                    else getattr(node.value, "elts", [])
                )
                names.update(
                    value.value
                    for value in values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(
                _decorator_name(decorator) == "_routed_op"
                for decorator in node.decorator_list
            ):
                names.add(node.name)
            if not (
                node.name.startswith(("configure_", "_select_"))
                or node.name
                in {"_load_ixaite_flash_attn_ops", "_collective_backend", "_bind_topk"}
            ):
                continue
            for call in ast.walk(node):
                if not (
                    isinstance(call, ast.Call)
                    and _call_func_name(call) in loader_names
                    and call.args
                ):
                    continue
                arg = call.args[0]
                # Grouped modules expose distinct conv and token-index ABIs.
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and len(call.args) > 1
                    and (
                        arg.value == "causal_conv1d"
                        or (
                            isinstance(call.args[1], ast.Constant)
                            and isinstance(call.args[1].value, str)
                            and call.args[1].value.startswith(arg.value)
                        )
                    )
                ):
                    arg = call.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
                elif isinstance(arg, ast.Name):
                    for loop in ast.walk(node):
                        if (
                            isinstance(loop, ast.For)
                            and isinstance(loop.target, ast.Name)
                            and loop.target.id == arg.id
                        ):
                            names.update(
                                value.value
                                for value in getattr(loop.iter, "elts", [])
                                if isinstance(value, ast.Constant)
                                and isinstance(value.value, str)
                            )
    return names
