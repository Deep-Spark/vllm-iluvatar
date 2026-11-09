# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tests.ixaite.benchmarks.suites  # noqa: F401
from tests.ixaite.benchmarks.core.model_catalog import (
    e2e_ops,
    l2_spec,
    list_models,
    load_model,
)
from tests.ixaite.benchmarks.core.plugin_op_suites import discover_plugin_op_names
from tests.ixaite.benchmarks.core.registry import get, list_suites

# Suite.name that is not an l2 key: iter_cases reads this profile key instead.
_SUITE_L2_KEY = {
    "scaled_int8_quant_asym": "scaled_int8_quant",
}


# These suites define their DS4 geometry in iter_cases without an L2 override.
_FIXED_GEOMETRY_SUITES = {
    "ds4_compute_swa_prefill_gather_lens",
    "ds4_dense_compressed_attn_fwd",
    "ds4_flash_mla_sparse_fwd",
    "ds4_flash_mla_with_kvcache",
    "ds4_fused_indexer_q_rope",
    "ds4_merge_attn_outputs",
    "ds4_paged_sparse_attn_fwd",
    "ds4_swa_attn_fwd",
}


def test_qwen3_8b_profile_is_discovered_for_l2_and_e2e() -> None:
    assert "qwen3_8b" in list_models()
    profile = load_model("qwen3_8b")
    assert profile["l2"]["attention"]["head_dim"] == 128
    assert profile["l2"]["attention"]["num_query_heads"] == 32
    assert profile["l2"]["attention"]["num_kv_heads"] == 8
    assert profile["l2"]["linear"]["attn_qkv_proj"]["N"] == 6144
    assert profile["l2"]["linear"]["lm_head"]["N"] == 151936
    assert profile["l2"]["silu_and_mul"]["d"] == 12288
    assert "flash_attn" in e2e_ops("qwen3_8b")
    assert "reshape_and_cache_flash" in e2e_ops("qwen3_8b")
    assert "rms_norm_qk" in e2e_ops("qwen3_8b")
    # Dense FA profile must not inherit Qwen3.5 GDN op lists.
    gdnish = {
        name
        for name in (*profile["l2"], *e2e_ops("qwen3_8b"))
        if "gdn" in name or "delta" in name or "conv1d" in name
    }
    assert not gdnish
    for suite_name in (
        "attention",
        "linear",
        "silu_and_mul",
        "embedding_gather",
        "fused_add_rms_norm",
        "rms_norm",
        "rms_norm_qk",
        "reshape_and_cache_flash",
    ):
        assert get(suite_name).iter_cases("qwen3_8b", None)


def test_qwen35_profile_is_discovered_for_l2_and_e2e() -> None:
    assert "qwen35_9b" in list_models()
    profile = load_model("qwen35_9b")
    assert profile["e2e"]["workloads"]
    assert "linear" in e2e_ops("qwen35_9b")
    assert "embedding_gather" in e2e_ops("qwen35_9b")
    assert "attn_output_sigmoid_gate" in e2e_ops("qwen35_9b")
    assert "gelu" in e2e_ops("qwen35_9b_vision")
    assert "vit_qkv_rope" in e2e_ops("qwen35_9b_vision")

    # The BF16 forward runs neither the quantized GEMM, the activation quant
    # feeding it, nor the int8 KV-cache writers. Those specs live on
    # qwen35_9b_w8a8 / qwen35_9b_kvint8 / qwen35_9b_kint8 instead.
    not_on_profile = {
        "w8a8",
        "scaled_int8_quant",
        "scaled_int8_quant_asym",
        "reshape_and_cache_with_scale",
        "reshape_and_cache_mix",
        "video_evs",
        "activation_dynamic_scaled_int8",
        "moe_w4a8_group_gemm",
        "moe_w4a8_group_gemv",
        "moe_compute_token_index",
        "moe_expand_input_dynamic_scaled_int8",
        "moe_fused_topk_bias",
        "moe_output_reduce_sum",
        # Dense Qwen3 (no GDN) norms live on qwen3_8b, not this GDN profile.
        "rms_norm",
        "rms_norm_qk",
        # qwen35_9b serves at --tensor-parallel-size 1. Collective shapes
        # live on the 27B TP profiles.
        "distributed",
    }
    # DeepSeek-only suites use the DS4 profile, including fixed-geometry cases.
    not_on_profile.update(
        suite for suite in list_suites() if suite.startswith(("ds4_", "dsa_"))
    )
    not_on_profile.update({"gemm_bf16_bf16_fp32", "marlin_w8a16"})
    missing = {suite for suite in list_suites() if suite not in profile["l2"]}
    assert missing == not_on_profile, (
        f"registered suites missing qwen35_9b model specs: {sorted(missing)}"
    )
    for suite_name in list_suites():
        if suite_name in not_on_profile:
            continue
        assert get(suite_name).iter_cases("qwen35_9b", None)


def test_qwen35_video_profile_is_greedy_correctness_only() -> None:
    assert "qwen35_9b_video" in list_models()
    profile = load_model("qwen35_9b_video")
    assert e2e_ops("qwen35_9b_video") == ("layer_norm", "linear", "video_evs")
    assert "workloads" not in profile["e2e"]
    assert profile["e2e"]["correctness"] == {
        "kind": "video_greedy_exact",
        "prompt": "Describe this video in one sentence.",
        "temperature": 0,
        "max_tokens": 32,
        "random_seed": 0,
        "video_bucket": [448, 448, 4],
        "request_timeout_s": 600,
        "provider_notes": {
            "conv3d": "plugin matmul CUDA path",
            "video_postprocess": (
                "ixaite EVS retention mask; community embedding assembly"
            ),
        },
    }
    assert "--compilation-config" not in profile["e2e"]["serve_args"]
    assert "--enforce-eager" in profile["e2e"]["serve_args"]
    assert l2_spec("qwen35_9b_video", "video_evs") == {
        "hidden_size": 3584,
        "grids": [[2, 28, 28]],
        "pruning_rates": [0.5],
    }


def test_every_registered_suite_has_a_model_profile() -> None:
    # Fixed-geometry suites may be claimed by e2e.ops instead of an L2 override.
    # Check that every claimed profile still produces runnable benchmark cases.
    missing = [
        suite
        for suite in list_suites()
        if not any(
            (
                _SUITE_L2_KEY.get(suite, suite) in load_model(model)["l2"]
                or (
                    suite in _FIXED_GEOMETRY_SUITES
                    and suite in load_model(model).get("e2e", {}).get("ops", [])
                )
            )
            and get(suite).iter_cases(model, None)
            for model in list_models()
        )
    ]
    assert not missing, f"registered suites with no model profile specs: {missing}"


def test_kint8_profile_covers_mixed_int8_kv_writer() -> None:
    assert "qwen35_9b_kint8" in list_models()
    assert e2e_ops("qwen35_9b_kint8") == ("reshape_and_cache_mix",)
    assert l2_spec("qwen35_9b_kint8", "reshape_and_cache_mix")["num_kv_heads"] == 4
    assert l2_spec("qwen35_9b_kint8", "reshape_and_cache_mix")["head_dim"] == 256
    assert get("reshape_and_cache_mix").iter_cases("qwen35_9b_kint8", None)
    assert "reshape_and_cache_mix" not in load_model("qwen35_9b")["l2"]
    assert "reshape_and_cache_mix" not in e2e_ops("qwen35_9b")


def test_kvint8_profile_covers_int8_kv_writer() -> None:
    assert "qwen35_9b_kvint8" in list_models()
    assert e2e_ops("qwen35_9b_kvint8") == ("reshape_and_cache_with_scale",)
    assert (
        l2_spec("qwen35_9b_kvint8", "reshape_and_cache_with_scale")["num_kv_heads"] == 4
    )
    assert (
        l2_spec("qwen35_9b_kvint8", "reshape_and_cache_with_scale")["head_dim"] == 256
    )
    assert get("reshape_and_cache_with_scale").iter_cases("qwen35_9b_kvint8", None)
    assert "reshape_and_cache_with_scale" not in load_model("qwen35_9b")["l2"]
    assert "reshape_and_cache_with_scale" not in e2e_ops("qwen35_9b")


def test_w4a8_moe_profile_covers_the_quantized_checkpoint() -> None:
    assert "qwen3_30b_a3b_w4a8" in list_models()
    assert e2e_ops("qwen3_30b_a3b_w4a8") == (
        "activation_dynamic_scaled_int8",
        "moe_w4a8_group_gemm",
        "moe_w4a8_group_gemv",
        "moe_compute_token_index_ep",
        "moe_expand_input_dynamic_scaled_int8",
        "moe_fused_topk_bias",
        "moe_output_reduce_sum",
    )
    assert get("activation_dynamic_scaled_int8").iter_cases("qwen3_30b_a3b_w4a8", None)
    assert get("moe_w4a8_group_gemm").iter_cases("qwen3_30b_a3b_w4a8", None)
    assert get("moe_w4a8_group_gemv").iter_cases("qwen3_30b_a3b_w4a8", None)
    hidden = l2_spec("qwen3_30b_a3b_w4a8", "activation_dynamic_scaled_int8")
    assert hidden["hidden_size"] == 768
    assert l2_spec("qwen3_30b_a3b_w4a8", "moe_w4a8_group_gemm")["calls"]["w13"] == {
        "N": 1536,
        "K": 2048,
        "tokens": [32, 0, 16, 8, 0, 24, 0, 16],
    }
    gemv_w13 = l2_spec("qwen3_30b_a3b_w4a8", "moe_w4a8_group_gemv")["calls"]["w13"]
    assert gemv_w13["K"] == 2048
    assert l2_spec("qwen3_30b_a3b_w4a8", "moe_w4a8_group_gemm")["calls"]["w2"] == {
        "N": 2048,
        "K": 768,
        "tokens": [32, 0, 16, 8, 0, 24, 0, 16],
    }
    gemv_w2 = l2_spec("qwen3_30b_a3b_w4a8", "moe_w4a8_group_gemv")["calls"]["w2"]
    assert gemv_w2 == {"N": 2048, "K": 768, "tokens": [1, 0, 1, 1, 0, 1, 0, 1]}
    assert "moe_w4a8_group_gemm" not in load_model("qwen35_9b")["l2"]
    assert "moe_w4a8_group_gemm" not in e2e_ops("qwen35_9b")
    for suite_name in (
        "moe_compute_token_index",
        "moe_expand_input_dynamic_scaled_int8",
        "moe_fused_topk_bias",
        "moe_output_reduce_sum",
    ):
        assert get(suite_name).iter_cases("qwen3_30b_a3b_w4a8", None)


def test_deepseek_v4_w4a8_covers_non_ep_token_index() -> None:
    assert "deepseek_v4_w4a8" in list_models()
    assert {
        "moe_compute_token_index",
        "moe_compute_token_index_for_decode",
    } <= set(e2e_ops("deepseek_v4_w4a8"))
    assert get("moe_compute_token_index").iter_cases("deepseek_v4_w4a8", None)
    variants = l2_spec("deepseek_v4_w4a8", "moe_compute_token_index")["variants"]
    assert "plain" in variants
    assert "decode" in variants
    assert "ep" not in variants


def test_w8a8_profile_covers_the_quantized_checkpoint() -> None:
    assert "qwen35_9b_w8a8" in list_models()
    # Both the quantized GEMM and the activation quant feeding it only run
    # against the quantized checkpoint, so both are claimed here.
    assert e2e_ops("qwen35_9b_w8a8") == ("w8a8", "scaled_int8_quant")
    assert get("w8a8").iter_cases("qwen35_9b_w8a8", None)
    assert get("scaled_int8_quant").iter_cases("qwen35_9b_w8a8", None)
    assert get("scaled_int8_quant_asym").iter_cases("qwen35_9b_w8a8", None)


def test_l2_suite_specs_come_from_model_catalog() -> None:
    assert l2_spec("qwen35_9b", "silu_and_mul")["d"] == 12288
    assert l2_spec("qwen35_9b", "linear")["attn_qkv_proj"]["N"] == 10240
    assert l2_spec("qwen35_9b", "embedding_gather")["vocab_size"] == 248320
    assert l2_spec("qwen35_9b", "embedding_gather")["hidden_size"] == 4096
    assert l2_spec("qwen35_9b", "attn_output_sigmoid_gate")["hidden_size"] == 4096
    assert l2_spec("qwen35_9b_w8a8", "w8a8")["attn_qkv_proj"]["K"] == 4096


def test_qwen36_27b_w8a8_linear_envelope_is_not_copied_from_9b() -> None:
    assert "qwen36_27b_w8a8_tp2" in list_models()
    assert "qwen36_27b_w8a8_tp4" in list_models()
    assert "linear" in e2e_ops("qwen36_27b_w8a8_tp2")
    assert "linear" in e2e_ops("qwen36_27b_w8a8_tp4")
    assert "silu_and_mul" in e2e_ops("qwen36_27b_w8a8_tp2")
    assert "silu_and_mul" in e2e_ops("qwen36_27b_w8a8_tp4")
    assert l2_spec("qwen36_27b_w8a8_tp2", "silu_and_mul")["d"] == 8704
    assert l2_spec("qwen36_27b_w8a8_tp4", "silu_and_mul")["d"] == 4352
    assert get("silu_and_mul").iter_cases("qwen36_27b_w8a8_tp2", None)
    assert get("silu_and_mul").iter_cases("qwen36_27b_w8a8_tp4", None)
    tp2 = l2_spec("qwen36_27b_w8a8_tp2", "linear")
    tp4 = l2_spec("qwen36_27b_w8a8_tp4", "linear")
    assert (tp2["gdn_out_proj"]["N"], tp2["gdn_out_proj"]["K"]) == (5120, 3072)
    assert (tp4["gdn_out_proj"]["N"], tp4["gdn_out_proj"]["K"]) == (5120, 1536)
    assert (tp2["gdn_in_proj_qkvz"]["N"], tp2["gdn_in_proj_qkvz"]["K"]) == (8192, 5120)
    assert (tp4["gdn_in_proj_qkvz"]["N"], tp4["gdn_in_proj_qkvz"]["K"]) == (4096, 5120)
    assert tp2["gdn_in_proj_ba"]["N"] == 48
    assert tp4["gdn_in_proj_ba"]["N"] == 24
    assert (tp2["lm_head"]["N"], tp2["lm_head"]["K"]) == (124160, 5120)
    assert (tp4["lm_head"]["N"], tp4["lm_head"]["K"]) == (62080, 5120)
    assert l2_spec("qwen36_27b_w8a8_tp2", "distributed")["hidden_size"] == 5120
    assert l2_spec("qwen36_27b_w8a8_tp4", "distributed")["hidden_size"] == 5120
    # Copied 9B O_PROJ_N=K onto hidden/TP.
    assert tp2["gdn_out_proj"]["K"] != 2560
    assert get("linear").iter_cases("qwen36_27b_w8a8_tp2", None)
    assert get("linear").iter_cases("qwen36_27b_w8a8_tp4", None)

    # Full-attn qkv is (Hq + 2*Hkv + Hq)*D because the sigmoid gate is fused
    # into the same projection: TP2 28*256=7168, TP4 14*256=3584. o_proj K
    # is Hq*D. MLP gate_up N is 2*(17408/TP); down K is 17408/TP.
    w8a8_tp2 = l2_spec("qwen36_27b_w8a8_tp2", "w8a8")
    w8a8_tp4 = l2_spec("qwen36_27b_w8a8_tp4", "w8a8")
    assert (w8a8_tp2["attn_qkv_proj"]["N"], w8a8_tp2["attn_qkv_proj"]["K"]) == (
        7168,
        5120,
    )
    assert (w8a8_tp4["attn_qkv_proj"]["N"], w8a8_tp4["attn_qkv_proj"]["K"]) == (
        3584,
        5120,
    )
    assert (w8a8_tp2["attn_o_proj"]["N"], w8a8_tp2["attn_o_proj"]["K"]) == (5120, 3072)
    assert (w8a8_tp4["attn_o_proj"]["N"], w8a8_tp4["attn_o_proj"]["K"]) == (5120, 1536)
    assert (w8a8_tp2["mlp_gate_up_proj"]["N"], w8a8_tp2["mlp_gate_up_proj"]["K"]) == (
        17408,
        5120,
    )
    assert (w8a8_tp4["mlp_gate_up_proj"]["N"], w8a8_tp4["mlp_gate_up_proj"]["K"]) == (
        8704,
        5120,
    )
    assert (w8a8_tp2["mlp_down_proj"]["N"], w8a8_tp2["mlp_down_proj"]["K"]) == (
        5120,
        8704,
    )
    assert (w8a8_tp4["mlp_down_proj"]["N"], w8a8_tp4["mlp_down_proj"]["K"]) == (
        5120,
        4352,
    )
    assert set(l2_spec("qwen36_27b_w8a8_tp2", "scaled_int8_quant")["hidden_sizes"]) == {
        5120,
        3072,
        8704,
    }
    assert set(l2_spec("qwen36_27b_w8a8_tp4", "scaled_int8_quant")["hidden_sizes"]) == {
        5120,
        1536,
        4352,
    }
    for model in ("qwen36_27b_w8a8_tp2", "qwen36_27b_w8a8_tp4"):
        assert l2_spec(model, "fused_add_rms_norm")["hidden_size"] == 5120
        assert l2_spec(model, "gemma_rms_norm")["hidden_size"] == 5120
        quant_ks = set(l2_spec(model, "scaled_int8_quant")["hidden_sizes"])
        w8a8_ks = {shape["K"] for shape in l2_spec(model, "w8a8").values()}
        assert quant_ks <= w8a8_ks
        assert get("w8a8").iter_cases(model, None)
        assert get("scaled_int8_quant").iter_cases(model, None)
        assert get("fused_add_rms_norm").iter_cases(model, None)
        assert get("gemma_rms_norm").iter_cases(model, None)

    expected_tokens = [1, 8, 32, 128, 256, 512, 1024, 2048]
    expected_workloads = {
        (input_len, output_len, batch_size)
        for input_len, output_len in (
            (128, 128),
            (128, 2048),
            (2048, 128),
            (2048, 2048),
            (1024, 1024),
        )
        for batch_size in (1, 8, 16, 32)
    }
    for model in ("qwen36_27b_w8a8_tp2", "qwen36_27b_w8a8_tp4"):
        embedding = l2_spec(model, "embedding_gather")
        assert embedding == {
            "hidden_size": 5120,
            "num_tokens": expected_tokens,
            "vocab_size": 248320,
        }
        expected_ops = {
            "silu_and_mul",
            "linear",
            "w8a8",
            "scaled_int8_quant",
            "embedding_gather",
            "distributed",
            "rms_norm_gated",
            "fused_add_rms_norm",
            "gemma_rms_norm",
            "reshape_and_cache_flash",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "gdn_post_conv_prep",
            "gdn_chunk",
            "gdn_packed_decode",
            "fused_qk_rmsnorm_rope_gate",
            "attn_output_sigmoid_gate",
            "flash_attn",
        }
        assert set(e2e_ops(model)) == expected_ops
        assert set(e2e_ops(model)) <= discover_plugin_op_names()

        e2e = load_model(model)["e2e"]
        assert e2e["correctness"] == {
            "kind": "tp_graph_vs_eager",
            "prompt": "Count from one to forty in words, separated by commas.",
            "temperature": 0,
            "max_tokens": 64,
            "random_seed": 0,
            "repeats": 4,
            "request_timeout_s": 600,
        }
        serve_args = e2e["serve_args"]
        config_index = serve_args.index("--compilation-config")
        assert serve_args[config_index + 1] == (
            '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
        )
        assert "--speculative-config" not in serve_args
        workloads = e2e["workloads"]
        assert len(workloads) == 20
        assert {
            (item["input_len"], item["output_len"], item["max_concurrency"])
            for item in workloads
        } == expected_workloads
        assert all(item["num_prompts"] == 32 for item in workloads)
        assert all(item.get("dataset", "random") == "random" for item in workloads)
        assert all(
            item["id"]
            == f"bs{item['max_concurrency']}_{item['input_len']}_{item['output_len']}"
            for item in workloads
        )


def test_qwen36_27b_w8a8_specialized_profiles_cover_common_hot_path() -> None:
    common_ops = {
        "distributed",
        "gemma_rms_norm",
        "scaled_int8_quant",
        "silu_and_mul",
        "w8a8",
    }
    for model in (
        "qwen36_27b_w8a8_tp2_chunked_prefill",
        "qwen36_27b_w8a8_tp2_prefix_cache",
    ):
        ops = set(e2e_ops(model))
        assert common_ops <= ops
        assert ops <= discover_plugin_op_names()


def _workload_keys(model: str) -> set[tuple[str, int, int, int]]:
    return {
        (item["id"], item["input_len"], item["output_len"], item["max_concurrency"])
        for item in load_model(model)["e2e"]["workloads"]
    }


def test_qwen36_27b_w8a8_ngram_variant_enables_spec() -> None:
    # Default W8A8 serve matches ixformer (no ngram). Spec comparison is a
    # sibling profile with the same matrix so gdn_spec_delta_update is hit.
    for model, tp in (
        ("qwen36_27b_w8a8_tp2_ngram", "2"),
        ("qwen36_27b_w8a8_tp4_ngram", "4"),
    ):
        parent = model.removesuffix("_ngram")
        e2e = load_model(model)["e2e"]
        serve_args = e2e["serve_args"]
        assert serve_args[serve_args.index("--tensor-parallel-size") + 1] == tp
        spec_index = serve_args.index("--speculative-config")
        assert '"num_speculative_tokens":3' in serve_args[spec_index + 1]
        assert '"method":"ngram"' in serve_args[spec_index + 1]
        assert e2e_ops(model) == ("gdn_spec_delta_update",)
        assert _workload_keys(model) == _workload_keys(parent)


def test_qwen36_35b_a3b_w4a8_tp2_shares_the_20_cell_e2e_matrix() -> None:
    model = "qwen36_35b_a3b_w4a8_tp2"
    assert model in list_models()
    expected_workloads = {
        (input_len, output_len, batch_size)
        for input_len, output_len in (
            (128, 128),
            (128, 2048),
            (2048, 128),
            (2048, 2048),
            (1024, 1024),
        )
        for batch_size in (1, 8, 16, 32)
    }
    e2e = load_model(model)["e2e"]
    serve_args = e2e["serve_args"]
    config_index = serve_args.index("--compilation-config")
    assert serve_args[config_index + 1] == (
        '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0,'
        '"cudagraph_capture_sizes":[1,8,16,32]}'
    )
    assert serve_args[serve_args.index("--tensor-parallel-size") + 1] == "2"
    assert e2e["env"].get("VLLM_ENFORCE_CUDA_GRAPH") == "1"
    workloads = e2e["workloads"]
    assert len(workloads) == 20
    assert {
        (item["input_len"], item["output_len"], item["max_concurrency"])
        for item in workloads
    } == expected_workloads
    assert all(item["num_prompts"] == 32 for item in workloads)
    assert "moe_w4a8_group_gemv" in e2e_ops(model)
    assert "moe_w4a8_group_gemm" in e2e_ops(model)
    embedding = l2_spec(model, "embedding_gather")
    assert embedding["hidden_size"] == 2048
    assert embedding["vocab_size"] == 248320


def test_qwen36_27b_w8a8_gdn_conv_dim_is_the_column_shard() -> None:
    # conv_dim = (2*Hk*Dk + Hv*Dv) / TP with the 27B global Hk=16, Hv=48,
    # Dk=Dv=128, so 10240/TP. The two ways to get this wrong are copying the
    # 9B specs (8192 / TP4 2048) and reading it as hidden/TP (5120 / 1280).
    for suite in ("causal_conv1d_fn", "causal_conv1d_update"):
        tp2 = l2_spec("qwen36_27b_w8a8_tp2", suite)
        tp4 = l2_spec("qwen36_27b_w8a8_tp4", suite)
        assert (tp2["dim"], tp2["width"]) == (5120, 4)
        assert (tp4["dim"], tp4["width"]) == (2560, 4)
        assert tp4["dim"] != 2048
        assert tp2["dim"] != l2_spec("qwen35_9b", suite)["dim"]
        assert get(suite).iter_cases("qwen36_27b_w8a8_tp2", None)
        assert get(suite).iter_cases("qwen36_27b_w8a8_tp4", None)
        assert suite in e2e_ops("qwen36_27b_w8a8_tp2")
        assert suite in e2e_ops("qwen36_27b_w8a8_tp4")


def test_quant_hidden_sizes_are_linear_input_dims() -> None:
    # Activation quant sees a linear's in_features, so every hidden size here
    # must be some linear's K. qkv_proj's N=10240 is the one that reads like a
    # plausible activation width and is not one.
    widths = set(l2_spec("qwen35_9b_w8a8", "scaled_int8_quant")["hidden_sizes"])
    assert widths == {4096, 12288}
    assert widths <= {shape["K"] for shape in l2_spec("qwen35_9b", "linear").values()}
    # The BF16 profile never quantizes activations, so it must not carry a
    # second copy of this spec for the default --model to pick up.
    assert "scaled_int8_quant" not in load_model("qwen35_9b")["l2"]


def test_qwen36_attention_l2_specs_share_the_27b_profiles() -> None:
    for model in ("qwen36_27b_w8a8_tp2", "qwen36_27b_w8a8_tp4"):
        assert model in list_models()
        assert l2_spec(model, "attention")


def test_qwen36_attention_uses_operation_specific_contexts() -> None:
    cases = get("attention").iter_cases("qwen36_27b_w8a8_tp4", None)
    prefill_contexts = {
        case.context_len for case in cases if case.operation == "flash_attn_varlen_func"
    }
    decode_contexts = {
        case.context_len
        for case in cases
        if case.operation == "flash_attn_with_kvcache"
    }
    assert 131072 not in prefill_contexts
    assert 131072 in decode_contexts


def test_qwen36_27b_reshape_and_cache_flash_is_tp_local_full_attn() -> None:
    assert (
        l2_spec("qwen36_27b_w8a8_tp2", "reshape_and_cache_flash")["num_kv_heads"] == 2
    )
    assert (
        l2_spec("qwen36_27b_w8a8_tp4", "reshape_and_cache_flash")["num_kv_heads"] == 1
    )
    for model, hkv in (("qwen36_27b_w8a8_tp2", 2), ("qwen36_27b_w8a8_tp4", 1)):
        spec = l2_spec(model, "reshape_and_cache_flash")
        assert spec["head_dim"] == 256
        assert spec["page_size"] == 16
        assert "num_kv_heads_tp" not in spec
        assert "reshape_and_cache_flash" in e2e_ops(model)
        cases = get("reshape_and_cache_flash").iter_cases(model, None)
        assert cases
        assert {case.num_kv_heads for case in cases} == {hkv}


def test_qwen36_27b_fused_qk_is_the_tp_local_full_attn_shard() -> None:
    # Global Hq=24, Hkv=4, D=256, rotary_dim=64. TP2/TP4 are 12/2 and 6/1,
    # not the 9B 16/4 tile and not hidden-derived head counts.
    for model, hq, hkv in (
        ("qwen36_27b_w8a8_tp2", 12, 2),
        ("qwen36_27b_w8a8_tp4", 6, 1),
    ):
        spec = l2_spec(model, "fused_qk_rmsnorm_rope_gate")
        assert (spec["num_query_heads"], spec["num_kv_heads"]) == (hq, hkv)
        assert spec["head_dim"] == 256
        assert spec["rotary_dim"] == 64
        assert spec["cache_rows"] == 4096
        assert "fused_qk_rmsnorm_rope_gate" in e2e_ops(model)
        cases = get("fused_qk_rmsnorm_rope_gate").iter_cases(model, None)
        assert cases
        assert {case.num_query_heads for case in cases} == {hq}
        assert {case.num_kv_heads for case in cases} == {hkv}


def test_qwen36_27b_attn_output_sigmoid_gate_is_hq_times_head_dim() -> None:
    # The multiply sits on attn_output before o_proj, so last dim is the
    # TP-local Hq*D. 9B's 4096 is 16*256 and only equals hidden by chance;
    # 27B hidden is 5120, which is not a serve width for this op.
    for model, width in (
        ("qwen36_27b_w8a8_tp2", 3072),
        ("qwen36_27b_w8a8_tp4", 1536),
    ):
        spec = l2_spec(model, "attn_output_sigmoid_gate")
        assert spec["hidden_size"] == width
        assert spec["hidden_size"] != 4096
        assert spec["hidden_size"] != 5120
        assert "attn_output_sigmoid_gate" in e2e_ops(model)
        cases = get("attn_output_sigmoid_gate").iter_cases(model, None)
        assert cases
        assert {case.hidden_size for case in cases} == {width}


def test_qwen36_27b_gdn_path_is_the_tp_local_shard() -> None:
    # Global Hk=16, Hv=48, Dk=Dv=128. TP2/TP4 are 8/24 and 4/12, not the
    # 9B 16/32 or its TP4 4/8. rms_norm_gated last dim is Dv, not hidden.
    for model, hk, hv in (
        ("qwen36_27b_w8a8_tp2", 8, 24),
        ("qwen36_27b_w8a8_tp4", 4, 12),
    ):
        prep = l2_spec(model, "fused_post_conv_prep")
        chunk = l2_spec(model, "chunk_gated_delta_rule")
        decode = l2_spec(model, "gdn_packed_decode")
        rms = l2_spec(model, "rms_norm_gated")
        for spec in (prep, chunk, decode):
            assert (spec["num_k_heads"], spec["num_v_heads"]) == (hk, hv)
            assert (spec["head_k_dim"], spec["head_v_dim"]) == (128, 128)
        assert prep["l2norm_min_t"] == 512
        assert chunk["lengths"] == [[512], [512, 1536]]
        assert decode["num_tokens"] == [1, 8, 128]
        assert (rms["heads"], rms["dim"]) == (hv, 128)
        assert rms["dim"] != 5120
        assert get("fused_post_conv_prep").iter_cases(model, None)
        assert get("chunk_gated_delta_rule").iter_cases(model, None)
        assert get("gdn_packed_decode").iter_cases(model, None)
        assert get("rms_norm_gated").iter_cases(model, None)
        for op in (
            "gdn_post_conv_prep",
            "gdn_chunk",
            "gdn_packed_decode",
            "rms_norm_gated",
            "flash_attn",
        ):
            assert op in e2e_ops(model)
        # Prep absorbs fused_gdn_gating / l2norm_fwd. Mixed spec+prefill
        # gating stays Triton and must not be claimed as an ixaite e2e op.
        for op in (
            "fused_gdn_gating",
            "l2norm_fwd",
        ):
            assert op not in e2e_ops(model)
        assert "gdn_spec_delta_update" not in e2e_ops(model)
        spec = l2_spec(model, "gdn_spec_delta_update")
        assert (spec["num_k_heads"], spec["num_v_heads"]) == (hk, hv)
        assert get("gdn_spec_delta_update").iter_cases(model, None)
        ngram = f"{model}_ngram"
        assert e2e_ops(ngram) == ("gdn_spec_delta_update",)
        assert get("gdn_spec_delta_update").iter_cases(ngram, None)


def test_qwen36_27b_bf16_covers_text_shapes() -> None:
    assert "qwen36_27b" in list_models()
    linear = l2_spec("qwen36_27b", "linear")
    assert (linear["attn_qkv_proj"]["N"], linear["attn_qkv_proj"]["K"]) == (7168, 5120)
    assert (linear["mlp_gate_up_proj"]["N"], linear["mlp_gate_up_proj"]["K"]) == (
        17408,
        5120,
    )
    assert (linear["mlp_down_proj"]["N"], linear["mlp_down_proj"]["K"]) == (5120, 8704)
    assert (linear["gdn_out_proj"]["N"], linear["gdn_out_proj"]["K"]) == (5120, 3072)
    assert l2_spec("qwen36_27b", "silu_and_mul")["d"] == 8704
    assert l2_spec("qwen36_27b", "gemma_rms_norm")["hidden_size"] == 5120
    assert l2_spec("qwen36_27b", "fused_add_rms_norm")["hidden_size"] == 5120
    assert l2_spec("qwen36_27b", "attn_output_sigmoid_gate")["hidden_size"] == 3072
    assert l2_spec("qwen36_27b", "rms_norm_gated") == {"dim": 128, "heads": 24}
    assert get("linear").iter_cases("qwen36_27b", None)
    assert get("silu_and_mul").iter_cases("qwen36_27b", None)
    assert get("distributed").iter_cases("qwen36_27b", None)
    assert "linear" in e2e_ops("qwen36_27b")
    assert "silu_and_mul" in e2e_ops("qwen36_27b")
    assert "flash_attn" in e2e_ops("qwen36_27b")
    assert "distributed" in e2e_ops("qwen36_27b")
    e2e = load_model("qwen36_27b")["e2e"]
    serve_args = e2e["serve_args"]
    assert serve_args[serve_args.index("--tensor-parallel-size") + 1] == "4"
    assert '{"image":0,"video":0}' in serve_args
    assert "--speculative-config" not in serve_args
    assert "gdn_spec_delta_update" not in e2e_ops("qwen36_27b")
    assert e2e["correctness"]["kind"] == "tp_graph_vs_eager"
    assert l2_spec("qwen36_27b", "distributed")["hidden_size"] == 5120


def test_qwen36_27b_vision_covers_tower_and_merger() -> None:
    assert "qwen36_27b_vision" in list_models()
    linear = l2_spec("qwen36_27b_vision", "linear")
    assert (linear["vis_merger_fc2"]["N"], linear["vis_merger_fc2"]["K"]) == (
        5120,
        2304,
    )
    assert linear["vis_merger_fc2"]["N"] != 4096
    assert l2_spec("qwen36_27b_vision", "layer_norm")["hidden_size"] == 1152
    assert l2_spec("qwen36_27b_vision", "vit_qkv_layout")["num_heads"] == [16, 8, 4]
    assert l2_spec("qwen36_27b_vision", "vit_qkv_rope")["num_heads"] == [16, 8, 4]
    assert get("linear").iter_cases("qwen36_27b_vision", None)
    assert get("layer_norm").iter_cases("qwen36_27b_vision", None)
    assert set(e2e_ops("qwen36_27b_vision")) == {
        "layer_norm",
        "gelu",
        "linear",
        "vit_qkv_layout",
        "vit_qkv_rope",
        "flash_attn",
    }
    attention = l2_spec("qwen36_27b_vision", "attention")
    assert attention["mode"] == "packed_encoder"
    layouts = {
        (item["num_query_heads"], item["head_dim"]) for item in attention["layouts"]
    }
    assert layouts == {(8, 72), (8, 128), (4, 128)}
    assert attention["softmax_head_dim"] == 72
    assert get("attention").iter_cases("qwen36_27b_vision", None)
    serve_args = load_model("qwen36_27b_vision")["e2e"]["serve_args"]
    assert serve_args[serve_args.index("--tensor-parallel-size") + 1] == "4"
    assert '{"image":1,"video":0}' in serve_args
    assert all("KERNEL_DISABLE" not in arg for arg in serve_args)


def test_qwen36_35b_attention_profiles_use_tp_local_heads() -> None:
    # 35B-A3B does not fit on one 32 GB card, so TP1 has no profile. TP2 keeps
    # both KV heads split and stays at GQA=8; TP4 replicates them, dropping the
    # TP-local ratio to 4 and exercising the pre-existing GQA4 path instead.
    for model, tp, hq, hkv, gqa in (
        ("qwen36_35b_a3b_w8a8_tp2", "2", 8, 1, 8),
        ("qwen36_35b_a3b_w8a8_tp4", "4", 4, 1, 4),
    ):
        assert model in list_models()
        attention = l2_spec(model, "attention")
        assert (
            attention["num_query_heads"],
            attention["num_kv_heads"],
            attention["head_dim"],
        ) == (hq, hkv, 256)
        assert hq // hkv == gqa
        assert 131072 not in attention["prefill_contexts"]
        assert 131072 in attention["decode_contexts"]
        assert get("attention").iter_cases(model, None)

        reshape = l2_spec(model, "reshape_and_cache_flash")
        assert (reshape["num_kv_heads"], reshape["head_dim"], reshape["page_size"]) == (
            hkv,
            256,
            16,
        )
        fused_qk = l2_spec(model, "fused_qk_rmsnorm_rope_gate")
        assert (
            fused_qk["num_query_heads"],
            fused_qk["num_kv_heads"],
            fused_qk["head_dim"],
            fused_qk["rotary_dim"],
        ) == (hq, hkv, 256, 64)
        assert l2_spec(model, "attn_output_sigmoid_gate")["hidden_size"] == hq * 256

        e2e = load_model(model)["e2e"]
        tp_index = e2e["serve_args"].index("--tensor-parallel-size")
        assert e2e["serve_args"][tp_index + 1] == tp
        spec_index = e2e["serve_args"].index("--speculative-config")
        assert '"num_speculative_tokens":3' in e2e["serve_args"][spec_index + 1]
        assert set(e2e_ops(model)) == {
            "flash_attn",
            "reshape_and_cache_flash",
            "fused_qk_rmsnorm_rope_gate",
            "attn_output_sigmoid_gate",
        }


# Profiles select a whole backend at the CLI; old kernel pins bypass hardware defaults.
def test_profiles_do_not_force_mixed_backend_controls():
    from tests.ixaite.benchmarks.core.model_catalog import e2e_spec, list_e2e_models

    for model in list_e2e_models():
        spec = e2e_spec(model)
        assert not {
            "VLLM_ILU_KERNEL_PROVIDER",
            "VLLM_ILU_IXAITE_OP_ALLOWLIST",
            "VLLM_ILU_GDN_ATTN_BACKEND",
        }.intersection(spec.get("env", {})), model
