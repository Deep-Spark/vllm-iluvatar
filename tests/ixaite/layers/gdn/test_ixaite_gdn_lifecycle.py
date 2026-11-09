# SPDX-License-Identifier: Apache-2.0
"""Compare stateful GDN and replay with an independent recurrence."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite.layers import causal_conv1d, gdn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Iluvatar GPU required"
)
_PREFIX = "model.layers.0.linear_attn"
_HK, _HV, _D = 4, 8, 128
_C = (2 * _HK + _HV) * _D


# Use the same TP-local shapes and states while avoiding unrelated model loading.
def _layer(module, conv_state, ssm_state, weights):
    layer = SimpleNamespace(
        prefix=_PREFIX,
        tp_size=1,
        num_k_heads=_HK,
        num_v_heads=_HV,
        key_dim=_HK * _D,
        value_dim=_HV * _D,
        head_k_dim=_D,
        head_v_dim=_D,
        enable_packed_recurrent_decode=True,
        activation="silu",
        conv1d=SimpleNamespace(weight=weights[0], bias=None),
        A_log=weights[1],
        dt_bias=weights[2],
        kv_cache=[conv_state, ssm_state],
        _qwen38_kernel=lambda: "plugin_fla",
    )
    for name in (
        "_forward_core",
        "rearrange_mixed_qkv",
        "_forward_core_decode_non_spec",
        "_forward_core_apc",
    ):
        method = getattr(module.IluQwenGatedDeltaNetAttention, name, None)
        if method is not None:
            setattr(layer, name, MethodType(method, layer))
    return layer


# Real convolution metadata is shared by the two implementations under comparison.
def _metadata(module, lengths, indices):
    starts_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0)], dtype=torch.int32)
    starts = starts_cpu.cuda()
    nums, batch, offsets = compute_causal_conv1d_metadata(starts_cpu, device="cuda")
    decodes = sum(length == 1 for length in lengths)
    return module.IluGDNAttentionMetadata(
        num_prefills=len(lengths) - decodes,
        num_prefill_tokens=sum(lengths) - decodes,
        num_decodes=decodes,
        num_decode_tokens=decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=sum(lengths),
        has_initial_state=torch.ones(len(lengths), device="cuda", dtype=torch.int32),
        non_spec_query_start_loc=starts,
        non_spec_query_start_loc_cpu=starts_cpu,
        non_spec_state_indices_tensor=indices,
        nums_dict=nums,
        batch_ptr=batch,
        token_chunk_offset_ptr=offsets,
    )


# Independent fp32 recurrence also checks convolution history and GQA head mapping.
def _reference(conv0, state0, weights, inputs, lengths, indices):
    conv = conv0.float().cpu().clone()
    state = state0.cpu().clone()
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    weight, log_a, bias = (tensor.float().cpu() for tensor in weights)
    packed, b, a = (tensor.float().cpu() for tensor in inputs)
    outputs = []
    token = 0
    # 1. Shift raw convolution history; apply activation and bf16 storage rounding.
    # 2. Normalize Q/K, map each value head, and update its independent SSM state.
    # 3. Project the current query and save both histories for the next step.
    for length, index in zip(lengths, indices.cpu().tolist()):
        for _ in range(length):
            window = torch.cat((conv[index], packed[token, :, None]), dim=-1)
            conv[index] = window[:, 1:]
            mixed = (
                F.silu((window * weight[:, 0, :]).sum(-1)).to(inputs[0].dtype).float()
            )
            q, k, v = mixed.split((_HK * _D, _HK * _D, _HV * _D))
            q, k = (tensor.reshape(_HK, _D) for tensor in (q, k))
            q, k = (
                (tensor * (tensor.square().sum(-1, keepdim=True) + 1e-6).rsqrt())
                .to(inputs[0].dtype)
                .float()
                .repeat_interleave(_HV // _HK, dim=0)
                for tensor in (q, k)
            )
            g = -log_a.exp() * F.softplus(a[token] + bias)
            beta = b[token].sigmoid().to(inputs[0].dtype).float()
            current = state[index] * g.exp()[:, None, None]
            prediction = torch.bmm(current, k[:, :, None]).squeeze(-1)
            delta = (v.reshape(_HV, _D) - prediction) * beta[:, None]
            current += delta[:, :, None] * k[:, None, :]
            state[index] = current
            outputs.append(torch.bmm(current, q[:, :, None]).squeeze(-1) * _D**-0.5)
            token += 1
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    return torch.stack(outputs).cuda().to(inputs[0].dtype), [
        conv.cuda().to(conv0.dtype),
        state.cuda(),
    ]


# Bind all executable lifecycle functions before the first state update.
def _bind():
    causal_conv1d.configure_causal_conv1d_provider()
    gdn.configure_gdn_post_conv_prep_provider()
    gdn.configure_gdn_chunk_provider()
    gdn.configure_gdn_packed_decode_provider()
    gdn.configure_gdn_spec_delta_update_provider()


# Protect outputs and both caches across prefill, decode, and repeated graph use.
@torch.inference_mode()
@pytest.mark.parametrize("path", ["prefill", "decode", "graph", "prefix"])
def test_stateful_gdn_matches_reference(monkeypatch, path):
    torch.manual_seed(71)
    dtype = torch.bfloat16
    lengths = [5, 3] if path in {"prefill", "prefix"} else [1, 1]
    indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    conv_shape = (5, _C, 3) if is_conv_state_dim_first() else (5, 3, _C)
    conv0 = torch.randn(conv_shape, dtype=dtype, device="cuda") * 0.1
    state0 = torch.randn(5, _HV, _D, _D, device="cuda") * 0.1
    weights = (
        torch.randn(_C, 1, 4, dtype=dtype, device="cuda") * 0.1,
        torch.randn(_HV, device="cuda") * 0.1,
        torch.randn(_HV, device="cuda") * 0.1,
    )
    selected = _layer(gdn, conv0.clone(), state0.clone(), weights)
    metadata = _metadata(gdn, lengths, indices)
    if path == "prefix":
        # Prefix sources remain intact; continuations write separate destinations.
        metadata.state_block_table = torch.tensor(
            [[1, 3], [2, 4]], dtype=torch.int32, device="cuda"
        )
        metadata.initial_state_idx = torch.zeros(2, dtype=torch.int32, device="cuda")
        metadata.block_idx_first_scheduled_token = torch.ones_like(indices)
        metadata.block_idx_last_scheduled_token = torch.ones_like(indices)
        metadata.output_state_indices = torch.tensor(
            [3, 4], device="cuda", dtype=torch.int32
        )
        metadata.num_computed_tokens_cpu = torch.full((2,), 256, dtype=torch.int32)
        metadata.num_computed_tokens = metadata.num_computed_tokens_cpu.cuda()
        metadata.cache_block_size = 256
    inputs = (
        torch.randn(sum(lengths), _C, dtype=dtype, device="cuda") * 0.1,
        torch.randn(sum(lengths), _HV, dtype=dtype, device="cuda") * 0.1,
        torch.randn(sum(lengths), _HV, dtype=dtype, device="cuda") * 0.1,
    )
    expected = torch.empty(sum(lengths), _HV, _D, dtype=dtype, device="cuda")
    actual = torch.empty_like(expected)
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={_PREFIX: metadata}),
    )
    expected, reference_caches = _reference(
        conv0, state0, weights, inputs, lengths, indices
    )
    # Decode convolution overwrites packed inputs; replay receives new raw tokens.
    raw_inputs = tuple(tensor.clone() for tensor in inputs)
    with forbid_ixformer_calls():
        _bind()
        selected._forward_core(*inputs, actual)
        if path == "graph":
            for tensor, raw in zip(inputs, raw_inputs):
                tensor.copy_(raw)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                selected._forward_core(*inputs, actual)
            for _ in range(2):
                selected.kv_cache[0].copy_(conv0)
                selected.kv_cache[1].copy_(state0)
                for tensor, raw in zip(inputs, raw_inputs):
                    tensor.copy_(raw)
                graph.replay()
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        destination = metadata.output_state_indices if path == "prefix" else indices
        for cache, ref_cache in zip(selected.kv_cache, reference_caches):
            torch.testing.assert_close(
                cache[destination], ref_cache[indices], rtol=3e-2, atol=3e-2
            )
        if path == "prefix":
            torch.testing.assert_close(
                selected.kv_cache[0][indices], conv0[indices], rtol=0, atol=0
            )
            torch.testing.assert_close(
                selected.kv_cache[1][indices], state0[indices], rtol=0, atol=0
            )


# Accepted-token selection and recurrent state writes retain the existing ABI gate.
@torch.inference_mode()
@pytest.mark.parametrize("use_graph", [False, True])
def test_speculative_state_and_replay(use_graph):
    import ixformer._C as ixf_c_ops

    from tests.ixaite.benchmarks.core.harness import clone_workload
    from tests.ixaite.benchmarks.suites.gdn_spec_delta_update import (
        GdnSpecDeltaUpdateCase,
    )

    case = GdnSpecDeltaUpdateCase("strict-state", 2, 3, _HK, _D, _HV, _D)
    original = case.make_workload(torch.device("cuda"), torch.bfloat16)
    selected = clone_workload(original)
    case.run(ixf_c_ops.infer.fused_recurrent_gated_delta_rule, original)
    with forbid_ixformer_calls():
        gdn.configure_gdn_spec_delta_update_provider()
        implementation = gdn._select_spec_delta_update(_HK, _D, _HV, _D)
        case.run(implementation, selected)
        if use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                case.run(implementation, selected)
            case.prepare_cuda_graph_replay(selected)
            graph.replay()
        case.check(original, selected)
