# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSV4.1 target FFN fusion for EP16 with TP1, TP2 or TP4, decode and eager replay."""

import os
import re
import weakref
from dataclasses import dataclass

import torch
from vllm import envs

_states = weakref.WeakValueDictionary()
_workspaces = {}


def enabled():
    return envs.VLLM_ILU_MOE_IXMM_FUSION


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def kernels():
    from ixformer.inference.functions import moe_ixmm

    return moe_ixmm


@dataclass
class FusedWeights:
    w1: object
    w2: object
    scale1: torch.Tensor
    scale2: torch.Tensor
    local_output: bool = False


def state_for(w1, w2):
    state = _states.get(w1.data_ptr())
    if state is not None:
        require(
            state.w1() is not None
            and state.w2() is not None
            and state.w2().data_ptr() == w2.data_ptr(),
            "stale ixmm weight registration",
        )
    return state


def _uses_local_output(tp_size):
    # TP2 can write into NVSM directly when the deployment opts into it.
    return tp_size == 4 or (
        tp_size == 2 and os.environ.get("IXFORMER_NVSM_W2_DIRECT_COMBINE", "0") == "0"
    )


@torch.no_grad()
def prepare(method, layer):
    if not enabled():
        return
    name = getattr(layer, "layer_name", None) or method._ixmm_layer_name or ""
    match = re.search(r"\blayers\.(\d+)\.", name)
    if not match or int(match[1]) >= 40:
        return  # Draft and other models retain their original weight layout.
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    require(
        layer.moe_config.ep_size == 16 and tp_size in (1, 2, 4),
        "ixmm fused MoE requires target EP16 and TP1, TP2 or TP4; enable only on D",
    )
    require(
        method.gemm_format == "NN" and method.group_size == -1 and method.version == 2,
        "ixmm fused MoE requires NN V2 per-channel W4A8 weights",
    )
    require(
        tuple(layer.w13_weight.shape) == (24, 5120, 2304)
        and tuple(layer.w2_weight.shape) == (24, 2304, 2560)
        and layer.swiglu_limit == 10.0,
        "unsupported ixmm fused MoE shape or activation clamp",
    )
    require(
        layer.w2_input_scale is None
        and all(
            getattr(layer, a) is None
            for a in (
                "w13_i8_weight_scale",
                "w2_i8_weight_scale",
                "w13_i8_weight_zero",
                "w2_i8_weight_zero",
                "w13_bias",
                "w2_bias",
            )
        ),
        "ixmm fused MoE requires dynamic activation scales without bias/zero point",
    )
    ops = kernels()
    require(
        ops.moe_ixmm_available(),
        "install the pinned MR ixmm package, then rebuild ixformer to detect its fusion APIs",
    )
    require(not hasattr(layer, "_ixmm_fused_weights"), "ixmm weights already converted")
    # Scale permutation and compensation match the NN V2 int4 representation.
    index = torch.arange(2304, device=layer.w13_weight.device).reshape(-1, 16)
    perm = torch.stack((index, index + 2304), 1).flatten()
    state = FusedWeights(
        weakref.ref(layer.w13_weight),
        weakref.ref(layer.w2_weight),
        (layer.w13_weight_scale.reshape(24, 4608)[:, perm] * 18).contiguous(),
        (layer.w2_weight_scale.reshape(24, 5120) * 18).contiguous(),
        local_output=_uses_local_output(tp_size),
    )
    for kind, weight in ((1, layer.w13_weight), (2, layer.w2_weight)):
        converted = torch.empty_like(weight)
        ops.moe_ixmm_reorder(weight, converted, kind)
        weight.copy_(converted)
        del converted
    layer._ixmm_fused_weights = state
    _states[layer.w13_weight.data_ptr()] = state


def scratch(device, rows, decode=False):
    key = (device, torch.cuda.current_stream().cuda_stream, rows if decode else 0)
    need = kernels().moe_ixmm_workspace_size(rows)
    entry = _workspaces.get(key)
    if entry is None or entry[0].numel() < need:
        entry = (
            torch.zeros(need, device=device, dtype=torch.int8),
            torch.empty((24, 3), device=device, dtype=torch.int32),
        )
        _workspaces[key] = entry
    return entry


def decode(experts, output, hidden_states, w1, w2, scales, metadata):
    state = state_for(w1, w2)
    if state is None:
        return False

    require(
        metadata is not None and scales is not None,
        "fused decode requires NVSM counts and scales",
    )
    counts = metadata.expert_num_tokens
    if state.local_output:
        require(
            counts.numel() == 24 and hidden_states.is_contiguous()
            and hidden_states.dtype == torch.int8 and hidden_states.shape[-1] == 5120,
            "invalid local expert activation layout",
        )
        activations = hidden_states.view(-1, 5120)
        rows = activations.size(0)
        require(0 < rows <= 16384 and rows % 24 == 0 and scales.numel() == rows,
                "invalid local expert slot capacity")
        ws, layout = scratch(w1.device, rows, decode=True)
        ops = kernels()
        ops.moe_ixmm_layout(counts, layout)
        ops.moe_ixmm_local(activations, w1, w2, scales.reshape(-1), state.scale1,
                           state.scale2, layout, ws, output.view_as(activations))
        return True

    from ixformer.distributed import _w2_direct

    indices, peers, offset, slots, stride = _w2_direct.fused_descriptor(counts)
    require(
        0 < slots <= 24 * 6
        and slots % 6 == 0
        and hidden_states.dtype == torch.int8
        and experts.out_dtype == torch.bfloat16,
        "unsupported fused NVSM decode capacity/dtype",
    )
    rx = hidden_states.reshape(24, -1, 5120)
    rs = scales.reshape(-1)
    require(
        rx.is_contiguous()
        and rs.is_contiguous()
        and indices.numel() == rs.numel() == rx.shape[0] * rx.shape[1]
        and rs.numel() == 24 * 16 * (slots // 6)
        and stride == 16 * 24 * slots,
        "fused decode requires the native NVSM fixed-slot layout",
    )
    ws, layout = scratch(w1.device, rs.numel(), decode=True)
    ops = kernels()
    ops.moe_ixmm_layout(counts, layout)
    ops.moe_ixmm_direct(
        rx.view(-1, 5120),
        w1,
        w2,
        rs,
        state.scale1,
        state.scale2,
        layout,
        ws,
        indices.view(-1),
        peers,
        offset,
        slots,
        stride,
    )
    _w2_direct.fused_complete(counts)
    # Only the subsequent NVSM combine may consume the output placeholder.
    return True


def reject_unhandled(w1, w2):
    require(
        state_for(w1, w2) is None,
        "converted ixmm weights reached an unsupported MoE path",
    )


def batches(counts):
    """Contiguous slices, <=24 groups, <=16384 rows, <=1023 rows/group."""
    start = 0
    base = 0
    groups = []
    for expert, count in enumerate(counts):
        require(count >= 0, "negative expert row count")
        while count:
            n = min(count, 1023)
            if groups and (len(groups) == 24 or start - base + n > 16384):
                yield base, start, groups
                base = start
                groups = []
            groups.append((start - base, start - base + n, expert))
            start += n
            count -= n
    if groups:
        yield base, start, groups


def compact_ffn(x, sa, w1, w2, s1, s2, counts):
    require(
        len(counts) == 24 and sum(counts) == x.shape[0],
        "expert counts must cover all replay rows",
    )
    require(
        x.is_contiguous() and sa.is_contiguous(),
        "replay rows and scales must be contiguous",
    )
    out = torch.empty((x.shape[0], 5120), dtype=torch.bfloat16, device=x.device)
    for lo, hi, groups in batches(counts):
        rows = hi - lo
        ws, _ = scratch(x.device, rows)
        # Stable descending order matches the decode producer; offsets remain compact.
        tuples = sorted(groups, key=lambda g: -(g[1] - g[0])) + [(0, 0, 0)] * (
            24 - len(groups)
        )
        layout = torch.tensor(tuples, device=x.device, dtype=torch.int32)
        kernels().moe_ixmm_replay(
            x[lo:hi], w1, w2, sa[lo:hi], s1, s2, layout, ws, out[lo:hi]
        )
    return out


def apply_standard(
    experts,
    output,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    global_num_experts,
    expert_map,
    a1q_scale,
    a2_scale,
    workspace13,
    workspace2,
    expert_tokens_meta=None,
    apply_router_weight_on_input=False,
    ep_rank=None,
    extra_residual=None,
):
    import ixformer.inference.functions as ixff
    from vllm.distributed import get_ep_group

    require(
        not torch.cuda.is_current_stream_capturing(),
        "compact replay requires eager execution",
    )
    require(
        experts.local_num_experts == 24
        and hidden_states.shape[1] == 5120
        and hidden_states.dtype == torch.bfloat16
        and experts.swiglu_limit == 10.0,
        "unsupported fused replay shape, dtype or clamp",
    )
    require(
        activation.value in ("silu", "swiglu") and not apply_router_weight_on_input,
        "unsupported fused replay activation/router weighting",
    )
    require(
        experts.w2_input_scale is None
        and all(
            getattr(experts, k) is None
            for k in ("g1_alphas", "g2_alphas", "w1_zp", "w2_zp", "w1_bias", "w2_bias")
        ),
        "unsupported fused replay scales, bias or zero point",
    )
    rank = get_ep_group().rank if ep_rank is None else ep_rank
    tokens, topk = topk_ids.shape
    src, dst, cgpu, ccpu, expanded = ixff.moe_compute_token_index_ep(
        topk_ids=topk_ids,
        num_experts=global_num_experts,
        start_expert_id=rank * 24,
        end_expert_id=min((rank + 1) * 24, global_num_experts),
    )
    # The baseline index API already supplies CPU counts; no new D2H copy.
    require(ccpu.device.type == "cpu", "replay index API must provide CPU counts")
    counts = ccpu.tolist()
    if sum(counts) == 0:
        output.zero_()
        if extra_residual is not None:
            output.copy_(extra_residual)
        return output
    x, sa = experts._expand_and_quant(
        input=hidden_states,
        expand_tokens=expanded,
        sorted_token_ids=dst,
        src_to_dst=src,
        topk=topk,
    )
    state = state_for(w1, w2)
    require(state is not None, "missing fused replay weights")
    s1, s2 = state.scale1, state.scale2
    y = compact_ffn(x, sa.reshape(-1), w1, w2, s1, s2, counts)
    scattered = torch.empty((tokens * topk, 5120), device=y.device, dtype=y.dtype)
    scattered.index_copy_(0, dst[: y.shape[0]].long(), y)
    result = ixff.moe_output_reduce_sum(
        input=scattered.view(tokens, topk, 5120),
        topk_weight=topk_weights,
        scaling_factor=experts.runner_scaling_factor,
        mask=src == -1,
        output=output,
        extra_residual=extra_residual,
    )
    return result


def apply_monolithic(
    method, layer, x, topk_weights, topk_ids, shared_experts_input, scale
):
    from types import SimpleNamespace

    require(
        not torch.cuda.is_current_stream_capturing(),
        "compact replay requires eager execution",
    )
    if not hasattr(method, "_ixmm_replay_experts"):
        e = SimpleNamespace(
            local_num_experts=layer.local_num_experts,
            swiglu_limit=layer.swiglu_limit,
            w2_input_scale=layer.w2_input_scale,
            runner_scaling_factor=scale,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        for attr, source in [
            ("g1_alphas", "w13_i8_weight_scale"),
            ("g2_alphas", "w2_i8_weight_scale"),
            ("w1_zp", "w13_i8_weight_zero"),
            ("w2_zp", "w2_i8_weight_zero"),
            ("w1_bias", "w13_bias"),
            ("w2_bias", "w2_bias"),
        ]:
            setattr(e, attr, getattr(layer, source))

        def expand(input, expand_tokens, sorted_token_ids, src_to_dst, topk):
            return method._expand_and_quant(
                layer=layer,
                input=input,
                expand_tokens=expand_tokens,
                sorted_token_ids=sorted_token_ids,
                src_to_dst=src_to_dst,
                output_format=0,
                check_valid=False,
            )

        e._expand_and_quant = expand
        method._ixmm_replay_experts = e
    e = method._ixmm_replay_experts
    return apply_standard(
        e,
        torch.empty_like(x),
        x,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
        topk_ids,
        layer.activation,
        layer.global_num_experts,
        layer.expert_map,
        None,
        None,
        None,
        None,
        ep_rank=layer.moe_config.ep_rank,
        extra_residual=shared_experts_input,
    )
