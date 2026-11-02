# SPDX-License-Identifier: Apache-2.0
"""Exercise the production runner skeleton with native NN GEMMs and collectives."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, pcp):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    sync = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    import ixformer.inference.functions as ops

    from tests.model_executor.layers.quantization.moe_test_utils import (
        make_nn_graph_method,
    )
    from vllm_iluvatar.distributed.device_communicators import all2all
    from vllm_iluvatar.distributed.device_communicators.ilu_communicator import (
        IluCommunicator,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.runner import (
        ilu_moe_runner as mod,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_shared_experts import (
        IluSharedExperts,
    )

    method, layer, _, _ = make_nn_graph_method(
        experts=16, hidden=6144, intermediate=256
    )
    method.moe_kernel = None

    # This test targets runner scheduling, collectives, overlap and graph
    # replay, not W4A8 kernel numerics. CI's ivcore11 ixformer wheel omits the
    # optional ixmm library, while ixformer's torch reference backend copies
    # group_layout to CPU and therefore cannot run during CUDA graph capture.
    # Keep the routed calculation GPU-only and graph-safe here; dedicated MoE
    # operator tests cover the real quantized GEMMs.
    def routed_apply(layer, x, topk_weights, topk_ids, *args):
        if isinstance(x, tuple):
            quantized, scale = x[:2]
            dtype = torch.bfloat16
        else:
            dtype = x.dtype
            quantized, scale = ops.scaled_int8_quant(x)
        x = (quantized.float() * scale.reshape(-1, 1)).to(dtype)
        route_gain = (topk_weights.float() * (topk_ids.remainder(7).float() + 1.0)).sum(
            dim=-1, keepdim=True
        )
        return (x.float() * route_gain).to(x.dtype)

    method.apply = routed_apply
    layer.top_k = 8
    layer.moe_config.ep_rank = rank
    layer.quant_method = method
    layer.custom_routing_function = None
    layer._ensure_moe_quant_config_init = lambda: None
    captured = {}

    def modular(**kw):
        captured.update(kw)
        output = method.apply(
            layer, kw["x"], kw["topk_weights"], kw["topk_ids"], None, None
        )
        captured["output"] = output
        return output

    layer.forward_modular = modular
    comm = SimpleNamespace(
        world_size=2, rank_in_group=rank, use_vllm_comm=True, ixfd_group=None
    )
    metadata = SimpleNamespace(all_ranks_decode_only=False)
    context = SimpleNamespace(dp_metadata=None if pcp else metadata)
    mod.get_forward_context = all2all.get_forward_context = lambda: context
    group = SimpleNamespace(rank_in_group=rank)
    group.all_gatherv = lambda ts, **kw: IluCommunicator.all_gatherv(comm, ts, **kw)
    group.reduce_scatterv = lambda x, **kw: IluCommunicator.reduce_scatterv(
        comm, x, **kw
    )
    all2all.get_dp_group = all2all.get_ep_group = lambda: group
    manager = all2all.AgRsAll2AllManager.__new__(all2all.AgRsAll2AllManager)
    group.dispatch_router_logits = manager.dispatch_router_logits
    group.combine = manager.combine
    mod.get_ep_group = mod.get_pcp_group = lambda: group
    group.all_gather = lambda x, dim: group.all_gatherv(x, dim=dim, sizes=sizes)
    group.reduce_scatter = lambda x, dim: group.reduce_scatterv(x, dim=dim, sizes=sizes)

    torch.manual_seed(111)
    gate_weight = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
    up = torch.randn(6144, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    down = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01

    class Shared(torch.nn.Module):
        def forward(self, value):
            if isinstance(value, tuple):
                q, scale, dtype = value
                value = (q.float() * scale.reshape(-1, 1)).to(dtype)
            return torch.nn.functional.silu(value @ up) @ down

    def router(**kw):
        values, ids = torch.topk(kw["router_logits"].float(), 8, dim=-1)
        return values.softmax(-1), ids.int()

    config = SimpleNamespace(
        dp_size=1 if pcp else 2,
        pcp_size=2 if pcp else 1,
        in_dtype=torch.bfloat16,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(
            all2all_backend="allgather_reducescatter", use_all2all_kernels=False
        ),
    )
    runner = mod.IluMoERunner.__new__(mod.IluMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = config
    runner.routed_experts = layer
    runner.routed_input_transform = runner.routed_output_transform = None
    runner.gate = lambda x: (x @ gate_weight.T, None)
    runner._fse_fuse_gate = False
    # This fixture bypasses IluMoERunner.__init__. Keep the model-opted
    # gather-before-gate path disabled so the test only exercises the
    # phase-aware AG/RS and shared-expert schedules it is intended to cover.
    runner._gather_before_gate_enabled = False
    runner._shared_experts = IluSharedExperts(Shared(), config, False, lambda: False)
    runner._sequence_parallel_context = nullcontext
    runner.router = SimpleNamespace(select_experts=router)

    for sizes in ((16, 16), (7, 3), (5, 0)) if not pcp else ((8, 8),):
        metadata.get_chunk_sizes_across_dp_rank = lambda sizes=sizes: sizes
        torch.manual_seed(112 + rank)
        x = torch.randn(sizes[rank], 6144, device="cuda", dtype=torch.bfloat16) * 0.1
        # DP=1/PCP=2 without SP MoE has no production DPMetadata.
        # Even decode uses pre-gate/INT8; do not inject a decode flag.
        for decode in (False,) if pcp else (False, True):
            metadata.all_ranks_decode_only = decode
            for reuse in (False, True):

                def shared_input(x=x, reuse=reuse):
                    if not reuse:
                        return x
                    if x.shape[0]:
                        q, scale = ops.scaled_int8_quant(x)
                    else:
                        q = torch.empty_like(x, dtype=torch.int8)
                        scale = torch.empty(0, 1, device=x.device)
                    return q, scale, x.dtype

                def run(x=x, shared_input=shared_input):
                    shared, routed = runner._forward_impl(x, None, shared_input())
                    assert runner._shared_experts._output == [None, None]
                    return shared + routed

                def reference(
                    x=x, sizes=sizes, shared_input=shared_input, decode=decode
                ):
                    full = group.all_gatherv(x, dim=0, sizes=sizes)
                    logits = (
                        full @ gate_weight.T
                        if decode
                        else group.all_gatherv(x @ gate_weight.T, dim=0, sizes=sizes)
                    )
                    weights, ids = router(router_logits=logits)
                    torch.testing.assert_close(
                        captured["topk_ids"], ids, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        captured["topk_weights"], weights, rtol=0, atol=0
                    )
                    if not decode:
                        full_q, full_scale = ops.scaled_int8_quant(full)
                        torch.testing.assert_close(
                            captured["x"][0], full_q, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            captured["x"][1].reshape(-1),
                            full_scale.reshape(-1),
                            rtol=0,
                            atol=0,
                        )
                    output = method.apply(layer, full, weights, ids, None, None)
                    torch.testing.assert_close(
                        captured["output"], output, rtol=0, atol=0
                    )
                    dist.all_reduce(output)
                    start = sum(sizes[:rank])
                    return output[
                        start : start + sizes[rank]
                    ] + runner._shared_experts._layer(shared_input())

                runner._ag_shared_overlap_enabled = False
                serial = run()
                torch.testing.assert_close(serial, reference(), rtol=0.008, atol=0.0001)
                runner._ag_shared_overlap_enabled = True
                for _ in range(3):
                    overlap = run()
                torch.testing.assert_close(overlap, serial, rtol=0, atol=0)
                torch.cuda.synchronize()
                dist.barrier(group=sync)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = run()
                for _ in range(2):
                    x.normal_(std=0.1)
                    runner._ag_shared_overlap_enabled = False
                    expected = run()
                    runner._ag_shared_overlap_enabled = True
                    graph.replay()
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                dist.barrier(group=sync)
                del graph, result
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two Iluvatar GPUs")
@pytest.mark.parametrize("pcp", [False, True])
def test_runner_phase_collectives_and_graph(tmp_path, pcp):
    context = mp.spawn(
        _worker, args=(f"file://{tmp_path / 'rdzv'}", pcp), nprocs=2, join=False
    )
    import time

    deadline = time.monotonic() + 150
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            for process in context.processes:
                process.join(timeout=5)
            pytest.fail("runner distributed test timed out")
