# SPDX-License-Identifier: Apache-2.0
"""Two-rank native packed INT8 dispatch with uneven/empty DP contributions."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import requires_corex30


def _worker(rank, rendezvous, sizes):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    sync = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    from vllm_iluvatar.distributed.device_communicators.ilu_communicator import (
        IluCommunicator,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.runner import (
        ilu_moe_runner as mod,
    )

    comm = SimpleNamespace(
        world_size=2, rank_in_group=rank, use_vllm_comm=True, ixfd_group=None
    )
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    group.all_gatherv = lambda x, **kwargs: IluCommunicator.all_gatherv(
        comm, x, **kwargs
    )
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: sizes)
    )
    mod.get_dp_group = lambda: group
    mod.get_forward_context = lambda: ctx
    q = torch.full((sizes[rank], 256), rank + 1, device="cuda", dtype=torch.int8)
    scale = torch.full((sizes[rank], 1), 0.125 * (rank + 1), device="cuda")
    logits = torch.full((sizes[rank], 8), rank + 0.5, device="cuda")
    ids = torch.arange(sizes[rank], device="cuda", dtype=torch.int32) + rank * 100

    def call():
        return mod.IluMoERunner._dispatch_packed_int8(
            None, (q, scale, torch.bfloat16), logits, ids
        )

    call()
    torch.cuda.synchronize()
    dist.barrier(group=sync)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed, out_logits, out_ids = call()
    q.add_(3)
    logits.add_(2)
    ids.add_(7)
    graph.replay()
    torch.cuda.synchronize()
    data, offset, dtype = packed
    assert offset == 256 and dtype == torch.bfloat16
    q_expected = torch.cat(
        [torch.full((n, 256), r + 4, dtype=torch.int8) for r, n in enumerate(sizes)]
    )
    s_expected = torch.cat(
        [torch.full((n, 1), 0.125 * (r + 1)) for r, n in enumerate(sizes)]
    )
    l_expected = torch.cat([torch.full((n, 8), r + 2.5) for r, n in enumerate(sizes)])
    i_expected = torch.cat(
        [torch.arange(n, dtype=torch.int32) + r * 100 + 7 for r, n in enumerate(sizes)]
    )
    assert torch.equal(data[:, :256].cpu(), q_expected)
    assert torch.equal(data[:, 256:260].clone().view(torch.float32).cpu(), s_expected)
    assert torch.equal(out_logits.cpu(), l_expected)
    assert torch.equal(out_ids.cpu(), i_expected)
    dist.barrier(group=sync)
    dist.destroy_process_group()


@requires_corex30
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
@pytest.mark.parametrize("sizes", [(4, 4), (0, 3), (2, 5)])
def test_packed_dispatch_graph(tmp_path, sizes):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    mp.spawn(_worker, args=(rendezvous, list(sizes)), nprocs=2, join=True)
