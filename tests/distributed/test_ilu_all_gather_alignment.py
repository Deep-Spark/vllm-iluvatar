# SPDX-License-Identifier: Apache-2.0
"""IluCommunicator.all_gather must stay correct inside a captured CUDA graph.

On this platform an all_gather whose per-rank message is not a multiple of 256
bytes returns wrong data once NCCL picks the SIMPLE protocol, but only inside a
CUDA graph. It is a platform issue rather than an ixformer one:
torch.distributed's own NCCL all_gather corrupts identically, eager is always
correct, and all_reduce is unaffected. The communicator pads the message to
dodge it; this test pins that.

32320 is the size that broke DSpark: 129280 vocab / 4 ranks in bf16 is 63.1 KiB,
which is past the protocol switch and 128 bytes short of alignment.

Run under torchrun with 4 ranks, matching the TP4 decoder:

    IXFORMER_COMM_ALL_GATHER_ALGO=NCCL \
    torchrun --nproc_per_node=4 -m pytest tests/distributed/\
test_ilu_all_gather_alignment.py
"""

from __future__ import annotations

import os

import pytest
import torch

UNALIGNED = [1000, 8200, 31808, 32320, 100000]
ALIGNED = [1024, 8192, 32384]


pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs torchrun with at least 2 ranks",
)


@pytest.fixture(scope="module")
def comm():
    import torch.distributed as dist

    from vllm_iluvatar.distributed.device_communicators.ilu_communicator import (
        IluCommunicator,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    cpu_group = dist.new_group(backend="gloo")
    communicator = IluCommunicator(
        cpu_group=cpu_group,
        device=torch.device("cuda", rank),
        device_group=dist.group.WORLD,
        unique_name="test_tp",
    )
    yield communicator
    dist.barrier()


def _expect(world, cols, dtype, device, it):
    def rank_data(r):
        base = (it * 1000 + r * 97 + 1) % 251
        flat = torch.arange(cols, device=device, dtype=torch.float32)
        return ((flat % 61) + base).to(dtype)

    return [rank_data(r) for r in range(world)]


@pytest.mark.parametrize("cols", UNALIGNED + ALIGNED)
def test_all_gather_in_cuda_graph(comm, cols):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16

    local = torch.empty((1, cols), dtype=dtype, device=device)
    out: list[torch.Tensor] = []

    def run():
        out.clear()
        out.append(comm.all_gather(local, dim=-1))

    local.copy_(_expect(world, cols, dtype, device, 0)[rank])
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    captured = out[0]
    torch.cuda.synchronize()

    for it in range(1, 6):
        parts = _expect(world, cols, dtype, device, it)
        local.copy_(parts[rank])
        graph.replay()
        torch.cuda.synchronize()
        want = torch.cat(parts).reshape(1, world * cols)
        assert torch.equal(captured, want), (
            f"rank{rank} cols={cols} iter={it}: "
            f"{(captured != want).sum().item()}/{want.numel()} elements wrong"
        )
