# SPDX-License-Identifier: Apache-2.0
"""Two-rank numerical checks for ordinary and stateless communicator paths."""

import os
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls


# A CPU oracle protects the 3D axis and uneven-row contracts even without transport.
@pytest.mark.parametrize("mode", ["ixaite"])
def test_axis_and_variable_lengths_on_cpu(monkeypatch, mode):
    import ixformer.distributed as ixfd

    from vllm_iluvatar.backends import resolve_device_communicator_cls

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    comm_cls = resolve_device_communicator_cls()
    comm = comm_cls.__new__(comm_cls)
    comm.world_size = 2
    comm.rank_in_group = 1
    comm.use_vllm_comm = True
    comm.ixfd_group = comm.collective_group = object()

    # Simulate the transport's leading-axis reduce; the plugin owns axis conversion.
    def reduce(output, input_, **kwargs):
        output.copy_((input_ * 1.5).chunk(2, dim=0)[1])

    # Each padded contribution carries its rank value, with one valid row on rank zero.
    def gather(output, input_, **kwargs):
        output.zero_()
        output[:1] = 1
        output[input_.shape[0] :] = input_

    monkeypatch.setattr(ixfd, "reduce_scatter_tensor", reduce)
    monkeypatch.setattr(ixfd, "all_gather_into_tensor", gather)
    comm._comm = SimpleNamespace(
        reduce_scatter_tensor=reduce, all_gather_into_tensor=gather
    )
    guard = forbid_ixformer_calls() if mode == "ixaite" else nullcontext()
    with guard:
        base = torch.arange(48.0).view(2, 3, 8)
        torch.testing.assert_close(
            comm.reduce_scatter(base * 2, dim=2), (base * 3)[..., 4:], rtol=0, atol=0
        )
        local = torch.full((3, 2), 2.0)
        gathered = comm.all_gatherv(local, dim=0, sizes=[1, 3])
        torch.testing.assert_close(
            gathered, torch.cat([torch.ones(1, 2), local]), rtol=0, atol=0
        )
        base = torch.arange(24.0).view(2, 3, 4)
        torch.testing.assert_close(
            comm.reduce_scatterv(base * 2, dim=2, sizes=[1, 3]),
            (base * 3)[..., 1:],
            rtol=0,
            atol=0,
        )


# Use exact integer-valued tensors so layout errors cannot hide in a tolerance.
def _check_collectives(comm, rank):
    device = torch.device("cuda", rank)
    x = torch.full((4,), rank + 1.0, device=device)
    torch.testing.assert_close(
        comm.all_reduce(x), torch.full_like(x, 3), rtol=0, atol=0
    )

    # The last axis of a 3D tensor exercises the gather/scatter axis transforms.
    x = torch.full((2, 3, 4), rank + 1.0, device=device)
    expected = torch.cat([torch.full_like(x, r + 1) for r in range(2)], dim=2)
    torch.testing.assert_close(comm.all_gather(x, dim=2), expected, rtol=0, atol=0)
    gathered = comm.gather(x, dst=1, dim=2)
    if rank == 1:
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    else:
        assert gathered is None
    base = torch.arange(48.0, device=device).view(2, 3, 8)
    scattered = comm.reduce_scatter(base * (rank + 1), dim=2)
    torch.testing.assert_close(
        scattered, (base * 3).chunk(2, dim=2)[rank], rtol=0, atol=0
    )

    # Include one empty rank and the equal-length fast path, plus list operands.
    for sizes in ([0, 3], [1, 3], [2, 2]):
        local = torch.full((sizes[rank], 2), rank + 1.0, device=device)
        gathered = comm.all_gatherv([local, local + 10], dim=0, sizes=sizes)
        expected = torch.cat(
            [torch.full((sizes[r], 2), r + 1.0, device=device) for r in range(2)]
        )
        torch.testing.assert_close(gathered[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(gathered[1], expected + 10, rtol=0, atol=0)
        full = torch.arange(6.0 * sum(sizes), device=device).view(2, 3, sum(sizes))
        reduced = comm.reduce_scatterv(full * (rank + 1), dim=2, sizes=sizes)
        begin = sum(sizes[:rank])
        torch.testing.assert_close(
            reduced, (full * 3)[..., begin : begin + sizes[rank]], rtol=0, atol=0
        )

    # A global source rank must be translated consistently for both broadcast APIs.
    x = torch.full((4,), 7.0 if rank == 1 else -1.0, device=device)
    torch.testing.assert_close(
        comm.broadcast(x, src=1), torch.full_like(x, 7), rtol=0, atol=0
    )
    x.fill_(11 if rank == 1 else -1)
    handle = comm.broadcast_device_tensor(x, src_global_rank=1)
    if handle is not None:
        handle.wait()
    torch.testing.assert_close(x, torch.full_like(x, 11), rtol=0, atol=0)


# Spawned ranks select the backend before platform discovery, as service workers do.
def _rank_check(rank, rendezvous, mode):
    os.environ["VLLM_ILU_KERNEL_PROVIDER"] = mode
    torch.cuda.set_device(rank)
    from vllm_iluvatar.backends import resolve_device_communicator_cls

    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    comm_cls = resolve_device_communicator_cls()
    comm = comm_cls.__new__(comm_cls)
    comm.world_size = 2
    comm.rank_in_group = rank
    comm.ranks = [0, 1]
    comm.device = torch.device("cuda", rank)
    comm.device_group = dist.group.WORLD
    comm.use_vllm_comm = True
    comm.is_stateless_group = False
    guard = forbid_ixformer_calls() if mode == "ixaite" else nullcontext()
    try:
        with guard:
            if mode == "ixaite":
                comm._comm = comm._collective_backend()
                comm.collective_group = comm._comm.init_comm_with_store(
                    dist.group.WORLD
                )
            else:
                import ixformer.distributed as ixfd

                comm.ixfd_group = ixfd.init_comm_with_store(dist.group.WORLD)
            _check_collectives(comm, rank)

            # Stateful transport shutdown must keep the owning Torch group alive.
            torch.cuda.synchronize()
            if mode == "ixaite":
                comm._comm.destroy_process_group(comm.collective_group)
            else:
                ixfd.destroy_process_group(comm.ixfd_group)
            dist.barrier()

            # Construct the actual stateless coordinator, including its selected class.
            from torch.distributed.distributed_c10d import _get_default_store
            from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

            if mode == "ixaite":
                from vllm_iluvatar.backends.ixaite.register import (
                    _apply_stateless as apply_stateless,
                )
            else:
                from vllm_iluvatar.ilu_patch.general_patches.patch_stateless_coordinator import (
                    _apply as apply_stateless,
                )

            apply_stateless()
            stateless = StatelessGroupCoordinator(
                [[0, 1]],
                rank,
                "nccl",
                True,
                _get_default_store(),
                group_name="backend_isolation",
                global_rank=rank,
                global_world_size=2,
            )
            try:
                assert type(stateless.device_communicator) is comm_cls
                _check_collectives(stateless.device_communicator, rank)
                torch.cuda.synchronize()
            finally:
                stateless.destroy()
    finally:
        dist.destroy_process_group()


# These corrected 3D/stateless contracts belong to ixaite; baseline keeps 9d6a55b.
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two Iluvatar GPUs required")
@pytest.mark.parametrize("mode", ["ixaite"])
def test_backend_collectives(tmp_path, monkeypatch, mode):
    # Interpreter startup hooks run before the spawned target sets its environment.
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    monkeypatch.delenv("VLLM_ILU_IXAITE_OP_ALLOWLIST", raising=False)
    monkeypatch.delenv("IXAITE_LARGE_AR_BACKEND", raising=False)
    rendezvous = (tmp_path / f"collectives-{mode}").as_uri()
    mp.spawn(_rank_check, args=(rendezvous, mode), nprocs=2, join=True)
