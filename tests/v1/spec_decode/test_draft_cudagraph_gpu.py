# SPDX-License-Identifier: Apache-2.0
"""Execute DraftGraphManager capture/replay on CQ, including two-rank NCCL.

The small deterministic draft substitutes model weights/attention math, not the
manager, CUDA graph wrapper, metadata staging, or collectives. It writes paged
KV and reuses index rows so stale graph inputs are observable without loading
GLM weights. This is a manager integration test, not model-quality validation.
"""

import time
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata

from tests.conftest import requires_corex30
from vllm_iluvatar.v1.spec_decode import draft_cudagraph as graphs
from vllm_iluvatar.v1.worker import dp_utils


@dataclass
class _Metadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_len: int
    num_prefills: int = 0


class _Draft(torch.nn.Module):
    def __init__(self, device, communicator=None):
        super().__init__()
        self.kv = torch.zeros(264, device=device)
        self.topk_indices_buffer = torch.full(
            (8,), -1, dtype=torch.int64, device=device
        )
        self.skip_topk = False
        self.calls = 0
        self.communicator = communicator

    @property
    def model(self):
        return self

    def set_skip_topk(self, skip):
        self.skip_topk = skip

    def forward(self, input_ids, positions, hidden_states):
        self.calls += 1  # Must not execute during graph replay.
        md = get_forward_context().attn_metadata["draft"]
        n = input_ids.numel()
        rows = torch.arange(n, device=input_ids.device) // md.query_len
        previous = (positions - 1).clamp_min(0)
        if not self.skip_topk:
            indices = md.block_table[rows, previous // 4].long() * 4 + previous % 4
            self.topk_indices_buffer[:n].copy_(indices)
        prior = self.kv[self.topk_indices_buffer[:n].clamp_min(0)]
        values = (
            input_ids.float()
            + positions.float()
            + hidden_states[:, 0]
            + md.seq_lens[rows].float()
            + prior
        )
        # Capture uses slots=-1. Redirect those writes to distinct scratch rows;
        # page zero and every real KV slot must remain unchanged during capture.
        slots = torch.where(
            md.slot_mapping >= 0,
            md.slot_mapping,
            256 + torch.arange(n, device=input_ids.device),
        )
        self.kv.index_copy_(0, slots, values)
        if self.communicator is not None:
            gathered = torch.empty(
                n * self.communicator.world_size, device=values.device
            )
            output = torch.empty_like(values)
            self.communicator.all_gather(gathered, values)
            self.communicator.reduce_scatter(output, gathered)
            return output
        return values


class _Proposer:
    def __init__(self, rank, world_size, communicator=None):
        self.device = torch.device("cuda", rank)
        self.num_speculative_tokens = 3
        self.max_model_len, self.block_size = 32, 4
        self.input_ids = torch.zeros(8, dtype=torch.int64, device=self.device)
        self.positions = torch.zeros_like(self.input_ids)
        self.hidden_states = torch.zeros(8, 1, device=self.device)
        self.model = _Draft(self.device, communicator)
        self.vllm_config = NS(
            scheduler_config=NS(max_num_seqs=4),
            compilation_config=NS(
                cudagraph_capture_sizes=[1, 2, 4, 8],
                fast_moe_cold_start=False,
                static_forward_context={},
            ),
            parallel_config=NS(
                data_parallel_size=world_size,
                data_parallel_rank=rank,
                all2all_backend="allgather_reducescatter",
                disable_nccl_for_dp_synchronization=True,
                num_ubatches=1,
                is_moe_model=True,
                use_sequence_parallel_moe=False,
            ),
        )

    def build_per_group_and_layer_attn_metadata(self, cad, draft_index=0):
        return None, {
            "draft": _Metadata(
                cad.seq_lens,
                cad.block_table_tensor,
                cad.slot_mapping,
                cad.max_query_len,
            )
        }

    def build_model_inputs_first_pass(self, n, *args):
        return dict(
            input_ids=self.input_ids[:n],
            positions=self.positions[:n],
            hidden_states=self.hidden_states[:n],
        ), None

    def _get_slot_mapping(self, n, slots):
        return {"draft": slots[:n]}


def _common(p, requests, q, start, pages):
    n = requests * q
    starts = torch.arange(requests + 1, dtype=torch.int32) * q
    positions = torch.arange(start, start + q).repeat(requests)
    table = torch.stack([torch.arange(8) + page for page in pages])
    rows = torch.arange(n) // q
    slots = table[rows, positions // 4].long() * 4 + positions % 4
    p.positions[:n].copy_(positions)
    return CommonAttentionMetadata(
        query_start_loc=starts.to(p.device),
        query_start_loc_cpu=starts,
        seq_lens=torch.full((requests,), start + q, dtype=torch.int32, device=p.device),
        seq_lens_cpu_upper_bound=torch.full((requests,), start + q, dtype=torch.int32),
        num_reqs=requests,
        num_actual_tokens=n,
        max_query_len=q,
        max_seq_len=32,
        block_table_tensor=table.to(device=p.device, dtype=torch.int32),
        slot_mapping=slots.to(p.device),
    )


@torch.inference_mode()
def _exercise(rank, world_size, communicator=None):
    p = _Proposer(rank, world_size, communicator)
    eager = _Draft(p.device, communicator)
    manager = graphs.DraftGraphManager(p)
    stream = torch.cuda.Stream(device=p.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        manager.capture()
        assert manager.ready
        for entry in manager.entries.values():
            assert entry.wrapper.concrete_cudagraph_entries
            assert all(
                isinstance(item.cudagraph, torch.cuda.CUDAGraph)
                for item in entry.wrapper.concrete_cudagraph_entries.values()
            )
        torch.testing.assert_close(p.model.kv[:256], eager.kv[:256], rtol=0, atol=0)
        captured_calls = p.model.calls
        manager.capture()  # Repeated capture is idempotent.
        assert p.model.calls == captured_calls

        # Reuse the same keys after advancing, rejecting/rolling back positions,
        # replacing page tables, shrinking/growing the batch and switching q.
        # The final round revisits the original key to catch cross-key clobbers.
        rounds = [
            (2, 4, 3, [1, 12]),
            (2, 4, 4, [1, 12]),
            (1, 4, 2, [23]),
            (4, 1, 3, [1, 12, 23, 34]),
            (1, 1, 2, [23]),
            (2, 1, 3, [12, 1]),
            (2, 4, 2, [23, 34]),
        ]
        oracle_kv = torch.zeros(256)
        replay_count = 0
        for round_id, (requests, q, start, pages) in enumerate(rounds):
            manager.reset()
            oracle_indices = None
            for draft_index in range(3):
                query_len = q if draft_index == 0 else 1
                position = start if draft_index == 0 else start + q + draft_index - 1
                cad = _common(p, requests, query_len, position, pages)
                n = cad.num_actual_tokens
                p.input_ids[:n].copy_(torch.arange(n) + 10 * round_id + rank)
                p.hidden_states[:n].fill_(draft_index + 1)
                _, md = manager.build_metadata(
                    _Proposer.build_per_group_and_layer_attn_metadata,
                    cad,
                    draft_index=draft_index,
                )
                mode, _, counts = manager.coordinate(n)
                assert mode == CUDAGraphMode.FULL
                for model in (p.model, eager):
                    if draft_index == 1:
                        last_rows = torch.arange(requests, device=p.device) * q + q - 1
                        model.topk_indices_buffer[:requests].copy_(
                            model.topk_indices_buffer[last_rows]
                        )
                    model.set_skip_topk(draft_index > 0)
                kwargs, _ = p.build_model_inputs_first_pass(n)
                # Independent scalar oracle checks physical KV writes, including
                # untouched speculative tail slots after a rejection rollback.
                positions = p.positions[:n].cpu()
                table = cad.block_table_tensor.cpu()
                if draft_index == 0:
                    oracle_indices = [
                        int(table[i // query_len, max(int(pos) - 1, 0) // 4]) * 4
                        + max(int(pos) - 1, 0) % 4
                        for i, pos in enumerate(positions)
                    ]
                elif draft_index == 1:
                    oracle_indices = [
                        oracle_indices[(i + 1) * q - 1] for i in range(requests)
                    ]
                expected = torch.tensor(
                    [
                        i
                        + 10 * round_id
                        + rank
                        + int(pos)
                        + draft_index
                        + 1
                        + position
                        + query_len
                        + float(oracle_kv[oracle_indices[i]])
                        for i, pos in enumerate(positions)
                    ]
                )
                oracle_kv[cad.slot_mapping.cpu()] = expected
                with set_forward_context(
                    md,
                    p.vllm_config,
                    num_tokens=n,
                    num_tokens_across_dp=counts,
                    cudagraph_runtime_mode=mode,
                    slot_mapping=p._get_slot_mapping(n, cad.slot_mapping),
                ):
                    context = get_forward_context()
                    original_md, original_desc = (
                        context.attn_metadata,
                        context.batch_descriptor,
                    )
                    actual = manager(**kwargs).clone()
                    assert context.attn_metadata is original_md
                    assert context.batch_descriptor is original_desc
                assert p.model.calls == captured_calls, "forward ran instead of replay"
                with set_forward_context(
                    md,
                    p.vllm_config,
                    num_tokens=n,
                    num_tokens_across_dp=counts,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ):
                    reference = eager(**kwargs)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.cpu(), expected * world_size, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    p.model.kv[:256].cpu(), oracle_kv, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    p.model.kv[:256], eager.kv[:256], rtol=0, atol=0
                )
                replay_count += 1
        assert replay_count == 21

        if world_size > 1:
            # Both keys exist, but unequal local token counts cannot replay
            # compatible collectives. The real coordinator must reject them.
            manager.first_key = graphs.DraftGraphKey(rank + 1, 1, False)
            manager.coordination_count = 0
            mode, _, counts = manager.coordinate(rank + 1)
            assert mode == CUDAGraphMode.NONE
            assert counts.tolist() == [1, 2]

        # Clearing one rank's real graphs must make every rank choose eager.
        manager.first_key = manager.key
        if rank == 0:
            manager.entries[manager.first_key].wrapper.clear_graphs()
        manager.coordination_count = 0
        mode, _, counts = manager.coordinate(manager.first_key.num_tokens)
        assert mode == CUDAGraphMode.NONE
        with set_forward_context(
            md, p.vllm_config, num_tokens=n, num_tokens_across_dp=counts
        ):
            manager(**kwargs)
        assert p.model.calls == captured_calls + 1
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def _worker(rank, world_size, rendezvous):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    if world_size == 1:
        _exercise(rank, world_size)
        return
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        # Production DP coordinator on its supported CPU group; the model's
        # captured AG+RS use the real GPU communicator, not Gloo substitutions.
        dp_utils.get_dp_group = lambda: NS(
            device=torch.device("cuda", rank),
            device_group=None,
            cpu_group=dist.group.WORLD,
        )
        comm = PyNcclCommunicator(dist.group.WORLD, device=rank)
        assert comm.available and not comm.disabled
        _exercise(rank, world_size, comm)
    finally:
        dist.destroy_process_group()


@requires_corex30
@pytest.mark.parametrize("world_size", [1, 2])
def test_real_draft_capture_replay_and_paged_kv(tmp_path, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CQ GPUs")
    context = mp.spawn(
        _worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri()),
        nprocs=world_size,
        join=False,
    )
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("draft capture/replay workers exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
