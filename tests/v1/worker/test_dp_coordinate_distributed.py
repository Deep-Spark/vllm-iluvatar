# SPDX-License-Identifier: Apache-2.0
"""Two real DP processes: coordination -> forward metadata -> GLM dispatch.

Uses the supported CPU/Gloo DP synchronization path. Only downstream MoE
compute is substituted: endpoint collectives verify that both ranks selected
runner, naive, or LL together; this does not exercise the DeepEP LL kernels.
"""

import os
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _coordinate_and_dispatch(rank, rendezvous, override, coordinator_api):
    if override is None:
        os.environ.pop("VLLM_ILU_DECODE_EP_BACKEND", None)
    else:
        os.environ["VLLM_ILU_DECODE_EP_BACKEND"] = override
    torch.set_num_threads(1)

    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_dp_utils,
    )

    patch_dp_utils.register()

    import vllm.forward_context as fc
    import vllm.v1.worker.dp_utils as coordinator
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import dp_utils as v2_coordinator

    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep
    from vllm_iluvatar.models.glm_moe_dsa.model import IluGlmMoeDsaMoE
    from vllm_iluvatar.v1.worker import dp_utils

    assert override == ep._DECODE_EP_BACKEND
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        # Supply group handles, while leaving the production all-reduce and
        # coordinator/metadata wiring intact.
        dp_utils.get_dp_group = lambda: SimpleNamespace(
            device="cpu", device_group=dist.group.WORLD, cpu_group=dist.group.WORLD
        )
        v2_coordinator.get_dp_group = lambda: SimpleNamespace(
            cpu_group=dist.group.WORLD
        )
        parallel = SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=rank,
            all2all_backend="allgather_reducescatter",
            disable_nccl_for_dp_synchronization=True,
            num_ubatches=1,
            is_moe_model=True,
            use_sequence_parallel_moe=False,
        )
        config = SimpleNamespace(
            parallel_config=parallel,
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False,
                static_forward_context={},
            ),
        )
        moe = IluGlmMoeDsaMoE.__new__(IluGlmMoeDsaMoE)
        moe.experts = MagicMock()
        moe.experts.is_internal_router = True
        moe.gate = MagicMock()

        # Repeated transitions expose stale cached phase bits. Reverse the
        # mixed ranks and include an empty shard to avoid a rank-0-only check.
        steps = [
            ((True, True), (3, 5)),
            ((True, False), (3, 5)),
            ((True, True), (3, 5)),
            ((False, True), (3, 5)),
            ((False, False), (3, 5)),
            ((True, False), (0, 5)),
            ((True, True), (3, 5)),
        ]
        for phases, rows in steps:
            decode = all(phases)
            if coordinator_api == "legacy":
                should_ubatch, sizes, graph_mode = (
                    coordinator.coordinate_batch_across_dp(
                        num_tokens_unpadded=rows[rank],
                        num_tokens_padded=8,
                        num_tokens_pre_cudagraph_padding=rows[rank],
                        allow_microbatching=False,
                        parallel_config=parallel,
                        uniform_decode=phases[rank],
                        cudagraph_mode=2,
                    )
                )
                assert not should_ubatch
                assert graph_mode == (2 if decode else 0)
                assert sizes.tolist() == ([8, 8] if decode else list(rows))
            else:
                batch_desc, synced = v2_coordinator.dispatch_cg_and_sync_dp(
                    cudagraph_manager=None,
                    num_reqs=int(rows[rank] > 0),
                    num_tokens=rows[rank],
                    uniform_token_count=1 if phases[rank] else None,
                    dp_size=2,
                    dp_rank=rank,
                    need_eager=True,
                    uniform_decode=phases[rank],
                )
                assert synced is not None
                sizes = synced.num_tokens_across_dp
                graph_mode = batch_desc.cg_mode.value
                assert graph_mode == CUDAGraphMode.NONE.value
                assert sizes.tolist() == list(rows)
            assert dp_utils.get_last_all_ranks_decode_only() is decode
            expected = {None: 0, "naive": 1, "deepep_low_latency": 2}[
                override if decode else None
            ]
            calls = []

            def endpoint(path, x, expected=expected, calls=calls):
                choices = [torch.empty(1, dtype=torch.int32) for _ in range(2)]
                dist.all_gather(choices, torch.tensor([path], dtype=torch.int32))
                assert [choice.item() for choice in choices] == [expected, expected]
                calls.append(path)
                return x + path

            moe.experts.side_effect = lambda *, hidden_states, router_logits: endpoint(
                0, hidden_states
            )
            moe.forward_dp_ep_naive = lambda x, *a, **kw: endpoint(1, x)
            moe.forward_dp_ep_deep_ep = lambda x, *a, **kw: endpoint(2, x)
            x = torch.zeros(int(sizes[rank]), 4, dtype=torch.bfloat16)
            # Neither all_ranks_decode_only nor get_forward_context is mocked.
            # Both DPMetadata.make and set_forward_context use the production
            # patch to propagate the last coordinated bit into this invocation.
            with fc.set_forward_context(
                None,
                config,
                num_tokens=x.shape[0],
                num_tokens_across_dp=sizes,
                cudagraph_runtime_mode=CUDAGraphMode(graph_mode),
            ):
                metadata = fc.get_forward_context().dp_metadata
                assert metadata.all_ranks_decode_only is decode
                torch.testing.assert_close(metadata.num_tokens_across_dp_cpu, sizes)
                output = moe.forward(x)
            assert calls == [expected]
            torch.testing.assert_close(output, x + expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("override", [None, "naive", "deepep_low_latency"])
@pytest.mark.parametrize("coordinator_api", ["legacy", "v2"])
def test_real_dp_coordination_and_glm_dispatch(tmp_path, override, coordinator_api):
    context = mp.spawn(
        _coordinate_and_dispatch,
        args=((tmp_path / "rendezvous").as_uri(), override, coordinator_api),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("DP coordination workers exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
