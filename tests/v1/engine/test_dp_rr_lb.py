# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the global round-robin DP load balancer."""

from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

from vllm_iluvatar.v1.engine import core_client


def _set_rr_env(module, rr_file: str) -> None:
    module.envs = SimpleNamespace(
        VLLM_ILU_DP_ROUND_ROBIN_LB=True,
        VLLM_ILU_DP_ROUND_ROBIN_LB_FILE=rr_file,
    )


def _clear_rr_cache() -> None:
    for fd, mm in core_client._DP_RR_MMAP_CACHE.values():
        mm.close()
        core_client.os.close(fd)
    core_client._DP_RR_MMAP_CACHE.clear()


def _rr_worker(rr_file: str, num_engines: int, calls: int, queue) -> None:
    from vllm_iluvatar.v1.engine import core_client as worker_core_client

    worker_core_client._DP_RR_MMAP_CACHE.clear()
    _set_rr_env(worker_core_client, rr_file)
    results = [
        worker_core_client._next_global_rr_engine_index(num_engines, 0)[0]
        for _ in range(calls)
    ]
    queue.put(results)


def test_next_global_rr_engine_index_uses_fallback_then_rotates(
    tmp_path,
):
    _clear_rr_cache()
    rr_file = tmp_path / "rr.state"
    _set_rr_env(core_client, str(rr_file))

    try:
        assert core_client._next_global_rr_engine_index(3, 1) == (1, 2)
        assert core_client._next_global_rr_engine_index(3, 99) == (2, 0)
        assert core_client._next_global_rr_engine_index(3, 99) == (0, 1)
    finally:
        _clear_rr_cache()


def test_next_global_rr_engine_index_is_shared_across_processes(
    tmp_path,
):
    _clear_rr_cache()
    rr_file = tmp_path / "rr-shared.state"
    _set_rr_env(core_client, str(rr_file))

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_rr_worker, args=(str(rr_file), 4, 2, queue))
        for _ in range(4)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0

        results = []
        for _ in processes:
            results.extend(queue.get(timeout=1))
        assert sorted(results) == [0, 0, 1, 1, 2, 2, 3, 3]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        _clear_rr_cache()
