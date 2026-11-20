# SPDX-License-Identifier: Apache-2.0
"""Eliminate DP dummy-batch blocking while waiting for async remote KV transfers.

Corresponds to: vllm/v1/worker/gpu/model_runner.py
                vllm/v1/executor/multiproc_executor.py
                vllm/v1/executor/abstract.py
                vllm/v1/engine/core.py
Baseline      : v0.31.0 (db9527a468)
Why           : Under DP+EP (+ MTP), a rank with requests parked in
                ``WAITING_FOR_REMOTE_KVS`` schedules 0 tokens and enters
                ``execute_dummy_batch()`` alongside idle DP peers:
                1) Even when ALL DP ranks are only running ``execute_dummy_batch``
                   (0 real tokens globally), every rank still ran a full target +
                   5-step MTP dummy pass (~120-150 ms) on GPU.
                2) ``_dummy_run`` disables ``kv_connector`` and discards worker
                   outputs, so completed NIXL transfers sitting in the worker
                   outbox were not harvested until the next step's ``no_forward``
                   *after* ``scheduler.schedule()`` had already returned 0 tokens,
                   forcing a second full ~120 ms ``execute_dummy_batch`` penalty.
How           : 1) ``merged_dp_sync`` / ``dp_utils`` short-circuits to 0 tokens
                   when all DP ranks are in runtime ``execute_dummy_batch``;
                   ``GPUModelRunner.{execute_model,_dummy_run}`` wrappers exit
                   cleanly before running target/MTP GPU work.
                2) ``IluWorker.execute_dummy_batch`` polls ``KVConnectorOutput``
                   after ``_dummy_run``; ``Executor`` / ``MultiprocExecutor``
                   aggregate it via ``kv_output_aggregator``; ``EngineCore``
                   feeds non-empty ``kv_connector_output`` into
                   ``scheduler.update_from_output`` immediately so the very next
                   ``scheduler.schedule()`` promotes ``WAITING_FOR_REMOTE_KVS``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p

ENGINE_CORE_DUMMY_SOURCE = """\
def execute_dummy_batch(self):
    self.model_executor.execute_dummy_batch()
"""

MULTIPROC_EXECUTOR_DUMMY_SOURCE = """\
def execute_dummy_batch(self) -> None:
    self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)
"""

BASE_EXECUTOR_DUMMY_SOURCE = """\
def execute_dummy_batch(self) -> None:
    self.collective_rpc("execute_dummy_batch")
"""


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DP idle dummy-batch all-idle short-circuit and KVConnector harvest",
    )


def _apply() -> None:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    import vllm_iluvatar.v1.engine.core as core_impl
    from vllm_iluvatar.v1.worker.gpu.model_runner import (
        make_idle_dummy_short_circuit_dummy_run,
        make_idle_dummy_short_circuit_execute_model,
    )

    p.check_fingerprint(
        "EngineCore.execute_dummy_batch",
        inspect.getsource(EngineCore.execute_dummy_batch),
        ENGINE_CORE_DUMMY_SOURCE,
        resync_hint="Review EngineCore.execute_dummy_batch before updating wrapper.",
    )
    p.check_fingerprint(
        "MultiprocExecutor.execute_dummy_batch",
        inspect.getsource(MultiprocExecutor.execute_dummy_batch),
        MULTIPROC_EXECUTOR_DUMMY_SOURCE,
        resync_hint="Review MultiprocExecutor.execute_dummy_batch before updating wrapper.",
    )
    p.check_fingerprint(
        "Executor.execute_dummy_batch",
        inspect.getsource(Executor.execute_dummy_batch),
        BASE_EXECUTOR_DUMMY_SOURCE,
        resync_hint="Review Executor.execute_dummy_batch before updating wrapper.",
    )

    mr_exec = make_idle_dummy_short_circuit_execute_model(GPUModelRunner.execute_model)
    mr_dummy = make_idle_dummy_short_circuit_dummy_run(GPUModelRunner._dummy_run)
    p.check_signatures(
        GPUModelRunner,
        SimpleNamespace(execute_model=mr_exec, _dummy_run=mr_dummy),
        ["execute_model", "_dummy_run"],
        strict=True,
    )
    GPUModelRunner.execute_model = mr_exec  # type: ignore[method-assign]
    GPUModelRunner._dummy_run = mr_dummy  # type: ignore[method-assign]

    mp_dummy = core_impl._make_multiproc_execute_dummy_batch(
        MultiprocExecutor.execute_dummy_batch
    )
    base_dummy = core_impl._make_base_executor_execute_dummy_batch(
        Executor.execute_dummy_batch
    )
    ec_dummy = core_impl._make_engine_core_execute_dummy_batch(
        EngineCore.execute_dummy_batch
    )

    p.check_signatures(
        MultiprocExecutor,
        SimpleNamespace(execute_dummy_batch=mp_dummy),
        ["execute_dummy_batch"],
        strict=True,
    )
    p.check_signatures(
        Executor,
        SimpleNamespace(execute_dummy_batch=base_dummy),
        ["execute_dummy_batch"],
        strict=True,
    )
    p.check_signatures(
        EngineCore,
        SimpleNamespace(execute_dummy_batch=ec_dummy),
        ["execute_dummy_batch"],
        strict=True,
    )

    MultiprocExecutor.execute_dummy_batch = mp_dummy  # type: ignore[method-assign]
    Executor.execute_dummy_batch = base_dummy  # type: ignore[method-assign]
    EngineCore.execute_dummy_batch = ec_dummy  # type: ignore[method-assign]
