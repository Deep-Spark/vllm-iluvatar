"""Paired GPU storage stays owned until native teardown proves it reusable."""

import asyncio
import gc
import weakref
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.lifecycle import (
    NixlDrainIncomplete,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    NixlConnectorWorker,
)


def worker():
    value = NixlConnectorWorker.__new__(NixlConnectorWorker)
    value.remote_sessions = {}
    value._send_requests = {}
    value._send_transfers = {}
    value._recv_requests = {}
    value.src_xfer_side_handles = {}
    value.dst_xfer_side_handles = {}
    value._remote_agents = {}
    value._registered_descs = [object()]
    tensor = torch.zeros(128)
    value._registered_cache_owners = {tensor.data_ptr(): tensor}
    owner = weakref.ref(tensor)
    value.producer_disconnect_drain_timeout = 0
    value._poll_send_transfers = lambda **_kwargs: None
    value._progress_sessions = lambda _now: None
    value._log_failure = lambda *_args, **_kwargs: None
    value.nixl_wrapper = NS(
        deregister_memory=lambda _desc: None,
        release_dlist_handle=lambda _handle: None,
        remove_remote_agent=lambda _name: None,
    )
    return value, owner


@pytest.mark.parametrize("direction", ["send", "recv"])
def test_live_transfer_blocks_deregistration_and_retains_storage(direction):
    value, owner = worker()
    transfers = value._send_transfers if direction == "send" else value._recv_requests
    transfers["live-request"] = object()
    with pytest.raises(NixlDrainIncomplete, match="transfers"):
        asyncio.run(value._drain_and_cleanup_on_loop())
    gc.collect()
    assert owner() is not None and value._registered_descs
    transfers.clear()  # Simulate the native generation's terminal proof.
    asyncio.run(value._drain_and_cleanup_on_loop())
    gc.collect()
    assert owner() is None and not value._registered_descs


def test_failed_deregistration_keeps_the_exact_owner_for_a_successful_retry():
    value, owner = worker()
    desc = value._registered_descs[0]

    def fail(_desc):
        raise RuntimeError("backend still owns registered region")

    value.nixl_wrapper.deregister_memory = fail
    with pytest.raises(NixlDrainIncomplete, match="deregistration"):
        asyncio.run(value._drain_and_cleanup_on_loop())
    assert value._registered_descs == [desc]
    gc.collect()
    assert owner() is not None
    value.nixl_wrapper.deregister_memory = lambda _desc: None
    asyncio.run(value._drain_and_cleanup_on_loop())
    gc.collect()
    assert owner() is None


def test_failed_native_descriptor_release_does_not_forget_registration():
    value, owner = worker()
    value.src_xfer_side_handles["source"] = object()

    def fail_release(_handle):
        raise RuntimeError("busy")

    value.nixl_wrapper.release_dlist_handle = fail_release
    with pytest.raises(NixlDrainIncomplete, match="descriptors"):
        asyncio.run(value._drain_and_cleanup_on_loop())
    assert "source" in value.src_xfer_side_handles
    assert value._registered_descs and owner() is not None
    value.nixl_wrapper.release_dlist_handle = lambda _: None
    asyncio.run(value._drain_and_cleanup_on_loop())
    gc.collect()
    assert owner() is None


def test_shutdown_delivers_queued_teardown_reports_before_releasing_owners():
    value, owner = worker()

    async def run():
        value._teardown_outbox = asyncio.Queue()
        value._teardown_outbox.put_nowait("terminal proof")

        async def report():
            await asyncio.sleep(0.01)
            assert owner() is not None
            assert await value._teardown_outbox.get() == "terminal proof"
            value._teardown_outbox.task_done()

        reporter = asyncio.create_task(report())
        await value._drain_and_cleanup_on_loop()
        await reporter

    asyncio.run(run())
    gc.collect()
    assert owner() is None
