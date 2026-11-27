# SPDX-License-Identifier: Apache-2.0
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.scheduler import (
    NixlConnectorScheduler,
)


@pytest.mark.parametrize(
    "submitted", [False, True], ids=["before-submit", "after-submit"]
)
def test_prefill_cancel_releases_blocks_without_a_worker_send(submitted):
    # Use the producer's local request hooks without initializing its transport.
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.is_consumer = False
    scheduler.prefix_state = None
    scheduler.kv_group_nums = 1
    scheduler.transfer_slices = (None,)
    scheduler._has_mamba = False
    scheduler._force_rank_wise_sends = True
    scheduler._kv_lease_duration = 30
    scheduler._reqs_to_send = {}
    scheduler._reqs_blocks = {}
    scheduler._reqs_to_send_time_record = {}
    scheduler._pending_worker_sends = set()
    scheduler._sends_completed_early = set()
    scheduler._sends_awaiting_release = set()

    params = {
        "do_remote_decode": True,
        "remote_engine_id": "decode",
        "transfer_id": "transfer",
    }
    request = Request(
        request_id="prefill",
        prompt_token_ids=[1] * 128,
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"kv_transfer_params": params}
        ),
        pooling_params=None,
    )
    num_scheduled_tokens = 128 if submitted else 64
    group = [KVCacheBlock(3), KVCacheBlock(8)] if submitted else [KVCacheBlock(3)]
    blocks = KVCacheBlocks((group,))
    block_ids = blocks.get_block_ids()

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)
    assert request.kv_transfer_params["do_remote_decode"] is False
    request.status = RequestStatus.RUNNING
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs.append(NewRequestData.from_request(request, block_ids))
    output.num_scheduled_tokens[request.request_id] = num_scheduled_tokens
    output.total_num_scheduled_tokens = num_scheduled_tokens
    metadata = scheduler.build_connector_meta(output)
    assert (request.request_id in metadata.reqs_to_send) is submitted

    request.status = RequestStatus.FINISHED_ABORTED
    delay_free, _ = scheduler.request_finished_all_groups(request, block_ids)

    assert delay_free is submitted, f"{delay_free=}, {submitted=}"
    assert not scheduler._reqs_to_send
    assert not scheduler._reqs_blocks
    assert scheduler.has_pending_push_work() is submitted
    if submitted:
        completion = KVConnectorOutput(finished_sending={request.request_id})
        scheduler.update_connector_output(completion)
        assert completion.finished_sending == {request.request_id}
        assert not scheduler.has_pending_push_work()
