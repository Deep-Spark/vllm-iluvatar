# SPDX-License-Identifier: Apache-2.0
"""Exercise the PP producer's remote KV completion fence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_iluvatar.v1.core.sched import prefill_supply


@pytest.mark.parametrize(
    "handler_name",
    ["_handle_blocked_waiting_request", "_try_promote_blocked_waiting_request"],
)
@pytest.mark.parametrize("finished_receiving", [False, True])
def test_remote_kv_is_promoted_only_after_completion(
    monkeypatch, handler_name, finished_receiving
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,1")
    request = SimpleNamespace(
        request_id="short",
        status=SimpleNamespace(name="WAITING_FOR_REMOTE_KVS"),
        has_encoder_inputs=False,
        num_computed_tokens=0,
        num_tokens=7,
    )
    running = SimpleNamespace(
        request_id="long",
        num_tokens_with_spec=32,
        num_computed_tokens=0,
        num_output_placeholders=0,
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=True, is_kv_consumer=False
            )
        ),
        policy=SimpleNamespace(name="FCFS"),
        _pause_state=SimpleNamespace(name="UNPAUSED"),
        lora_config=None,
        hash_block_size=16,
        running=[running],
        skipped_waiting=(),
        waiting=_WaitingQueue([request]),
        finished_recving_kv_req_ids={"short"} if finished_receiving else set(),
    )

    def promote(req):
        scheduler.finished_recving_kv_req_ids.remove(req.request_id)
        req.status = SimpleNamespace(name="WAITING")

    handler = Mock(side_effect=promote)
    setattr(scheduler, handler_name, handler)
    plan = prefill_supply.plan(scheduler, budget=16)

    if finished_receiving:
        handler.assert_called_once_with(request)
        assert not scheduler.finished_recving_kv_req_ids
        assert plan.short_ids == {"short"}
        assert plan.limit("short", 16) == 7
        assert plan.limit("long", 16) == 9
    else:
        handler.assert_not_called()
        assert request.status.name == "WAITING_FOR_REMOTE_KVS"
        assert plan is None


class _WaitingQueue(list):
    def peek_request(self):
        return self[0]
