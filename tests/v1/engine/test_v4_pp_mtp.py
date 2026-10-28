# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for batch-bound PP MTP proposals; no model/GPU execution."""

import pickle
from types import SimpleNamespace

import pytest
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput

from vllm_iluvatar.v1.engine.ilu_pp_mtp import (
    needs_paired_drafts,
    wrap_post_step,
    wrap_sample_tokens,
    wrap_update_from_output,
)


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4")
        ),
        speculative_config=SimpleNamespace(method="mtp"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )


@pytest.mark.parametrize(
    "excluded", ["model", "no_spec", "method", "pp1", "v1", "async"]
)
def test_scope_exclusions(excluded):
    cfg = config()
    assert needs_paired_drafts(cfg)
    if excluded == "model":
        cfg.model_config.hf_config.model_type = "other"
    elif excluded == "no_spec":
        cfg.speculative_config = None
    elif excluded == "method":
        cfg.speculative_config.method = "eagle"
    elif excluded == "pp1":
        cfg.parallel_config.pipeline_parallel_size = 1
    elif excluded == "v1":
        cfg.use_v2_model_runner = False
    else:
        cfg.scheduler_config.async_scheduling = True
    assert not needs_paired_drafts(cfg)


class PendingOutput(AsyncModelRunnerOutput):
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def get_output(self):
        self.calls += 1
        return self.output


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batch_identity_snapshot_and_accounting_order(asynchronous):
    output = ModelRunnerOutput(
        req_ids=["first"], req_id_to_index={"first": 0}, sampled_token_ids=[[7]]
    )
    pending = PendingOutput(output)
    proposals = DraftTokenIds(["first"], [[11]])
    worker = SimpleNamespace(
        vllm_config=config(),
        model_runner=SimpleNamespace(
            is_last_pp_rank=True, take_draft_token_ids=lambda: proposals
        ),
    )
    sample = wrap_sample_tokens(
        lambda self, grammar: pending if asynchronous else output
    )
    paired = sample(worker, None)
    # A later worker step must not overwrite this queued batch's proposals.
    proposals.req_ids[0] = "later"
    proposals.draft_token_ids[0][0] = 99
    if asynchronous:
        assert pending.calls == 0
        paired = paired.get_output()
        assert pending.calls == 1
    assert paired is output
    paired = pickle.loads(pickle.dumps(paired))  # Worker response MQ serialization.
    events = []
    scheduler = SimpleNamespace(
        vllm_config=config(),
        requests={
            "first": SimpleNamespace(num_stale_output_tokens=0, num_output_tokens=0)
        },
        update_draft_token_ids=lambda d: events.append(
            ("draft", d.req_ids, d.draft_token_ids)
        ),
    )
    result = object()

    def account(self, scheduled, sampled):
        assert sampled.sampled_token_ids == [[7]]
        events.append("samples_accounted")
        self.requests["first"].num_output_tokens += 1
        return result

    update = wrap_update_from_output(account)
    assert update(scheduler, object(), paired) is result
    assert events == ["samples_accounted", ("draft", ["first"], [[11]])]


@pytest.mark.parametrize("model_executed", [False, True])
def test_post_step_avoids_unpaired_latest_draft_rpc(model_executed):
    calls = []
    post = wrap_post_step(lambda self, executed: calls.append(executed))
    engine = SimpleNamespace(vllm_config=config())
    assert post(engine, model_executed) is None
    assert not calls
    engine.vllm_config.parallel_config.pipeline_parallel_size = 1
    post(engine, model_executed)
    assert calls == [model_executed]


def test_missing_pair_fails_before_output_accounting():
    events = []
    update = wrap_update_from_output(lambda *args: events.append("accounted"))
    scheduler = SimpleNamespace(vllm_config=config())
    with pytest.raises(RuntimeError, match="missing its batch-bound"):
        update(scheduler, None, ModelRunnerOutput(["request"], {"request": 0}))
    assert not events


def test_request_identity_mismatch_rejected():
    worker = SimpleNamespace(
        vllm_config=config(),
        model_runner=SimpleNamespace(
            is_last_pp_rank=True,
            take_draft_token_ids=lambda: DraftTokenIds(["other"], [[11]]),
        ),
    )
    sample = wrap_sample_tokens(
        lambda *args: ModelRunnerOutput(["sampled"], {"sampled": 0})
    )
    with pytest.raises(RuntimeError, match="identities differ"):
        sample(worker, None)


@pytest.mark.parametrize("case", ["nonlast", "none", "other_model"])
def test_sample_delegation_unaffected(case):
    cfg = config()
    if case == "other_model":
        cfg.model_config.hf_config.model_type = "other"
    output = None if case == "none" else object()

    def forbidden():
        raise AssertionError("unrelated draft handler must not be read")

    worker = SimpleNamespace(
        vllm_config=cfg,
        model_runner=SimpleNamespace(
            is_last_pp_rank=case != "nonlast", take_draft_token_ids=forbidden
        ),
    )
    assert wrap_sample_tokens(lambda *args: output)(worker, None) is output


@pytest.mark.parametrize(
    "case", ["empty_sample", "stale", "discarded", "finished", "replaced"]
)
def test_discarded_batch_does_not_publish_old_proposals(case):
    request = SimpleNamespace(
        num_stale_output_tokens=1 if case == "stale" else 0, num_output_tokens=4
    )
    published = []
    scheduler = SimpleNamespace(
        vllm_config=config(),
        requests={"r": request},
        update_draft_token_ids=published.append,
    )
    output = ModelRunnerOutput(
        ["r"], {"r": 0}, sampled_token_ids=[[] if case == "empty_sample" else [7]]
    )
    worker = SimpleNamespace(
        vllm_config=config(),
        model_runner=SimpleNamespace(
            is_last_pp_rank=True,
            take_draft_token_ids=lambda: DraftTokenIds(["r"], [[11]]),
        ),
    )
    output = wrap_sample_tokens(lambda *args: output)(worker, None)

    def account(self, scheduled, sampled):
        if case == "finished":
            request.num_output_tokens += 1
            del self.requests["r"]
        elif case == "replaced":
            request.num_output_tokens += 1
            self.requests["r"] = SimpleNamespace(num_output_tokens=5)
        elif case == "stale":
            request.num_output_tokens += 1  # Even delivered stale output has old KV.

    wrap_update_from_output(account)(scheduler, None, output)
    assert not published
