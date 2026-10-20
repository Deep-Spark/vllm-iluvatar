"""Sampler/dummy phase contracts without GPU or process-group initialization."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_iluvatar.distributed import dp_vocab_projection as impl


@pytest.fixture
def harness(monkeypatch):
    def original(processor, head, hidden, bias):
        return F.linear(hidden, head.weight, bias)

    monkeypatch.setattr(LogitsProcessor, "_apply_head", original)
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 16
    head = SimpleNamespace(weight=weight)
    projector = SimpleNamespace(dummy=torch.zeros(1, 4), calls=[])

    def project(hidden, current_weight):
        projector.calls.append((hidden.clone(), current_weight))
        return F.linear(hidden, current_weight)

    projector.project = project
    controller = impl.DPVocabController(projector, head, head)
    impl._enable_scoped_head_dispatch()
    processor = SimpleNamespace(head_dtype=None)

    def head_call(hidden, bias=None):
        return LogitsProcessor._apply_head(processor, head, hidden, bias)

    sequential = controller.wrap_sequential(lambda hidden: head_call(hidden))
    propose = controller.wrap_propose(lambda hidden, **kwargs: sequential(hidden))
    sample = controller.wrap_sample(head_call)
    return SimpleNamespace(controller=controller, projector=projector, head=head,
                           head_call=head_call, sample=sample, propose=propose)


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 12, 24])
def test_target_and_draft_have_one_projection_each(harness, rows):
    x = torch.ones(rows, 4)
    expected = F.linear(x, harness.head.weight)
    torch.testing.assert_close(harness.sample(x), expected)
    torch.testing.assert_close(harness.propose(x), expected)
    assert len(harness.projector.calls) == 2
    assert not harness.controller.pending_target
    assert impl._active_projection.get() is None


def test_idle_dummy_replaces_the_missing_target_projection(harness):
    for idle in (False, True, True, False):
        x = torch.ones(2, 4)
        if not idle:
            harness.sample(x)
        harness.propose(x, dummy_run=idle)
    assert len(harness.projector.calls) == 8
    assert harness.projector.calls[2][0].shape[0] == 1
    assert not harness.controller.pending_target


def test_unscoped_prompt_logits_keep_the_original_path(harness):
    x = torch.ones(2, 4)
    harness.head_call(x)
    assert not harness.projector.calls


def test_parameter_replacement_is_resolved_at_the_call(harness):
    weight = torch.full_like(harness.head.weight, 3)
    harness.head.weight = weight
    expected = F.linear(torch.ones(1, 4), weight)
    torch.testing.assert_close(harness.sample(torch.ones(1, 4)), expected)
    harness.propose(torch.ones(1, 4))
    assert all(call[1] is weight for call in harness.projector.calls)


def test_real_draft_requires_target_phase(harness):
    with pytest.raises(RuntimeError, match="preceding target"):
        harness.propose(torch.ones(1, 4))


def test_duplicate_target_or_dummy_is_rejected(harness):
    harness.sample(torch.ones(1, 4))
    with pytest.raises(RuntimeError, match="Previous target"):
        harness.sample(torch.ones(1, 4))
    with pytest.raises(RuntimeError, match="duplicate"):
        harness.propose(torch.ones(1, 4), dummy_run=True)


def test_scope_is_cleared_after_failure(harness):
    with pytest.raises(ValueError):
        with harness.controller.scope():
            raise ValueError("test")
    assert impl._active_projection.get() is None


def test_dispatch_install_is_idempotent(harness):
    original = LogitsProcessor._apply_head
    impl._enable_scoped_head_dispatch()
    assert LogitsProcessor._apply_head is original


def test_projection_scope_rejects_bias_without_changing_default(harness):
    x = torch.ones(1, 4)
    with pytest.raises(ValueError, match="precision or bias"):
        harness.sample(x, torch.ones(8))
    expected = F.linear(x, harness.head.weight, torch.ones(8))
    torch.testing.assert_close(harness.head_call(x, torch.ones(8)), expected)
