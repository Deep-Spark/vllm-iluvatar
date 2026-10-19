import weakref

from vllm_iluvatar.v1.worker.prefill_state.consumer import ConsumerPrefixState


def test_consumer_does_not_retain_warmup_batch():
    class Batch:
        pass

    state = ConsumerPrefixState.__new__(ConsumerPrefixState)
    state._p21_ready = False
    state._p21_batch = None
    batch = Batch()
    reference = weakref.ref(batch)

    state.prepare_inputs(batch)
    del batch
    assert reference() is None

    state._p21_ready = True
    batch = Batch()
    reference = weakref.ref(batch)
    state.prepare_inputs(batch)
    del batch
    assert reference() is not None
    state.end_step()
    assert reference() is None
