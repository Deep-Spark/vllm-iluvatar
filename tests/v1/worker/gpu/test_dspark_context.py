# SPDX-License-Identifier: Apache-2.0
"""CPU tests for direct-context capability dispatch and proposal fallback."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark.speculator import (
    load_draft_model,
    propose,
)


def test_direct_propose_passes_shared_buffer_and_fallback_preserves_inputs():
    hidden = torch.empty(8, 16)
    original = torch.ones(3, 16)
    aux = [original]

    def community(*args, **kwargs):
        return args[3], args[4]

    owner = SimpleNamespace(hidden_states=hidden, _ilu_community_propose=community)
    batch = SimpleNamespace(num_tokens=3)
    got, got_aux = propose(owner, batch, None, None, original, aux)
    assert got is original and got_aux is aux
    owner._ilu_target_context_direct = True
    got, got_aux = propose(owner, batch, None, None, original, [])
    assert got.data_ptr() == hidden.data_ptr() and got.shape[0] == 3 and got_aux is None
    with pytest.raises(ValueError, match="unexpectedly exported"):
        propose(owner, batch, None, None, original, aux)


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "use_ubatching",
        "lora",
    ],
)
@pytest.mark.parametrize("capability", [None, False, True])
def test_load_dispatches_optional_capability_without_model_types(
    unsupported, capability
):
    parallel = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
    )
    config = SimpleNamespace(parallel_config=parallel, lora_config=None)
    if unsupported == "lora":
        config.lora_config = object()
    elif unsupported:
        setattr(parallel, unsupported, True if unsupported == "use_ubatching" else 2)
    calls = []
    model = SimpleNamespace()
    if capability is not None:

        def bind(target, output):
            calls.append((target, output))
            return capability

        model.try_bind_target_context = bind
    owner = SimpleNamespace(
        _ilu_community_load_draft_model=lambda *args: model,
        vllm_config=config,
        hidden_states=object(),
        _ilu_target_context_direct=True,
    )
    target = object()
    assert load_draft_model(owner, target, []) is model
    expected = unsupported is None and capability is not None
    assert bool(calls) is expected
    assert owner._ilu_target_context_direct is (expected and capability is True)
    if expected:
        assert calls == [(target, owner.hidden_states)]


def test_registration_twice_preserves_community_methods(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as community

    from vllm_iluvatar.ilu_patch import general_patch
    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_dspark_speculator as wiring,
    )
    from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import speculator as impl

    originals = {
        name: object() for name in ("_sample_sequential", "load_draft_model", "propose")
    }
    target = SimpleNamespace(**originals)
    monkeypatch.setattr(community, "DSparkSpeculator", target)
    monkeypatch.setattr(
        general_patch, "_APPLIED", general_patch._APPLIED - {wiring.__name__}
    )
    wiring.register()
    wiring.register()
    for name, original in originals.items():
        saved = "_ilu_community_" + name.lstrip("_")
        assert getattr(target, saved) is original
        assert getattr(target, name) is getattr(impl, name.lstrip("_"))
