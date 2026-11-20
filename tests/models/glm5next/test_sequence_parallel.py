# SPDX-License-Identifier: Apache-2.0
"""Target SP must not leak into the MTP model or unsupported parallel layouts."""

from types import SimpleNamespace

import pytest

from vllm_iluvatar.models.glm5next import model


def _config(**changes):
    parallel = dict(
        use_sequence_parallel_moe=True,
        all2all_backend="allgather_reducescatter",
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
        enable_eplb=False,
        enable_elastic_ep=False,
    )
    parallel.update(changes)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(mhc=True)),
        parallel_config=SimpleNamespace(**parallel),
    )


def test_target_sp_preserves_draft_layout(monkeypatch):
    monkeypatch.setattr(
        model.envs, "VLLM_ILU_DECODE_EP_BACKEND", "allgather_reducescatter"
    )
    assert model._target_sequence_parallel(_config())
    assert not model._target_sequence_parallel(_config(), is_mtp_layer=True)


@pytest.mark.parametrize(
    "change",
    [
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"use_sequence_parallel_moe": False},
        {"all2all_backend": "deepep_low_latency"},
        {"enable_dbo": True},
        {"enable_eplb": True},
        {"enable_elastic_ep": True},
    ],
)
def test_target_sp_preserves_unsupported_layout(monkeypatch, change):
    assert not model._target_sequence_parallel(_config(**change))


def test_target_sp_preserves_models_without_mhc():
    config = _config()
    config.model_config.hf_config.mhc = False
    assert not model._target_sequence_parallel(config)


def test_target_sp_preserves_decode_deepep(monkeypatch):
    monkeypatch.setattr(model.envs, "VLLM_ILU_DECODE_EP_BACKEND", "deepep_low_latency")
    assert not model._target_sequence_parallel(_config())
