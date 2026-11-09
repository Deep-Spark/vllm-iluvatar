# SPDX-License-Identifier: Apache-2.0
"""Keep the DeepSeek single-tensor contract covered alongside backend selection."""

from types import SimpleNamespace

from vllm_iluvatar.v1.spec_decode.llm_base_proposer import make_model_returns_tuple


def _fake_proposer(method, architectures):
    """Minimal ``SpecDecodeBaseProposer`` stand-in for the attributes read by
    both the ilu wrapper and the community implementation."""
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
    )


def _orig(calls, *, ret):
    def model_returns_tuple(self):
        calls.append(self)
        return ret

    return model_returns_tuple


def test_ilu_deepseek_v4_mtp_delegates_single_tensor_contract():
    calls = []
    wrapped = make_model_returns_tuple(_orig(calls, ret=False))

    self_ = _fake_proposer("mtp", ["IluDeepSeekV4MTPModel"])

    assert wrapped(self_) is False
    assert calls == [self_]
