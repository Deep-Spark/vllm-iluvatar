# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar ``SpecDecodeBaseProposer.model_returns_tuple``
override.

Two layers are covered:

1. The pure ``make_model_returns_tuple`` wrapper factory
   (``vllm_iluvatar.v1.spec_decode.llm_base_proposer``) — exercised with a fake
   ``orig`` and a minimal proposer stand-in.
2. The general-patch wiring (``patch_llm_base_proposer``) — because the
   community ``model_returns_tuple`` only reads ``self.method`` and
   ``self.draft_model_config.hf_config.architectures``, we can drive the real
   patched class method with a fake ``self`` end-to-end (both the ilu-short
   circuit and the community delegation paths).

No GPU required.
"""

from __future__ import annotations

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


# --- factory: ilu short-circuit --------------------------------------------


def test_returns_true_for_ilu_glm_mtp_without_delegating():
    calls = []
    wrapped = make_model_returns_tuple(_orig(calls, ret="ORIG"))

    self_ = _fake_proposer("mtp", ["IluGlmMoeDsaMTPModel"])

    assert wrapped(self_) is True
    assert calls == [], "orig must not be called on the ilu short-circuit"


def test_returns_true_when_ilu_arch_among_multiple():
    wrapped = make_model_returns_tuple(_orig([], ret=False))

    self_ = _fake_proposer("mtp", ["SomethingElse", "IluGlmMoeDsaMTPModel"])

    assert wrapped(self_) is True


# --- factory: delegation ----------------------------------------------------


def test_mtp_non_ilu_arch_delegates_to_orig():
    calls = []
    wrapped = make_model_returns_tuple(_orig(calls, ret=True))

    self_ = _fake_proposer("mtp", ["DeepSeekMTPModel"])

    assert wrapped(self_) is True
    assert calls == [self_]


def test_mtp_non_ilu_arch_returns_orig_false():
    wrapped = make_model_returns_tuple(_orig([], ret=False))

    self_ = _fake_proposer("mtp", ["SomeOtherMTPModel"])

    assert wrapped(self_) is False


def test_non_mtp_method_delegates_even_with_ilu_arch():
    calls = []
    wrapped = make_model_returns_tuple(_orig(calls, ret="ORIG"))

    self_ = _fake_proposer("eagle", ["IluGlmMoeDsaMTPModel"])

    assert wrapped(self_) == "ORIG"
    assert calls == [self_]


def test_mtp_none_architectures_delegates_without_crash():
    calls = []
    wrapped = make_model_returns_tuple(_orig(calls, ret=False))

    self_ = _fake_proposer("mtp", None)

    assert wrapped(self_) is False
    assert calls == [self_]


def test_preserves_orig_metadata():
    def orig(self):
        return False

    orig.__name__ = "model_returns_tuple"
    wrapped = make_model_returns_tuple(orig)

    assert wrapped.__wrapped__ is orig
    assert wrapped.__name__ == "model_returns_tuple"


# --- wiring: real patched class + fake self --------------------------------


def test_patch_binds_wrapper_on_class():
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    from vllm_iluvatar.ilu_patch.general_patches import patch_llm_base_proposer

    patch_llm_base_proposer.register()

    assert hasattr(SpecDecodeBaseProposer.model_returns_tuple, "__wrapped__"), (
        "model_returns_tuple not wrapped by the ilu override"
    )


def test_patch_short_circuits_ilu_arch_via_class():
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    from vllm_iluvatar.ilu_patch.general_patches import patch_llm_base_proposer

    patch_llm_base_proposer.register()

    fake = _fake_proposer("mtp", ["IluGlmMoeDsaMTPModel"])
    assert SpecDecodeBaseProposer.model_returns_tuple(fake) is True


def test_patch_delegates_to_community_for_deepseek_via_class():
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    from vllm_iluvatar.ilu_patch.general_patches import patch_llm_base_proposer

    patch_llm_base_proposer.register()

    # Community path: only DeepSeekMTPModel returns True for the mtp method.
    assert (
        SpecDecodeBaseProposer.model_returns_tuple(
            _fake_proposer("mtp", ["DeepSeekMTPModel"])
        )
        is True
    )
    assert (
        SpecDecodeBaseProposer.model_returns_tuple(
            _fake_proposer("mtp", ["NotAnMTPModel"])
        )
        is False
    )


def test_patch_register_is_idempotent():
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    from vllm_iluvatar.ilu_patch.general_patches import patch_llm_base_proposer

    patch_llm_base_proposer.register()
    first = SpecDecodeBaseProposer.model_returns_tuple
    patch_llm_base_proposer.register()
    patch_llm_base_proposer.register()

    assert SpecDecodeBaseProposer.model_returns_tuple is first
