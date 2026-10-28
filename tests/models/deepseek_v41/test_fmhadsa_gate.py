from types import SimpleNamespace

from vllm_iluvatar.models.deepseek_v41 import attention as attn


def test_prefill_fmhadsa_off_when_env_disabled(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_V41_PREFILL_FMHADSA_ENV", False)
    monkeypatch.setattr(attn, "is_ivcore30", lambda: True)
    monkeypatch.setattr(attn, "ixfops", SimpleNamespace(fmhadsa=object()))
    assert attn._v41_prefill_fmhadsa() is False


def test_prefill_fmhadsa_requires_ivcore30(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_V41_PREFILL_FMHADSA_ENV", True)
    monkeypatch.setattr(attn, "is_ivcore30", lambda: False)
    monkeypatch.setattr(attn, "ixfops", SimpleNamespace(fmhadsa=object()))
    assert attn._v41_prefill_fmhadsa() is False


def test_prefill_fmhadsa_requires_op(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_V41_PREFILL_FMHADSA_ENV", True)
    monkeypatch.setattr(attn, "is_ivcore30", lambda: True)
    monkeypatch.setattr(attn, "ixfops", SimpleNamespace())
    assert attn._v41_prefill_fmhadsa() is False


def test_prefill_fmhadsa_on_ivcore30_when_op_exists(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_V41_PREFILL_FMHADSA_ENV", True)
    monkeypatch.setattr(attn, "is_ivcore30", lambda: True)
    monkeypatch.setattr(attn, "ixfops", SimpleNamespace(fmhadsa=object()))
    assert attn._v41_prefill_fmhadsa() is True
