"""Installed-package git identity baked at wheel build time."""

import re
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches.patch_version_print import (
    _wrap_collect_env,
    _wrap_log_version_and_model,
)
from vllm_iluvatar.version import __git_id__, __version__, format_identity

_SELF_ID_RE = re.compile(r"^\[.+\] [0-9a-f]{40}$")


def test_git_id_defined():
    assert __git_id__ is not None
    if __git_id__ == "unknown":
        return
    assert _SELF_ID_RE.match(__git_id__), __git_id__


def test_format_identity_includes_version():
    ident = format_identity()
    assert ident.startswith(__version__)
    if __git_id__ != "unknown":
        assert __git_id__ in ident


def test_wrap_collect_env_appends_plugin_identity():
    mod = SimpleNamespace(get_vllm_version=lambda: "0.27.1 [main] abc")
    _wrap_collect_env(mod)
    out = mod.get_vllm_version()
    assert out.startswith("0.27.1")
    assert "vllm_iluvatar" in out
    assert format_identity() in out
    assert mod.get_vllm_version().count("vllm_iluvatar") == 1


def test_wrap_log_version_and_model_logs_plugin_line():
    recorded: list[str] = []

    def orig(lgr, version, model_name):
        lgr.info("vLLM server version %s, serving model %s", version, model_name)

    class _Lgr:
        def info(self, msg, *args):
            recorded.append(msg % args if args else msg)

    mod = SimpleNamespace(log_version_and_model=orig)
    _wrap_log_version_and_model(mod)
    mod.log_version_and_model(_Lgr(), "0.27.1", "llama")
    assert recorded[0].startswith("vLLM server version 0.27.1")
    assert recorded[1] == f"vllm_iluvatar {format_identity()}"
