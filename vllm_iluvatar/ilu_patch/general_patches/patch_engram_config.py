# SPDX-License-Identifier: Apache-2.0
"""Whitelist the Iluvatar V4.1 arch for community Engram.

Corresponds to: vllm/config/engram.py, vllm/config/vllm.py,
                vllm/distributed/parallel_state.py

ILU general-plugins patch, auto-discovered from
``ilu_patch.general_patches`` during ``register_general``.

``dev_dsv41_0.30.1_gyf_dev`` has no Engram architecture gate. It shards the
table with the ordinary DP group. vLLM 0.31 only recognizes
``DeepseekV41ForCausalLM``: that name is required both to create
``EngramConfig`` and to build the node-local Engram DP group. The W4A8
checkpoint keeps ``IluDeepseekV41ForCausalLM``, so the group stays unset and
Engram reports that the DP replicas are not on one node.

Target   : vllm.config.engram._NGRAM_LAYER_FIELDS
           vllm.config.vllm.VllmConfig._resolve_and_verify_engram_config
           vllm.distributed.parallel_state.initialize_model_parallel
Baseline : v0.31.0 (341bf7d4)
Why      : the W4A8 checkpoint declares ``IluDeepseekV41ForCausalLM``.
How      : add that arch to the n-gram field map, apply the upstream DBO
           rejection, and present the upstream arch while the Engram DP group
           is built. The community tree is left unchanged.
Scope    : EngramConfig and the node-local Engram DP group for Iluvatar V4.1.
Remove   : when community whitelists ``IluDeepseekV41ForCausalLM`` itself.
"""

from __future__ import annotations

from contextlib import contextmanager

from vllm_iluvatar.ilu_patch import general_patch as p

_ILU_ENGRAM_ARCH = "IluDeepseekV41ForCausalLM"
_UPSTREAM_ENGRAM_ARCH = "DeepseekV41ForCausalLM"
_ENGRAM_FIELD = "engram_layer_ids"


def register() -> None:
    """Register the Iluvatar Engram architecture whitelist."""
    p.register_once(
        __name__,
        _apply,
        detail="IluDeepseekV41ForCausalLM EngramConfig and DP group",
    )


def _reject_ilu_engram_ubatch(vllm_config) -> None:
    """Reject DBO for the Iluvatar arch the same way upstream rejects it."""
    model_config = vllm_config.model_config
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is not None
        and model_config is speculative_config.draft_model_config
    ):
        model_config = speculative_config.target_model_config
    if (
        model_config is not None
        and model_config.architecture == _ILU_ENGRAM_ARCH
        and getattr(model_config.hf_text_config, _ENGRAM_FIELD, None)
        and vllm_config.parallel_config.use_ubatching
    ):
        raise ValueError(
            "DeepSeek V4.1 Engram does not support DBO or microbatching. "
            "Disable --enable-dbo and set --ubatch-size to 0."
        )


@contextmanager
def _as_upstream_engram_arch(model_config):
    """Let the Iluvatar arch pass the community Engram DP-group check.

    ``initialize_model_parallel`` builds that group only for
    ``DeepseekV41ForCausalLM``. The public name is restored before return.
    """
    if model_config is None or model_config.architecture != _ILU_ENGRAM_ARCH:
        yield
        return
    saved = model_config._architecture
    model_config._architecture = _UPSTREAM_ENGRAM_ARCH
    try:
        yield
    finally:
        model_config._architecture = saved


def _apply() -> None:
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config.engram import _NGRAM_LAYER_FIELDS
    from vllm.config.vllm import VllmConfig

    _NGRAM_LAYER_FIELDS[_ILU_ENGRAM_ARCH] = _ENGRAM_FIELD
    orig = VllmConfig._resolve_and_verify_engram_config

    def _resolve_and_verify_engram_config(self):
        _reject_ilu_engram_ubatch(self)
        return orig(self)

    VllmConfig._resolve_and_verify_engram_config = _resolve_and_verify_engram_config

    orig_init = parallel_state.initialize_model_parallel

    def _initialize_model_parallel(*args, **kwargs):
        from vllm.config import get_current_vllm_config

        model = getattr(get_current_vllm_config(), "model_config", None)
        with _as_upstream_engram_arch(model):
            return orig_init(*args, **kwargs)

    p.rebind_module_func(
        parallel_state,
        "initialize_model_parallel",
        _initialize_model_parallel,
    )
