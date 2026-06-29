# SPDX-License-Identifier: Apache-2.0
"""Centralised late-phase general patch wiring.

Modules in this package expose ``register()`` and are auto-discovered by
``vllm_iluvatar.ilu_patch.apply_general_patches`` during ``register_general``.
They contain only register-time wiring: drift/signature guards, lazy imports,
and rebinding onto community symbols.

Override implementations stay in the upstream-mirrored packages
(``vllm_iluvatar.distributed.*``, ``vllm_iluvatar.v1.*``,
``vllm_iluvatar.model_executor.*``, etc.) so implementation files remain easy
to diff against community vLLM.
"""
