# SPDX-License-Identifier: Apache-2.0
"""Iluvatar logging wiring.

ILU modules live under the ``vllm_iluvatar.*`` namespace, which is a sibling of
(not a child of) the community ``vllm`` logger tree. vLLM only configures the
logger named ``"vllm"`` (with its handler and ``propagate=False``), so a logger
named ``vllm_iluvatar.foo`` would not inherit vLLM's formatting/handlers and its
INFO/DEBUG records would never emit.

:func:`configure_ilu_logging` fixes this once per process by reparenting the
``vllm_iluvatar`` root logger under ``vllm``. After it runs, every ILU module can
use the community convention ``from vllm.logger import init_logger;
logger = init_logger(__name__)`` (or even bare ``logging.getLogger(__name__)``)
and still print through vLLM's stream/formatter/level — the same reason
community ``vllm.*`` modules work that way. Because it uses live parent-based
inheritance (not a handler snapshot), it also tracks any later vLLM logging
reconfiguration.
"""

from __future__ import annotations

import logging


def configure_ilu_logging() -> None:
    """Reparent the ``vllm_iluvatar`` logger tree under ``vllm``.

    By default ``logging.getLogger("vllm_iluvatar")`` is a top-level logger whose
    parent is the root logger (which vLLM never configures), so its INFO/DEBUG
    records never emit. Setting its parent to the ``vllm`` logger makes the whole
    ``vllm_iluvatar.*`` subtree inherit vLLM's configured handler(s)/level live,
    while ``vllm``'s own ``propagate=False`` keeps records from spilling to root
    (no double-print via lastResort). Idempotent.
    """
    logging.getLogger("vllm_iluvatar").parent = logging.getLogger("vllm")
