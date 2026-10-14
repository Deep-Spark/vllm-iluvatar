# SPDX-License-Identifier: Apache-2.0
"""Honor the coordinator's block-drop switch for explicitly tagged drafts."""

import inspect
from dataclasses import replace
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="honor disabled EAGLE prefix block drop")


def _apply() -> None:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    original = KVCacheCoordinator.__init__
    signature = inspect.signature(original)

    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        # The scheduler passes use_eagle_block_drop(), not use_eagle(), here.
        # Upstream otherwise honors explicit is_eagle_group tags even when
        # that switch is False. Use a coordinator-local copy so worker/draft
        # metadata remains unchanged and the opt-in switch works for both
        # explicit tags and the coordinator's fallback group selection.
        config = bound.arguments["kv_cache_config"]
        if not bound.arguments["use_eagle"] and any(
            group.is_eagle_group for group in config.kv_cache_groups
        ):
            bound.arguments["kv_cache_config"] = replace(
                config,
                kv_cache_groups=[
                    replace(group, is_eagle_group=False)
                    for group in config.kv_cache_groups
                ],
            )
        original(*bound.args, **bound.kwargs)

    KVCacheCoordinator.__init__ = initialize
