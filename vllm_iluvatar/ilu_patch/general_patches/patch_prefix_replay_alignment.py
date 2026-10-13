# SPDX-License-Identifier: Apache-2.0
"""Allow connector-validated even remote-prefix replay hits.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target   : Scheduler.schedule prefix-replay alignment
Baseline : v0.30.0rc0
Why      : SWA bounded replay rounds every remote hit down to a whole block.
           A complete DeepSeek V4.1 remote-prefill can keep its partial
           blocks when the connector validates even alignment. That behavior
           stays in this plugin so community vLLM is unchanged.
How      : ``Scheduler.schedule`` is rewritten from source exactly once, by
           ``patch_pp_spec_schedule``. This module contributes
           ``patch_schedule_source``, which swaps the block-only rounding for
           a call to ``align_prefix_replay_ext_tokens``, then verifies the
           rewrite landed.
Scope    : DeepSeek V4.1 consumers whose connector reports alignment 2
           (``VLLM_ILU_V41_REMOTE_REPLAY_ALIGN2``); every other hit keeps
           whole-block rounding.
Remove   : when community vLLM lets a connector declare replay alignment.
"""

from __future__ import annotations

from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

# Dedented ``Scheduler.schedule`` body (``textwrap.dedent`` drops the class indent).
_OLD_BLOCK = """\
                    if self.prefix_replay_tokens:
                        # SWA bounded replay recomputes the hit's last
                        # window, from the block holding its first
                        # token; the sliding-window groups retire
                        # whole blocks below the window of the hit's
                        # next token. Hits end on a block boundary, or
                        # a hit ending one token short of one would
                        # retire the block the replay starts in.
                        ext_tokens -= ext_tokens % self.block_size
                        load_kv_async = load_kv_async and ext_tokens > 0
"""

_HELPER = "_ilu_align_prefix_replay_ext_tokens"
_NEW_CALL = f"""\
                    ext_tokens, load_kv_async = {_HELPER}(
                        self,
                        request,
                        ext_tokens,
                        load_kv_async,
                        block_aligned_local,
                    )
"""


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="connector-validated even remote-prefix replay alignment",
    )


def align_prefix_replay_ext_tokens(
    scheduler: Any,
    request: Any,
    ext_tokens: int,
    load_kv_async: bool,
    block_aligned_local: int,
) -> tuple[int, bool]:
    """Round a remote prefix hit for SWA bounded replay.

    Whole blocks remain the default. Even alignment is used only when the
    connector advertises it and the hit is a complete remote prompt.
    """
    if not scheduler.prefix_replay_tokens:
        return ext_tokens, load_kv_async

    alignment = scheduler.block_size
    connector = scheduler.connector
    remote_alignment = getattr(connector, "prefix_replay_alignment", None)
    params = request.kv_transfer_params or {}
    if (
        load_kv_async
        and remote_alignment == 2
        and scheduler.prefix_replay_tokens == 128
        and block_aligned_local % 2 == 0
        and params.get("do_remote_prefill")
        and (params.get("transfer_id") or params.get("remote_request_id"))
        and block_aligned_local + ext_tokens == request.num_prompt_tokens
    ):
        alignment = remote_alignment
    ext_tokens -= ext_tokens % alignment
    return ext_tokens, load_kv_async and ext_tokens > 0


def patch_schedule_source(source: str) -> str:
    """Replace block-only prefix replay rounding in dedented schedule source."""
    if _HELPER in source:
        return source
    if _OLD_BLOCK in source:
        return source.replace(_OLD_BLOCK, _NEW_CALL, 1)
    if "if self.prefix_replay_tokens:" in source:
        raise RuntimeError(
            "Scheduler.schedule prefix-replay block no longer matches "
            "patch_prefix_replay_alignment.py; re-sync the patch"
        )
    return source


def _apply() -> None:
    """Drive the shared schedule rewrite, then verify this patch is in it.

    ``register_once`` makes the dependency idempotent, so this does not rely
    on the auto-discovery order of the two modules.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_schedule

    patch_pp_spec_schedule.register()

    source = getattr(Scheduler.schedule, "_ilu_pp_spec_schedule_source", None)
    if source is None or _HELPER not in source:
        raise RuntimeError(
            "Scheduler.schedule was not rewritten with even remote-prefix "
            "replay alignment; re-sync patch_pp_spec_schedule.py"
        )
