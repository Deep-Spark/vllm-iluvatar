# SPDX-License-Identifier: Apache-2.0
"""HY-V4 reasoning parser (``<think:opensource>`` / ``</think:opensource>``).

Target   : the ``Hy4-FP4-testing`` chat format. Its template wraps reasoning in
           ``<think:opensource> ... </think:opensource>`` (single special tokens
           120029 / 120030) and emits the opening tag in the prompt, so the model
           output contains only the reasoning text followed by the closing tag
           and then the answer.
Why      : the community ``hy_v3`` parser already handles exactly this shape
           (including the "start token lives in the prompt" streaming case) via a
           configurable ``token_suffix``. But this checkpoint's
           ``tokenizer_config.json`` leaves ``token_suffix`` unset, so ``hy_v3``
           would look for plain ``<think>`` tags and fail; and ``hy_v3`` falls
           back to a no-split IdentityReasoningParser whenever ``reasoning_effort``
           is absent, which it always is at server-construction time.
How      : subclass ``hy_v3``, pin the suffix to ``:opensource`` and force the
           real (non-identity) parser. Registered as ``hy_v4``; select it with
           ``--reasoning-parser hy_v4``.
Scope    : HY-V4 chat checkpoints using the ``:opensource`` think tags.
Remove   : when the checkpoint ships ``token_suffix`` in its tokenizer config and
           the serving layer forwards ``reasoning_effort`` to the parser.
"""

from __future__ import annotations

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.hy_v3_reasoning_parser import HYV3ReasoningParser

_HY_V4_TOKEN_SUFFIX = ":opensource"


@ReasoningParserManager.register_module("hy_v4")
class HYV4ReasoningParser(HYV3ReasoningParser):
    """``hy_v3`` with the suffix pinned and the identity fallback disabled."""

    def __init__(self, tokenizer, *args, **kwargs):
        # ``HYV3ReasoningParser.__init__`` reads ``token_suffix`` from the
        # tokenizer's ``init_kwargs`` *before* the base class resolves the think
        # token ids, so inject it here (the checkpoint omits it).
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict) and not init_kwargs.get("token_suffix"):
            init_kwargs["token_suffix"] = _HY_V4_TOKEN_SUFFIX
        super().__init__(tokenizer, *args, **kwargs)
        # Guard the case where ``init_kwargs`` was missing/not a dict: the base
        # already resolved the token ids, but keep ``suffix`` consistent.
        self.suffix = _HY_V4_TOKEN_SUFFIX
        # Always use the real thinking parser. ``hy_v3`` installs a no-split
        # IdentityReasoningParser when ``reasoning_effort == "no_think"``, and
        # the server never passes ``reasoning_effort`` at construction, so the
        # fallback would disable reasoning extraction entirely. Token-based
        # splitting is correct in every mode: the template emits
        # ``<think:opensource>`` in the prompt and, for no_think, an empty
        # ``<think:opensource></think:opensource>`` pair (reasoning stays empty).
        self._identity_parser = None
