# SPDX-License-Identifier: Apache-2.0
"""DSpark drafting under pipeline parallelism, on the vLLM 0.30.1 loader.

Community ``load_dspark_model`` already pins the draft to
``pipeline_parallel_size=1`` and shares the target embedding through
``maybe_share_target_embed``. That share raises on the last PP rank: the
target's ``embed_tokens`` is a ``PPMissingLayer`` there, and the DeepSeek
DSpark draft is marked ``has_own_embed_tokens = False``.

The drafter only runs on the last rank, so this module keeps the community
loader and changes two call sites:

* skip the embed alias when the target embedding is missing, and
* while loading draft weights, keep the checkpoint ``embed.weight`` for the
  draft's own ``embed_tokens``.

``lm_head`` sharing is unchanged: the last rank owns a real head.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.utils import PPMissingLayer

# Names the checkpoint (or a weights mapper) may use for the input embedding.
_EMBED_CKPT_NAMES = frozenset(
    {
        "embed.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
    }
)


def _load_embed_weight(draft_model: Any, loaded_weight: Any) -> None:
    param = draft_model.model.embed_tokens.weight
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        param.data.copy_(loaded_weight)
        return
    weight_loader(param, loaded_weight)


def iter_weights_keeping_pp_embed(
    draft_model: Any,
    weights: Iterable[tuple[str, Any]],
    loaded_names: list[str],
) -> Iterator[tuple[str, Any]]:
    """Yield draft weights, pulling the input embedding out for this rank."""
    for name, loaded_weight in weights:
        if name in _EMBED_CKPT_NAMES:
            _load_embed_weight(draft_model, loaded_weight)
            loaded_names.append("model.embed_tokens.weight")
            continue
        yield name, loaded_weight


def load_weights_keeping_pp_embed(
    orig: Callable[..., set[str]],
) -> Callable[..., set[str]]:
    """Wrap a DSpark ``load_weights`` so PP>1 keeps ``embed.weight``."""

    def load_weights(self: Any, weights: Iterable[tuple[str, Any]]) -> set[str]:
        if get_pp_group().world_size <= 1:
            return orig(self, weights)
        loaded_names: list[str] = []
        loaded = orig(
            self,
            iter_weights_keeping_pp_embed(self, weights, loaded_names),
        )
        loaded.update(loaded_names)
        return loaded

    return load_weights


def maybe_share_target_embed_pp(
    orig: Callable[..., None],
) -> Callable[..., None]:
    """Wrap community embed sharing so a missing PP-stage embed is not an error.

    PP=1, and any rank that actually has the target embedding, still go through
    ``orig``. The last PP rank hits ``PPMissingLayer`` and returns: the draft
    keeps the embedding ``load_weights_keeping_pp_embed`` just loaded.
    """

    def maybe_share_target_embed(
        draft_model: Any, draft_inner: Any, target_inner: Any
    ) -> None:
        if get_pp_group().world_size > 1:
            target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
                target_inner, "embedding", None
            )
            if target_embed is None or isinstance(target_embed, PPMissingLayer):
                return
        orig(draft_model, draft_inner, target_inner)

    return maybe_share_target_embed
