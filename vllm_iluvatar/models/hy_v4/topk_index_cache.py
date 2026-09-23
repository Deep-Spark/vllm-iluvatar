# SPDX-License-Identifier: Apache-2.0
"""Per-step reuse of the converted sparse top-k indices for HY V4.

HY V4 has 78 backbone layers but only 21 ``indexer_types == "full"`` layers.
The other 57 are "shared": they run sparse attention on the indices the closest
preceding full layer left in ``topk_indices_buffer``, without building an
indexer of their own. Everything derived from that buffer is therefore also
shared, but the sparse MLA impl used to rebuild it per layer:

* ``triton_convert_req_index_to_global_index`` — per-request index to global
  slot, one ``[num_tokens, index_topk]`` int32 kernel;
* ``(indices >= 0).sum(-1)`` — the ``topk_length`` the ixformer kernel needs,
  one reduction over the same shape.

Both only depend on the buffer contents plus ``req_id_per_token`` /
``block_table``, which are identical for every layer of a step, so a shared
layer can read what the full layer already computed. That removes 57 x 2
kernels per forward step -- real work a CUDA graph cannot replay away, unlike
launch overhead.

Ownership is deliberately one object per model, created next to
``topk_indices_buffer`` in ``HYV4Model`` and handed to every layer's impl. The
MTP draft block is a full layer (``layer_id >= num_hidden_layers`` always
builds an indexer), so it refreshes the entry rather than reading a stale
target-pass one, and sharing a single cache with the target is what keeps the
two passes consistent.
"""

from __future__ import annotations

import torch


class HYV4TopkIndexCache:
    """One step's converted global top-k indices and their valid lengths.

    Not keyed by step: a full layer always overwrites the entry before any of
    the shared layers that follow it read it, so "last write wins" is already
    the correct policy. ``num_tokens`` is still checked because the target and
    MTP draft passes run different token counts through the same cache.
    """

    __slots__ = ("_indices", "_topk_length", "_num_tokens")

    def __init__(self) -> None:
        self._indices: torch.Tensor | None = None
        self._topk_length: torch.Tensor | None = None
        self._num_tokens: int = -1

    def store(
        self,
        num_tokens: int,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
    ) -> None:
        self._num_tokens = num_tokens
        self._indices = indices
        self._topk_length = topk_length

    def load(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the cached pair, or None when this layer must recompute.

        A miss is always safe: the caller falls back to the full conversion, so
        a shared layer that somehow runs before any full layer (or after a
        token-count change) stays correct and merely loses the saving.
        """
        if self._indices is None or self._topk_length is None:
            return None
        if self._num_tokens != num_tokens:
            return None
        return self._indices, self._topk_length
