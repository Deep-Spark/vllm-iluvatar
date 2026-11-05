# Source: vocab_parallel_embedding.py at 47fb804; only ixaite registers this layer.
# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/layers/vocab_parallel_embedding.py

Iluvatar OOT override routing the embedding gather through ixaite FlyDSL.

Target   : vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding
Why      : the gather half of the composite ``embedding_lm_head``
           entry had no Iluvatar hook (``F.embedding`` only), while the lm_head
           GEMM half already routes through the ixaite linear provider. This
           adds the missing gather route with a startup-bound implementation.
How      : ``@VocabParallelEmbedding.register_oot`` subclass that overrides only
           ``forward``; the gather call goes to the bound ixaite op. TP masking and all_reduce keep
           their community positions around the gather, so vocab sharding is
           untouched. ``ParallelLMHead`` registers under its own class name and
           is not affected by this OOT entry.
Scope    : runs whenever a ``VocabParallelEmbedding`` is instantiated on the
           Iluvatar platform (e.g. Qwen3.5-9B ``embed_tokens``).

Registration: import-time decorator via
``vllm_iluvatar.ilu_register.apply_oot_registrations``; the provider binding
itself happens in ``configure_embedding_gather_provider`` (called from
``register_general`` after OOT registrations, mirroring the GDN providers).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

logger = init_logger("vllm." + __name__)

# ---------------------------------------------------------------------------
# The ixaite FlyDSL op is bound once at platform init via
# configure_embedding_gather_provider(). Forward never reselects.
#
# The gather op is TP-agnostic: TP masking zeroes out-of-shard ids before the
# gather and the all_reduce follows it, exactly like the community path.
# ---------------------------------------------------------------------------
_IXAITE_EMBEDDING_GATHER: Callable[..., torch.Tensor] | None = None


# Bind the gather implementation during platform init, mirroring
# configure_gdn_post_conv_prep_provider for the GDN ops.
def configure_embedding_gather_provider() -> None:
    global _IXAITE_EMBEDDING_GATHER
    selected, reason = load_ixaite_python_op("embedding_gather", "embedding_gather")
    if selected is None:
        raise RuntimeError(f"ixaite embedding_gather is unavailable: {reason}")
    _IXAITE_EMBEDDING_GATHER = selected


@VocabParallelEmbedding.register_oot()
class IluVocabParallelEmbedding(VocabParallelEmbedding):
    """VocabParallelEmbedding whose gather routes through the bound provider.

    Only the gather call itself is overridden; the TP mask/all_reduce
    scaffolding is copied from the community forward to preserve vocab sharding.
    """

    def forward(self, input_):
        gather = _IXAITE_EMBEDDING_GATHER
        # Quantized embedding methods own their gather path; only the plain
        # unquantized [V_local, H] table has an ixaite implementation.
        if gather is None:
            raise RuntimeError("ixaite embedding gather was not initialized")
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("ixaite embedding requires an unquantized table")
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings through the provider-selected ixaite FlyDSL op.
        output_parallel = gather(masked_input.long(), self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        return tensor_model_parallel_all_reduce(output_parallel)
