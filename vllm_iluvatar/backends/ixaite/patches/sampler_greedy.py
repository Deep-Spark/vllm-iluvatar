# SPDX-License-Identifier: Apache-2.0
"""Wiring for the V1 Sampler greedy-path argmax override.

Corresponds to: vllm/v1/sample/sampler.py

ILU general-plugins patch, applied from
``configure_after_registration`` during ``register_general``.

Target   : vllm.v1.sample.sampler.Sampler.greedy_sample
Baseline : main_61ac3680210
Why      : the community greedy path is a single pass
           ``logits.argmax(dim=-1)`` over the full fp32 vocab. At decode the
           batch is tiny (num_reqs rows, e.g. 8) while the vocab is huge
           (151936 for Qwen3-8B), so the reduction launches one block per row
           and starves the GPU: ~306us per step measured on Iluvatar.
           The V2 runner avoids this with a two-stage
           gumbel/argmax kernel (~50us); the ixaite backend pins the V1
           runner, so port the same two-stage shape here with pure aten ops.
How      : ``setattr`` a staticmethod override onto the class; the call site
           (``self.greedy_sample(logits)``) resolves through the class.
           The override views the vocab as [n, num_blocks, BLOCK] (padding to
           a BLOCK multiple with -inf when needed), reduces each block with
           ``max`` (1192+ row-blocks -> full occupancy), argmaxes the block
           winners, and recombines indices. Tie-break stays first-occurrence
           like the documented single-pass argmax.
Scope    : V1-runner greedy sampling only (``not all_random`` batches).
           Random sampling goes through TopKTopPSampler untouched; V2-runner
           processes use their own sampler module and are unaffected.
Remove   : when the ixaite backend runs on the V2 model runner, which already
           has the fused two-stage kernel.
"""

from __future__ import annotations

import torch

from vllm_iluvatar.ilu_patch import general_patch as p

# 128 measured fastest on Iluvatar for Qwen3-8B decode (151936 = 128*1187,
# exact divide skips the pad copy; other vocab sizes pad at most 127 floats).
_BLOCK_SIZE = 128


def register() -> None:
    """Swap ``Sampler.greedy_sample`` for the two-stage block argmax."""
    p.register_once(
        __name__,
        _apply,
        detail="Sampler.greedy_sample (two-stage block argmax)",
    )


def _apply() -> None:
    from vllm.v1.sample.sampler import Sampler

    # [ILU override] Sampler.greedy_sample — delta: single-pass full-vocab
    # argmax -> two-stage blockwise argmax to fix grid starvation at decode
    # batch sizes. Staticmethod swap on the class; call sites resolve through
    # the class at call time.
    Sampler.greedy_sample = staticmethod(_two_stage_argmax)


def _two_stage_argmax(logits: torch.Tensor) -> torch.Tensor:
    n, vocab = logits.shape
    num_blocks = (vocab + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    if vocab % _BLOCK_SIZE:
        logits = torch.nn.functional.pad(
            logits, (0, num_blocks * _BLOCK_SIZE - vocab), value=float("-inf")
        )
    block_vals, block_idxs = logits.reshape(n, num_blocks, _BLOCK_SIZE).max(dim=-1)
    win = block_vals.argmax(dim=-1)
    idx = block_idxs.gather(-1, win.unsqueeze(-1)).squeeze(-1)
    return (win * _BLOCK_SIZE + idx).view(-1)
