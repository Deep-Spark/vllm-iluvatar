# SPDX-License-Identifier: Apache-2.0
"""Context buffers owned by a sequential target/draft execution stream."""

import torch


class IluTargetDraftBuffer:
    """One target/draft stream owns these buffers until the draft step finishes.

    Pack once into reusable FC input; project straight into the speculator's
    fixed context buffer. The target graph need not export auxiliary tensors.
    """

    def __init__(self, fc, output, num_layers):
        self.weight = fc.weight
        self.output = output
        self.num_layers = num_layers
        self.packed = torch.empty(
            (output.shape[0], fc.input_size), dtype=output.dtype, device=output.device
        )

    def __call__(self, states):
        if len(states) != self.num_layers:
            raise ValueError("Target/draft auxiliary layer count changed")
        tokens = states[0].shape[0]
        if tokens > self.output.shape[0]:
            raise ValueError("Target/draft context capacity exceeded")
        packed = self.packed[:tokens]
        torch.cat(states, dim=-1, out=packed)
        torch.mm(packed, self.weight.t(), out=self.output[:tokens])
