# SPDX-License-Identifier: Apache-2.0
"""Publishing kernel-private weights derived during checkpoint loading."""

import torch


def publish_derived_weight(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    """Attach a load-time-derived tensor under ``name``, as a buffer.

    Not ``replace_parameter``: the loader calls
    ``process_weights_after_loading`` inside ``device_loading_context``, which
    records every parameter that was on CPU on entry and, on exit, moves
    whatever still carries that *name* back to CPU. Since the ix30 FP4 image is
    published under the checkpoint's own names, keeping it a Parameter means the
    context evicts it to host memory the moment the context closes, and the
    first ``group_gemm_fp4`` then fails with "weight must be on the same CUDA
    device as a".

    A non-persistent buffer is both immune to that (the restore loop walks
    ``named_parameters()``) and the honest description of what this is: a
    derived, kernel-private tensor, not a checkpoint parameter, so it should
    also stay out of ``state_dict()``.
    """
    layer._parameters.pop(name, None)
    layer._buffers.pop(name, None)
    layer.register_buffer(name, tensor, persistent=False)
