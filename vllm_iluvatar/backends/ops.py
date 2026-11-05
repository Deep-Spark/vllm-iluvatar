# SPDX-License-Identifier: Apache-2.0
"""Publish one custom-op implementation module before consumers copy references."""

from vllm_iluvatar.backends import provider_mode

if provider_mode() == "ixaite":
    from vllm_iluvatar.backends.ixaite import ops as implementation
else:
    from vllm_iluvatar import _custom_ops as implementation

__all__ = ["implementation"]
