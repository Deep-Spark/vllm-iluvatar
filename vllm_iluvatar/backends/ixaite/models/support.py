# SPDX-License-Identifier: Apache-2.0
"""Check fixed model dependencies before allocating or transforming weights."""

from vllm_iluvatar.backends.ixaite.ops import _load_ixaite_routed_op


# The library validates architecture support; model constructors only require availability.
def require_model_ops(*names: str) -> None:
    for name in names:
        implementation, reason = _load_ixaite_routed_op(name)
        if implementation is None:
            raise RuntimeError(f"ixaite model requires {name}: {reason}")
