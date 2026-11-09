# SPDX-License-Identifier: Apache-2.0
# Source: ilu_causal_conv1d.py at 47fb804; independent convolution bindings.
"""Direct GDN convolution ABI selected before layer imports."""

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

causal_conv1d_fn = None
causal_conv1d_update = None


# Publish the pair atomically so loading cannot observe an incomplete lifecycle.
def configure_causal_conv1d_provider() -> None:
    global causal_conv1d_fn, causal_conv1d_update
    implementations = []
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        implementation, reason = load_ixaite_python_op("causal_conv1d", name)
        if implementation is None:
            raise RuntimeError(f"ixaite {name} is unavailable: {reason}")
        implementations.append(implementation)
    causal_conv1d_fn, causal_conv1d_update = implementations
