# SPDX-License-Identifier: Apache-2.0
# Identify both checkpoint scheme classes only inside the ixaite route.
_W8A8_LINEAR_SCHEME_NAMES = frozenset(
    {
        "IluCompressedTensorsW8A8Int8",
        "IxaiteCompressedTensorsW8A8Int8",
    }
)

# Match either layout owner while keeping this compatibility check off baseline.
def is_w8a8_linear_scheme(scheme: object) -> bool:
    """True for either Iluvatar W8A8 linear scheme (ixformer or ixaite)."""
    return scheme is not None and type(scheme).__name__ in _W8A8_LINEAR_SCHEME_NAMES
