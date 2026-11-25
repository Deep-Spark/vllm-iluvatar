# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Layer name parsing shared by the Iluvatar NIXL connectors."""


def find_number_on_layer_name(text: str) -> list[str]:
    """Return the nonnegative integer components between dots, in name order."""
    indices = []
    for part in text.split(".")[1:-1]:
        try:
            value = int(part)
        except ValueError:
            continue
        # Keep the connector's existing unsigned, unpadded layer name format.
        if value >= 0 and str(value) == part:
            indices.append(part)
    return indices
