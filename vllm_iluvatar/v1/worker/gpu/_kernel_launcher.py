# SPDX-License-Identifier: Apache-2.0
"""Adapter that preserves Triton's ``kernel[grid](...)`` call protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class KernelLauncher:
    def __init__(self, launch: Callable[..., None]) -> None:
        self._launch = launch

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        def invoke(*args: Any, **kwargs: Any) -> None:
            self._launch(grid, *args, **kwargs)

        return invoke
