# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by benchmark suites."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

_CaseT = TypeVar("_CaseT")

RUNTIME_T: tuple[int, ...] = (1, 8, 128, 256, 512, 2048)
DECODE_GRAPH_T: tuple[int, ...] = (1, 8, 128, 256)
PREFILL_T: tuple[int, ...] = (256, 512, 2048)


def filter_cases(cases: Sequence[_CaseT], case_filter: str | None) -> list[_CaseT]:
    if not case_filter:
        return list(cases)
    return [case for case in cases if case_filter in case.case_id]  # type: ignore[attr-defined]


def initialize_ixaite():
    import ixaite

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    arch = cuda_arch()
    if arch is None:
        raise RuntimeError("Iluvatar GPU architecture cannot be detected")
    ixaite.initialize(arch)
    return ixaite
