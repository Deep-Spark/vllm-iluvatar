# SPDX-License-Identifier: Apache-2.0
"""Register reasoning parsers through the official vLLM registry."""


def register_reasoning_parsers() -> None:
    from vllm.reasoning import ReasoningParserManager

    from vllm_iluvatar.reasoning.minimax_m3_reasoning_parser import (
        IluMiniMaxM3ReasoningParser,
    )

    ReasoningParserManager.register_module(
        "minimax_m3",
        module=IluMiniMaxM3ReasoningParser,
        force=True,
    )
