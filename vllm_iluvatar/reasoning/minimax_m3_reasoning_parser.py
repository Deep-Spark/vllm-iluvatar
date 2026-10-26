# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/reasoning/minimax_m3_reasoning_parser.py

Target   : MiniMaxM3ReasoningParser.is_reasoning_end
Baseline : v0.31.0 (db9527a468)
Why      : adaptive prompts include literal think markers in instructions.
How      : inspect only the current assistant turn; inherit other parser behavior.
Scope    : the minimax_m3 parser registered by ilu_register.
Remove   : when upstream scopes prompt reasoning markers to the current turn.
"""

from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


class IluMiniMaxM3ReasoningParser(MiniMaxM3ReasoningParser):
    def is_reasoning_end(self, input_ids):
        text = self._decode_text(input_ids)
        boundary = "]~b]ai\n"
        if boundary not in text:
            return super().is_reasoning_end(input_ids)
        # Only the current assistant turn may close reasoning. Instructions
        # and completed historical turns also contain the marker token IDs.
        current = text.rsplit(boundary, 1)[1]
        end = current.rfind(self.end_token)
        return end >= 0 and end > current.rfind(self.start_token)
