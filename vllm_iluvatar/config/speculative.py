# SPDX-License-Identifier: Apache-2.0
"""FLy configuration for the Iluvatar ModelRunnerV2 verifier."""

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self
from vllm.config.speculative import SpeculativeConfig as CommunitySpeculativeConfig
from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
class SpeculativeConfig(CommunitySpeculativeConfig):
    """Extend the community schema without changing its standard verifier."""

    rejection_sample_method: Literal["standard", "synthetic", "block", "fly"] = (
        "standard"
    )
    """Verification algorithm. FLy adds entropy-gated deferred verification."""

    fly_window_size: int | None = Field(default=None, ge=1)
    """Subsequent native accepts required to confirm a deferred rejection.
    Defaults to ``min(6, num_speculative_tokens - 1)`` when FLy is selected."""

    fly_entropy_threshold: float = Field(default=0.3, ge=0)
    """Target top-k entropy lower bound for opening a deferred window."""

    fly_entropy_top_k: int = Field(default=3, ge=1)
    """Number of top target probabilities used in the entropy sum."""

    fly_min_target_probability: float = Field(
        default=0.0, ge=0, le=1, allow_inf_nan=False
    )
    """Minimum processed target p(draft) for an initial FLy rescue.
    Zero disables this gate. A positive floor requires probabilistic drafting;
    native accepts and pending-window confirmations are unaffected."""

    @model_validator(mode="after")
    def _verify_fly_args(self) -> Self:
        if self.fly_min_target_probability > 0:
            if self.rejection_sample_method != "fly":
                raise ValueError(
                    "FLy target probability floor requires FLy verification"
                )
            if self.draft_sample_method != "probabilistic":
                raise ValueError(
                    "FLy target probability floor requires probabilistic drafting"
                )

        if self.rejection_sample_method == "fly":
            if self.uses_extract_hidden_states():
                raise ValueError(
                    "FLy is not compatible with method='extract_hidden_states', "
                    "which does not perform speculative decoding."
                )
            if self.num_speculative_tokens < 2:
                raise ValueError(
                    "FLy requires num_speculative_tokens to be at least 2."
                )
            if self.fly_window_size is None:
                self.fly_window_size = min(6, self.num_speculative_tokens - 1)
            elif self.fly_window_size >= self.num_speculative_tokens:
                raise ValueError(
                    "FLy requires fly_window_size to be smaller than "
                    "num_speculative_tokens."
                )
            logger.warning_once(
                "FLy verification is a lossy speculative decoding method and "
                "may degrade model outputs."
            )
        return self
