# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Step3.5 MTP model overrides.

Corresponds to: vllm/model_executor/models/step3p5_mtp.py
Target   : arch ``Step3p5MTP``
Baseline : v0.31.0 (341bf7d4d1) residual / W4A8 load semantics
Why      : Iluvatar Step3.5 decoder layers retain a separate residual stream
           and return ``(hidden_states, residual)``. W4A8 Step-3.7 checkpoints
           pack MTP attn/MLP as int8. ``eh_proj``, enorm, hnorm, and
           ``shared_head`` stay BF16 via their own ignore entries.
How      : Promote the MTP decoder block, add residual back into the hidden
           state, and skip missing optional attention quant params. MTP
           attn/MLP follow ``quantization_config``; the W4A8 recipe does not
           ignore ``mtp_block``.
Scope    : Step-3.5 / Step-3.7 speculative decoding on Iluvatar.
Remove   : when upstream exposes the residual-stream decoder contract and
           loads W4A8 MTP attention metadata.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictor as _Step3p5AMultiTokenPredictor,
)
from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictorLayer as _Step3p5AMultiTokenPredictorLayer,
)
from vllm.model_executor.models.step3p5_mtp import Step3p5MTP as _Step3p5MTP
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)

from vllm_iluvatar.model_executor.models.step3p5 import IluStep3p5DecoderLayer

logger = init_logger("vllm." + __name__)

# Step W4A8 exports commonly omit this MTP attention quant metadata. The
# Iluvatar attention path supplies calibrated/default scales at runtime.
_OPTIONAL_ATTN_QUANT_SUFFIXES = (
    ".attn.k_scale",
    ".attn.q_scale",
    ".attn.v_scale",
    ".attn.prob_scale",
    ".attn.q_zero_point",
    ".attn.k_zero_point",
    ".attn.v_zero_point",
)

# Optional packed-quant metadata the constructed graph may still omit.
# Skip those tensors so community load_weights does not KeyError.
_UNOWNED_QUANT_SUFFIXES = (
    ".weight_scale",
    ".input_scale",
    ".output_scale",
    ".weight_zero_point",
    ".weight_packed",
    ".weight_shape",
    ".g_idx",
)

# Checkpoint shards that community load_weights packs into fused Linear names.
_STACKED_WEIGHT_SHARDS = (
    ("qkv_proj", "q_proj"),
    ("qkv_proj", "k_proj"),
    ("qkv_proj", "v_proj"),
    ("gate_up_proj", "gate_proj"),
    ("gate_up_proj", "up_proj"),
)

_QUANT_IGNORE_ATTRS = ("ignore", "ignored_layers")


def _quant_ignore_entries(quant_config: object | None) -> list[str]:
    if quant_config is None:
        return []
    for attr in _QUANT_IGNORE_ATTRS:
        entries = getattr(quant_config, attr, None)
        if entries is not None:
            return list(entries)
    return []


def _checkpoint_name_owned_by_graph(name: str, params_dict: dict[str, object]) -> bool:
    if name in params_dict:
        return True
    for fused, shard in _STACKED_WEIGHT_SHARDS:
        if shard not in name:
            continue
        if name.replace(shard, fused) in params_dict:
            return True
    return False


class IluStep3p5AMultiTokenPredictorLayer(_Step3p5AMultiTokenPredictorLayer):
    """MTP layer that consumes the Iluvatar decoder's residual pair."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(vllm_config, prefix)
        IluStep3p5DecoderLayer.promote(self.mtp_block)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        quant_name = (
            quant_config.get_name()
            if quant_config is not None and hasattr(quant_config, "get_name")
            else None
        )
        ignore_now = _quant_ignore_entries(quant_config)
        eh_ignored = any("eh_proj" in str(item) for item in ignore_now)
        if (
            isinstance(self.eh_proj, nn.Linear)
            and quant_name != "fp8"
            and not eh_ignored
        ):
            self.eh_proj = ReplicatedLinear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.eh_proj",
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        return residual + hidden_states


class IluStep3p5AMultiTokenPredictor(_Step3p5AMultiTokenPredictor):
    """Community MTP predictor with promoted Iluvatar blocks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): IluStep3p5AMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


class IluStep3p5MTP(_Step3p5MTP):
    """OOT Step3.5 MTP model compatible with Iluvatar decoder output."""

    # V2 load_eagle_model replaces every layer shared_head with the target
    # lm_head unless this flag is set. Other MTP models leave it unset.
    has_own_shared_head = True
    # V2 sample_draft calls compute_logits(hidden) and therefore stays on
    # MTP layer 0. Only this class opts into passing the draft step.
    ilu_pass_spec_step_idx = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        from copy import copy

        from vllm.config import replace

        # V2 constructs the draft with the target VllmConfig. Step-3.7 keeps
        # vocab and MTP fields on text_config; the speculative draft config
        # already promotes that text config to step3p5_mtp. Draft PP stays 1.
        cfg = vllm_config
        spec = vllm_config.speculative_config
        draft_model_config = (
            getattr(spec, "draft_model_config", None) if spec is not None else None
        )
        updates: dict[str, object] = {}
        if draft_model_config is not None:
            updates["model_config"] = draft_model_config
        target_pc = vllm_config.parallel_config
        if target_pc.pipeline_parallel_size > 1:
            draft_pc = copy(target_pc)
            draft_pc.pipeline_parallel_size = 1
            updates["parallel_config"] = draft_pc
        if updates:
            cfg = replace(vllm_config, **updates)
        nn.Module.__init__(self)
        self.config = cfg.model_config.hf_config
        self.vllm_config = cfg
        self.model = IluStep3p5AMultiTokenPredictor(
            vllm_config=cfg,
            prefix=maybe_prefix(prefix, "model"),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        def _drop_unowned_quant_metadata(
            src: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in src:
                spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
                check = name
                if spec_layer is not None:
                    check = self._rewrite_spec_layer_name(spec_layer, name)
                if not _checkpoint_name_owned_by_graph(
                    check, params_dict
                ) and check.endswith(_UNOWNED_QUANT_SUFFIXES):
                    continue
                yield name, tensor

        with disable_mtp_completeness_check():
            loaded_params = super().load_weights(_drop_unowned_quant_metadata(weights))

        params_dict = dict(self.named_parameters())
        missing = set(params_dict) - set(loaded_params)
        skipped = sorted(
            name for name in missing if name.endswith(_OPTIONAL_ATTN_QUANT_SUFFIXES)
        )
        if skipped:
            preview = tuple(skipped[:10])
            logger.warning_once(
                "Ilu Step3p5MTP load_weights: skip %d missing optional attn "
                "quant params with suffixes %s (showing first %d): %s",
                len(skipped),
                _OPTIONAL_ATTN_QUANT_SUFFIXES,
                len(preview),
                preview,
            )
            missing -= set(skipped)

        if missing:
            example = next(iter(missing))
            raise RuntimeError(
                "Some parameters like "
                f"{example} are not in the checkpoint and will falsely "
                "use random initialization"
            )
        return loaded_params


def register_model() -> None:
    """Register the Iluvatar Step3.5 MTP implementation."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Step3p5MTP",
        "vllm_iluvatar.model_executor.models.step3p5_mtp:IluStep3p5MTP",
    )
