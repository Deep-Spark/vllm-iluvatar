"""Step5 three-module MTP with independent checkpoint heads and native inputs.

Each decoder owns ordinary vLLM 0.31 Attention objects. Draft cache allocation,
slot mappings, rejection updates and re-prefill remain the runner's responsibility.
"""

import copy
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.step3p5_mtp import Step3p5MTP as _Step3p5MTP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from .config import Step5Config
from .logits import Step5LogitsProcessor
from .model import Step5DecoderLayer, Step5RMSNorm, _checkpoint_parameter_audit
from .native_layers import Step5VocabParallelEmbedding
from .runtime_contract import validate_step5_runtime


def draft_decoder_config(vllm_config: VllmConfig) -> VllmConfig:
    """Expose global MTP indices and keep the actual BF16 dense draft unquantized."""
    result = copy.copy(vllm_config)
    spec = getattr(vllm_config, "speculative_config", None)
    source = spec.draft_model_config if spec is not None else vllm_config.model_config
    result.model_config = copy.copy(source)
    config = copy.deepcopy(source.hf_config)
    if not isinstance(config, Step5Config):
        raise TypeError("Step5 MTP requires Step5Config")
    if config.hidden_size != 4096 or not config.zero_centered:
        raise ValueError("Step5 MTP native input requires zero-centered D4096 norms")
    start, count = (config.num_hidden_layers, config.num_nextn_predict_layers)
    types = config.layer_types_with_mtp
    if count < 1 or types is None or len(types) != start + count:
        raise ValueError("Step5 MTP requires complete main + draft layer_types")
    config.layer_types = list(types)
    moe_layers = set(getattr(config, "moe_layer_list", None) or [])
    if any(index in moe_layers for index in range(start, start + count)):
        raise ValueError("Step5 MTP checkpoint must have dense draft layers")
    result.model_config.hf_config = config
    result.quant_config = None
    return result


class Step5MTPOutputHead(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.norm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=prefix + ".output",
        )

    def forward(self, hidden_states):
        return self.norm(
            hidden_states, output_dtype=torch.bfloat16, round_input_to_bf16=True
        )


class Step5MTPLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.enorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=prefix + ".eh_proj",
        )
        self.mtp_block = Step5DecoderLayer(vllm_config, prefix=prefix)
        if self.mtp_block.use_moe:
            raise ValueError("Step5 draft decoder unexpectedly constructed MoE")
        self.output_head = Step5MTPOutputHead(
            config, prefix + ".transformer.shared_head"
        )

    def forward(self, positions, previous_hidden_states, inputs_embeds):
        from ixformer.inference.functions.models.step5.mtp.float.mtp_norm_concat import (
            mtp_norm_concat,
        )

        fused = mtp_norm_concat(
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
            backend="ixformer_native",
        )
        hidden_states, _ = self.eh_proj(fused)
        return self.mtp_block(positions=positions, hidden_states=hidden_states)


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.embed_tokens = Step5VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleDict(
            {
                str(index): Step5MTPLayer(vllm_config, f"{prefix}.layers.{index}")
                for index in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.logits_processor = Step5LogitsProcessor(config.vocab_size)
        self._last_spec_step_idx = 0

    def _layer(self, spec_step_idx):
        if not isinstance(spec_step_idx, int) or spec_step_idx < 0:
            raise ValueError("Step5 MTP spec_step_idx must be a nonnegative Python int")
        return self.layers[
            str(self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers)
        ]

    def forward(
        self,
        input_ids,
        positions,
        previous_hidden_states,
        inputs_embeds=None,
        spec_step_idx=0,
    ):
        layer = self._layer(spec_step_idx)
        self._last_spec_step_idx = spec_step_idx
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return layer(positions, previous_hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states, spec_step_idx=None):
        layer = self._layer(
            self._last_spec_step_idx if spec_step_idx is None else spec_step_idx
        )
        return self.logits_processor(
            layer.output_head.head, layer.output_head(hidden_states)
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)


class Step5MTP(_Step3p5MTP):
    _enable_weights_track_by_default = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    has_own_embed_tokens = True
    has_own_lm_head = True
    is_dense_mtp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.vllm_config = draft_decoder_config(vllm_config)
        validate_step5_runtime(self.vllm_config, draft=True)
        self.config = self.vllm_config.model_config.hf_config
        with set_current_vllm_config(self.vllm_config):
            self.model = Step5MultiTokenPredictor(
                self.vllm_config,
                maybe_prefix(prefix, "model"),
            )
        self.lm_head = self.model._layer(0).output_head.head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ):
        if intermediate_tensors is not None:
            raise ValueError("Step5 MTP draft uses a single PP stage")
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int | None = None
    ):
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int | None, name: str) -> str:
        name = name.replace(".transformer.shared_head.norm.", ".output_head.norm.")
        name = name.replace(".transformer.shared_head.output.", ".output_head.head.")
        if spec_layer is not None:
            prefix = f"model.layers.{spec_layer}."
            suffix = name.removeprefix(prefix)
            if name.startswith(prefix) and (
                not suffix.startswith(("enorm.", "hnorm.", "eh_proj.", "output_head."))
            ):
                name = prefix + "mtp_block." + suffix
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        expected, _ = _checkpoint_parameter_audit(self)
        missing = expected - loaded
        if missing:
            raise ValueError(
                f"Step5 MTP checkpoint parameters not loaded: {sorted(missing)}"
            )
        return loaded


def register_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model("Step5MTP", "vllm_iluvatar.models.step5.mtp:Step5MTP")


__all__ = ["Step5MTP", "Step5MultiTokenPredictor", "Step5MTPLayer"]
