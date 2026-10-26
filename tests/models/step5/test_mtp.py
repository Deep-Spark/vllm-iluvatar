# SPDX-License-Identifier: Apache-2.0
"""Independent MTP config, checkpoint heads and native-preprocessing integration."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.step5.config import Step5Config, Step5MTPConfig, Step5VLConfig
from vllm_iluvatar.models.step5.mtp import (
    Step5MTP,
    Step5MultiTokenPredictor,
    draft_decoder_config,
)


def config():
    return Step5Config(
        num_hidden_layers=92,
        num_nextn_predict_layers=3,
        hidden_size=4096,
        moe_layer_list=list(range(3, 91)),
        layer_types=["sliding_attention"] * 95,
    )


def test_draft_config_preserves_target_and_checkpoint_indices():
    target = config()
    method = object()
    original = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target), quant_config=method
    )
    result = draft_decoder_config(original)
    assert original.model_config.hf_config is target and len(target.layer_types) == 92
    assert original.quant_config is method
    assert result.quant_config is None and result.model_config.hf_config is not target
    assert len(result.model_config.hf_config.layer_types) == 95
    assert result.model_config.hf_config.num_hidden_layers == 92
    assert result.model_config.hf_config.num_nextn_predict_layers == 3
    with pytest.raises(ValueError, match="dense draft"):
        target.moe_layer_list.append(92)
        draft_decoder_config(original)


@pytest.mark.parametrize("vl", [False, True])
def test_speculative_routing_preserves_main_and_unquantized_draft(vl):
    from typing import get_args

    import vllm.config.speculative as spec_module
    from vllm.config.speculative import SpeculativeConfig

    from vllm_iluvatar.ilu_patch.general_patches.patch_speculative_config import (
        register,
    )

    register()
    target = config()
    target.quantization_config = {
        "quant_method": "fp8",
        "modules_to_not_convert": ["model.layers.3.moe.share_expert.gate_proj"],
    }
    target.architectures = ["Step5ForCausalLM"]
    if vl:
        target = Step5VLConfig(text_config=target)
        target.architectures = ["Step5ForConditionalGeneration"]
    before = target.to_dict()
    draft = SpeculativeConfig.hf_config_override(target)
    assert isinstance(draft, Step5MTPConfig)
    assert draft.model_type == "step5_mtp" and draft.architectures == ["Step5MTP"]
    assert draft.n_predict == 3 and len(draft.layer_types_with_mtp) == 95
    assert draft.num_hidden_layers == 92 and len(draft.layer_types) == 92
    assert draft.quantization_config is None
    assert target.to_dict() == before
    assert "step5_mtp" in get_args(spec_module.MTPModelTypes)
    # An already converted config retains all depth and layer metadata.
    again = SpeculativeConfig.hf_config_override(draft)
    assert again.to_dict() == draft.to_dict() and again is not draft


def test_speculative_routing_rejects_incomplete_draft_metadata():
    from vllm_iluvatar.config.speculative import make_hf_config_override

    wrap = make_hf_config_override(lambda value: value)
    target = config()
    target.num_nextn_predict_layers = 0
    with pytest.raises(ValueError, match="no MTP"):
        wrap(target)
    target = config()
    target.layer_types_with_mtp = target.layer_types_with_mtp[:94]
    with pytest.raises(ValueError, match="all main and draft"):
        wrap(target)


def test_draft_builder_selects_speculative_config_over_target():
    target = config()
    draft = Step5MTPConfig.from_target(target)
    draft.max_position_embeddings = 4096
    original = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target),
        quant_config=object(),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft)
        ),
    )
    result = draft_decoder_config(original)
    assert result.model_config.hf_config.max_position_embeddings == 4096
    assert result.model_config.hf_config.model_type == "step5_mtp"
    assert (
        original.model_config.hf_config is target and original.quant_config is not None
    )
    assert result.quant_config is None


def test_checkpoint_key_mapping_matches_actual_dense_mtp_layout():
    module = Step5MTP.__new__(Step5MTP)
    nn.Module.__init__(module)
    for index in (92, 93, 94):
        base = f"model.layers.{index}."
        for name in ("enorm.weight", "hnorm.weight", "eh_proj.weight"):
            assert module._rewrite_spec_layer_name(index, base + name) == base + name
        assert (
            module._rewrite_spec_layer_name(
                index, base + "transformer.shared_head.output.weight"
            )
            == base + "output_head.head.weight"
        )
        assert (
            module._rewrite_spec_layer_name(
                index, base + "transformer.shared_head.norm.weight"
            )
            == base + "output_head.norm.weight"
        )
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_norm.weight",
            "mlp.up_proj.weight",
        ):
            assert (
                module._rewrite_spec_layer_name(index, base + name)
                == base + "mtp_block." + name
            )
    assert (
        module._rewrite_spec_layer_name(None, "model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


class Head(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2) * multiplier)

    def forward(self, x):
        return x


class Layer(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step
        self.output_head = Head(step + 1)

    def forward(self, positions, previous, embedding):
        return previous + self.step


def predictor():
    result = Step5MultiTokenPredictor.__new__(Step5MultiTokenPredictor)
    nn.Module.__init__(result)
    result.mtp_start_layer_idx = 92
    result.num_mtp_layers = 3
    result._last_spec_step_idx = 0
    result.layers = nn.ModuleDict({str(92 + i): Layer(i) for i in range(3)})
    result.logits_processor = lambda head, x: head(x)
    return result


@torch.inference_mode()
def test_three_steps_use_distinct_heads_without_explicit_logits_index():
    model = predictor()
    hidden = torch.tensor([[1.0, 2.0]])
    for step in (0, 1, 2, 0, 2, 1, 3, 5):
        feedback = model(
            torch.tensor([1]),
            torch.tensor([0]),
            hidden,
            inputs_embeds=torch.zeros_like(hidden),
            spec_step_idx=step,
        )
        expected = (hidden + step % 3) * (step % 3 + 1)
        torch.testing.assert_close(model.compute_logits(feedback), expected)
    torch.testing.assert_close(model.compute_logits(hidden, spec_step_idx=0), hidden)
    for invalid in (-1, torch.tensor(1)):
        with pytest.raises(ValueError):
            model._layer(invalid)


@torch.inference_mode()
def test_generic_top_level_head_reassignment_preserves_all_draft_heads():
    draft = Step5MTP.__new__(Step5MTP)
    nn.Module.__init__(draft)
    draft.model = predictor()
    draft.lm_head = draft.model._layer(0).output_head.head
    old = [layer.output_head.head for layer in draft.model.layers.values()]
    # vLLM may replace this compatibility alias. The actual layer heads are
    # deliberately not called shared_head and remain independent modules.
    draft.lm_head = nn.Linear(2, 2, bias=False)
    for step in range(3):
        layer = draft.model._layer(step)
        assert layer.output_head.head is old[step]
        assert not hasattr(layer, "shared_head")
        x = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(draft.compute_logits(x, step), x * (step + 1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mtp_layer_uses_native_preprocessing_before_projection(monkeypatch, dtype):
    import importlib

    ops = importlib.import_module(
        "ixformer.inference.functions.models.step5.mtp.float.mtp_norm_concat"
    )

    from vllm_iluvatar.models.step5.model import Step5RMSNorm
    from vllm_iluvatar.models.step5.mtp import Step5MTPLayer

    layer = Step5MTPLayer.__new__(Step5MTPLayer)
    nn.Module.__init__(layer)
    layer.enorm = Step5RMSNorm(4096, 1e-5)
    layer.hnorm = Step5RMSNorm(4096, 1e-5)
    embedding = torch.randn(2, 4096, dtype=torch.bfloat16)
    hidden = torch.randn(2, 4096, dtype=dtype)
    positions = torch.arange(2)
    fused = torch.empty(2, 8192, dtype=torch.bfloat16)
    projected = torch.empty_like(embedding)
    calls = []

    def preprocess(e, h, ew, hw, eps, *, backend):
        assert e is embedding and h is hidden
        assert ew is layer.enorm.weight and hw is layer.hnorm.weight
        assert eps == layer.enorm.variance_epsilon and backend == "ixformer_native"
        calls.append("native")
        return fused

    def project(value):
        assert value is fused
        calls.append("projection")
        return projected, None

    def decode(*, positions, hidden_states):
        assert hidden_states is projected
        calls.append("decoder")
        return positions

    monkeypatch.setattr(ops, "mtp_norm_concat", preprocess)
    layer.eh_proj = project
    layer.mtp_block = decode
    assert layer(positions, hidden, embedding) is positions
    assert calls == ["native", "projection", "decoder"]
