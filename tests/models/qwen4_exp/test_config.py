# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.config import get_config
from vllm_iluvatar.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def test_qwen4_exp_framework_defaults_enable_architecture_features() -> None:
    config = _text_config()

    assert config.hc_count == 2
    assert config.output_gate_type == "sigmoid"


def test_qwen4_exp_qsa_preserves_indexer_config() -> None:
    config = _text_config(
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )

    assert config.indexer_n_heads == 2
    assert config.indexer_compress_ratio == 4


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"indexer_n_heads": 2}, "missing required fields"),
        (
            {
                "indexer_n_heads": 0,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
            "must be positive",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 2,
                "indexer_head_dim": 8,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
            "indexer_kv_heads=1",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 1025,
                "indexer_compress_ratio": 2,
            },
            "must be divisible",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 2,
                "indexer_compress_ratio": 4,
            },
            "must be divisible",
        ),
    ],
)
def test_qwen4_exp_rejects_invalid_qsa_config(
    overrides: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _text_config(**overrides)


@pytest.mark.skip(reason="Qwen4Exp MTP is gated off on this BI-V150 port")
def test_qwen4_exp_mtp_override_preserves_text_backbone_layout() -> None:
    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config(),
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == draft_config.text_config.hc_count == 2
    assert draft_config.to_dict()["hc_mult"] == 2
    assert draft_config.n_predict == 1
    assert draft_config.index_share_for_mtp_iteration is False
    assert draft_config.text_config.num_hidden_layers == 2
    assert draft_config.text_config.layer_types == [
        "linear_attention",
        "full_attention",
    ]


@pytest.mark.skip(reason="Qwen4Exp MTP is gated off on this BI-V150 port")
def test_qwen4_exp_text_mtp_override_sets_hc_mult() -> None:
    config = _text_config(architectures=["Qwen4ExpForCausalLM"])

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.hc_mult == draft_config.hc_count == 2
    assert draft_config.to_dict()["hc_mult"] == 2
    assert draft_config.index_share_for_mtp_iteration is False


@pytest.mark.skip(reason="Qwen4Exp MTP is gated off on this BI-V150 port")
@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_exposes_index_share_flag(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True


def test_qwen4_exp_checkpoint_names_load_without_overrides(tmp_path) -> None:
    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config().to_dict(),
    )
    config.to_json_file(tmp_path / "config.json", use_diff=False)

    loaded_config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(loaded_config, Qwen4ExpConfig)
    assert isinstance(loaded_config.text_config, Qwen4ExpTextConfig)
    assert loaded_config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert loaded_config.text_config.hc_count == 2
    assert loaded_config.text_config.output_gate_type == "sigmoid"


def test_qwen4_exp_config_classes_are_reachable_without_touching_vllm() -> None:
    """The HF config classes must resolve through the community fallbacks only.

    ``_register_configs()`` runs at plugin import; upstream vLLM carries no
    qwen4_exp entry, so if either registry were missed the checkpoint would
    fail to load with an unknown ``model_type``.
    """
    from transformers import AutoConfig
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert _CONFIG_REGISTRY["qwen4_exp"] is Qwen4ExpConfig
    assert _CONFIG_REGISTRY["qwen4_exp_text"] is Qwen4ExpTextConfig
    assert AutoConfig.for_model("qwen4_exp").__class__ is Qwen4ExpConfig


def test_qwen4_exp_register_model_points_at_the_plugin_package() -> None:
    from vllm import ModelRegistry

    from vllm_iluvatar.models.qwen4_exp import model as model_hook

    model_hook.register_model()
    for architecture in (
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    ):
        assert architecture in ModelRegistry.get_supported_archs()


def test_qwen4_exp_registers_v2_model_state() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from vllm_iluvatar.models.qwen4_exp.model_state import Qwen4ExpModelState

    assert Qwen4ExpForCausalLM.get_model_state_cls() is Qwen4ExpModelState
    assert Qwen4ExpForConditionalGeneration.get_model_state_cls() is Qwen4ExpModelState


def test_qwen4_exp_uses_local_moe_metadata() -> None:
    from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
    from vllm_iluvatar.models.qwen4_exp.model import (
        Qwen4ExpDecoderLayer,
        Qwen4ExpMixtureOfExperts,
        Qwen4ExpSparseMoeBlock,
    )

    assert not issubclass(Qwen4ExpMixtureOfExperts, QwenNextMixtureOfExperts)

    layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
    torch.nn.Module.__init__(layer)
    moe = Qwen4ExpSparseMoeBlock.__new__(Qwen4ExpSparseMoeBlock)
    torch.nn.Module.__init__(moe)
    moe.experts = torch.nn.Identity()
    moe.n_logical_experts = 8
    moe.n_physical_experts = 10
    moe.n_local_physical_experts = 5
    moe.n_routed_experts = 8
    moe.n_shared_experts = 1
    moe.n_redundant_experts = 2
    layer.mlp = moe

    metadata = Qwen4ExpMixtureOfExperts()
    metadata.set_moe_parameters([layer])

    assert metadata.moe_layers == [moe.experts]
    assert metadata.num_moe_layers == 1
    assert metadata.num_logical_experts == 8
    assert metadata.num_physical_experts == 10
    assert metadata.num_shared_experts == 1


def _supported_vllm_config(**overrides) -> SimpleNamespace:
    """Minimal VllmConfig stand-in that ``_verify_supported_config`` accepts."""
    parallel = {
        "pipeline_parallel_size": 1,
        "enable_eplb": False,
        "enable_dbo": False,
        "ubatch_size": 1,
    }
    parallel.update(overrides.pop("parallel_config", {}))
    config = {
        "parallel_config": SimpleNamespace(**parallel),
        "speculative_config": None,
        "compilation_config": SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.NONE,
        ),
        "cache_config": SimpleNamespace(
            mamba_ssm_cache_dtype="auto",
            mamba_cache_dtype="auto",
            cpu_offload_gb=0,
        ),
        "model_config": SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(mamba_ssm_dtype="float32"),
        ),
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_qwen4_exp_accepts_the_supported_configuration() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    with patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True):
        _verify_supported_config(_supported_vllm_config())


def test_qwen4_exp_accepts_decode_only_full_cudagraph_without_compile() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        mode=CompilationMode.NONE,
    )
    with patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True):
        _verify_supported_config(
            _supported_vllm_config(compilation_config=compilation_config)
        )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_qwen4_exp_rejects_unsupported_cudagraph_modes(
    cudagraph_mode: CUDAGraphMode,
) -> None:
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    compilation_config = SimpleNamespace(
        cudagraph_mode=cudagraph_mode,
        mode=CompilationMode.NONE,
    )
    with (
        patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True),
        pytest.raises(NotImplementedError, match="FULL_DECODE_ONLY"),
    ):
        _verify_supported_config(
            _supported_vllm_config(compilation_config=compilation_config)
        )


def test_qwen4_exp_rejects_compile_with_decode_only_cudagraph() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        mode=CompilationMode.VLLM_COMPILE,
    )
    with (
        patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True),
        pytest.raises(NotImplementedError, match="mode=0"),
    ):
        _verify_supported_config(
            _supported_vllm_config(compilation_config=compilation_config)
        )


def test_qwen4_exp_refuses_the_v1_model_runner() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    with (
        patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", False),
        pytest.raises(NotImplementedError, match="VLLM_USE_V2_MODEL_RUNNER=1"),
    ):
        _verify_supported_config(_supported_vllm_config())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"parallel_config": {"pipeline_parallel_size": 2}}, "pipeline parallelism"),
        ({"parallel_config": {"enable_eplb": True}}, "EPLB"),
        ({"parallel_config": {"enable_dbo": True}}, "dual-batch overlap"),
        ({"parallel_config": {"ubatch_size": 2}}, "microbatching"),
        ({"speculative_config": SimpleNamespace()}, "speculative decoding"),
        (
            {
                "cache_config": SimpleNamespace(
                    mamba_ssm_cache_dtype="auto",
                    mamba_cache_dtype="auto",
                    cpu_offload_gb=4,
                )
            },
            "CPU offload",
        ),
    ],
)
def test_qwen4_exp_rejects_features_it_does_not_implement(
    overrides: dict, match: str
) -> None:
    """These gates lived in ``MODELS_CONFIG_MAP`` in tree.

    Out of tree there is no per-architecture config hook, so they moved into the
    model's ``__init__``; each one would otherwise fail late and obscurely, or
    silently roll a per-request carry twice.
    """
    from vllm_iluvatar.models.qwen4_exp.model import _verify_supported_config

    with (
        patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True),
        pytest.raises(NotImplementedError, match=match),
    ):
        _verify_supported_config(_supported_vllm_config(**overrides))


def test_qwen4_exp_gdn_ssm_state_falls_back_to_the_checkpoint_dtype() -> None:
    """``mamba_ssm_cache_dtype="auto"`` must not silently demote the GDN state.

    In tree ``MODELS_CONFIG_MAP`` copied the checkpoint's ``mamba_ssm_dtype``
    onto ``cache_config``; without that hook, "auto" would leave the recurrent
    state at the model dtype instead of the float32 this checkpoint asks for.
    """
    from vllm_iluvatar.models.qwen4_exp.model import Qwen4ExpForCausalLM

    _, ssm_dtype = Qwen4ExpForCausalLM.get_gdn_mamba_state_dtype_from_config(
        _supported_vllm_config()
    )
    assert ssm_dtype == torch.float32


def test_qwen4_exp_gdn_ssm_state_honours_an_explicit_override() -> None:
    from vllm_iluvatar.models.qwen4_exp.model import Qwen4ExpForCausalLM

    config = _supported_vllm_config(
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="bfloat16",
            mamba_cache_dtype="auto",
            cpu_offload_gb=0,
        )
    )
    _, ssm_dtype = Qwen4ExpForCausalLM.get_gdn_mamba_state_dtype_from_config(config)
    assert ssm_dtype == torch.bfloat16


def test_qwen4_exp_text_config_strips_mrope() -> None:
    """QSA derives a group's RoPE position arithmetically, so positions must be 1D."""
    config = _text_config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
        }
    )

    assert "mrope_section" not in config.rope_parameters
    assert "mrope_interleaved" not in config.rope_parameters
