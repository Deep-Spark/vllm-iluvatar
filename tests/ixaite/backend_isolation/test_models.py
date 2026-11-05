# SPDX-License-Identifier: Apache-2.0
"""Keep model construction and weight conversion inside the selected backend."""

import os
from types import SimpleNamespace

import pytest
import torch


# Unsupported packages must publish a constructor failure instead of a vendor class.
@pytest.mark.parametrize(
    "architecture",
    [
        "GlmMoeDsaForCausalLM",
        "HYV4ForCausalLM",
        "Qwen4ExpForCausalLM",
    ],
)
def test_unavailable_models_fail_before_parameters(architecture):
    from vllm_iluvatar.backends.ixaite.models.unsupported import UnsupportedIxaiteModel

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        )
    )
    with pytest.raises(RuntimeError, match=architecture):
        UnsupportedIxaiteModel(vllm_config=config)


# The ixaite classifier rejects FP8 before construction can enter a vendor path.
def test_ds4_own_dispatch_rejects_fp8_before_import(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch import IluDeepseekV4Arch

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
        ),
        quant_config=None,
    )
    with pytest.raises(RuntimeError, match="FP8.*unavailable"):
        IluDeepseekV4Arch(vllm_config=config)


# Missing routing is checked before the model reaches TP groups or parameter creation.
def test_ds4_hash_dependency_is_checked_at_construction(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4 import checkpoint, model

    calls = []

    # Recording the required op also proves construction stopped at the first gap.
    def missing(*names):
        calls.extend(names)
        raise RuntimeError("missing topk_hash_softplus_sqrt")

    monkeypatch.setattr(checkpoint, "require_model_ops", missing)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hash_layers=3)),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
    )
    with pytest.raises(RuntimeError, match="topk_hash_softplus_sqrt"):
        model.IluDeepseekV4ForCausalLM(vllm_config=config)
    assert calls == ["topk_hash_softplus_sqrt"]


# A rejected cache contract must not require model parameters or GPU buffers.
def test_ds4_fp8_cache_rejected_at_construction():
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint import (
        validate_attention_contract,
    )

    with pytest.raises(RuntimeError, match="cache layout.*fp8_ds_mla"):
        validate_attention_contract(SimpleNamespace(cache_dtype="fp8_ds_mla"))


# Both checkpoint parameters must survive a missing repacker unchanged.
def test_ds4_repack_does_not_enter_baseline_or_mutate_weights(monkeypatch):
    from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4 import checkpoint
    from vllm_iluvatar.backends.ixaite.quantization.config import (
        IxaiteCompressedTensorsLinearMethod,
    )

    monkeypatch.setattr(
        checkpoint, "load_ixaite_python_op", lambda *_: (None, "catalog missing")
    )
    layer = torch.nn.Module()
    layer.bmm_geom = SimpleNamespace(n_local_groups=1, o_lora_rank=2)
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    original_weight, original_scale = layer.weight, layer.weight_scale
    method = object.__new__(IxaiteCompressedTensorsLinearMethod)
    with forbid_ixformer_calls(), pytest.raises(
        RuntimeError, match="Marlin W8 repack.*catalog missing"
    ):
        method.process_weights_after_loading(layer)
    assert layer.weight is original_weight and layer.weight_scale is original_scale
    assert not hasattr(layer, "W_repacked")
    assert torch.equal(layer.weight, torch.tensor([[1, 2], [3, 4]], dtype=torch.int8))


# Fixed router choices must fail before the parent's expert-map/weight lifecycle.
@pytest.mark.parametrize(
    "options,reason",
    [
        ({"use_grouped_topk": True}, "grouped_topk"),
        ({"scoring_func": "sqrtsoftplus"}, "sqrtsoftplus"),
    ],
)
def test_own_routed_experts_reject_missing_routing(options, reason):
    from vllm_iluvatar.backends.ixaite.moe.routed_experts import IluRoutedExperts

    with pytest.raises(RuntimeError, match=reason):
        IluRoutedExperts("experts", torch.bfloat16, None, None, None, **options)


# Actual registration paths must reference the independent model implementations.
def test_own_model_registration_paths(monkeypatch):
    import vllm

    from vllm_iluvatar.backends.ixaite.models import (
        pixtral,
        step3p5,
        step3p5_mtp,
        unsupported,
    )
    from vllm_iluvatar.backends.ixaite.models.deepseek_v4 import arch, model
    from vllm_iluvatar.backends.ixaite.models.minimax_m3 import model as minimax_model

    calls = {}
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda name, path: calls.__setitem__(name, path),
    )
    for module in (
        pixtral,
        step3p5,
        step3p5_mtp,
        unsupported,
        arch,
        model,
        minimax_model,
    ):
        module.register_model()
    for architecture in (
        "PixtralForConditionalGeneration",
        "Step3p5ForCausalLM",
        "Step3p5MTP",
        "DeepseekV4ForCausalLM",
        "DeepSeekV4MTPModel",
        "IluDeepseekV4ForCausalLM",
        "IluDSparkDraftModel",
        "GlmMoeDsaForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "IluMiniMaxM3SparseForCausalLM",
        "IluMiniMaxM3SparseForConditionalGeneration",
    ):
        assert calls[architecture].startswith("vllm_iluvatar.backends.ixaite.models.")


# A decode-only vendor override must be rejected before communicator buffers exist.
def test_own_communicator_rejects_decode_ep_override(monkeypatch):
    import vllm.config
    from vllm import envs

    from vllm_iluvatar.backends.ixaite.distributed import IluIxaiteCommunicator

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(all2all_backend="naive")
        ),
    )
    monkeypatch.setattr(envs, "VLLM_ILU_DECODE_EP_BACKEND", "deepep_low_latency")
    with pytest.raises(ValueError, match="vendor decode EP override"):
        IluIxaiteCommunicator(None, unique_name="ep:0", use_all2all=True)


# Native AWQ/GPTQ/FP8 loaders must be rejected before they repack weights.
@pytest.mark.parametrize("quantization", ["awq", "gptq", "fp8", "ilu_runtime"])
def test_own_platform_rejects_vendor_quantization(quantization):
    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform

    with pytest.raises(ValueError, match="ixaite does not support quantization"):
        IxaitePlatform.verify_quantization(quantization)


# These formats have ixaite-owned config classes; individual schemes validate later.
@pytest.mark.parametrize(
    "quantization",
    ["compressed-tensors", "ilu_deepseek_v4_w4a8", "ilu_minimax_m3_flash"],
)
def test_own_platform_accepts_owned_quantization_configs(quantization):
    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform

    IxaitePlatform.verify_quantization(quantization)


# Peer access is decided from the resolved node count, before NCCL init.
@pytest.mark.parametrize("nnodes,expected", [(1, None), (2, "1")])
def test_config_sets_peer_access_from_node_count(monkeypatch, nnodes, expected):
    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform
    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.setattr(IluPlatform, "check_and_update_config", lambda *_: None)
    config = SimpleNamespace(
        lora_config=None,
        additional_config={},
        parallel_config=SimpleNamespace(
            worker_cls="vllm_iluvatar.backends.ixaite.worker.IxaiteWorker",
            nnodes=nnodes,
        ),
    )
    IxaitePlatform.check_and_update_config(config)
    assert os.environ.get("NCCL_P2P_DISABLE") == expected


# No vendor LoRA workspace can be constructed for an ixaite service.
def test_own_platform_rejects_lora_before_parent_configuration(monkeypatch):
    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform
    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    monkeypatch.setattr(
        IluPlatform,
        "check_and_update_config",
        lambda *_: pytest.fail("parent config entered"),
    )
    with pytest.raises(ValueError, match="ixaite does not support LoRA"):
        IxaitePlatform.check_and_update_config(SimpleNamespace(lora_config=object()))


# Runtime vendor conversion cannot bypass the checkpoint quantization check.
def test_own_platform_rejects_runtime_quant_before_parent_configuration(monkeypatch):
    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform
    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    monkeypatch.setattr(
        IluPlatform,
        "check_and_update_config",
        lambda *_: pytest.fail("parent config entered"),
    )
    config = SimpleNamespace(lora_config=None, additional_config={"ilu_quant": {}})
    with pytest.raises(ValueError, match="ixaite does not support runtime ilu_quant"):
        IxaitePlatform.check_and_update_config(config)


# Custom workers must retain the owned lifecycle without constructing GPU state here.
@pytest.mark.parametrize("owned", [True, False])
def test_custom_worker_preserves_ixaite_lifecycle(monkeypatch, owned):
    from vllm.utils import import_utils

    from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform
    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker
    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(IluPlatform, "check_and_update_config", lambda *_: None)
    worker = type("CustomWorker", (IxaiteWorker if owned else object,), {})
    monkeypatch.setattr(import_utils, "resolve_obj_by_qualname", lambda _: worker)
    config = SimpleNamespace(
        lora_config=None,
        additional_config={},
        parallel_config=SimpleNamespace(worker_cls="extension.CustomWorker"),
    )
    if owned:
        IxaitePlatform.check_and_update_config(config)
        assert config.parallel_config.worker_cls == "extension.CustomWorker"
    else:
        with pytest.raises(ValueError, match="must derive from IxaiteWorker"):
            IxaitePlatform.check_and_update_config(config)
