# Source: models/deepseek_v4/arch.py at 47fb804; checkpoint fixes are confined to ixaite.
# SPDX-License-Identifier: Apache-2.0
"""Community DeepSeek-V4 arch names dispatched by checkpoint quant.

``DeepseekV4ForCausalLM`` is shared by the raw FP8 release and FAE W4A8
compressed-tensors packs. Register a dispatcher so 0731-style checkpoints
keep their ``config.json`` and still land on the Iluvatar W4A8 model
(ixaite MHC).

``ModelRegistry.inspect_model_cls`` runs on this class *before* construct,
so the stubs below must look like a text-generation model (``embed_input_ids``,
``forward``, ``compute_logits``, ``__init__(vllm_config=...)``). ``__new__``
returns the real FP8 / W4A8 instance; those stubs are never executed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint import (
    classify_deepseek_v4_vllm_config,
)

logger = init_logger(__name__)


def _require_kind(vllm_config, *, what: str) -> str:
    kind = classify_deepseek_v4_vllm_config(vllm_config)
    if kind in ("fp8", "w4a8"):
        return kind
    hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    qc = getattr(hf, "quantization_config", None) or getattr(
        hf, "compression_config", None
    )
    raise ValueError(
        f"{what}: cannot choose FP8 vs W4A8 for DeepSeek-V4. "
        f"Need quantization_config/compression_config with "
        f"quant_method=fp8 or compressed-tensors W4A8 (or iluvatar_quant). "
        f"Got quant blob={qc!r}."
    )


class IluDeepseekV4Arch(nn.Module, SupportsPP):
    """Registered as community ``DeepseekV4ForCausalLM``."""

    def __new__(cls, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            raise TypeError(
                "IluDeepseekV4Arch requires vllm_config= (keyword or first arg)"
            )
        kind = _require_kind(vllm_config, what="IluDeepseekV4Arch")
        logger.info("ILU: DeepseekV4ForCausalLM dispatch kind=%s", kind)
        if kind == "w4a8":
            from vllm_iluvatar.backends.ixaite.models.deepseek_v4.model import (
                IluDeepseekV4ForCausalLM,
            )

            return IluDeepseekV4ForCausalLM(*args, **kwargs)
        raise RuntimeError(
            "ixaite DeepSeek-V4 FP8 requires unavailable FP8 attention and MoE operators"
        )

    # Registry inspect only. ``__new__`` returns another class, so these never run.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("IluDeepseekV4Arch stub; __new__ must return real model")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        raise RuntimeError("IluDeepseekV4Arch stub; __new__ must return real model")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise RuntimeError("IluDeepseekV4Arch stub; __new__ must return real model")

    def make_empty_intermediate_tensors(self, *args, **kwargs):
        raise RuntimeError("IluDeepseekV4Arch stub; __new__ must return real model")


class IluDeepSeekV4MTPArch(nn.Module):
    """Registered as community ``DeepSeekV4MTPModel``."""

    def __new__(cls, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            raise TypeError(
                "IluDeepSeekV4MTPArch requires vllm_config= (keyword or first arg)"
            )
        kind = _require_kind(vllm_config, what="IluDeepSeekV4MTPArch")
        logger.info("ILU: DeepSeekV4MTPModel dispatch kind=%s", kind)
        if kind == "w4a8":
            from vllm_iluvatar.backends.ixaite.models.deepseek_v4.mtp import (
                IluDeepSeekV4MTP,
            )

            return IluDeepSeekV4MTP(*args, **kwargs)
        raise RuntimeError(
            "ixaite DeepSeek-V4 FP8 MTP requires unavailable FP8 attention and MoE operators"
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("IluDeepSeekV4MTPArch stub; __new__ must return real model")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        raise RuntimeError("IluDeepSeekV4MTPArch stub; __new__ must return real model")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise RuntimeError("IluDeepSeekV4MTPArch stub; __new__ must return real model")


def register_model() -> None:
    """Bind community DS-V4 arch names to the quant dispatcher.

    Invoked by ``backends.ixaite.oot_registration._register_models``. Converted
    W4A8 packs still register ``IluDeepseekV4ForCausalLM`` from ``model.py``.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch:IluDeepseekV4Arch",
    )
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.arch:IluDeepSeekV4MTPArch",
    )
