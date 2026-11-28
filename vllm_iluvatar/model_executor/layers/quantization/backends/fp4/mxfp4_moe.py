# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP4 MoE quantization method.

Corresponds to: vllm/model_executor/layers/quantization/mxfp4.py

Target   : vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod
Baseline : v0.31.0 (db9527a468)
Why      : community ``Mxfp4MoEMethod`` picks a backend through
           ``select_deepseek_v4_mxfp4_moe_backend``, and off ROCm that function
           can only return NVIDIA-specific backends (TRTLLM / DeepGEMM / Marlin /
           FlashInfer) or raise ``NotImplementedError``. Iluvatar needs the
           community weight layout with an Iluvatar compute.
How      : subclass the community Method. The selected compute backend declares
           a final weight contract; the MXFP4 loader converts paired checkpoint
           shards directly into those parameters. Post-load only finalizes and
           binds compute.
Scope    : routed MXFP4 experts on Iluvatar; selected by
           ``IluDeepseekV4FP8Config.get_quant_method``.
Remove   : when community exposes a vendor MXFP4 MoE backend hook.

The Method is named after the source quantization format. Model routing and
activation parameters come from ``moe_config`` / the router. Reuse by another
model still requires compatible checkpoint shard, activation and communication
contracts; the shared MXFP4 name alone does not establish compatibility.

Why ``experts_cls`` must be *absent* rather than ``None``
--------------------------------------------------------
Community ``FusedMoEMethodBase.is_monolithic``
(``fused_moe_method_base.py:163-170``) reads::

    if self.moe_kernel is None:
        if hasattr(self, "experts_cls"):
            return self.experts_cls.is_monolithic()
        return False

``Mxfp4MoEMethod.__init__`` (``mxfp4.py:481``) assigns ``self.experts_cls =
None``, so the ``hasattr`` guard passes and the attribute access raises
``AttributeError: 'NoneType' object has no attribute 'is_monolithic'``. That is
exactly the EP-only crash the retired patch P3
(``patch_dsv4_moe_modular_kernel``) worked around. Not setting the attribute
makes ``hasattr`` false; the mixin's explicit ``is_monolithic`` makes the
answer independent of that subtlety altogether. ``__init__`` asserts both.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from vllm.config import get_current_vllm_config_or_none

# Same import surface community ``quantization/mxfp4.py:8-15`` uses -- the
# package root, not the defining modules, which is what keeps this free of the
# fused_moe <-> quantization import cycle.
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    MoEComputeBindingMixin,
    pad_online_w4a8_config,
)


class IluMxfp4MoEMethod(MoEComputeBindingMixin, Mxfp4MoEMethod):
    """MXFP4 checkpoint shards converted straight into the selected contract.

    The source's counterpart of ``IluFp8MoEMethod`` / ``IluMxFp8MoEMethod``:
    the loader writes the compute backend's final parameters as shards arrive,
    and everything after that is the shared lifecycle. MXFP4 keeps its own
    ``create_weights`` because it has its own loader and because streaming
    needs a sequential checkpoint reader.

    ``selection`` is injected rather than resolved lazily: ``make_mxfp4_method``
    owns the policy, and the loader the constructor has to reach needs the
    answer.
    """

    compute_source = "MXFP4"

    # [ILU override] Mxfp4MoEMethod.__init__ — delta: skip the parent body (it
    # calls select_deepseek_v4_mxfp4_moe_backend, which off ROCm either returns
    # an NVIDIA-only backend or raises) and mirror its remaining field
    # initialisation, minus ``experts_cls``. Then record the selected backend's
    # identity, which ``deepseek_v4/model.py`` reads to decide hybrid EP and
    # the lifecycle below reads to build the loader and the compute.
    #
    # Mirrors community ``mxfp4.py:478-489``. If that constructor grows a field
    # the asserts below will not catch it, but every consumer of a missing field
    # raises ``AttributeError`` loudly rather than computing a wrong number.
    def __init__(self, moe: FusedMoEConfig, selection):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            get_moe_backend,
        )

        # Skip Mxfp4MoEMethod.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "mxfp4"
        # NONE is still required (not merely convenient): community
        # ``maybe_roundup_sizes`` (mxfp4.py:507), ``skip_forward_padding``
        # (:494) and ``has_unpadded_output`` (:499) all read this attribute, and
        # ``maybe_roundup_sizes`` runs during FusedMoE construction. Under NONE
        # the round-up degrades to ``round_up(intermediate, 64)``
        # (``oracle/mxfp4.py:671-672``), a no-op for DeepSeek-V4's 4096 / 2048.
        self.mxfp4_backend = Mxfp4MoeBackend.NONE
        self.max_capture_size = moe.max_capture_size
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel = None
        self.w13_precision_config = None
        self.w2_precision_config = None

        assert not hasattr(self, "experts_cls"), (
            f"{type(self).__name__} must not define experts_cls; community "
            "is_monolithic() would dereference it (see module docstring)"
        )
        assert self.is_monolithic is False

        self.selection = selection
        self._compute = None
        self._runtime_weights = None
        self._checkpoint_loader = None
        self.ilu_compute_backend = selection.name
        self.ilu_weight_format = selection.layout
        self.ilu_prefer_hybrid_ep = get_moe_backend(selection).prefer_hybrid_ep

    def _compute_config(self, layer):
        config = super()._compute_config(layer)
        unpadded = layer.moe_config.intermediate_size_per_partition_unpadded
        if (
            self.selection.name == "ixmm_fp4fp8"
            and unpadded is not None
            and unpadded != config.intermediate
        ):
            # Gate/up and activation use the logical width. Only the down
            # projection needs extra K alignment in the weight contract.
            return replace(config, intermediate=unpadded)
        return pad_online_w4a8_config(config, unpadded, self.selection)

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            make_moe_compute,
        )
        from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
            Mxfp4LoadSpec,
            make_mxfp4_loader,
        )

        if self._checkpoint_loader is not None:
            raise RuntimeError("MXFP4 Method cannot create weights twice")
        if (
            self.moe.has_bias
            or not self.moe.is_act_and_mul
            or self.moe.hidden_dim_unpadded != hidden_size
            or (
                self.selection.name not in ("ixmm_fp4fp8", "ixinfer_w4a8", "ixmm_w4a8")
                and self.moe.intermediate_size_per_partition_unpadded
                not in (None, intermediate_size_per_partition)
            )
        ):
            raise ValueError(
                "MXFP4 streaming requires unpadded bias-free gated experts"
            )
        config = get_current_vllm_config_or_none()
        if config is not None:
            fmt = config.load_config.load_format
            if str(getattr(fmt, "value", fmt)) not in (
                "auto",
                "safetensors",
                "pt",
                "npcache",
            ):
                raise ValueError(
                    "MXFP4 streaming requires a sequential checkpoint loader"
                )
        backend = make_moe_compute(self._compute_config(layer), self.selection)
        if backend.requires_contiguous_experts and layer.expert_map is not None:
            start = layer.moe_config.ep_rank * layer.local_num_experts
            expected = torch.full_like(layer.expert_map, -1)
            expected[start : start + layer.local_num_experts] = torch.arange(
                layer.local_num_experts, device=expected.device, dtype=expected.dtype
            )
            if not torch.equal(layer.expert_map, expected):
                raise ValueError("W4A8 compute requires contiguous expert placement")
        source_width = self.moe.intermediate_size_per_partition_unpadded
        if source_width is None:
            source_width = intermediate_size_per_partition
        loader = make_mxfp4_loader(
            backend.weight_spec,
            Mxfp4LoadSpec(
                num_experts,
                hidden_size,
                backend.intermediate,
                self.moe.tp_size,
                self.moe.tp_rank,
                source_width,
                backend.config.w2_k_padded,
                backend.config.w13_k_padded,
                backend.config.w2_n_padded,
            ),
        )
        layer.params_dtype = params_dtype
        self.num_experts = layer.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        loader.create_weights(layer, attrs)
        self._checkpoint_loader = loader
        self.supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer):
        if self._checkpoint_loader is None:
            raise RuntimeError("MXFP4 weights must be created once before finalization")
        if self._checkpoint_loader._layer is not layer:
            raise ValueError("MXFP4 Method can only finalize its own layer")
        # Runner-owned output scaling is installed after create_weights. The
        # mixin freezes it into the compute now, before any Experts adapter or
        # graph captures the backend.
        self._bind_compute(layer, self._checkpoint_loader.finalize())


def select_mxfp4_compute(arch, mode="auto"):
    from vllm_iluvatar.config.ilu_compute import select_four_bit_expert_compute

    return select_four_bit_expert_compute(arch, mode, source="MXFP4")


def make_mxfp4_method(layer):
    """The MXFP4 routed-expert Method for this build's selected compute."""
    from vllm.logger import init_logger

    from vllm_iluvatar import envs
    from vllm_iluvatar.config.ilu_compute import compute_options
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    options = compute_options("moe")
    arch = cuda_arch()
    mode = options.get("mode", "auto")
    # Checked whatever else is configured: the point is that a launch script
    # carrying this setting learns it stopped meaning anything, and someone who
    # also passes ilu_compute.moe is exactly who has a stale script to clean up.
    if envs.VLLM_ILU_MXFP4_MOE_BACKEND in ("torch", "ixformer_native"):
        raise ValueError(
            "VLLM_ILU_MXFP4_MOE_BACKEND=torch/ixformer_native no longer "
            "selects a routed-expert method; routed MXFP4 always converts "
            "to a compute backend's contract. Use ilu_compute.moe.mode to "
            "pick one."
        )
    if not options and arch == "ivcore11":
        # Preserve the existing default/legacy layout override when no explicit
        # MoE configuration is supplied. An explicit mode always takes
        # precedence.
        layout = envs.VLLM_ILU_MOE_RUNTIME_LAYOUT
        if layout:
            mode = {"NN": "ixinfer_w4a8_nn_v2", "TN": "ixinfer_w4a8_tn_v2"}.get(
                layout, layout
            )
    selection = select_mxfp4_compute(arch, mode)
    init_logger(__name__).info_once(
        "MXFP4 compute: %s / %s on %s (streaming weights)",
        selection.name,
        selection.layout,
        arch,
    )
    return IluMxfp4MoEMethod(layer.moe_config, selection)


def select_mxfp4_method(original, runtime=None):
    """Let a validated ``ilu_quant`` conversion replace the checkpoint's method."""
    from vllm_iluvatar.config.ilu_compute import compute_options

    if runtime is None:
        return original
    if any(value != "auto" for value in compute_options("moe").values()):
        raise ValueError(
            "Explicit ilu_quant MoE conversion cannot be combined with "
            "ilu_compute.moe overrides yet"
        )
    return runtime
