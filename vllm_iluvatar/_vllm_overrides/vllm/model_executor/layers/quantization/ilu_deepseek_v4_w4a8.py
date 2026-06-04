# SPDX-License-Identifier: Apache-2.0
"""Iluvatar mixed W4A8 quantization config for DeepSeek-V4.

Schema (in checkpoint's ``config.json::quantization_config``)::

    {
        "quant_method": "ilu_deepseek_v4_w4a8",
        "format": "ilu-mixed-w4a8",
        "version": 1 | 2,

        "routed_experts": {...},        # INT4 W4A8 MoE (v1 + v2)
        "shared_experts": {...},        # INT8 W8A8 MoE / Linear (v1 + v2)

        # NEW in v2: dense Linear (main MLA / MTP / indexer.wq_b) -- INT8
        # W8A8 channel-symmetric + per-token dynamic INT8 activation.
        # Same schema shape as ``shared_experts`` (decoded by the same
        # helper ``_quant_args_from_section``).
        "dense_linears": {              # optional; missing = v1 behaviour
            "targets": ["re:.*\\.attn\\.(fused_wqa_wkv|wq_b|wo_b)$", ...],
            "num_bits": 8, "strategy": "channel", "symmetric": true, ...,
            "input_activations": {
                "num_bits": 8, "strategy": "token", "dynamic": true,
                "symmetric": true, "type": "int"
            }
        },

        # NEW in v2: ``attn.wo_a`` -- grouped einsum / BMM, dispatched to
        # ``ixf_ops.marlin_w8a16`` (input stays BF16, weight is INT8 then
        # repacked into Marlin layout at process_weights_after_loading time).
        "bmm_w8a16_layers": {           # optional; missing = v1 behaviour
            "targets": ["re:.*\\.attn\\.wo_a$"],
            "num_bits": 8, "strategy": "channel", "symmetric": true, ...,
            "input_activations": null,
            "gemm_format": "k16n16",
            "input_dtype": "bfloat16"
        },

        "bf16_layers": [<regex...>],    # informational; runtime same as ignore
        "ignore":      [<regex...>]
    }

Dispatch (``get_quant_method``):

- ``FusedMoE`` whose prefix matches a routed-experts target ->
  :class:`IluCompressedTensorsW4A8Int8MoEMethod`
- ``FusedMoE`` whose prefix matches a shared-experts target ->
  :class:`IluCompressedTensorsW8A8Int8MoEMethod`
- ``LinearBase`` whose prefix matches a shared-experts target ->
  :class:`_IluV4W8A8LinearMethod` (W8A8 INT8 channel via
  :class:`CompressedTensorsW8A8Int8` scheme).
- (v2) ``LinearBase`` whose prefix matches ``bmm_w8a16_layers.targets`` ->
  :class:`_IluV4W8A16BMMLinearMethod` (INT8 weight + bf16 act via
  ``ixf_ops.marlin_w8a16``; the layer stores W_repacked / W_scale and the
  attention wrapper does the marlin GEMM directly).
- (v2) ``LinearBase`` whose prefix matches ``dense_linears.targets`` ->
  :class:`_IluV4W8A8LinearMethod` (same path as shared experts).
- Everything else (``bf16_layers`` / ``ignore`` / unmatched) ->
  :class:`UnquantizedLinearMethod` / :class:`UnquantizedFusedMoEMethod`.

Note on target matching: schema-side ``shared_experts.targets`` use the
*source* checkpoint naming (``.shared_experts.w1``); vLLM's internal prefix is
the *post-mapper* naming. For shared experts we use stable substring tests on
the post-mapper prefix and ignore the schema regex. ``dense_linears.targets``
and ``bmm_w8a16_layers.targets`` are written against the post-mapper prefix
already (e.g. ``fused_wqa_wkv`` is the runtime fused name) and matched with
``re.search``.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)


# Match both forms:
#   - FusedMoE container layer:        ".../ffn.experts"      (vLLM dispatch)
#   - Individual expert linear leaf:   ".../ffn.experts.5.w1" (offline tooling)
# vLLM only invokes ``get_quant_method`` on the container, but we keep the
# leaf form for forward-compat with any future per-expert module.
_ROUTED_EXPERT_PREFIX_RE = re.compile(r"\.experts(?:$|\.)")


def _is_routed_expert(prefix: str) -> bool:
    return (
        _ROUTED_EXPERT_PREFIX_RE.search(prefix) is not None and "shared" not in prefix
    )


def _is_shared_expert(prefix: str) -> bool:
    return ".shared_experts." in prefix or prefix.endswith(".shared_experts")


def _build_quant_args(
    *,
    num_bits: int,
    strategy: str,
    group_size: int | None,
    symmetric: bool,
    qtype: str = "int",
    dynamic: bool = False,
) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=num_bits,
        type=qtype,
        symmetric=symmetric,
        strategy=strategy,
        group_size=group_size,
        dynamic=dynamic,
    )


def _quant_args_from_section(
    section: dict[str, Any], default_strategy: str
) -> tuple[QuantizationArgs, QuantizationArgs]:
    """Decode ``routed_experts`` / ``shared_experts`` section into
    ``(weight_quant, input_quant)`` :class:`QuantizationArgs` pair."""

    weight_quant = _build_quant_args(
        num_bits=int(section["num_bits"]),
        strategy=str(section.get("strategy", default_strategy)),
        group_size=section.get("group_size"),
        symmetric=bool(section.get("symmetric", True)),
        qtype=str(section.get("type", "int")),
        dynamic=False,
    )

    act_section = section.get("input_activations") or {}
    input_quant = _build_quant_args(
        num_bits=int(act_section.get("num_bits", 8)),
        strategy=str(act_section.get("strategy", "token")),
        group_size=act_section.get("group_size"),
        symmetric=bool(act_section.get("symmetric", True)),
        qtype=str(act_section.get("type", "int")),
        dynamic=bool(act_section.get("dynamic", True)),
    )
    return weight_quant, input_quant


def _compile_targets(section: dict[str, Any] | None) -> list[re.Pattern]:
    """Compile ``section.targets`` (``["re:<pattern>", ...]``) into regexes.

    Non-``re:`` literal targets are treated as substring matches by wrapping
    them in ``re.escape``. Empty/missing section yields ``[]`` so dispatch
    naturally falls through to the unquantised branch.
    """
    if not section:
        return []
    out: list[re.Pattern] = []
    for t in section.get("targets", []) or []:
        if not isinstance(t, str):
            continue
        if t.startswith("re:"):
            out.append(re.compile(t[3:]))
        else:
            out.append(re.compile(re.escape(t)))
    return out


class IluDeepseekV4W4A8Config(QuantizationConfig):
    """Custom mixed-precision quant config for DeepSeek-V4 in ilu vllm.

    v1 layout: routed experts W4A8 INT4, shared experts W8A8 INT8, every
    other Linear left as BF16.

    v2 adds two optional schema sections (``dense_linears`` /
    ``bmm_w8a16_layers``) that move the main MLA / MTP / indexer Linear
    weights from BF16 to INT8 (W8A8 for plain linears, W8A16 marlin for
    ``attn.wo_a``'s grouped einsum). Both sections are optional -- a v1 ckpt
    with no such sections continues to dispatch identically.
    """

    def __init__(
        self,
        routed_section: dict[str, Any],
        shared_section: dict[str, Any] | None,
        bf16_layers: list[str],
        ignore: list[str],
        *,
        format_str: str,
        version: int,
        gemm_format: str,
        pack_version: int,
        dense_section: dict[str, Any] | None = None,
        bmm_w8a16_section: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.format_str = format_str
        self.version = version

        # Stash raw sections for downstream use / debugging.
        self.routed_section = routed_section
        self.shared_section = shared_section or {}
        self.dense_section = dense_section or {}
        self.bmm_w8a16_section = bmm_w8a16_section or {}
        self.bf16_layers_raw: list[str] = list(bf16_layers)
        self.ignore_raw: list[str] = list(ignore)

        # ixformer kernel parameters; surfaced so the routed MoE method can
        # read them deterministically without env vars.
        self.gemm_format = gemm_format
        self.pack_version = pack_version

        # Pre-build QuantizationArgs pairs.
        (
            self.routed_weight_quant,
            self.routed_input_quant,
        ) = _quant_args_from_section(
            routed_section,
            default_strategy=QuantizationStrategy.CHANNEL.value,
        )
        if shared_section:
            (
                self.shared_weight_quant,
                self.shared_input_quant,
            ) = _quant_args_from_section(
                shared_section,
                default_strategy=QuantizationStrategy.CHANNEL.value,
            )
        else:
            self.shared_weight_quant = None
            self.shared_input_quant = None

        # v2 dense linears (W8A8 INT8 channel) -- optional, missing => v1 behaviour.
        self.dense_targets: list[re.Pattern] = _compile_targets(self.dense_section)
        # v2 bmm w8a16 (attn.wo_a) -- optional.
        self.bmm_w8a16_targets: list[re.Pattern] = _compile_targets(
            self.bmm_w8a16_section
        )
        # ``input_dtype`` / ``gemm_format`` for the wo_a marlin path; kept on
        # the config so the W8A16 method can read them without env vars.
        self.bmm_w8a16_input_dtype = str(
            self.bmm_w8a16_section.get("input_dtype", "bfloat16")
        )
        self.bmm_w8a16_gemm_format = str(
            self.bmm_w8a16_section.get("gemm_format", "k16n16")
        )

    # ----- QuantizationConfig boilerplate ----------------------------------

    @classmethod
    def get_name(cls):  # type: ignore[override]
        return "ilu_deepseek_v4_w4a8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # ixformer w4a8 / w8a8 int8 kernels target sm75+ in practice; keep
        # generous to let ilu Iluvatar / consumer Hopper run.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IluDeepseekV4W4A8Config":
        routed = config.get("routed_experts", {}) or {}
        shared = config.get("shared_experts", {}) or {}
        # v2 optional sections; absent in v1 ckpts (defaults to {}).
        dense = config.get("dense_linears", {}) or {}
        bmm_w8a16 = config.get("bmm_w8a16_layers", {}) or {}
        bf16_layers = list(config.get("bf16_layers", []) or [])
        ignore = list(config.get("ignore", []) or [])
        format_str = str(config.get("format", "ilu-mixed-w4a8"))
        version = int(config.get("version", 1))
        gemm_format = str(routed.get("gemm_format", "TN"))
        pack_version = int(routed.get("pack_version", 2))
        return cls(
            routed,
            shared,
            bf16_layers,
            ignore,
            format_str=format_str,
            version=version,
            gemm_format=gemm_format,
            pack_version=pack_version,
            dense_section=dense,
            bmm_w8a16_section=bmm_w8a16,
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: Any,
        user_quant: str | None,
        hf_config: Any = None,
    ):
        """Activate this config when the checkpoint declares
        ``quant_method=ilu_deepseek_v4_w4a8`` (regardless of any user-passed
        ``--quantization`` flag)."""
        if (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") == cls.get_name()
        ):
            return cls.get_name()
        return None

    # ----- Dispatch --------------------------------------------------------

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Lazy imports to avoid heavy dependencies at config-class import time.
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        if isinstance(layer, FusedMoE):
            if _is_routed_expert(prefix):
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.iluvatar.ilu_compressed_tensors_moe_w4a8_int8 import (  # noqa: E501
                    IluCompressedTensorsW4A8Int8MoEMethod,
                )

                return IluCompressedTensorsW4A8Int8MoEMethod(
                    weight_quant=self.routed_weight_quant,
                    input_quant=self.routed_input_quant,
                    moe=layer.moe_config,
                    layer_name=prefix,
                )
            if _is_shared_expert(prefix):
                # In V4 shared_experts is a plain MLP, not a FusedMoE; we keep
                # this branch in case a future variant re-routes shared_experts
                # through FusedMoE.
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.iluvatar.ilu_compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
                    IluCompressedTensorsW8A8Int8MoEMethod,
                )

                assert self.shared_weight_quant is not None
                assert self.shared_input_quant is not None
                return IluCompressedTensorsW8A8Int8MoEMethod(
                    weight_quant=self.shared_weight_quant,
                    input_quant=self.shared_input_quant,
                    moe=layer.moe_config,
                    layer_name=prefix,
                )
            return UnquantizedFusedMoEMethod(layer.moe_config)

        if isinstance(layer, LinearBase):
            if _is_shared_expert(prefix):
                return _IluV4W8A8LinearMethod(self)
            # v2: ``attn.wo_a`` (grouped einsum / BMM) -- W8A16 marlin.
            # Match before dense_linears so the more specific regex wins.
            if any(p.search(prefix) for p in self.bmm_w8a16_targets):
                return _IluV4W8A16BMMLinearMethod(self)
            # v2: dense Linear (main MLA / MTP / indexer.wq_b) -- W8A8.
            if any(p.search(prefix) for p in self.dense_targets):
                return _IluV4W8A8LinearMethod(self)
            # v1 fallthrough -- BF16 (UnquantizedLinearMethod) for every
            # Linear that hasn't been claimed by routed/shared/dense/bmm.
            return UnquantizedLinearMethod()

        return None


class _IluV4W8A8LinearMethod(LinearMethodBase):
    """Thin W8A8 INT8 channel-symmetric linear method.

    Reuses ``CompressedTensorsW8A8Int8`` as the per-layer scheme so we get
    the same kernel selection / state layout as the compressed-tensors
    config path, without paying for the full ``CompressedTensorsConfig``
    plumbing.
    """

    def __init__(self, parent_config: IluDeepseekV4W4A8Config) -> None:
        # Stored for introspection only; the scheme below carries all the
        # state actually consulted at create_weights / apply time.
        self.parent_config = parent_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
            CompressedTensorsW8A8Int8,
        )

        scheme = CompressedTensorsW8A8Int8(
            strategy=QuantizationStrategy.CHANNEL.value,
            is_static_input_scheme=False,
            input_symmetric=True,
            is_w4a8_linear=False,
        )
        layer.scheme = scheme  # type: ignore[attr-defined]

        weight_loader = extra_weight_attrs.get("weight_loader")
        scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scheme = layer.scheme  # type: ignore[attr-defined]
        return scheme.apply_weights(layer, x, bias=bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scheme = getattr(layer, "scheme", None)
        if scheme is not None:
            scheme.process_weights_after_loading(layer)


class _IluV4W8A16BMMLinearMethod(LinearMethodBase):
    """W8A16 grouped-BMM linear method for ``attn.wo_a`` (V4 v2 only).

    DeepSeek-V4 collapses its multi-head latent down-projection into a
    grouped einsum::

        z = einsum("bhr,hdr->bhd", o_view, wo_a.weight.view(groups, N, K))

    A 2D ``ixf_ops.w8a8`` GEMM does not apply. We follow R1's
    ``_v_up_proj_w8a8`` / ``_k_up_proj_w8a8`` template
    (vllm/v1/attention/backends/mla/ilu_mla.py) and use
    ``ixf_ops.marlin_w8a16``: weight is stored as INT8 per-channel on disk
    (identical to the dense W8A8 path), but the input stays BF16 and the
    weight is repacked into Marlin layout in
    :meth:`process_weights_after_loading`.

    The GEMM itself is not invoked here -- the attention wrapper picks up
    ``layer.W_repacked`` / ``layer.W_scale_repacked`` and calls
    ``ixf_ops.marlin_w8a16(...)`` directly so the grouped 3D shape is
    preserved without going through ``LinearBase.forward``.
    """

    def __init__(self, parent_config: IluDeepseekV4W4A8Config) -> None:
        self.parent_config = parent_config
        self.gemm_format = parent_config.bmm_w8a16_gemm_format

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.parameter import (
            ChannelQuantScaleParameter,
            ModelWeightParameter,
        )

        out_total = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(out_total, input_size_per_partition, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((out_total, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Reshape + Marlin-repack ``wo_a.weight`` for grouped W8A16 GEMM.

        The attention wrapper attaches ``layer.bmm_geom`` (a SimpleNamespace
        with ``n_local_groups``, ``o_lora_rank``, ``heads_per_group``,
        ``head_dim``) at construction time. We use it to view the flat 2D
        ``[n_local_groups * o_lora_rank, heads_per_group * head_dim]`` weight
        into the grouped layout Marlin expects.

        Result attributes (on the layer):
        - ``W_repacked``       -- INT8 in Marlin layout, ready for marlin_w8a16
        - ``W_scale_repacked`` -- per-channel FP32 scale matching W_repacked
        - ``W_format``         -- copied from quant_config (e.g. "k16n16_grouped_n")
        - ``layer.weight`` / ``layer.weight_scale`` are deleted to free RAM.
        """
        import ixformer.inference.functions as ixf_ops

        bmm_geom = getattr(layer, "bmm_geom", None)
        assert bmm_geom is not None, (
            "_IluV4W8A16BMMLinearMethod requires layer.bmm_geom to be set "
            "by the attention wrapper before process_weights_after_loading."
        )
        n_local_groups: int = bmm_geom.n_local_groups
        o_lora_rank: int = bmm_geom.o_lora_rank
        K_per_group: int = bmm_geom.heads_per_group * bmm_geom.head_dim

        weight = layer.weight.data  # [n_local_groups * o_lora_rank, K_per_group]
        weight_scale = layer.weight_scale.data  # [n_local_groups * o_lora_rank, 1]

        assert weight.shape == (n_local_groups * o_lora_rank, K_per_group), (
            f"wo_a weight shape {tuple(weight.shape)} does not match "
            f"({n_local_groups} * {o_lora_rank}, {K_per_group}) implied by "
            f"bmm_geom={bmm_geom!r}"
        )
        assert weight_scale.shape == (n_local_groups * o_lora_rank, 1), (
            f"wo_a weight_scale shape {tuple(weight_scale.shape)} != "
            f"({n_local_groups * o_lora_rank}, 1)"
        )

        # einsum is "bhr, hdr -> bhd": ckpt weight is [groups, N=o_lora, K].
        # Marlin expects [groups, K, N] (K-axis-1, N-axis-2), so permute.
        w_grouped = (
            weight.view(n_local_groups, o_lora_rank, K_per_group)
            .permute(0, 2, 1)
            .contiguous()
        )  # [groups, K, N]
        # Per-row scale on disk == per-N (output channel) per group.
        s_grouped = weight_scale.view(
            n_local_groups, o_lora_rank, 1
        ).contiguous()  # [groups, N, 1]

        layer.W_repacked = ixf_ops.marlin_w8_weight_repack(
            weights=w_grouped,
            weight_format="int8",
            reformat=self.gemm_format,
        )
        layer.W_scale_repacked = s_grouped
        layer.W_format = self.gemm_format

        # Free the now-redundant flat parameters so the wrapper code only ever
        # reads W_repacked / W_scale_repacked.
        delattr(layer, "weight")
        delattr(layer, "weight_scale")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``LinearBase.forward`` should never reach here -- the attention
        # wrapper bypasses it and calls ``ixf_ops.marlin_w8a16`` directly on
        # ``layer.W_repacked`` / ``layer.W_scale_repacked``. Raise loudly if
        # something does try to invoke this Linear normally.
        raise NotImplementedError(
            "_IluV4W8A16BMMLinearMethod.apply must not be called; "
            "wo_a is consumed by IluDeepseekV4MultiHeadLatentAttentionWrapper "
            "via marlin_w8a16(W_repacked, W_scale_repacked)."
        )
