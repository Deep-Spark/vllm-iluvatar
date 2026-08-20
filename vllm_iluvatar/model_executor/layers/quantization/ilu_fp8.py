# SPDX-License-Identifier: Apache-2.0
"""Iluvatar FP8 linear and block-FP8 MoE quantization methods.

Corresponds to: vllm/model_executor/layers/quantization/fp8.py

Target   : vllm.model_executor.layers.quantization.fp8.Fp8Config
           (registered under the name ``fp8``), its serialized linear branch,
           and its block-wise ``Fp8MoEMethod`` routed-expert branch.
Baseline : main_9c12a81ac
Why      : the block-FP8 checkpoint (``quant_method="fp8"`` +
           ``weight_block_size=[128,128]`` + ``activation_scheme="dynamic"``)
           routes routed experts to community ``Fp8MoEMethod.__init__``, which
           calls ``select_fp8_moe_backend`` -- every candidate (FlashInfer /
           DeepGEMM / Triton / Marlin) is NVIDIA-only, so off NVIDIA it either
           raises or lands on a backend that crashes at runtime (same failure
           class as NVFP4 landing on MARLIN).
How      : ``@register_quantization_config("fp8")`` on a subclass whose
           ``get_quant_method`` returns ``IluFp8LinearMethod`` for serialized
           linear layers, ``IluFp8PerChannelOnlineLinearMethod`` for online
           linear layers, and ``IluFp8MoEMethod`` for block-wise routed experts.
           Linear weights dispatch directly to FP8 GEMM/GEMV. The MoE method
           converts weights to ix30 FP4. DP+EP uses
           ``IluMoEBatchedExpertsFp8`` with FP8 activations; the legacy path calls
           ``ilu_mxfp4_fused_experts()``. Attention, KV cache and skipped layers
           delegate to ``super()``.
Scope    : online and serialized FP8 linear layers plus block-wise FP8 routed
           experts on Iluvatar (ivcore30).
Remove   : when community exposes a vendor FP8 MoE backend hook, or a native
           grouped block-FP8 GEMM lands (plan phase 2, removing the fp8 -> fp4
           requant loss).

Reusing the MXFP4/NVFP4 ix30 compute
------------------------------------
``ilu_mxfp4_fused_experts(backend="ixmm")`` consumes the ix30 packed format
(``uint8 [E, k/2, n]`` weight + ``float32 [E, k/256, n]`` running-acc scale).
That format is produced by dequantising the checkpoint and re-quantising per
256-element group -- a step agnostic to the *source* scale protocol. Block-FP8
(per-128x128 f32 ``weight_scale_inv``) therefore yields ix30 tensors byte
compatible with the ones the MXFP4/NVFP4 compute path already accepts, so the
whole GEMM / SwiGLU / gather pipeline is reused with no change; the only
FP8-specific work is the block dequant in
``preprocess_fp8_block_weight_for_ixmm`` below. Both legacy and modular DP+EP
paths consume this FP4 weight representation with FP8 activations.

Why the community name (not a new ``ilu_*``)
--------------------------------------------
The weights are the official block-FP8 release: their config resolves to ``fp8``
through community checkpoint detection. This is the "community format, Iluvatar
implementation" case, mirroring ``IluModelOptNvFp4Config`` covering
``modelopt_fp4`` and ``IluDeepseekV4FP8Config`` covering ``deepseek_v4_fp8``.
``register_quantization_config`` permits shadowing a built-in name;
``get_quantization_config`` resolves the customized map last, so our class wins.
Since the shadow covers *all* ``fp8`` models, ``get_quant_method`` explicitly
narrows the overrides to serialized linear and block-wise routed-expert branches.
"""

from __future__ import annotations

import copy
from typing import Any

import ixformer.inference.functions as ixfops
import torch
from ixformer.inference.functions.fp8 import fp8_per_token_dynamic_scaled
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.online.fp8 import _Fp8OnlineLinearBase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.ilu_modelopt_nvfp4 import (
    _SWIGLUSTEP_CLAMP,
    _get_ilu_output_scale,
)

# Process this many experts per dequant/requant temporary. Dequanting w13 across
# all local experts at once would allocate several GB of fp32; mirror the
# NVFP4/MXFP4 methods' chunking (``ilu_modelopt_nvfp4.py:77``).
_IXMM_EXPERT_CHUNK = 4
_IXMM_GROUP_SIZE = 256
_FP4_MAX_MAG = 6.0
_IXMM_FP8_GEMV_MAX_M = 16

logger = init_logger(__name__)


@IluvatarQuantRegistry.register_method
class IluFp8PerChannelOnlineLinearMethod(_Fp8OnlineLinearBase):
    """Load floating-point weights and quantize them for ixmm after loading.

    The public name follows vLLM's online per-channel method. ixmm's FP8 linear
    epilogue consumes one weight scale per output channel, so the materialized
    ``[N, K]`` weight is quantized row-wise.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: IluFp8Config) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.gemm_format = quant_config.gemm_format.upper()

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        weight_loader = extra_weight_attrs.get("weight_loader")
        scale = create_fp8_scale_parameter(
            ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        qweight, weight_scale = fp8_per_token_dynamic_scaled(layer.weight)
        layer._can_use_gemm = (
            qweight.shape[0] % 32 == 0
            if self.gemm_format == "TN"
            else qweight.shape[1] % 64 == 0
        )

        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        qx, scale_x = fp8_per_token_dynamic_scaled(x_2d)

        output = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            dtype=torch.bfloat16,
            device=x.device,
        )
        bias = None if bias is None else bias.to(torch.bfloat16)

        scale_w = layer.weight_scale
        m = qx.shape[0]
        can_use_gemm = getattr(layer, "_can_use_gemm", False)

        if can_use_gemm and m <= _IXMM_FP8_GEMV_MAX_M:
            ixfops.fp8_per_channel_gemv(
                qx,
                layer.weight,
                scale_x,
                scale_w,
                output,
                bias,
                format=self.gemm_format,
                backend="ixmm",
            )
        else:
            ixfops.fp8_per_channel_mm(
                qx, layer.weight, scale_x, scale_w, output, bias, backend="ixmm"
            )

        result = output.to(x.dtype) if x.dtype != torch.bfloat16 else output
        return result.view(*x.shape[:-1], layer.output_size_per_partition)


@IluvatarQuantRegistry.register_method
class IluFp8LinearMethod(Fp8LinearMethod):
    """Serialized FP8 linear method backed directly by GEMM/GEMV.

    Checkpoint weights remain in their native TN layout, ``[N, K]``. Block
    quantization dispatches to DeepGEMM; non-block checkpoints are exposed
    to per-channel epilogue.
    """

    def __init__(self, quant_config: IluFp8Config) -> None:
        super().__init__(quant_config)
        self.gemm_format = quant_config.gemm_format.upper()

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        weight_loader = extra_weight_attrs.get("weight_loader")
        scale = create_fp8_scale_parameter(
            ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
        )
        replace_parameter(layer, "weight_scale", None)
        layer.register_parameter("weight_scale", scale)
        weight = layer.weight
        layer._can_use_gemm = (
            weight.shape[0] % 32 == 0
            if self.gemm_format == "TN"
            else weight.shape[1] % 64 == 0
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Keep weights unchanged and materialize per-channel scales."""

        scale = layer.weight_scale.to(torch.float32).reshape(-1)
        n = layer.output_size_per_partition
        if scale.numel() == n:
            channel_scale = scale.contiguous()
        elif scale.numel() == 1:
            channel_scale = scale.expand(n).contiguous()
        elif scale.numel() == len(layer.logical_widths):
            channel_scale = torch.repeat_interleave(
                scale,
                torch.tensor(
                    layer.logical_widths,
                    dtype=torch.int64,
                    device=scale.device,
                ),
            ).contiguous()
        else:
            raise ValueError(
                f"Cannot map weight_scale shape {tuple(layer.weight_scale.shape)} "
                f"to {n} output channels"
            )

        replace_parameter(layer, "weight_scale", channel_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        qx, scale_x = fp8_per_token_dynamic_scaled(x_2d)

        output = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            dtype=torch.bfloat16,
            device=x.device,
        )
        bias = None if bias is None else bias.to(torch.bfloat16)

        scale_w = layer.weight_scale
        m = qx.shape[0]
        can_use_gemm = getattr(layer, "_can_use_gemm", False)

        if can_use_gemm and m <= _IXMM_FP8_GEMV_MAX_M:
            ixfops.fp8_per_channel_gemv(
                qx,
                layer.weight,
                scale_x,
                scale_w,
                output,
                bias,
                format=self.gemm_format,
                backend="ixmm",
            )
        else:
            ixfops.fp8_per_channel_mm(
                qx, layer.weight, scale_x, scale_w, output, bias, backend="ixmm"
            )

        result = output.to(x.dtype) if x.dtype != torch.bfloat16 else output
        return result.view(*x.shape[:-1], layer.output_size_per_partition)


def preprocess_fp8_block_weight_for_ixmm(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int],
    allow_k_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one expert-chunk of block-FP8 weights to the ixmm ix30 format.

    Block-FP8 counterpart of ``preprocess_nvfp4_weight_for_ixmm``
    (``ilu_modelopt_nvfp4.py``): only step 2's dequant differs (a per-128x128
    float32 ``weight_scale_inv`` block scale instead of NVFP4's per-16 e4m3 block
    scale times a per-expert global). Steps 3-7 reuse ixformer's leaf helpers
    verbatim, which is what guarantees the output is byte-compatible with the
    MXFP4/NVFP4 ix30 tensors the compute path already accepts.

    Args:
        weight:           ``[E, n, k]`` float8_e4m3fn (un-packed).
        weight_scale_inv: ``[E, ceil(n/bn), ceil(k/bk)]`` float32 block scale.
        block:            ``(bn, bk)`` the ``weight_block_size`` (e.g. 128x128).

    Returns:
        ixmm_weight: ``[E, k/2, n]`` uint8 in ix30 packed layout.
        ixmm_scale:  ``[E, k/256, n]`` float32 running-acc ratio scale.
    """
    # Lazy import keeps this module importable without ixformer (CPU-only unit
    # tests), matching the NVFP4/MXFP4 methods' lazy-import discipline.
    from ixformer.inference.functions.fp4.gemm import gemm as ixmm_gemm

    expert, n_dim, k = weight.shape
    if allow_k_padding:
        padded_k = (k + _IXMM_GROUP_SIZE - 1) // _IXMM_GROUP_SIZE * _IXMM_GROUP_SIZE
    else:
        assert k % _IXMM_GROUP_SIZE == 0, (
            f"k={k} must be a multiple of {_IXMM_GROUP_SIZE}"
        )
        padded_k = k

    # 1-2. Block dequant: fp8 magnitude * per-block float32 scale. Expand the
    # block scale back to element granularity, clipping the trailing partial
    # block when n / k are not exact multiples of the block size.
    block_n, block_k = block
    # Pin the block grid explicitly. Without this a transposed or mis-sized
    # scale can survive: ``repeat_interleave`` followed by a slice silently
    # truncates whenever the expanded tensor is longer than the weight.
    expected_scale_shape = (
        expert,
        (n_dim + block_n - 1) // block_n,
        (k + block_k - 1) // block_k,
    )
    assert tuple(weight_scale_inv.shape) == expected_scale_shape, (
        f"weight_scale_inv must be {expected_scale_shape} for a [{expert}, "
        f"{n_dim}, {k}] weight with block {block}, got "
        f"{tuple(weight_scale_inv.shape)}"
    )
    scale = weight_scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(block_n, dim=1)[:, :n_dim, :]
    scale = scale.repeat_interleave(block_k, dim=2)[:, :, :k]
    weight_float = weight.to(torch.float32) * scale

    # 3. Transpose [E, n, k] -> [E, k, n], then may zero-pad K to the ixmm
    # group boundary. The compute path pads the corresponding activation to
    # the packed weight's K, so the added columns contribute exactly zero.
    weight_float = weight_float.transpose(1, 2).contiguous()
    if padded_k != k:
        padded_weight = weight_float.new_zeros((expert, padded_k, n_dim))
        padded_weight[:, :k, :] = weight_float
        weight_float = padded_weight

    # 4. Per-256 absolute scale: amax / FP4_MAX_MAG.
    groups_256 = padded_k // _IXMM_GROUP_SIZE
    grouped = weight_float.reshape(expert, groups_256, _IXMM_GROUP_SIZE, n_dim)
    amax = grouped.abs().amax(dim=2)  # [E, groups, n]
    ixmm_scale = (amax / _FP4_MAX_MAG).clamp(min=1e-12)

    # 5. Re-quantise to fp4 nibbles [E, k, n].
    normalized = grouped / ixmm_scale.unsqueeze(2)
    new_nibbles = ixmm_gemm._quantize_fp4(normalized.reshape(expert, padded_k, n_dim))

    # 6. Pack into ix30 layout [E, k/2, n].
    ixmm_weight = ixmm_gemm._pack_nibbles_ix30(new_nibbles)

    # 7. Absolute scale -> running-acc ratio for the ixmm kernel.
    ixmm_scale = ixmm_gemm.preprocess_group_scale_for_running_acc(ixmm_scale)

    return ixmm_weight, ixmm_scale


@IluvatarQuantRegistry.register_method
class IluFp8MoEMethod(Fp8MoEMethod):
    """Block-FP8 routed-expert method backed by ``ilu_mxfp4_fused_experts()``."""

    # [ILU override] Fp8MoEMethod.__init__ — delta: skip the parent body (it
    # calls select_fp8_moe_backend, which off NVIDIA raises or lands on an
    # unusable backend) and mirror only the field initialisation the inherited
    # ``create_weights`` needs.
    #
    # ``experts_cls`` is deliberately left unset (not None) so community
    # ``FusedMoEMethodBase.is_monolithic`` never dereferences it; the explicit
    # ``is_monolithic`` override below makes the answer independent of that
    # subtlety regardless. See ``ilu_modelopt_nvfp4.py``.
    def __init__(
        self,
        quant_config: Fp8Config | Any,
        layer: RoutedExperts,
        ilu_params: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(quant_config, Fp8Config):
            ilu_params = ilu_params or {}
            weight_block_size = ilu_params.get("weight_block_size", [128, 128])
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=weight_block_size,
            )

        # Skip Fp8MoEMethod.__init__, run its base directly.
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        # Block quant stores per-block scales under ``*_weight_scale_inv``; the
        # inherited ``create_weights`` reads this name.
        self.weight_scale_name = "weight_scale_inv"
        # No community backend is selected: the ix30 path is our own compute.
        self.fp8_backend = None
        self.moe_kernel = None

        assert self.block_quant, (
            "IluFp8MoEMethod only handles block-wise FP8 routed experts; "
            "IluFp8Config.get_quant_method must not route per-tensor FP8 here"
        )
        assert not hasattr(self, "experts_cls"), (
            "IluFp8MoEMethod must not define experts_cls; community "
            "is_monolithic() would dereference it"
        )
        assert self.is_monolithic is False

    # [ILU override] FusedMoEMethodBase.is_monolithic — delta: constant False.
    # We always take the modular path (router first, then
    # ``RoutedExperts.forward_modular`` -> ``apply``).
    @property
    def is_monolithic(self) -> bool:
        return False

    # [ILU override] Fp8MoEMethod.supports_eplb — delta: community returns True.
    # Phase 1's ixmm expert loop records no expert load, so claiming EPLB
    # support would silently disable rebalancing; matches the NVFP4 method.
    @property
    def supports_eplb(self) -> bool:
        return False

    # [ILU override] Fp8MoEMethod.maybe_make_prepare_finalize — delta: None.
    # Community *raises* here (it assumes the modular-kernel init path);
    # Iluvatar dispatch/combine is the naive EP-group path, so opt out.
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        return FusedMoEMethodBase.maybe_make_prepare_finalize(self, routing_tables)

    # [ILU override] Fp8MoEMethod.get_fused_moe_quant_config — delta: None. A
    # ``FusedMoEQuantConfig`` only configures a community modular kernel, which
    # we do not build. Returning None also keeps ``topk_indices_dtype`` at None
    # so the router emits int32 ids -- the dtype ``ilu_mxfp4_fused_experts``
    # expects.
    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        activation = getattr(layer.activation, "value", layer.activation)
        swiglu_limit = _SWIGLUSTEP_CLAMP if activation == "swiglustep" else None
        return mxfp4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=swiglu_limit,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> FusedMoEExpertsModular:
        from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_batch_experts import (
            IluMoEBatchedExpertsFp8,
        )

        if prepare_finalize.activation_format != FusedMoEActivationFormat.BatchedExperts:
            raise NotImplementedError(
                "IluMoEBatchedExpertsFp8 requires the batched activation format, "
                f"got {prepare_finalize.activation_format}"
            )
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        if max_num_tokens is None:
            raise ValueError("batched FP8 experts require max_num_tokens")
        assert self.moe_quant_config is not None
        return IluMoEBatchedExpertsFp8(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            swiglu_limit=self.moe_quant_config.gemm1_clamp_limit,
        )

    # [ILU override] Fp8MoEMethod.process_weights_after_loading — delta: replace
    # community's ``select_fp8_moe_backend`` + kernel build with a block-FP8 ->
    # ix30 conversion, then drop the now-unused ``*_weight_scale_inv`` so they do
    # not stay resident (see plan §4). The dynamic activation scheme means there
    # are no ``*_input_scale`` params to drop.
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if not _ixmm_backend_available():
            raise RuntimeError(
                "IluFp8MoEMethod requires the ixmm backend (ix30 group_gemm_fp4 "
                "only accepts the preprocessed uint8 layout). Install the ixmm "
                "wheel and rebuild ixformer with HAVE_IXMM on ivcore30."
            )

        block = (self.weight_block_size[0], self.weight_block_size[1])

        for w_name, s_in_name, s_out_name in (
            ("w13_weight", "w13_weight_scale_inv", "w13_weight_scale"),
            ("w2_weight", "w2_weight_scale_inv", "w2_weight_scale"),
        ):
            w = getattr(layer, w_name)
            s = getattr(layer, s_in_name)
            assert w.dtype is torch.float8_e4m3fn, (
                f"{w_name} expected float8_e4m3fn, got {w.dtype}"
            )
            assert s.dtype is torch.float32, (
                f"{s_in_name} expected float32, got {s.dtype}"
            )

            weight_parts: list[torch.Tensor] = []
            scale_parts: list[torch.Tensor] = []
            allow_k_padding = w_name == "w2_weight"
            for begin in range(0, w.shape[0], _IXMM_EXPERT_CHUNK):
                end = min(begin + _IXMM_EXPERT_CHUNK, w.shape[0])
                ixmm_weight, ixmm_scale = preprocess_fp8_block_weight_for_ixmm(
                    w[begin:end],
                    s[begin:end],
                    block,
                    allow_k_padding,
                )
                weight_parts.append(ixmm_weight)
                scale_parts.append(ixmm_scale)

            ixmm_weight = torch.cat(weight_parts, dim=0).contiguous()
            ixmm_scale = torch.cat(scale_parts, dim=0).contiguous()
            if not allow_k_padding:
                # Un-packed input [E, n, k] -> ix30 weight [E, k/2, n] + scale
                # [E, k/256, n].
                expected_weight_shape = (w.shape[0], w.shape[2] // 2, w.shape[1])
                expected_scale_shape = (
                    w.shape[0],
                    w.shape[2] // _IXMM_GROUP_SIZE,
                    w.shape[1],
                )
            else:
                padded_k = (
                    (w.shape[2] + _IXMM_GROUP_SIZE - 1)
                    // _IXMM_GROUP_SIZE
                    * _IXMM_GROUP_SIZE
                )
                expected_weight_shape = (w.shape[0], padded_k // 2, w.shape[1])
                expected_scale_shape = (
                    w.shape[0],
                    padded_k // _IXMM_GROUP_SIZE,
                    w.shape[1],
                )

            if tuple(ixmm_weight.shape) != expected_weight_shape:
                raise ValueError(
                    f"preprocess_fp8_block_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_weight.shape)} for {w_name}, expected "
                    f"{expected_weight_shape}"
                )
            if tuple(ixmm_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"preprocess_fp8_block_weight_for_ixmm returned shape "
                    f"{tuple(ixmm_scale.shape)} for {s_in_name}, expected "
                    f"{expected_scale_shape}"
                )
            replace_parameter(layer, w_name, ixmm_weight)
            # ix30 scale lives under a new name (``*_weight_scale``); the
            # community ``*_weight_scale_inv`` is dropped below.
            layer.register_parameter(
                s_out_name,
                torch.nn.Parameter(ixmm_scale, requires_grad=False),
            )

        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)

        layer._ilu_mxfp4_ixmm = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [ILU override] Fp8MoEMethod.apply — delta: wiring only. Community delegates
    # to ``self.moe_kernel.apply``; we call the plugin's ixmm compute with
    # explicit tensors, selecting the Step-3.5 ``step3_swiglustep`` variant and
    # deriving its numeric clamp from ``layer.activation``. Block-FP8 routed
    # experts carry no bias, and the shared output arrives through
    # ``shared_experts_input`` (``extra_residual``), so ``shared_experts`` is
    # accepted for signature compatibility but unused.
    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = getattr(layer, "activation", "silu")
        swiglu_limit = _SWIGLUSTEP_CLAMP if activation == "swiglustep" else 0.0
        return ilu_mxfp4_fused_experts(
            hidden_states=x,
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            swiglu_limit=swiglu_limit,
            expert_map=layer.expert_map,
            output_scale=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
            w13_bias=None,
            w2_bias=None,
            backend="ixmm",
            swiglu_variant="step3_swiglustep",
        )


@register_quantization_config("fp8")
class IluFp8Config(Fp8Config):
    """FP8 config whose routed experts and linear layers use Iluvatar kernels.

    ``weight_block_size`` describes routed experts. The optional
    ``none_moe_weight_block_size`` independently selects block scaling for
    ordinary Linear layers; ``null`` selects per-channel scaling.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
        gemm_format: str = "TN",
        none_moe_use_weight_block_size: bool = False,
        none_moe_is_checkpoint_fp8_serialized: bool = False,
    ) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )
        self.gemm_format = gemm_format
        self.none_moe_use_weight_block_size = none_moe_use_weight_block_size
        self.none_moe_is_checkpoint_fp8_serialized = (
            none_moe_is_checkpoint_fp8_serialized
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        gemm_format = cls.get_from_keys_or(config, ["gemm_format"], "TN")
        gemm_format = gemm_format.upper()
        # for models that not the all weights are fp8, those layer's name are passed with ignored_layers
        # if we want to use fp8 for whose layers, we can do:
        #    1. offline quant those layers and remove those layer's names from ignored_layers,
        #       add none_moe_use_weight_block_size: False to config.json
        #    2. remove those layer's names from ignored_layers and add none_moe_is_checkpoint_fp8_serialized: False and
        #       none_moe_use_weight_block_size: False to config.json to enable online fp8 quantization for those layers
        # default to True so official vllm can use the default behavior.
        none_moe_use_weight_block_size = cls.get_from_keys_or(
            config, ["none_moe_use_weight_block_size"], True
        )
        none_moe_is_checkpoint_fp8_serialized = bool(
            cls.get_from_keys_or(
                config, ["none_moe_is_checkpoint_fp8_serialized"], True
            )
        )
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        if gemm_format != "TN":
            raise ValueError("IluFp8Config only supports tn gemm_format for now")
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
            gemm_format=gemm_format,
            none_moe_use_weight_block_size=none_moe_use_weight_block_size,
            none_moe_is_checkpoint_fp8_serialized=none_moe_is_checkpoint_fp8_serialized,
        )

    def _is_layer_skipped(self, prefix: str) -> bool:
        """Match exact exclusions and parent-module exclusions."""
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return True
        return any(prefix.startswith(f"{name}.") for name in self.ignored_layers)

    # [ILU override] Fp8Config.get_quant_method — delta: online LinearBase
    # layers use IluFp8PerTensorOnlineLinearMethod, serialized LinearBase layers
    # use IluFp8LinearMethod, and block-wise routed experts use IluFp8MoEMethod.
    # Skipped layers, attention and KV cache remain delegated to the community.
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                return UnquantizedLinearMethod()

            new_quant = copy.deepcopy(self)
            new_quant.weight_block_size = (
                self.weight_block_size if self.none_moe_use_weight_block_size else None
            )
            if (
                self.none_moe_is_checkpoint_fp8_serialized
                and self.is_checkpoint_fp8_serialized
            ):
                method = IluFp8LinearMethod(new_quant)
            else:
                method = IluFp8PerChannelOnlineLinearMethod(new_quant)
            return method

        if (
            isinstance(layer, RoutedExperts)
            and self.is_checkpoint_fp8_serialized
            and self.weight_block_size is not None
            and self.store_dtype != "mxfp4"
            and not self._is_layer_skipped(prefix)
        ):
            return IluFp8MoEMethod(self, layer)
        return super().get_quant_method(layer, prefix)
