# SPDX-License-Identifier: Apache-2.0
"""ivcore30 backend for serialized block-FP8 routed experts."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    resolve_fp8_moe_weight_block_shape,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.layers.fused_moe.activation import swiglustep_limit
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,
    ilu_mxfp4_fused_experts,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp4.nvfp4_moe import (
    _get_ilu_output_scale,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

# Process this many experts per dequant/requant temporary. Dequanting w13 across
# all local experts at once would allocate several GB of fp32; mirror the
# NVFP4/MXFP4 methods' chunking (``nvfp4_moe.py:44``).
_IXMM_EXPERT_CHUNK = 4
_IXMM_GROUP_SIZE = 256
_FP4_MAX_MAG = 6.0
logger = init_logger(__name__)


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

    This is the bulk path: it runs once in ``process_weights_after_loading``,
    with the whole FP8 layer resident. MoE also has a load-time path
    (``ilu_fp8_to_fp4_moe.py``) that converts each checkpoint shard as it
    lands, through ixformer's ``preprocess_fp8_block_weight_for_ixmm`` op.
    Both produce the same bytes -- ``test_ilu_fp8_to_fp4_moe.py`` asserts it --
    so the two can be compared directly when debugging.

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
    from ixformer.inference.functions.ops.gemm.fp4.gemm import gemm as ixmm_gemm

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


def _shrink_ll_max_tokens_to_decode(
    prepare_finalize: FusedMoEPrepareAndFinalizeModular,
) -> None:
    """Size the batched expert layout for decode instead of prefill.

    The community all2all factory builds DeepEP LL with
    ``moe_config.max_num_tokens`` -- the prefill bound
    (``max_num_batched_tokens``). When LL only carries decode (a
    ``VLLM_ILU_DECODE_EP_BACKEND`` override on top of a non-LL configured
    backend) the real bound is the uniform decode batch, which is also what the
    NVSHMEM buffer in ``ilu_fused_moe_ep`` is built for. The batched layout
    holds ``max_tokens_per_rank * num_dispatchers`` rows per expert, so keeping
    the prefill bound costs GiBs of workspace and output per captured graph.
    """
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
        _DECODE_EP_BACKEND,
        _decode_ll_max_tokens_per_rank,
    )

    if _DECODE_EP_BACKEND != "deepep_low_latency":
        return

    current = getattr(prepare_finalize, "max_tokens_per_rank", None)
    decode_max = _decode_ll_max_tokens_per_rank()
    if current is None or decode_max >= current:
        return

    prepare_finalize.max_tokens_per_rank = decode_max
    logger.info_once(
        "Sized DeepEP LL batched experts for decode: max_tokens_per_rank "
        "%d -> %d, matching the NVSHMEM dispatch buffer.",
        current,
        decode_max,
    )


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

        self.weight_scale_refine: tuple[int, int] | None = None
        self.moe_block_shape = self.weight_block_size

        # Set weight key and activation key for kernel compatibility
        assert self.weight_block_size is not None
        # Adapt the checkpoint block shape to TP sharding before
        # building the weight key.
        self.moe_block_shape, self.weight_scale_refine = (
            resolve_fp8_moe_weight_block_shape(
                self.moe,
                self.weight_block_size,
                kFp8Dynamic128Sym,
                self.quant_config.is_checkpoint_fp8_serialized,
            )
        )

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
        swiglu_limit = (
            swiglustep_limit(getattr(layer, "swiglu_limit", None))
            if activation == "swiglustep"
            else None
        )
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
        from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp4.batch_experts import (
            IluMoEBatchedExpertsFp8,
        )

        if (
            prepare_finalize.activation_format
            != FusedMoEActivationFormat.BatchedExperts
        ):
            raise NotImplementedError(
                "IluMoEBatchedExpertsFp8 requires the batched activation format, "
                f"got {prepare_finalize.activation_format}"
            )
        _shrink_ll_max_tokens_to_decode(prepare_finalize)
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

        # Use the runtime (TP-refined) block shape, NOT the checkpoint one.
        # When TP slicing leaves ``intermediate_size_per_partition`` indivisible
        # by the checkpoint block, ``resolve_fp8_moe_weight_block_shape``
        # (mirrored in ``__init__``) refines the grid (e.g. 128 -> 32) and the
        # inherited ``create_weights`` + loader upsample ``weight_scale_inv``
        # onto that finer grid. Dequant stays mathematically identical (every
        # fine sub-block inherits its parent scale), but the element-wise scale
        # expansion below must use the grid the parameter actually carries --
        # passing the checkpoint block here would repeat-and-truncate into
        # silently wrong scales.
        block = (self.moe_block_shape[0], self.moe_block_shape[1])

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
        activation = getattr(activation, "value", activation)
        swiglu_limit = (
            swiglustep_limit(getattr(layer, "swiglu_limit", None))
            if activation == "swiglustep"
            else 0.0
        )
        # Step5 routes each token to distinct experts. For 17–64 gathered
        # rows, mode 1 can cover every W13 expert; smaller batches already use
        # the small tile automatically. W2 keeps the backend's default.
        w13_gemm_mode = (
            1
            if getattr(layer, "_ilu_step5_native_activation", False)
            and ilu_envs.VLLM_ILU_STEP5_FP4_SMALL_TILES
            and topk_ids.shape[1] == 8
            and 16 < topk_ids.shape[0] <= 64
            else None
        )
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
            w13_gemm_mode=w13_gemm_mode,
            swiglu_variant="step3_swiglustep",
        )
