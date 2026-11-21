# SPDX-License-Identifier: Apache-2.0
"""Serialized block-FP8 routed experts, including paired whole-layer checkpoints."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    resolve_fp8_moe_weight_block_shape,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.layers.fused_moe.activation import swiglustep_limit
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,  # noqa: F401
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
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

    This retained diagnostic reference is not used by the Method lifecycle.
    Historically it ran in ``process_weights_after_loading``,
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
    """Block-FP8 source -> paired loader -> MoEWeights -> shared local compute.

    Checkpoint tensor organization never selects execution. Model loaders adapt
    their names/views to the community per-expert callback. All original FP8
    experts use the same lifecycle and communication adapters as MXFP4.
    """

    def __init__(self, quant_config, layer, ilu_params=None):
        if not isinstance(quant_config, Fp8Config):
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=(ilu_params or {}).get(
                    "weight_block_size", [128, 128]
                ),
            )
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        if self.weight_block_size is None:
            raise ValueError("IluFp8MoEMethod requires block-FP8 weights")
        self.block_quant = True
        self.weight_scale_name = "weight_scale_inv"
        self.moe_block_shape, self.weight_scale_refine = (
            resolve_fp8_moe_weight_block_shape(
                self.moe,
                self.weight_block_size,
                kFp8Dynamic128Sym,
                self.quant_config.is_checkpoint_fp8_serialized,
            )
        )
        self.fp8_backend = self.moe_kernel = None
        self.selection = None
        self._checkpoint_loader = self._compute = self._runtime_weights = None
        self.supports_pre_processed_weights = False

    @property
    def is_monolithic(self):
        return False

    @property
    def supports_eplb(self):
        return False

    def _select_compute(self):
        if self.selection is None:
            from vllm_iluvatar.config.ilu_compute import compute_options
            from vllm_iluvatar.ilu_util.arch import cuda_arch

            self.selection = select_fp8_compute(
                cuda_arch(), compute_options("moe").get("mode", "auto")
            )
        return self.selection

    def _compute_config(self, layer):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEComputeConfig,
        )

        activation = getattr(layer.activation, "value", layer.activation)
        limit = getattr(layer, "swiglu_limit", None)
        if activation == "swiglustep":
            limit = swiglustep_limit(limit)
        return MoEComputeConfig(
            hidden=layer.moe_config.hidden_dim,
            intermediate=layer.moe_config.intermediate_size_per_partition,
            local_num_experts=layer.local_num_experts,
            global_num_experts=layer.global_num_experts,
            ep_rank=layer.moe_config.ep_rank,
            activation=activation,
            limit=limit,
            output_scale=float(_get_ilu_output_scale(layer)),
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )

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
            get_moe_backend,
        )
        from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
            BlockFp8LoadSpec,
            make_block_fp8_loader,
        )

        if self._checkpoint_loader is not None:
            raise RuntimeError("FP8 Method cannot create weights twice")
        if (
            self.moe.has_bias
            or not self.moe.is_act_and_mul
            or self.moe.hidden_dim_unpadded != hidden_size
            or self.moe.intermediate_size_per_partition_unpadded
            not in (None, intermediate_size_per_partition)
        ):
            raise ValueError(
                "Block-FP8 loading requires unpadded bias-free gated experts"
            )
        selection = self._select_compute()
        backend = get_moe_backend(selection)
        error = backend.support_error(self._compute_config(layer), selection)
        if error:
            raise ValueError(error)
        if backend.requires_contiguous_experts and layer.expert_map is not None:
            start = self.moe.ep_rank * num_experts
            expected = torch.full_like(layer.expert_map, -1)
            expected[start : start + num_experts] = torch.arange(
                num_experts, device=expected.device, dtype=expected.dtype
            )
            if not torch.equal(expected, layer.expert_map):
                raise ValueError("W4A8 compute requires contiguous expert placement")
        loader = make_block_fp8_loader(
            backend.weight_specs[selection.layout],
            BlockFp8LoadSpec(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                tuple(self.weight_block_size),
                self.moe.tp_size,
                self.moe.tp_rank,
            ),
        )
        layer.params_dtype = params_dtype
        self.num_experts = layer.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        loader.create_weights(layer, attrs)
        self._checkpoint_loader = loader

    def _bind_compute(self, layer, weights=None):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEWeights,
            make_moe_compute,
        )

        if self._compute is not None:
            raise RuntimeError("FP8 compute has already been bound")
        compute = make_moe_compute(self._compute_config(layer), self._select_compute())
        if weights is None:
            weights = MoEWeights.from_layer(layer, compute.weight_spec)
        compute.validate_weights(weights)
        self._runtime_weights, self._compute = weights, compute
        layer._ilu_mxfp4_ixmm = self.selection.name == "ixmm_fp4fp8"

    def process_weights_after_loading(self, layer):
        loader = self._checkpoint_loader
        if loader is None or loader._layer is not layer or self._compute is not None:
            raise RuntimeError("FP8 finalization requires its own newly loaded layer")
        self._bind_compute(layer, loader.finalize())

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return FusedMoEMethodBase.maybe_make_prepare_finalize(self, routing_tables)

    def get_fused_moe_quant_config(self, layer):
        if self._select_compute().name == "ixmm_fp4fp8":
            return mxfp4_w4a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                gemm1_clamp_limit=self._compute_config(layer).limit,
            )
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4",
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
            bind_compute_experts,
        )

        if self._compute is None:
            raise RuntimeError(
                "FP8 weights must be finalized before binding communication"
            )
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            _shrink_ll_max_tokens_to_decode(prepare_finalize)
        return bind_compute_experts(
            self._compute,
            self._runtime_weights,
            self.moe,
            self.moe_quant_config,
            prepare_finalize,
        )

    @property
    def supports_prequantized_input(self):
        # Available before weight loading, when models choose fused norm output.
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            get_moe_backend,
        )

        return get_moe_backend(self._select_compute()).supports_prequantized_int8

    @property
    def supports_packed_int8_input(self):
        return self.supports_prequantized_input

    @property
    def supports_decode_packed_fp8_dispatch(self):
        return self._compute is not None and self._compute.supports_prequantized_fp8

    @property
    def supports_prequant_fp8_dispatch(self):
        return self._compute is not None and self._compute.supports_prequantized_fp8

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self._compute is None:
            raise RuntimeError("FP8 weights have not been finalized")
        compute_kwargs = {}
        output_dtype = x.dtype if isinstance(x, torch.Tensor) else x[2]
        if isinstance(x, tuple):
            q, scale, _ = x
            packed = type(scale) is int
            valid_dtype = (
                q.dtype == torch.float8_e4m3fn
                and self._compute.supports_prequantized_fp8
            ) or (q.dtype == torch.int8 and self._compute.supports_prequantized_int8)
            if (
                not valid_dtype
                or q.ndim != 2
                or output_dtype not in (torch.float16, torch.bfloat16)
            ):
                raise ValueError(
                    "Selected compute cannot consume this prequantized input"
                )
            if packed:
                if (
                    scale != self._compute.hidden
                    or q.shape[1] < scale + 4
                    or not q.is_contiguous()
                ):
                    raise ValueError("Invalid inline activation scale")
            elif (
                q.shape[1] != self._compute.hidden
                or scale.dtype != torch.float32
                or scale.device != q.device
                or scale.numel() != q.shape[0]
            ):
                raise ValueError("Prequantized input requires one FP32 scale per token")
            else:
                scale = scale.reshape(-1)
            x = q
            compute_kwargs["quantized_input"] = (q, scale)
            compute_kwargs["output"] = torch.empty(
                (q.shape[0], self._compute.hidden), device=q.device, dtype=output_dtype
            )
        residual = shared_experts_input
        if residual is not None and residual.shape[0] != x.shape[0]:
            residual = None
        # Preserve the measured Step5 W13 tile choice without selecting a
        # different weight format, quantization lifecycle or compute backend.
        mode = (
            1
            if getattr(layer, "_ilu_step5_native_activation", False)
            and ilu_envs.VLLM_ILU_STEP5_FP4_SMALL_TILES
            and topk_ids.shape[1] == 8
            and 16 < topk_ids.shape[0] <= 64
            else None
        )
        workspace = self._compute.workspace_spec(
            topk_ids.numel(), output_dtype
        ).allocate(x.device)
        return self._compute.run_standard(
            x,
            self._runtime_weights,
            topk_weights,
            topk_ids,
            layer.expert_map,
            workspace=workspace,
            residual=residual,
            w13_gemm_mode=mode,
            **compute_kwargs,
        )


def select_fp8_compute(arch, mode="auto"):
    from vllm_iluvatar.config.ilu_compute import select_four_bit_expert_compute

    return select_four_bit_expert_compute(arch, mode, source="block-FP8")


def select_fp8_method(original, runtime):
    """Resolve explicit conversion before original-source compute preferences."""
    if runtime is not None:
        from vllm_iluvatar.config.ilu_compute import compute_options

        if any(value != "auto" for value in compute_options("moe").values()):
            raise ValueError(
                "Explicit ilu_quant MoE conversion cannot be combined with ilu_compute.moe overrides yet"
            )
        return runtime
    return original
