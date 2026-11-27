# SPDX-License-Identifier: Apache-2.0
"""Serialized block-FP8 routed experts, including paired whole-layer checkpoints."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    resolve_fp8_moe_weight_block_shape,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym

# Re-exported: ``ilu_runtime/moe/fp8_to_fp4.py`` and ``mxfp4_moe.py`` reach these
# through this module, which predates the shared binding mixin.
from vllm_iluvatar.model_executor.layers.fused_moe.activation import (  # noqa: F401
    swiglustep_limit,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    _ixmm_backend_available,  # noqa: F401
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
    _get_ilu_output_scale,  # noqa: F401
)
from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    MoEComputeBindingMixin,
    pad_online_w4a8_config,
)
from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    shrink_ll_max_tokens_to_decode as _shrink_ll_max_tokens_to_decode,  # noqa: F401
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


@IluvatarQuantRegistry.register_method
class IluFp8MoEMethod(MoEComputeBindingMixin, Fp8MoEMethod):
    """Block-FP8 source -> paired loader -> MoEWeights -> shared local compute.

    Checkpoint tensor organization never selects execution. Model loaders adapt
    their names/views to the community per-expert callback. All original FP8
    experts use the same lifecycle and communication adapters as MXFP4.

    ``compute_source`` and ``_make_loader`` are the only places the source
    checkpoint format enters; a sibling source overrides those two and inherits
    the rest of the lifecycle (see ``backends/mxfp8/moe.py``).
    """

    compute_source = "block-FP8"

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

    def _compute_config(self, layer):
        config = super()._compute_config(layer)
        return pad_online_w4a8_config(
            config,
            layer.moe_config.intermediate_size_per_partition_unpadded,
            self._select_compute(),
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

        if self._checkpoint_loader is not None:
            raise RuntimeError("FP8 Method cannot create weights twice")
        selection = self._select_compute()
        unpadded = self.moe.intermediate_size_per_partition_unpadded
        if (
            self.moe.has_bias
            or not self.moe.is_act_and_mul
            or self.moe.hidden_dim_unpadded != hidden_size
            or (
                selection.name not in ("ixinfer_w4a8", "ixmm_w4a8")
                and unpadded not in (None, intermediate_size_per_partition)
            )
        ):
            raise ValueError(
                f"{self.compute_source} loading requires unpadded bias-free "
                "gated experts"
            )
        backend = get_moe_backend(selection)
        compute = self._compute_config(layer)
        error = backend.support_error(compute, selection)
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
        loader = self._make_loader(
            backend.weight_specs[selection.layout],
            num_experts,
            hidden_size,
            intermediate_size_per_partition if unpadded is None else unpadded,
            compute,
        )
        layer.params_dtype = params_dtype
        self.num_experts = layer.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        loader.create_weights(layer, attrs)
        self._checkpoint_loader = loader

    def _make_loader(
        self, weight_spec, num_experts, hidden_size, intermediate, compute
    ):
        from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
            BlockFp8LoadSpec,
            make_block_fp8_loader,
        )

        return make_block_fp8_loader(
            weight_spec,
            BlockFp8LoadSpec(
                num_experts,
                hidden_size,
                intermediate,
                tuple(self.weight_block_size),
                self.moe.tp_size,
                self.moe.tp_rank,
                runtime_intermediate=compute.intermediate,
                w13_k_padded=compute.w13_k_padded,
                w2_n_padded=compute.w2_n_padded,
                w2_k_padded=compute.w2_k_padded,
            ),
        )

    def process_weights_after_loading(self, layer):
        loader = self._checkpoint_loader
        if loader is None or loader._layer is not layer or self._compute is not None:
            raise RuntimeError("FP8 finalization requires its own newly loaded layer")
        self._bind_compute(layer, loader.finalize())

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
