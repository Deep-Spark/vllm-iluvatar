# SPDX-License-Identifier: Apache-2.0
"""Serialized MXFP8 routed experts, on whichever contract compute selected.

MXFP8 is block-FP8 on a ``[1, 32]`` grid, so it reaches the shared lifecycle
through the same paired streaming load: ``ixinfer_w4a8_nn_v2`` on ivcore11,
``ixmm_fp4fp8`` on ivcore30. The architecture enters only through
``select_four_bit_expert_compute``; nothing here branches on it.

Only the source loader differs from block-FP8, so this module is two overrides
on ``IluFp8MoEMethod``.
"""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
    IluFp8MoEMethod,
)


class IluMxFp8MoEMethod(IluFp8MoEMethod):
    """MXFP8 routed experts converted to the selected expert contract.

    Inherits the whole block-FP8 lifecycle -- backend selection, paired
    streaming load, ``finalize`` plus ``validate_weights``, compute binding and
    ``apply``. The source format enters only through ``compute_source`` and
    ``_make_loader``.
    """

    compute_source = "MXFP8"

    # [ILU override] IluFp8MoEMethod.__init__ -- delta: the source is MXFP8, so
    # there is no Fp8Config to synthesize and no community block-shape refining
    # to run. The grid is fixed at [1, 32] by the OCP format and the scale is
    # named ``weight_scale``, not DeepSeek's ``weight_scale_inv``.
    #
    # Takes ``moe_config`` rather than the layer because community's
    # ``ModelOptQuantConfigBase.get_quant_method`` instantiates
    # ``FusedMoEMethodCls(quant_config=..., moe_config=layer.moe_config)``.
    def __init__(self, quant_config, moe_config):
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        self.block_quant = True
        self.weight_scale_name = "weight_scale"
        # ``ilu_routed_experts`` reads these through ``getattr(..., None)``.
        # MXFP8 needs no refined grid: [1, 32] already divides every partition.
        self.moe_block_shape = tuple(self.weight_block_size)
        self.weight_scale_refine = None
        self.fp8_backend = self.moe_kernel = None
        self.selection = None
        self._checkpoint_loader = self._compute = self._runtime_weights = None
        self.supports_pre_processed_weights = False

    def _make_loader(
        self, weight_spec, num_experts, hidden_size, intermediate, compute
    ):
        from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
            BlockFp8LoadSpec,
        )
        from vllm_iluvatar.model_executor.layers.quantization.common.mxfp8_loader import (
            MXFP8_BLOCK_SHAPE,
            make_mxfp8_loader,
        )

        return make_mxfp8_loader(
            weight_spec,
            BlockFp8LoadSpec(
                num_experts,
                hidden_size,
                intermediate,
                MXFP8_BLOCK_SHAPE,
                self.moe.tp_size,
                self.moe.tp_rank,
                runtime_intermediate=compute.intermediate,
                w13_k_padded=compute.w13_k_padded,
                w2_n_padded=compute.w2_n_padded,
                w2_k_padded=compute.w2_k_padded,
            ),
        )


__all__ = ["IluMxFp8MoEMethod"]
