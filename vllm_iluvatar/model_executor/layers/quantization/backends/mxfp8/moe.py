# SPDX-License-Identifier: Apache-2.0
"""ivcore11 MXFP8 routed experts executed as IxInfer W4A8.

ivcore30 keeps the ix30 FP4 methods in ``backends/fp4/mxfp8_moe.py``. On
ivcore11 there is no ixmm, and the default expert backend is
``ixinfer_w4a8_nn_v2``; MXFP8 reaches it through the same lifecycle block-FP8
already uses, because MXFP8 is block-FP8 on a ``[1, 32]`` grid. Only the source
loader differs, so this module is the dispatch gate plus two overrides.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
    IluFp8MoEMethod,
)

logger = init_logger(__name__)


class IluMxFp8ToW4A8MoEMethod(IluFp8MoEMethod):
    """MXFP8 routed experts converted to the selected IxInfer W4A8 contract.

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
    def __init__(self, quant_config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
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
        logger.info_once("MXFP8 MoE execution backend: ixinfer_w4a8")

    def _make_loader(self, weight_spec, num_experts, hidden_size, intermediate):
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
            ),
        )


def mxfp8_w4a8_moe_method(quant_config, layer, prefix):
    """Claim MXFP8 routed experts on ivcore11; return None to defer."""
    from vllm_iluvatar.ilu_util.arch import IVCORE11, cuda_arch

    if not isinstance(layer, RoutedExperts):
        return None
    if cuda_arch() != IVCORE11:
        return None
    if quant_config.is_layer_excluded(prefix):
        return None
    return IluMxFp8ToW4A8MoEMethod(quant_config, layer)


__all__ = ["IluMxFp8ToW4A8MoEMethod", "mxfp8_w4a8_moe_method"]
