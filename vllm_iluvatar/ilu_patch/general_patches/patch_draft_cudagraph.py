# SPDX-License-Identifier: Apache-2.0
"""Let V2 DFlash/DSpark run eagerly while target decode graphs stay enabled.

Community DFlashSpeculator.init_cudagraph_manager selects graphs from the
target mode without checking SpeculativeConfig.enforce_eager. Wrap its input
mode in the plugin; DSpark inherits the same method.
"""

from vllm_iluvatar.ilu_patch import general_patch as p


def register():
    p.register_once(__name__, _apply, detail="V2 draft CUDA graphs honor enforce_eager")


def _apply():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm_iluvatar.v1.worker.gpu.spec_decode.draft_cudagraph import (
        make_init_cudagraph_manager,
    )

    DFlashSpeculator.init_cudagraph_manager = make_init_cudagraph_manager(
        DFlashSpeculator.init_cudagraph_manager
    )
