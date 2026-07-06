# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull scheduler for the Ilu NIXL connector."""

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)

from .base_scheduler import IluNixlBaseConnectorScheduler
from .pp_utils import infer_pp_split


class IluNixlPullConnectorScheduler(
    NixlPullConnectorScheduler, IluNixlBaseConnectorScheduler
):
    """Pull scheduler that includes PP topology in transfer params."""

    set_xfer_handshake_metadata = (
        IluNixlBaseConnectorScheduler.set_xfer_handshake_metadata
    )
    _nixl_handshake_listener = staticmethod(
        IluNixlBaseConnectorScheduler._nixl_handshake_listener
    )
    on_new_request = IluNixlBaseConnectorScheduler.on_new_request
    build_connector_meta = IluNixlBaseConnectorScheduler.build_connector_meta

    def request_finished(self, request, block_ids):
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if params is not None:
            params["pp_size"] = self.vllm_config.parallel_config.pipeline_parallel_size
            params["pp_split"] = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(
                self.vllm_config
            )
        return delay_free_blocks, params
