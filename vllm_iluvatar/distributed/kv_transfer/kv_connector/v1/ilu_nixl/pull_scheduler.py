# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull scheduler for the Ilu NIXL connector."""

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)

from .base_scheduler import IluNixlBaseConnectorScheduler


class IluNixlPullConnectorScheduler(
    NixlPullConnectorScheduler, IluNixlBaseConnectorScheduler
):
    """Pull scheduler that includes PP topology in transfer params."""

    def request_finished(self, request, block_ids):
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if params is not None:
            params["pp_size"] = self.vllm_config.parallel_config.pipeline_parallel_size
            params["pp_split"] = envs.VLLM_PP_LAYER_PARTITION or ""
        return delay_free_blocks, params
