# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull scheduler for the Ilu NIXL connector."""

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.logger import init_logger

from .base_scheduler import IluNixlBaseConnectorScheduler
from .metadata import format_kv_group_page_log
from .pp_utils import infer_pp_split

logger = init_logger(__name__)


class IluNixlPullConnectorScheduler(
    IluNixlBaseConnectorScheduler, NixlPullConnectorScheduler
):
    """Pull scheduler that includes PP topology in transfer params."""

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        params = request.kv_transfer_params
        if self.fp8_wire_staging_enabled and params and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request
            self._fp8_staging_block_ids[request.request_id] = [
                list(group) for group in blocks.get_block_ids()
            ]
            logger.info(
                "[PD-P] alloc req=%s prompt=%s groups=%s",
                request.request_id,
                request.num_prompt_tokens,
                format_kv_group_page_log(
                    self.kv_cache_config.kv_cache_groups,
                    self._fp8_staging_block_ids[request.request_id],
                    blocks_per_sw=self.blocks_per_sw,
                ),
            )

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params
        if self.fp8_wire_staging_enabled and params and params.get("do_remote_decode"):
            from vllm.v1.request import RequestStatus

            if request.status in (
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_STOPPED,
            ):
                exported = self._fp8_exported_block_ids.pop(request.request_id, None)
                if exported is None:
                    raise RuntimeError(
                        "Request finished before its FP8 wire staging export: "
                        f"{request.request_id}"
                    )
                block_ids = exported
                logger.info(
                    "[PD-P] publish remote_block_ids req=%s status=%s "
                    "pages=%s groups=%s",
                    request.request_id,
                    request.status.name,
                    ",".join(str(len(group)) for group in block_ids),
                    format_kv_group_page_log(
                        getattr(
                            getattr(self, "kv_cache_config", None),
                            "kv_cache_groups",
                            (),
                        ),
                        block_ids,
                        blocks_per_sw=getattr(self, "blocks_per_sw", None),
                    ),
                )
            else:
                self._fp8_staging_block_ids.pop(request.request_id, None)
                self._fp8_exported_block_ids.pop(request.request_id, None)

        delay_free_blocks, params = super().request_finished(request, block_ids)
        if params is not None:
            params["pp_size"] = self.vllm_config.parallel_config.pipeline_parallel_size
            params["pp_split"] = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(
                self.vllm_config
            )
            remote_ids = params.get("remote_block_ids")
            if remote_ids is not None:
                logger.info(
                    "[PD-P] request_finished params req=%s "
                    "remote_block_ids_pages=%s pp_size=%s pp_split=%s",
                    request.request_id,
                    ",".join(str(len(group)) for group in remote_ids),
                    params.get("pp_size"),
                    params.get("pp_split"),
                )
        return delay_free_blocks, params
