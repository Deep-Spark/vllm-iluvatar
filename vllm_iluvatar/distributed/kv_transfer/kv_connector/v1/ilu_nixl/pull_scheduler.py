# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull scheduler for the Ilu NIXL connector."""

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.logger import init_logger

from ..ilu_nixl_common.types import (
    encode_block_ids_rle,
    resolve_remote_block_ids,
)
from .base_scheduler import IluNixlBaseConnectorScheduler
from .fp8_layout import get_fp8_wire_block_size
from ..ilu_nixl_common.dsv41 import compute_prefix_replay_alignment
from .metadata import format_kv_group_page_log
from .pp_utils import infer_pp_split

logger = init_logger(__name__)

_DEFAULT_RLE_MIN_BLOCKS = 16


class IluNixlPullConnectorScheduler(
    IluNixlBaseConnectorScheduler, NixlPullConnectorScheduler
):
    """Pull scheduler that includes PP, V4.1 topology, E25 receipts, and RLE blocks."""

    def __init__(self, vllm_config, engine_id: str, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        transfer_cfg = getattr(vllm_config, "kv_transfer_config", None)
        is_consumer = (
            transfer_cfg is not None and transfer_cfg.kv_role == "kv_consumer"
        )
        get_extra = getattr(transfer_cfg, "get_from_extra_config", None)
        rle_enabled_raw = (
            True if get_extra is None else get_extra("enable_rle_block_ids", True)
        )
        self._rle_enabled = str(rle_enabled_raw).lower() not in ("0", "false", "no")
        self._rle_min_blocks = int(
            _DEFAULT_RLE_MIN_BLOCKS
            if get_extra is None
            else get_extra("rle_min_blocks", _DEFAULT_RLE_MIN_BLOCKS)
        )
        self._fp8_wire_group_ratios: list[int] = []
        if self.fp8_wire_staging_enabled:
            wire_bs = get_fp8_wire_block_size(vllm_config)
            if wire_bs is not None:
                for group in getattr(
                    self.kv_cache_config, "kv_cache_groups", ()
                ):
                    spec = group.kv_cache_spec
                    compute_bs = int(getattr(spec, "block_size", 0) or 0)
                    if (
                        compute_bs > wire_bs
                        and compute_bs % wire_bs == 0
                    ):
                        self._fp8_wire_group_ratios.append(compute_bs // wire_bs)
                    else:
                        self._fp8_wire_group_ratios.append(1)
        self.prefix_replay_alignment = compute_prefix_replay_alignment(
            vllm_config,
            kv_cache_config,
            is_consumer=is_consumer,
            has_mamba=self._has_mamba,
        )
        if self.prefix_replay_alignment is not None:
            logger.info(
                "IluNixl remote-prefix replay alignment: %s",
                self.prefix_replay_alignment,
            )

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        params = request.kv_transfer_params
        if params is not None:
            resolve_remote_block_ids(params)
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        if params is not None:
            resolve_remote_block_ids(params)
        capture_starts = (
            num_external_tokens > 0
            and bool(params)
            and bool(params.get("do_remote_prefill"))
            and bool(params.get("remote_block_ids"))
        )
        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            prefix_state.allocated(request, blocks, num_external_tokens)

        super().update_state_after_alloc(request, blocks, num_external_tokens)

        if capture_starts and request.request_id in self._reqs_need_recv:
            self._remote_block_starts[request.request_id] = (
                self.compute_remote_block_starts(blocks)
            )
        if self.fp8_wire_staging_enabled and params and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request
            self._fp8_staging_block_ids[request.request_id] = [
                list(group) for group in blocks.get_block_ids()
            ]
            raw_block_objs = getattr(blocks, "blocks", None)
            if raw_block_objs is not None and hasattr(self, "_fp8_block_hashes"):
                self._fp8_block_hashes[request.request_id] = [
                    {
                        int(blk.block_id): getattr(blk, "block_hash", None)
                        for blk in grp
                        if not getattr(blk, "is_null", False)
                    }
                    for grp in raw_block_objs
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
                if hasattr(self, "_fp8_block_hashes"):
                    self._fp8_block_hashes.pop(request.request_id, None)
                if hasattr(self, "_fp8_staged_blocks"):
                    self._fp8_staged_blocks.pop(request.request_id, None)

        remote_starts = getattr(self, "_remote_block_starts", None)
        if remote_starts is not None:
            remote_starts.pop(request.request_id, None)
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if params is not None:
            params["pp_size"] = self.vllm_config.parallel_config.pipeline_parallel_size
            params["pp_split"] = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(
                self.vllm_config
            )
            remote_ids = params.get("remote_block_ids")
            if remote_ids is not None:
                if (
                    self.fp8_wire_staging_enabled
                    and self._fp8_wire_group_ratios
                    and len(remote_ids) == len(self._fp8_wire_group_ratios)
                    and any(r > 1 for r in self._fp8_wire_group_ratios)
                ):
                    remote_ids = tuple(
                        [bid * ratio + sub for bid in grp for sub in range(ratio)]
                        if ratio > 1 and grp
                        else list(grp)
                        for grp, ratio in zip(
                            remote_ids, self._fp8_wire_group_ratios
                        )
                    )
                    params["remote_block_ids"] = remote_ids
                total_pages = sum(len(group) for group in remote_ids)
                logger.info(
                    "[PD-P] request_finished params req=%s "
                    "remote_block_ids_pages=%s pp_size=%s pp_split=%s",
                    request.request_id,
                    ",".join(str(len(group)) for group in remote_ids),
                    params.get("pp_size"),
                    params.get("pp_split"),
                )
                if (
                    getattr(self, "_rle_enabled", False)
                    and total_pages >= getattr(self, "_rle_min_blocks", _DEFAULT_RLE_MIN_BLOCKS)
                ):
                    params["remote_block_ids_rle"] = encode_block_ids_rle(remote_ids)
                    params["remote_block_ids"] = ()
        return delay_free_blocks, params
